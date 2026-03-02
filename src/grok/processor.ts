import type { GrokSettings, GlobalSettings } from "../settings";
import type { OpenAIToolChoice, OpenAIToolDefinition } from "./conversation";

type GrokNdjson = Record<string, unknown>;
type ParsedToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
  index?: number;
};
type ToolStreamEvent =
  | { kind: "text"; text: string }
  | { kind: "tool"; toolCall: ParsedToolCall };

type UsagePayload = {
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  input_tokens_details: {
    text_tokens: number;
    image_tokens: number;
  };
  completion_tokens_details?: {
    reasoning_tokens?: number;
    text_tokens?: number;
    audio_tokens?: number;
  };
};

const NDJSON_DEBUG_SAMPLE_LIMIT = 40;
const NDJSON_DEBUG_RAW_PREVIEW_LIMIT = 120;
const NDJSON_DEBUG_TOKEN_PREVIEW_LIMIT = 160;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function readWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  ms: number,
): Promise<ReadableStreamReadResult<Uint8Array> | { timeout: true }> {
  if (ms <= 0) return { timeout: true };
  return Promise.race([
    reader.read(),
    sleep(ms).then(() => ({ timeout: true }) as const),
  ]);
}

function makeChunkWithDelta(
  id: string,
  created: number,
  model: string,
  delta: Record<string, unknown>,
  finish_reason?: "stop" | "tool_calls" | "error" | null,
): string {
  const payload: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finish_reason ?? null,
      },
    ],
  };
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function makeChunk(
  id: string,
  created: number,
  model: string,
  content: string,
  finish_reason?: "stop" | "tool_calls" | "error" | null,
): string {
  return makeChunkWithDelta(
    id,
    created,
    model,
    content ? { role: "assistant", content } : {},
    finish_reason,
  );
}

function makeToolChunk(
  id: string,
  created: number,
  model: string,
  toolCalls: ParsedToolCall[],
): string {
  return makeChunkWithDelta(id, created, model, { tool_calls: toolCalls }, null);
}

function makeDone(): string {
  return "data: [DONE]\n\n";
}

function toImgProxyUrl(globalCfg: GlobalSettings, origin: string, path: string): string {
  const baseUrl = (globalCfg.base_url ?? "").trim() || origin;
  return `${baseUrl}/images/${path}`;
}

function buildVideoTag(src: string): string {
  return `<video src="${src}" controls="controls" width="500" height="300"></video>\n`;
}

function buildVideoPosterPreview(videoUrl: string, posterUrl?: string): string {
  const href = String(videoUrl || "").replace(/"/g, "&quot;");
  const poster = String(posterUrl || "").replace(/"/g, "&quot;");
  if (!href) return "";
  if (!poster) return `<a href="${href}" target="_blank" rel="noopener noreferrer">${href}</a>\n`;
  return `<a href="${href}" target="_blank" rel="noopener noreferrer" style="display:inline-block;position:relative;max-width:100%;text-decoration:none;">
  <img src="${poster}" alt="video" style="max-width:100%;height:auto;border-radius:12px;display:block;" />
  <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
    <span style="width:64px;height:64px;border-radius:9999px;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
      <span style="width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:18px solid #fff;margin-left:4px;"></span>
    </span>
  </span>
</a>\n`;
}

function buildVideoHtml(args: { videoUrl: string; posterUrl?: string; posterPreview: boolean }): string {
  if (args.posterPreview) return buildVideoPosterPreview(args.videoUrl, args.posterUrl);
  return buildVideoTag(args.videoUrl);
}

function base64UrlEncode(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function encodeAssetPath(raw: string): string {
  try {
    const u = new URL(raw);
    // Keep full URL (query etc.) to avoid lossy pathname-only encoding (some URLs may encode the real path in query).
    return `u_${base64UrlEncode(u.toString())}`;
  } catch {
    const p = raw.startsWith("/") ? raw : `/${raw}`;
    return `p_${base64UrlEncode(p)}`;
  }
}

function normalizeGeneratedAssetUrls(input: unknown): string[] {
  if (!Array.isArray(input)) return [];

  const out: string[] = [];
  for (const v of input) {
    if (typeof v !== "string") continue;
    const s = v.trim();
    if (!s) continue;
    if (s === "/") continue;

    try {
      const u = new URL(s);
      if (u.pathname === "/" && !u.search && !u.hash) continue;
    } catch {
      // ignore (path-style strings are allowed)
    }

    out.push(s);
  }

  return out;
}

const TOOL_CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;

function stripCodeFences(text: string): string {
  const cleaned = text.trim();
  if (!cleaned.startsWith("```")) return cleaned;
  return cleaned.replace(/^```[a-zA-Z0-9_-]*\s*/, "").replace(/\s*```$/, "").trim();
}

function extractJsonObject(text: string): string {
  const start = text.indexOf("{");
  if (start < 0) return text;
  const end = text.lastIndexOf("}");
  if (end < start) return text;
  return text.slice(start, end + 1);
}

function removeTrailingCommas(text: string): string {
  return text.replace(/,\s*([}\]])/g, "$1");
}

function balanceBraces(text: string): string {
  let open = 0;
  let close = 0;
  let inString = false;
  let escaped = false;
  for (const ch of text) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\" && inString) {
      escaped = true;
      continue;
    }
    if (ch === "\"") {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") open += 1;
    else if (ch === "}") close += 1;
  }
  if (open > close) return `${text}${"}".repeat(open - close)}`;
  return text;
}

function repairJson(raw: string): unknown | null {
  const cleaned = balanceBraces(
    removeTrailingCommas(extractJsonObject(stripCodeFences(raw)).replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\n/g, " ")),
  );
  try {
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

function toArgumentsString(input: unknown): string {
  if (typeof input === "string") return input;
  try {
    return JSON.stringify(input ?? {});
  } catch {
    return String(input ?? "");
  }
}

function validToolNames(tools?: OpenAIToolDefinition[]): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(tools)) return names;
  for (const tool of tools) {
    const name = String(tool?.function?.name ?? "").trim();
    if (name) names.add(name);
  }
  return names;
}

function parseToolCallBlock(raw: string, tools?: OpenAIToolDefinition[]): ParsedToolCall | null {
  if (!raw.trim()) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    parsed = repairJson(raw);
  }
  if (!parsed || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;
  const name = String(obj.name ?? "").trim();
  if (!name) return null;

  const allowed = validToolNames(tools);
  if (allowed.size > 0 && !allowed.has(name)) return null;

  const args = toArgumentsString(obj.arguments ?? {});
  return {
    id: `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
    type: "function",
    function: { name, arguments: args },
  };
}

function parseToolCalls(
  content: string,
  tools?: OpenAIToolDefinition[],
): { textContent: string | null; toolCalls: ParsedToolCall[] | null } {
  if (!content) return { textContent: content, toolCalls: null };
  const matches = [...content.matchAll(TOOL_CALL_RE)];
  if (matches.length === 0) return { textContent: content, toolCalls: null };

  const toolCalls: ParsedToolCall[] = [];
  for (const match of matches) {
    const block = String(match[1] ?? "").trim();
    const parsed = parseToolCallBlock(block, tools);
    if (parsed) toolCalls.push(parsed);
  }
  if (toolCalls.length === 0) return { textContent: content, toolCalls: null };

  const textParts: string[] = [];
  let lastEnd = 0;
  for (const match of matches) {
    const start = match.index ?? 0;
    const before = content.slice(lastEnd, start).trim();
    if (before) textParts.push(before);
    const full = String(match[0] ?? "");
    lastEnd = start + full.length;
  }
  const trailing = content.slice(lastEnd).trim();
  if (trailing) textParts.push(trailing);
  return { textContent: textParts.length > 0 ? textParts.join("\n") : null, toolCalls };
}

function isToolChoiceRequired(toolChoice?: OpenAIToolChoice): boolean {
  if (toolChoice === "required") return true;
  if (toolChoice && typeof toolChoice === "object") {
    const forced = String(toolChoice.function?.name ?? "").trim();
    return String(toolChoice.type ?? "").trim() === "function" && Boolean(forced);
  }
  return false;
}

function num(...values: unknown[]): number | null {
  for (const v of values) {
    const n = Number(v);
    if (Number.isFinite(n) && n >= 0) return n;
  }
  return null;
}

function zeroUsage(): UsagePayload {
  return {
    total_tokens: 0,
    input_tokens: 0,
    output_tokens: 0,
    input_tokens_details: { text_tokens: 0, image_tokens: 0 },
  };
}

function normalizeUsage(input: unknown): UsagePayload | null {
  if (!input || typeof input !== "object") return null;
  const src = input as Record<string, unknown>;

  const prompt = num(
    src.input_tokens,
    src.prompt_tokens,
    src.inputTokens,
    src.promptTokens,
    src.inputTokenCount,
  );
  const completion = num(
    src.output_tokens,
    src.completion_tokens,
    src.outputTokens,
    src.completionTokens,
    src.outputTokenCount,
  );
  const total = num(
    src.total_tokens,
    src.totalTokens,
    src.totalTokenCount,
    prompt !== null && completion !== null ? prompt + completion : null,
  );
  const textInput = num(
    (src.input_tokens_details as Record<string, unknown> | undefined)?.text_tokens,
    (src.prompt_tokens_details as Record<string, unknown> | undefined)?.text_tokens,
  );
  const imageInput = num(
    (src.input_tokens_details as Record<string, unknown> | undefined)?.image_tokens,
    (src.prompt_tokens_details as Record<string, unknown> | undefined)?.image_tokens,
  );
  const reasoning = num(
    (src.completion_tokens_details as Record<string, unknown> | undefined)?.reasoning_tokens,
    src.reasoning_tokens,
    src.reasoningTokens,
  );

  const hasAny =
    prompt !== null ||
    completion !== null ||
    total !== null ||
    textInput !== null ||
    imageInput !== null ||
    reasoning !== null;
  if (!hasAny) return null;

  const usage: UsagePayload = {
    total_tokens: total ?? 0,
    input_tokens: prompt ?? 0,
    output_tokens: completion ?? 0,
    input_tokens_details: {
      text_tokens: textInput ?? 0,
      image_tokens: imageInput ?? 0,
    },
  };
  if (reasoning !== null) {
    usage.completion_tokens_details = { reasoning_tokens: reasoning, text_tokens: 0, audio_tokens: 0 };
  }
  return usage;
}

function suffixPrefix(text: string, tag: string): number {
  if (!text || !tag) return 0;
  const maxKeep = Math.min(text.length, tag.length - 1);
  for (let keep = maxKeep; keep > 0; keep -= 1) {
    if (text.endsWith(tag.slice(0, keep))) return keep;
  }
  return 0;
}

function previewText(input: string, limit: number): string {
  const compact = input.replace(/\s+/g, " ").trim();
  if (!compact) return "";
  if (compact.length <= limit) return compact;
  return `${compact.slice(0, limit)}...`;
}

function buildNdjsonDebugSummary(lines: string[]): Record<string, unknown> {
  let parsedLines = 0;
  let parseErrors = 0;
  let thinkingTrueLines = 0;
  let thinkingFalseLines = 0;
  let tokenLines = 0;
  let webSearchLines = 0;
  let modelResponseLines = 0;
  const sample: Record<string, unknown>[] = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] ?? "";
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(line) as Record<string, unknown>;
      parsedLines += 1;
    } catch {
      parseErrors += 1;
      if (sample.length < NDJSON_DEBUG_SAMPLE_LIMIT) {
        sample.push({
          line: i + 1,
          parse_error: true,
          raw_preview: previewText(line, NDJSON_DEBUG_RAW_PREVIEW_LIMIT),
        });
      }
      continue;
    }

    const grok = (data.result as Record<string, unknown> | undefined)?.response as
      | Record<string, unknown>
      | undefined;
    const modelResponse = grok?.modelResponse as Record<string, unknown> | undefined;
    const rawToken = grok?.token;
    const tokenType = Array.isArray(rawToken) ? "array" : typeof rawToken;
    const isThinking = typeof grok?.isThinking === "boolean" ? grok.isThinking : undefined;
    if (isThinking === true) thinkingTrueLines += 1;
    if (isThinking === false) thinkingFalseLines += 1;

    if (typeof rawToken === "string" || Array.isArray(rawToken)) tokenLines += 1;
    if (modelResponse) modelResponseLines += 1;
    const webSearchResults = (grok?.webSearchResults as Record<string, unknown> | undefined)?.results;
    const webSearchCount = Array.isArray(webSearchResults) ? webSearchResults.length : 0;
    if (webSearchCount > 0) webSearchLines += 1;

    if (sample.length >= NDJSON_DEBUG_SAMPLE_LIMIT) continue;
    sample.push({
      line: i + 1,
      has_response: Boolean(grok),
      is_thinking: isThinking,
      message_tag: typeof grok?.messageTag === "string" ? grok.messageTag : undefined,
      has_token: tokenType === "string" ? Boolean(rawToken) : tokenType === "array",
      token_type: tokenType,
      token_preview: typeof rawToken === "string" ? previewText(rawToken, NDJSON_DEBUG_TOKEN_PREVIEW_LIMIT) : undefined,
      model:
        typeof (grok?.userResponse as Record<string, unknown> | undefined)?.model === "string"
          ? (grok?.userResponse as Record<string, unknown>).model
          : typeof modelResponse?.model === "string"
            ? modelResponse.model
            : undefined,
      has_model_message: typeof modelResponse?.message === "string",
      web_search_results: webSearchCount || undefined,
      has_error:
        Boolean((data.error as Record<string, unknown> | undefined)?.message) ||
        (typeof modelResponse?.error === "string" && modelResponse.error.trim().length > 0),
    });
  }

  return {
    total_lines: lines.length,
    parsed_lines: parsedLines,
    parse_errors: parseErrors,
    thinking_true_lines: thinkingTrueLines,
    thinking_false_lines: thinkingFalseLines,
    token_lines: tokenLines,
    web_search_lines: webSearchLines,
    model_response_lines: modelResponseLines,
    sample,
  };
}

export function createOpenAiStreamFromGrokNdjson(
  grokResp: Response,
  opts: {
    cookie: string;
    settings: GrokSettings;
    global: GlobalSettings;
    origin: string;
    requestedModel: string;
    tools?: OpenAIToolDefinition[];
    toolChoice?: OpenAIToolChoice;
    onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
  },
): ReadableStream<Uint8Array> {
  const { settings, global, origin } = opts;
  const fallbackModel =
    typeof opts.requestedModel === "string" && opts.requestedModel.trim()
      ? opts.requestedModel.trim()
      : "grok-4";
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);

  const filteredTags = (settings.filtered_tags ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const showThinking = settings.show_thinking !== false;

  const firstTimeoutMs = Math.max(0, (settings.stream_first_response_timeout ?? 30) * 1000);
  const chunkTimeoutMs = Math.max(0, (settings.stream_chunk_timeout ?? 120) * 1000);
  const totalTimeoutMs = Math.max(0, (settings.stream_total_timeout ?? 600) * 1000);

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const body = grokResp.body;
      if (!body) {
        controller.enqueue(encoder.encode(makeChunk(id, created, fallbackModel, "Empty response", "error")));
        controller.enqueue(encoder.encode(makeDone()));
        controller.close();
        return;
      }

      const reader = body.getReader();
      const startTime = Date.now();
      let finalStatus = 200;
      let lastChunkTime = startTime;
      let firstReceived = false;

      let currentModel = fallbackModel;
      let isImage = false;
      let isThinking = false;
      let thinkingFinished = false;
      let videoProgressStarted = false;
      let lastVideoProgress = -1;

      let buffer = "";
      const toolStreamEnabled =
        Array.isArray(opts.tools) && opts.tools.length > 0 && opts.toolChoice !== "none";
      let toolState: "text" | "tool" = "text";
      let toolBuffer = "";
      let toolPartial = "";
      let toolCallsSeen = false;
      let toolCallIndex = 0;
      const requireToolCall = isToolChoiceRequired(opts.toolChoice);

      const withToolIndex = (call: ParsedToolCall): ParsedToolCall => {
        if (typeof call.index === "number") return call;
        return { ...call, index: toolCallIndex++ };
      };

      const handleToolStream = (chunk: string): ToolStreamEvent[] => {
        const events: ToolStreamEvent[] = [];
        if (!chunk) return events;
        const startTag = "<tool_call>";
        const endTag = "</tool_call>";
        let data = `${toolPartial}${chunk}`;
        toolPartial = "";

        while (data) {
          if (toolState === "text") {
            const startIdx = data.indexOf(startTag);
            if (startIdx < 0) {
              const keep = suffixPrefix(data, startTag);
              const emit = keep > 0 ? data.slice(0, -keep) : data;
              if (emit) events.push({ kind: "text", text: emit });
              toolPartial = keep > 0 ? data.slice(-keep) : "";
              break;
            }
            const before = data.slice(0, startIdx);
            if (before) events.push({ kind: "text", text: before });
            data = data.slice(startIdx + startTag.length);
            toolState = "tool";
            continue;
          }

          const endIdx = data.indexOf(endTag);
          if (endIdx < 0) {
            const keep = suffixPrefix(data, endTag);
            const append = keep > 0 ? data.slice(0, -keep) : data;
            if (append) toolBuffer += append;
            toolPartial = keep > 0 ? data.slice(-keep) : "";
            break;
          }

          toolBuffer += data.slice(0, endIdx);
          data = data.slice(endIdx + endTag.length);
          const parsed = parseToolCallBlock(toolBuffer, opts.tools);
          if (parsed) {
            events.push({ kind: "tool", toolCall: withToolIndex(parsed) });
            toolCallsSeen = true;
          }
          toolBuffer = "";
          toolState = "text";
        }

        return events;
      };

      const flushToolStream = (): ToolStreamEvent[] => {
        const events: ToolStreamEvent[] = [];
        if (toolState === "text") {
          if (toolPartial) {
            events.push({ kind: "text", text: toolPartial });
            toolPartial = "";
          }
          return events;
        }
        const raw = `${toolBuffer}${toolPartial}`;
        const parsed = parseToolCallBlock(raw, opts.tools);
        if (parsed) {
          events.push({ kind: "tool", toolCall: withToolIndex(parsed) });
          toolCallsSeen = true;
        } else if (raw) {
          events.push({ kind: "text", text: `<tool_call>${raw}` });
        }
        toolBuffer = "";
        toolPartial = "";
        toolState = "text";
        return events;
      };

      const flushStop = () => {
        if (toolStreamEnabled && requireToolCall && !toolCallsSeen) {
          controller.enqueue(
            encoder.encode(
              makeChunk(
                id,
                created,
                currentModel,
                "tool_choice is 'required' but model did not produce tool_calls.",
                "error",
              ),
            ),
          );
          controller.enqueue(encoder.encode(makeDone()));
          return;
        }
        const finishReason = toolStreamEnabled && toolCallsSeen ? "tool_calls" : "stop";
        if (toolStreamEnabled) {
          for (const event of flushToolStream()) {
            if (event.kind === "text" && event.text) {
              controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, event.text)));
            } else if (event.kind === "tool") {
              controller.enqueue(encoder.encode(makeToolChunk(id, created, currentModel, [event.toolCall])));
            }
          }
        }
        controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, "", finishReason)));
        controller.enqueue(encoder.encode(makeDone()));
      };

      try {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const now = Date.now();
          const elapsed = now - startTime;
          if (!firstReceived && elapsed > firstTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          if (totalTimeoutMs > 0 && elapsed > totalTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          const idle = now - lastChunkTime;
          if (firstReceived && idle > chunkTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const perReadTimeout = Math.min(
            firstReceived ? chunkTimeoutMs : firstTimeoutMs,
            totalTimeoutMs > 0 ? Math.max(0, totalTimeoutMs - elapsed) : Number.POSITIVE_INFINITY,
          );

          const res = await readWithTimeout(reader, perReadTimeout);
          if ("timeout" in res) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const { value, done } = res;
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });

          let idx: number;
          while ((idx = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) continue;

            let data: GrokNdjson;
            try {
              data = JSON.parse(line) as GrokNdjson;
            } catch {
              continue;
            }

            firstReceived = true;
            lastChunkTime = Date.now();

            const err = (data as any).error;
            if (err?.message) {
              finalStatus = 500;
              controller.enqueue(
                encoder.encode(makeChunk(id, created, currentModel, `Error: ${String(err.message)}`, "stop")),
              );
              controller.enqueue(encoder.encode(makeDone()));
              if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
              controller.close();
              return;
            }

            const grok = (data as any).result?.response;
            if (!grok) continue;

            const userRespModel = grok.userResponse?.model;
            if (typeof userRespModel === "string" && userRespModel.trim()) currentModel = userRespModel.trim();

            // Video generation stream
            const videoResp = grok.streamingVideoGenerationResponse;
            if (videoResp) {
              const progress = typeof videoResp.progress === "number" ? videoResp.progress : 0;
              const videoUrl = typeof videoResp.videoUrl === "string" ? videoResp.videoUrl : "";
              const thumbUrl = typeof videoResp.thumbnailImageUrl === "string" ? videoResp.thumbnailImageUrl : "";

              if (progress > lastVideoProgress) {
                lastVideoProgress = progress;
                if (showThinking) {
                  let msg = "";
                  if (!videoProgressStarted) {
                    msg = `<think>视频已生成${progress}%\n`;
                    videoProgressStarted = true;
                  } else if (progress < 100) {
                    msg = `视频已生成${progress}%\n`;
                  } else {
                    msg = `视频已生成${progress}%</think>\n`;
                  }
                  controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, msg)));
                }
              }

              if (videoUrl) {
                const videoPath = encodeAssetPath(videoUrl);
                const src = toImgProxyUrl(global, origin, videoPath);

                let poster: string | undefined;
                if (thumbUrl) {
                  const thumbPath = encodeAssetPath(thumbUrl);
                  poster = toImgProxyUrl(global, origin, thumbPath);
                }

                controller.enqueue(
                  encoder.encode(
                    makeChunk(
                      id,
                      created,
                      currentModel,
                      buildVideoHtml({
                        videoUrl: src,
                        posterPreview: settings.video_poster_preview === true,
                        ...(poster ? { posterUrl: poster } : {}),
                      }),
                    ),
                  ),
                );
              }
              continue;
            }

            if (grok.imageAttachmentInfo) isImage = true;
            const rawToken = grok.token;

            if (isImage) {
              const modelResp = grok.modelResponse;
              if (modelResp) {
                const urls = normalizeGeneratedAssetUrls(modelResp.generatedImageUrls);
                if (urls.length) {
                  const linesOut: string[] = [];
                  for (const u of urls) {
                    const imgPath = encodeAssetPath(u);
                    const imgUrl = toImgProxyUrl(global, origin, imgPath);
                    linesOut.push(`![Generated Image](${imgUrl})`);
                  }
                  controller.enqueue(
                    encoder.encode(makeChunk(id, created, currentModel, linesOut.join("\n"), "stop")),
                  );
                  controller.enqueue(encoder.encode(makeDone()));
                  if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
                  controller.close();
                  return;
                }
              } else if (typeof rawToken === "string" && rawToken) {
                controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, rawToken)));
              }
              continue;
            }

            // Text chat stream
            if (Array.isArray(rawToken)) continue;
            if (typeof rawToken !== "string" || !rawToken) continue;
            let token = rawToken;

            if (filteredTags.some((t) => token.includes(t))) continue;

            const currentIsThinking = Boolean(grok.isThinking);
            const messageTag = grok.messageTag;

            if (thinkingFinished && currentIsThinking) continue;

            if (grok.toolUsageCardId && grok.webSearchResults?.results && Array.isArray(grok.webSearchResults.results)) {
              if (currentIsThinking) {
                if (showThinking) {
                  let appended = "";
                  for (const r of grok.webSearchResults.results) {
                    const title = typeof r.title === "string" ? r.title : "";
                    const url = typeof r.url === "string" ? r.url : "";
                    const preview = typeof r.preview === "string" ? r.preview.replace(/\n/g, "") : "";
                    appended += `\n- [${title}](${url} \"${preview}\")`;
                  }
                  token += `${appended}\n`;
                } else {
                  continue;
                }
              } else {
                continue;
              }
            }

            let content = token;
            if (messageTag === "header") content = `\n\n${token}\n\n`;

            let shouldSkip = false;
            if (!isThinking && currentIsThinking) {
              if (showThinking) content = `<think>\n${content}`;
              else shouldSkip = true;
            } else if (isThinking && !currentIsThinking) {
              if (showThinking) content = `\n</think>\n${content}`;
              thinkingFinished = true;
            } else if (currentIsThinking && !showThinking) {
              shouldSkip = true;
            }

            if (!shouldSkip) {
              if (toolStreamEnabled && !currentIsThinking) {
                for (const event of handleToolStream(content)) {
                  if (event.kind === "text" && event.text) {
                    controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, event.text)));
                  } else if (event.kind === "tool") {
                    controller.enqueue(encoder.encode(makeToolChunk(id, created, currentModel, [event.toolCall])));
                  }
                }
              } else {
                controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, content)));
              }
            }
            isThinking = currentIsThinking;
          }
        }

        if (toolStreamEnabled) {
          for (const event of flushToolStream()) {
            if (event.kind === "text" && event.text) {
              controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, event.text)));
            } else if (event.kind === "tool") {
              controller.enqueue(encoder.encode(makeToolChunk(id, created, currentModel, [event.toolCall])));
            }
          }
        }
        if (toolStreamEnabled && requireToolCall && !toolCallsSeen) {
          controller.enqueue(
            encoder.encode(
              makeChunk(
                id,
                created,
                currentModel,
                "tool_choice is 'required' but model did not produce tool_calls.",
                "error",
              ),
            ),
          );
          controller.enqueue(encoder.encode(makeDone()));
          if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
          controller.close();
          return;
        }
        controller.enqueue(
          encoder.encode(
            makeChunk(id, created, currentModel, "", toolStreamEnabled && toolCallsSeen ? "tool_calls" : "stop"),
          ),
        );
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } catch (e) {
        finalStatus = 500;
        controller.enqueue(
          encoder.encode(
            makeChunk(id, created, currentModel, `处理错误: ${e instanceof Error ? e.message : String(e)}`, "error"),
          ),
        );
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
      }
    },
  });
}

export async function parseOpenAiFromGrokNdjson(
  grokResp: Response,
  opts: {
    cookie: string;
    settings: GrokSettings;
    global: GlobalSettings;
    origin: string;
    requestedModel: string;
    tools?: OpenAIToolDefinition[];
    toolChoice?: OpenAIToolChoice;
    debugNdjsonSample?: boolean;
  },
): Promise<Record<string, unknown>> {
  const { global, origin, requestedModel, settings } = opts;
  const text = await grokResp.text();
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const debugNdjson = opts.debugNdjsonSample ? buildNdjsonDebugSummary(lines) : null;

  let content = "";
  let model = requestedModel;
  let usage: UsagePayload | null = null;
  for (const line of lines) {
    let data: GrokNdjson;
    try {
      data = JSON.parse(line) as GrokNdjson;
    } catch {
      continue;
    }

    const err = (data as any).error;
    if (err?.message) throw new Error(String(err.message));

    const grok = (data as any).result?.response;
    if (!grok) continue;

    if (!usage) {
      usage =
        normalizeUsage((data as any).usage) ||
        normalizeUsage((data as any).result?.usage) ||
        normalizeUsage(grok.usage) ||
        normalizeUsage(grok.modelResponse?.usage) ||
        normalizeUsage(grok.tokenUsage) ||
        null;
    }

    const videoResp = grok.streamingVideoGenerationResponse;
    if (videoResp?.videoUrl && typeof videoResp.videoUrl === "string") {
      const videoPath = encodeAssetPath(videoResp.videoUrl);
      const src = toImgProxyUrl(global, origin, videoPath);

      let poster: string | undefined;
      if (typeof videoResp.thumbnailImageUrl === "string" && videoResp.thumbnailImageUrl) {
        const thumbPath = encodeAssetPath(videoResp.thumbnailImageUrl);
        poster = toImgProxyUrl(global, origin, thumbPath);
      }

      content = buildVideoHtml({
        videoUrl: src,
        posterPreview: settings.video_poster_preview === true,
        ...(poster ? { posterUrl: poster } : {}),
      });
      model = requestedModel;
      break;
    }

    const modelResp = grok.modelResponse;
    if (!modelResp) continue;
    if (typeof modelResp.error === "string" && modelResp.error) throw new Error(modelResp.error);

    if (typeof modelResp.model === "string" && modelResp.model) model = modelResp.model;
    if (typeof modelResp.message === "string") content = modelResp.message;

    const rawUrls = modelResp.generatedImageUrls;
    const urls = normalizeGeneratedAssetUrls(rawUrls);
    if (urls.length) {
      for (const u of urls) {
        const imgPath = encodeAssetPath(u);
        const imgUrl = toImgProxyUrl(global, origin, imgPath);
        content += `\n![Generated Image](${imgUrl})`;
      }
      break;
    }

    // If upstream emits placeholder/empty generatedImageUrls in intermediate frames, keep scanning.
    if (Array.isArray(rawUrls)) continue;

    // For normal chat replies, the first modelResponse is enough.
    break;
  }

  let finishReason: "stop" | "tool_calls" = "stop";
  let messageContent: string | null = content;
  let toolCalls: ParsedToolCall[] | null = null;
  if (Array.isArray(opts.tools) && opts.tools.length > 0 && opts.toolChoice !== "none") {
    const parsed = parseToolCalls(content, opts.tools);
    if (parsed.toolCalls && parsed.toolCalls.length > 0) {
      finishReason = "tool_calls";
      toolCalls = parsed.toolCalls.map((call, index) => ({ ...call, index }));
      messageContent = parsed.textContent;
    }
  }
  if (isToolChoiceRequired(opts.toolChoice) && Array.isArray(opts.tools) && opts.tools.length > 0 && !toolCalls) {
    throw new Error("tool_choice is 'required' but model did not produce tool_calls.");
  }

  const message: Record<string, unknown> = { role: "assistant", content: messageContent };
  if (toolCalls) message.tool_calls = toolCalls;

  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
      },
    ],
    usage: usage ?? zeroUsage(),
    ...(debugNdjson ? { _debug_ndjson: debugNdjson } : {}),
  };
}
