import type { GrokSettings } from "../settings";
import { getDynamicHeaders } from "./headers";
import { getModelInfo, toGrokModel } from "./models";

export interface OpenAIToolDefinition {
  type?: string;
  function?: {
    name?: string;
    description?: string;
    parameters?: unknown;
  };
}

export type OpenAIToolChoice =
  | "auto"
  | "required"
  | "none"
  | {
      type?: string;
      function?: {
        name?: string;
      };
    };

export interface OpenAIChatToolCall {
  id?: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: unknown;
  };
}

export interface OpenAIChatContentItem {
  type: string;
  text?: string;
  image_url?: { url?: string };
}

export interface OpenAIChatMessage {
  role: string;
  content: string | OpenAIChatContentItem[] | OpenAIChatContentItem | null;
  tool_calls?: OpenAIChatToolCall[];
  tool_call_id?: string;
  name?: string;
}

export interface OpenAIChatRequestBody {
  model: string;
  messages: OpenAIChatMessage[];
  stream?: boolean;
  tools?: OpenAIToolDefinition[];
  tool_choice?: OpenAIToolChoice;
  parallel_tool_calls?: boolean;
  video_config?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
}

export const CONVERSATION_API = "https://grok.com/rest/app-chat/conversations/new";

function stringifyJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return String(value ?? "");
  }
}

function getTextPartsFromContent(
  content: OpenAIChatMessage["content"],
  images: string[],
): string[] {
  const parts: string[] = [];
  if (Array.isArray(content)) {
    for (const item of content) {
      if (item?.type === "text") {
        const t = item.text ?? "";
        if (t.trim()) parts.push(t);
      }
      if (item?.type === "image_url") {
        const url = item.image_url?.url;
        if (url) images.push(url);
      }
    }
    return parts;
  }
  if (content && typeof content === "object") {
    return getTextPartsFromContent([content], images);
  }
  const text = String(content ?? "");
  if (text.trim()) parts.push(text);
  return parts;
}

function buildToolPrompt(
  tools: OpenAIToolDefinition[],
  toolChoice?: OpenAIToolChoice,
  parallelToolCalls = true,
): string {
  if (!Array.isArray(tools) || tools.length === 0) return "";
  if (toolChoice === "none") return "";

  const lines: string[] = [
    "# Available Tools",
    "",
    "You have access to the following tools. To call a tool, output a <tool_call> block with a JSON object containing \"name\" and \"arguments\".",
    "",
    "Format:",
    "<tool_call>",
    "{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}",
    "</tool_call>",
    "",
  ];

  if (parallelToolCalls) {
    lines.push("You may make multiple tool calls in a single response by using multiple <tool_call> blocks.");
    lines.push("");
  }

  lines.push("## Tool Definitions");
  lines.push("");

  for (const tool of tools) {
    if ((tool?.type ?? "") !== "function") continue;
    const name = String(tool.function?.name ?? "").trim();
    const description = String(tool.function?.description ?? "").trim();
    const parameters = tool.function?.parameters;
    if (!name) continue;
    lines.push(`### ${name}`);
    if (description) lines.push(description);
    if (parameters !== undefined) lines.push(`Parameters: ${stringifyJson(parameters)}`);
    lines.push("");
  }

  if (toolChoice === "required") {
    lines.push("IMPORTANT: You MUST call at least one tool in your response. Do not respond with only text.");
  } else if (toolChoice && typeof toolChoice === "object") {
    const forcedName = String(toolChoice.function?.name ?? "").trim();
    if (forcedName) {
      lines.push(`IMPORTANT: You MUST call the tool "${forcedName}" in your response.`);
    }
  } else {
    lines.push(
      "Decide whether to call a tool based on the user's request. If you don't need a tool, respond normally with text only.",
    );
  }

  lines.push("");
  lines.push(
    "When you call a tool, you may include text before or after the <tool_call> blocks, but the tool call blocks must be valid JSON.",
  );
  return lines.join("\n");
}

function formatToolHistory(messages: OpenAIChatMessage[]): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = [];
  for (const msg of messages) {
    const role = String(msg.role ?? "");
    if (role === "assistant" && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
      const parts = getTextPartsFromContent(msg.content, []);
      for (const call of msg.tool_calls) {
        const name = String(call?.function?.name ?? "");
        const argsRaw = call?.function?.arguments;
        const args = typeof argsRaw === "string" ? argsRaw : stringifyJson(argsRaw);
        parts.push(`<tool_call>{"name":"${name}","arguments":${args}}</tool_call>`);
      }
      out.push({ role: "assistant", content: parts.join("\n") });
      continue;
    }
    if (role === "tool") {
      const toolName = String(msg.name ?? "unknown");
      const callId = String(msg.tool_call_id ?? "");
      let content = "";
      if (typeof msg.content === "string") content = msg.content;
      else if (msg.content !== null && msg.content !== undefined) content = stringifyJson(msg.content);
      out.push({ role: "user", content: `tool (${toolName}, ${callId}): ${content}` });
      continue;
    }
    out.push(msg);
  }
  return out;
}

export function extractContent(
  messages: OpenAIChatMessage[],
  opts?: {
    tools?: OpenAIToolDefinition[];
    toolChoice?: OpenAIToolChoice;
    parallelToolCalls?: boolean;
  },
): { content: string; images: string[] } {
  const images: string[] = [];
  const extracted: Array<{ role: string; text: string }> = [];
  const normalizedMessages =
    Array.isArray(opts?.tools) && opts.tools.length > 0 ? formatToolHistory(messages) : messages;

  for (const msg of normalizedMessages) {
    const role = msg.role ?? "user";
    const content = msg.content;
    const parts = getTextPartsFromContent(content, images);

    // Keep tool traces in transcript when assistant message only carries tool_calls.
    if (role === "assistant" && parts.length === 0 && Array.isArray(msg.tool_calls)) {
      for (const call of msg.tool_calls) {
        const name = String(call?.function?.name ?? call?.id ?? "tool");
        const argsRaw = call?.function?.arguments;
        const args = typeof argsRaw === "string" ? argsRaw : stringifyJson(argsRaw);
        parts.push(`[tool_call] ${name} ${args}`.trim());
      }
    }

    if (parts.length) extracted.push({ role, text: parts.join("\n") });
  }

  let lastUserIndex: number | null = null;
  for (let i = extracted.length - 1; i >= 0; i--) {
    if (extracted[i]!.role === "user") {
      lastUserIndex = i;
      break;
    }
  }

  const out: string[] = [];
  for (let i = 0; i < extracted.length; i++) {
    const role = extracted[i]!.role || "user";
    const text = extracted[i]!.text;
    if (i === lastUserIndex) out.push(text);
    else out.push(`${role}: ${text}`);
  }

  let combined = out.join("\n\n");
  if (!combined.trim() && images.length > 0) {
    combined = "Refer to the following content:";
  }
  const tools = opts?.tools ?? [];
  const toolPrompt = buildToolPrompt(tools, opts?.toolChoice, opts?.parallelToolCalls ?? true);
  if (toolPrompt) {
    combined = combined ? `${toolPrompt}\n\n${combined}` : toolPrompt;
  }

  return { content: combined, images };
}

export function buildConversationPayload(args: {
  requestModel: string;
  content: string;
  imgIds: string[];
  imgUris: string[];
  postId?: string;
  videoConfig?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
  settings: GrokSettings;
}): { payload: Record<string, unknown>; referer?: string; isVideoModel: boolean } {
  const { requestModel, content, imgIds, imgUris, postId, settings } = args;
  const cfg = getModelInfo(requestModel);
  const { grokModel, mode, isVideoModel } = toGrokModel(requestModel);

  if (cfg?.is_video_model) {
    if (!postId) throw new Error("视频模型缺少 postId（需要先创建 media post）");

    const aspectRatio = (args.videoConfig?.aspect_ratio ?? "").trim() || "3:2";
    const videoLengthRaw = Number(args.videoConfig?.video_length ?? 6);
    const videoLength = Number.isFinite(videoLengthRaw) ? Math.max(1, Math.floor(videoLengthRaw)) : 6;
    const resolution = (args.videoConfig?.resolution ?? "SD") === "HD" ? "HD" : "SD";
    const preset = (args.videoConfig?.preset ?? "normal").trim();

    let modeFlag = "--mode=custom";
    if (preset === "fun") modeFlag = "--mode=extremely-crazy";
    else if (preset === "normal") modeFlag = "--mode=normal";
    else if (preset === "spicy") modeFlag = "--mode=extremely-spicy-or-crazy";

    const prompt = `${String(content || "").trim()} ${modeFlag}`.trim();

    return {
      isVideoModel: true,
      referer: "https://grok.com/imagine",
      payload: {
        temporary: true,
        modelName: "grok-3",
        message: prompt,
        toolOverrides: { videoGen: true },
        enableSideBySide: true,
        responseMetadata: {
          experiments: [],
          modelConfigOverride: {
            modelMap: {
              videoGenModelConfig: {
                parentPostId: postId,
                aspectRatio,
                videoLength,
                videoResolution: resolution,
              },
            },
          },
        },
      },
    };
  }

  return {
    isVideoModel,
    payload: {
      temporary: settings.temporary ?? true,
      modelName: grokModel,
      message: content,
      fileAttachments: imgIds,
      imageAttachments: [],
      disableSearch: false,
      enableImageGeneration: true,
      returnImageBytes: false,
      returnRawGrokInXaiRequest: false,
      enableImageStreaming: true,
      imageGenerationCount: 2,
      forceConcise: false,
      toolOverrides: {},
      enableSideBySide: true,
      sendFinalMetadata: true,
      isReasoning: false,
      webpageUrls: [],
      disableTextFollowUps: true,
      responseMetadata: { requestModelDetails: { modelId: grokModel } },
      disableMemory: false,
      forceSideBySide: false,
      modelMode: mode,
      isAsyncChat: false,
    },
  };
}

export async function sendConversationRequest(args: {
  payload: Record<string, unknown>;
  cookie: string;
  settings: GrokSettings;
  referer?: string;
}): Promise<Response> {
  const { payload, cookie, settings, referer } = args;
  const headers = getDynamicHeaders(settings, "/rest/app-chat/conversations/new");
  headers.Cookie = cookie;
  if (referer) headers.Referer = referer;
  const body = JSON.stringify(payload);

  return fetch(CONVERSATION_API, { method: "POST", headers, body });
}
