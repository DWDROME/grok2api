import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Env } from "../env";
import { requireApiAuth } from "../auth";
import { getSettings, normalizeCfCookie } from "../settings";
import { isValidModel, MODEL_CONFIG } from "../grok/models";
import {
  extractContent,
  buildConversationPayload,
  sendConversationRequest,
  type OpenAIToolChoice,
  type OpenAIToolDefinition,
} from "../grok/conversation";
import { uploadImage } from "../grok/upload";
import { getDynamicHeaders } from "../grok/headers";
import { createMediaPost, createPost } from "../grok/create";
import { createOpenAiStreamFromGrokNdjson, parseOpenAiFromGrokNdjson } from "../grok/processor";
import {
  IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
  generateImagineWs,
  resolveAspectRatio,
  resolveImageGenerationMethod,
  sendExperimentalImageEditRequest,
} from "../grok/imagineExperimental";
import { addRequestLog } from "../repo/logs";
import { applyCooldown, recordTokenFailure, selectBestToken } from "../repo/tokens";
import type { ApiAuthInfo } from "../auth";
import { getApiKeyLimits } from "../repo/apiKeys";
import { localDayString, tryConsumeDailyUsage, tryConsumeDailyUsageMulti } from "../repo/apiKeyUsage";
import { nextLocalMidnightExpirationSeconds } from "../kv/cleanup";
import { nowMs } from "../utils/time";
import { arrayBufferToBase64 } from "../utils/base64";
import { upsertCacheRow } from "../repo/cache";

function openAiError(message: string, code: string): Record<string, unknown> {
  return { error: { message, type: "invalid_request_error", code } };
}

function getClientIp(req: Request): string {
  return (
    req.headers.get("CF-Connecting-IP") ||
    req.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
    "0.0.0.0"
  );
}

async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = [];
  const queue = items.slice();
  const workers = Array.from({ length: Math.max(1, limit) }, async () => {
    while (queue.length) {
      const item = queue.shift() as T;
      results.push(await fn(item));
    }
  });
  await Promise.all(workers);
  return results;
}

async function runTasksSettledWithLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  if (!items.length) return [];
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  let nextIndex = 0;
  const workerCount = Math.max(1, Math.min(Math.floor(limit || 1), items.length));
  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      const idx = nextIndex++;
      if (idx >= items.length) break;
      try {
        const value = await fn(items[idx] as T);
        results[idx] = { status: "fulfilled", value };
      } catch (reason) {
        results[idx] = { status: "rejected", reason };
      }
    }
  });
  await Promise.all(workers);
  return results;
}

export const openAiRoutes = new Hono<{ Bindings: Env; Variables: { apiAuth: ApiAuthInfo } }>();

openAiRoutes.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Authorization", "Content-Type", "X-API-Key", "X-Grok-Debug-NDJSON"],
    allowMethods: ["GET", "POST", "OPTIONS"],
    maxAge: 86400,
  }),
);

openAiRoutes.use("/*", requireApiAuth);

function parseIntSafe(v: string | undefined, fallback: number): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.floor(n);
}

function quotaError(bucket: string): Record<string, unknown> {
  return openAiError(`Daily quota exceeded: ${bucket}`, "daily_quota_exceeded");
}

function isContentModerationMessage(message: string): boolean {
  const m = String(message || "").toLowerCase();
  return (
    m.includes("content moderated") ||
    m.includes("content-moderated") ||
    m.includes("wke=grok:content-moderated")
  );
}

async function enforceQuota(args: {
  env: Env;
  apiAuth: ApiAuthInfo;
  model: string;
  kind: "chat" | "image" | "video";
  imageCount?: number;
}): Promise<{ ok: true } | { ok: false; resp: Response }> {
  const key = args.apiAuth.key;
  if (!key) return { ok: true };
  if (args.apiAuth.is_admin) return { ok: true };

  const limits = await getApiKeyLimits(args.env.DB, key);
  if (!limits) return { ok: true };

  const tz = parseIntSafe(args.env.CACHE_RESET_TZ_OFFSET_MINUTES, 480);
  const day = localDayString(nowMs(), tz);
  const atMs = nowMs();
  const jsonHeaders = { "content-type": "application/json; charset=utf-8" };

  if (args.model === "grok-4-heavy") {
    const ok = await tryConsumeDailyUsageMulti({
      db: args.env.DB,
      key,
      day,
      atMs,
      updates: [
        { field: "heavy_used", inc: 1, limit: limits.heavy_limit },
        { field: "chat_used", inc: 1, limit: limits.chat_limit },
      ],
    });
    if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("heavy/chat")), { status: 429, headers: jsonHeaders }) };
    return { ok: true };
  }

  if (args.kind === "video") {
    const ok = await tryConsumeDailyUsage({
      db: args.env.DB,
      key,
      day,
      atMs,
      field: "video_used",
      inc: 1,
      limit: limits.video_limit,
    });
    if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("video")), { status: 429, headers: jsonHeaders }) };
    return { ok: true };
  }

  if (args.kind === "image") {
    const inc = Math.max(1, Math.floor(Number(args.imageCount ?? 1) || 1));
    const ok = await tryConsumeDailyUsage({
      db: args.env.DB,
      key,
      day,
      atMs,
      field: "image_used",
      inc,
      limit: limits.image_limit,
    });
    if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("image")), { status: 429, headers: jsonHeaders }) };
    return { ok: true };
  }

  // chat
  const ok = await tryConsumeDailyUsage({
    db: args.env.DB,
    key,
    day,
    atMs,
    field: "chat_used",
    inc: 1,
    limit: limits.chat_limit,
  });
  if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("chat")), { status: 429, headers: jsonHeaders }) };
  return { ok: true };
}

function base64UrlEncodeString(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function encodeAssetPath(raw: string): string {
  try {
    const u = new URL(raw);
    return `u_${base64UrlEncodeString(u.toString())}`;
  } catch {
    const p = raw.startsWith("/") ? raw : `/${raw}`;
    return `p_${base64UrlEncodeString(p)}`;
  }
}

function toProxyUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/$/, "")}/images/${path}`;
}

type ImageResponseFormat = "url" | "base64" | "b64_json";

function resolveResponseFormat(raw: unknown, defaultMode: string): ImageResponseFormat | null {
  const fallback = String(defaultMode || "url").trim().toLowerCase();
  const candidate =
    typeof raw === "string" && raw.trim() ? raw.trim().toLowerCase() : fallback;
  if (candidate === "url" || candidate === "base64" || candidate === "b64_json") {
    return candidate;
  }
  return null;
}

function responseFieldName(format: ImageResponseFormat): ImageResponseFormat {
  return format;
}

function toBool(input: unknown): boolean {
  if (typeof input === "boolean") return input;
  if (typeof input === "number") return input === 1;
  if (typeof input !== "string") return false;
  const normalized = input.trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
}

function normalizeGeneratedImageUrls(input: unknown): string[] {
  if (!Array.isArray(input)) return [];
  return input
    .filter((u): u is string => typeof u === "string")
    .map((u) => u.trim())
    .filter((u) => Boolean(u && u !== "/"));
}

function dedupeImages(images: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of images) {
    if (!value || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

function pickImageResults(images: string[], n: number): string[] {
  if (images.length >= n) {
    const pool = images.slice();
    const picked: string[] = [];
    while (picked.length < n && pool.length) {
      const idx = Math.floor(Math.random() * pool.length);
      const [item] = pool.splice(idx, 1);
      if (item) picked.push(item);
    }
    return picked;
  }
  const picked = images.slice();
  while (picked.length < n) picked.push("error");
  return picked;
}

function normalizeImageMime(mime: string): string {
  const m = (mime || "").trim().toLowerCase();
  if (m === "image/jpg") return "image/jpeg";
  return m;
}

function mimeFromFilename(filename: string): string | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".webp")) return "image/webp";
  return null;
}

async function fetchImageAsBase64(args: {
  rawUrl: string;
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
}): Promise<string> {
  let url: URL;
  try {
    url = new URL(args.rawUrl);
  } catch {
    const p = args.rawUrl.startsWith("/") ? args.rawUrl : `/${args.rawUrl}`;
    url = new URL(`https://assets.grok.com${p}`);
  }

  const headers = getDynamicHeaders(args.settings, url.pathname || "/");
  headers.Cookie = args.cookie;
  delete headers["Content-Type"];
  headers.Accept = "image/avif,image/webp,image/*,*/*;q=0.8";
  headers["Sec-Fetch-Dest"] = "image";
  headers["Sec-Fetch-Mode"] = "no-cors";
  headers["Sec-Fetch-Site"] = "same-site";
  headers.Referer = "https://grok.com/";

  const resp = await fetch(url.toString(), { method: "GET", headers, redirect: "follow" });
  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`Image download failed: ${resp.status} ${txt.slice(0, 200)}`);
  }
  return arrayBufferToBase64(await resp.arrayBuffer());
}

async function convertRawUrlByFormat(
  rawUrl: string,
  responseFormat: ImageResponseFormat,
  args: {
    baseUrl: string;
    cookie: string;
    settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  },
): Promise<string> {
  if (responseFormat === "url") {
    return toProxyUrl(args.baseUrl, encodeAssetPath(rawUrl));
  }
  return fetchImageAsBase64({ rawUrl, cookie: args.cookie, settings: args.settings });
}

async function collectImageUrls(resp: Response): Promise<string[]> {
  const text = await resp.text();
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const allUrls: string[] = [];
  for (const line of lines) {
    let data: any;
    try {
      data = JSON.parse(line);
    } catch {
      continue;
    }
    const err = data?.error;
    if (err?.message) throw new Error(String(err.message));
    const grok = data?.result?.response;
    const urls = normalizeGeneratedImageUrls(grok?.modelResponse?.generatedImageUrls);
    if (urls.length) allUrls.push(...urls);
  }
  return allUrls;
}

function buildImageSse(event: string, data: Record<string, unknown>): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function createImageEventStream(args: {
  upstream: Response;
  responseFormat: ImageResponseFormat;
  baseUrl: string;
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  n: number;
  onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
}): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const responseField = responseFieldName(args.responseFormat);
  const targetIndex = args.n === 1 ? Math.floor(Math.random() * 2) : null;

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const startedAt = Date.now();
      const body = args.upstream.body;
      if (!body) {
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.close();
        return;
      }

      const reader = body.getReader();
      const finalImages: string[] = [];
      let buffer = "";
      let failed = false;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });
          let idx = buffer.indexOf("\n");
          while (idx >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) {
              idx = buffer.indexOf("\n");
              continue;
            }

            let data: any;
            try {
              data = JSON.parse(line);
            } catch {
              idx = buffer.indexOf("\n");
              continue;
            }

            const err = data?.error;
            if (err?.message) throw new Error(String(err.message));

            const resp = data?.result?.response ?? {};
            const progressInfo = resp.streamingImageGenerationResponse;
            if (progressInfo) {
              const imageIndex = Number(progressInfo.imageIndex ?? 0);
              const progress = Number(progressInfo.progress ?? 0);
              if (args.n === 1 && imageIndex !== targetIndex) {
                idx = buffer.indexOf("\n");
                continue;
              }
              const outIndex = args.n === 1 ? 0 : imageIndex;
              controller.enqueue(
                encoder.encode(
                  buildImageSse("image_generation.partial_image", {
                    type: "image_generation.partial_image",
                    [responseField]: "",
                    index: outIndex,
                    progress,
                  }),
                ),
              );
            }

            const rawUrls = normalizeGeneratedImageUrls(resp?.modelResponse?.generatedImageUrls);
            if (rawUrls.length) {
              for (const rawUrl of rawUrls) {
                const converted = await convertRawUrlByFormat(rawUrl, args.responseFormat, {
                  baseUrl: args.baseUrl,
                  cookie: args.cookie,
                  settings: args.settings,
                });
                finalImages.push(converted);
              }
            }
            idx = buffer.indexOf("\n");
          }
        }

        for (let i = 0; i < finalImages.length; i++) {
          if (args.n === 1 && i !== targetIndex) continue;
          const outIndex = args.n === 1 ? 0 : i;
          controller.enqueue(
            encoder.encode(
              buildImageSse("image_generation.completed", {
                type: "image_generation.completed",
                [responseField]: finalImages[i] ?? "",
                index: outIndex,
                usage: {
                  total_tokens: 50,
                  input_tokens: 25,
                  output_tokens: 25,
                  input_tokens_details: { text_tokens: 5, image_tokens: 20 },
                },
              }),
            ),
          );
        }
        if (args.onFinish) {
          await args.onFinish({ status: 200, duration: (Date.now() - startedAt) / 1000 });
        }
      } catch (e) {
        failed = true;
        console.error("Image stream processing failed:", e);
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.error(e);
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
        if (!failed) controller.close();
      }
    },
  });
}

function imageResponseData(field: ImageResponseFormat, values: string[]) {
  return values.map((v) => ({ [field]: v }));
}

function getTokenSuffix(token: string): string {
  return token.length >= 6 ? token.slice(-6) : token;
}

const IMAGE_GENERATION_MODEL_ID = "grok-imagine-1.0";
const IMAGE_EDIT_MODEL_ID = "grok-imagine-1.0-edit";

function parseImageCount(input: unknown): number {
  const raw = Number(input ?? 1);
  if (!Number.isFinite(raw)) return 1;
  return Math.max(1, Math.min(10, Math.floor(raw)));
}

function parseImagePrompt(input: unknown): string {
  return String(input ?? "").trim();
}

function parseImageModel(input: unknown, fallback: string): string {
  return String(input ?? fallback).trim() || fallback;
}

function parseImageStream(input: unknown): boolean {
  return toBool(input);
}

function parseImageSize(input: unknown): string {
  return String(input ?? "1024x1024").trim() || "1024x1024";
}

function parseImageConcurrencyOrError(
  input: unknown,
): { value: number } | { error: { message: string; code: string } } {
  if (input === undefined || input === null || String(input).trim() === "") {
    return { value: 1 };
  }
  const parsed = Number(input);
  if (!Number.isFinite(parsed)) {
    return {
      error: { message: "concurrency must be between 1 and 3", code: "invalid_concurrency" },
    };
  }
  const value = Math.floor(parsed);
  if (value < 1 || value > 3) {
    return {
      error: { message: "concurrency must be between 1 and 3", code: "invalid_concurrency" },
    };
  }
  return { value };
}

function parseAllowedImageMime(file: File): string | null {
  const byMime = normalizeImageMime(String(file.type || ""));
  if (byMime === "image/png" || byMime === "image/jpeg" || byMime === "image/webp") return byMime;
  const byName = mimeFromFilename(String(file.name || ""));
  if (byName) return byName;
  return null;
}

function buildCookie(token: string, cf: string): string {
  return cf ? `sso-rw=${token};sso=${token};${cf}` : `sso-rw=${token};sso=${token}`;
}

async function runImageCall(args: {
  requestModel: string;
  prompt: string;
  fileIds: string[];
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  responseFormat: ImageResponseFormat;
  baseUrl: string;
}): Promise<string[]> {
  const { payload, referer } = buildConversationPayload({
    requestModel: args.requestModel,
    content: args.prompt,
    imgIds: args.fileIds,
    imgUris: [],
    settings: args.settings,
  });
  const upstream = await sendConversationRequest({
    payload,
    cookie: args.cookie,
    settings: args.settings,
    ...(referer ? { referer } : {}),
  });
  if (!upstream.ok) {
    const txt = await upstream.text().catch(() => "");
    throw new Error(`Upstream ${upstream.status}: ${txt.slice(0, 200)}`);
  }
  const rawUrls = await collectImageUrls(upstream);
  const converted = await Promise.all(
    rawUrls.map((rawUrl) =>
      convertRawUrlByFormat(rawUrl, args.responseFormat, {
        baseUrl: args.baseUrl,
        cookie: args.cookie,
        settings: args.settings,
      }),
    ),
  );
  return converted.filter(Boolean);
}

async function runImageStreamCall(args: {
  requestModel: string;
  prompt: string;
  fileIds: string[];
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
}): Promise<Response> {
  const { payload, referer } = buildConversationPayload({
    requestModel: args.requestModel,
    content: args.prompt,
    imgIds: args.fileIds,
    imgUris: [],
    settings: args.settings,
  });
  return sendConversationRequest({
    payload,
    cookie: args.cookie,
    settings: args.settings,
    ...(referer ? { referer } : {}),
  });
}

function imageGenerationMethod(settingsBundle: Awaited<ReturnType<typeof getSettings>>) {
  return resolveImageGenerationMethod(settingsBundle.grok.image_generation_method);
}

async function collectExperimentalGenerationImages(args: {
  prompt: string;
  n: number;
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  responseFormat: ImageResponseFormat;
  baseUrl: string;
  aspectRatio: string;
  concurrency: number;
}): Promise<string[]> {
  const calls = Math.ceil(Math.max(1, args.n) / 4);
  const plans = Array.from({ length: calls }, (_, i) => {
    const alreadyPlanned = i * 4;
    const chunkN = Math.max(1, Math.min(4, args.n - alreadyPlanned));
    return { chunkN };
  });

  const settled = await runTasksSettledWithLimit(
    plans,
    Math.min(plans.length, Math.max(1, args.concurrency || 1)),
    async (plan) =>
      generateImagineWs({
        prompt: args.prompt,
        n: plan.chunkN,
        cookie: args.cookie,
        settings: args.settings,
        aspectRatio: args.aspectRatio,
      }),
  );
  const rawUrls: string[] = [];
  for (const item of settled) {
    if (item.status === "fulfilled") rawUrls.push(...item.value);
  }
  if (!rawUrls.length) {
    const firstRejected = settled.find(
      (item): item is PromiseRejectedResult => item.status === "rejected",
    );
    if (firstRejected) throw firstRejected.reason;
    throw new Error("Experimental imagine websocket returned no images");
  }
  const dedupedRawUrls = dedupeImages(rawUrls);

  const converted = await Promise.all(
    dedupedRawUrls.map((rawUrl) =>
      convertRawUrlByFormat(rawUrl, args.responseFormat, {
        baseUrl: args.baseUrl,
        cookie: args.cookie,
        settings: args.settings,
      }),
    ),
  );
  return dedupeImages(converted.filter(Boolean));
}

async function runExperimentalImageEditCall(args: {
  prompt: string;
  fileUris: string[];
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  responseFormat: ImageResponseFormat;
  baseUrl: string;
}): Promise<string[]> {
  const upstream = await sendExperimentalImageEditRequest({
    prompt: args.prompt,
    fileUris: args.fileUris,
    cookie: args.cookie,
    settings: args.settings,
  });
  const rawUrls = await collectImageUrls(upstream);
  const converted = await Promise.all(
    rawUrls.map((rawUrl) =>
      convertRawUrlByFormat(rawUrl, args.responseFormat, {
        baseUrl: args.baseUrl,
        cookie: args.cookie,
        settings: args.settings,
      }),
    ),
  );
  return converted.filter(Boolean);
}

function createSyntheticImageEventStream(args: {
  selected: string[];
  responseField: ImageResponseFormat;
  onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
}): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const startedAt = Date.now();
      try {
        let emitted = false;
        for (let i = 0; i < args.selected.length; i++) {
          const value = args.selected[i];
          if (!value || value === "error") continue;
          emitted = true;

          controller.enqueue(
            encoder.encode(
              buildImageSse("image_generation.partial_image", {
                type: "image_generation.partial_image",
                [args.responseField]: "",
                index: i,
                progress: 100,
              }),
            ),
          );
          controller.enqueue(
            encoder.encode(
              buildImageSse("image_generation.completed", {
                type: "image_generation.completed",
                [args.responseField]: value,
                index: i,
                usage: {
                  total_tokens: 50,
                  input_tokens: 25,
                  output_tokens: 25,
                  input_tokens_details: { text_tokens: 5, image_tokens: 20 },
                },
              }),
            ),
          );
        }

        if (!emitted) {
          controller.enqueue(
            encoder.encode(
              buildImageSse("image_generation.completed", {
                type: "image_generation.completed",
                [args.responseField]: "error",
                index: 0,
                usage: {
                  total_tokens: 0,
                  input_tokens: 0,
                  output_tokens: 0,
                  input_tokens_details: { text_tokens: 0, image_tokens: 0 },
                },
              }),
            ),
          );
        }

        if (args.onFinish) {
          await args.onFinish({ status: 200, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.close();
      } catch (e) {
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.error(e);
      }
    },
  });
}

function createStreamErrorImageEventStream(args: {
  message: string;
  responseField: ImageResponseFormat;
  onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
}): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const startedAt = Date.now();
      try {
        controller.enqueue(
          encoder.encode(
            buildImageSse("image_generation.error", {
              type: "image_generation.error",
              message: args.message,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            buildImageSse("image_generation.completed", {
              type: "image_generation.completed",
              [args.responseField]: "error",
              index: 0,
              usage: {
                total_tokens: 0,
                input_tokens: 0,
                output_tokens: 0,
                input_tokens_details: { text_tokens: 0, image_tokens: 0 },
              },
            }),
          ),
        );
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.close();
      } catch (e) {
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.error(e);
      }
    },
  });
}

function createExperimentalImageEventStream(args: {
  prompt: string;
  n: number;
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  responseFormat: ImageResponseFormat;
  responseField: ImageResponseFormat;
  baseUrl: string;
  aspectRatio: string;
  concurrency: number;
  onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
}): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const safeN = Math.max(1, Math.floor(args.n || 1));
  const concurrency = Math.max(1, Math.min(3, Math.floor(args.concurrency || 1)));

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const startedAt = Date.now();
      const completedByIndex = new Map<number, string>();

      const emitPartial = (index: number, progress: number) => {
        if (index < 0 || index >= safeN) return;
        const pct = Math.max(0, Math.min(100, Number(progress) || 0));
        controller.enqueue(
          encoder.encode(
            buildImageSse("image_generation.partial_image", {
              type: "image_generation.partial_image",
              [args.responseField]: "",
              index,
              progress: pct,
            }),
          ),
        );
      };

      const emitCompleted = (index: number, value: string) => {
        if (index < 0 || index >= safeN) return;
        if (completedByIndex.has(index)) return;
        const finalValue = String(value || "").trim() || "error";
        completedByIndex.set(index, finalValue);
        const isError = finalValue === "error";
        controller.enqueue(
          encoder.encode(
            buildImageSse("image_generation.completed", {
              type: "image_generation.completed",
              [args.responseField]: finalValue,
              index,
              usage: {
                total_tokens: isError ? 0 : 50,
                input_tokens: isError ? 0 : 25,
                output_tokens: isError ? 0 : 25,
                input_tokens_details: {
                  text_tokens: isError ? 0 : 5,
                  image_tokens: isError ? 0 : 20,
                },
              },
            }),
          ),
        );
      };

      const toOutIndex = (offset: number, localIndex: number) =>
        Math.max(0, Math.min(safeN - 1, offset + Math.max(0, Math.floor(localIndex || 0))));

      try {
        const callCount = Math.ceil(safeN / 4);
        const plans = Array.from({ length: callCount }, (_, i) => {
          const offset = i * 4;
          const chunkN = Math.max(1, Math.min(4, safeN - offset));
          return { offset, chunkN };
        });

        const settled = await runTasksSettledWithLimit(
          plans,
          Math.min(plans.length, concurrency),
          async (plan) => {
            const rawUrls = await generateImagineWs({
              prompt: args.prompt,
              n: plan.chunkN,
              cookie: args.cookie,
              settings: args.settings,
              aspectRatio: args.aspectRatio,
              progressCb: ({ index, progress }) => {
                emitPartial(toOutIndex(plan.offset, index), progress);
              },
              completedCb: async ({ index, url }) => {
                const converted = await convertRawUrlByFormat(url, args.responseFormat, {
                  baseUrl: args.baseUrl,
                  cookie: args.cookie,
                  settings: args.settings,
                });
                if (converted) {
                  emitCompleted(toOutIndex(plan.offset, index), converted);
                }
              },
            });
            return { plan, rawUrls };
          },
        );

        for (const item of settled) {
          if (item.status !== "fulfilled") continue;
          const { plan, rawUrls } = item.value;
          for (let i = 0; i < rawUrls.length; i++) {
            const outIndex = toOutIndex(plan.offset, i);
            if (completedByIndex.has(outIndex)) continue;
            const converted = await convertRawUrlByFormat(rawUrls[i] ?? "", args.responseFormat, {
              baseUrl: args.baseUrl,
              cookie: args.cookie,
              settings: args.settings,
            });
            if (converted) {
              emitCompleted(outIndex, converted);
            }
          }
        }

        if (!Array.from(completedByIndex.values()).some((v) => v && v !== "error")) {
          try {
            const allImages = await collectExperimentalGenerationImages({
              prompt: args.prompt,
              n: safeN,
              cookie: args.cookie,
              settings: args.settings,
              responseFormat: args.responseFormat,
              baseUrl: args.baseUrl,
              aspectRatio: args.aspectRatio,
              concurrency,
            });
            const selected = pickImageResults(dedupeImages(allImages), safeN);
            for (let i = 0; i < selected.length; i++) {
              const value = selected[i] ?? "error";
              if (value !== "error") emitPartial(i, 100);
              emitCompleted(i, value);
            }
          } catch (fallbackErr) {
            const message = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
            controller.enqueue(
              encoder.encode(
                buildImageSse("image_generation.error", {
                  type: "image_generation.error",
                  message,
                }),
              ),
            );
          }
        }

        for (let i = 0; i < safeN; i++) {
          if (!completedByIndex.has(i)) {
            emitCompleted(i, "error");
          }
        }

        const success = Array.from(completedByIndex.values()).some((v) => v !== "error");
        if (args.onFinish) {
          await args.onFinish({ status: success ? 200 : 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.close();
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        controller.enqueue(
          encoder.encode(
            buildImageSse("image_generation.error", {
              type: "image_generation.error",
              message,
            }),
          ),
        );
        if (!completedByIndex.has(0)) emitCompleted(0, "error");
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.close();
      }
    },
  });
}

function streamHeaders(): Record<string, string> {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
  };
}

function isValidImageModel(model: string): boolean {
  if (!isValidModel(model)) return false;
  const cfg = MODEL_CONFIG[model];
  return Boolean(cfg?.is_image_model);
}

function invalidResponseFormatMessage(): string {
  return "response_format must be one of [\"b64_json\", \"base64\", \"url\"]";
}

function invalidStreamNMessage(): string {
  return "Streaming is only supported when n=1 or n=2";
}

function imageUsagePayload(values: string[]) {
  return {
    total_tokens: 0 * values.filter((v) => v !== "error").length,
    input_tokens: 0,
    output_tokens: 0 * values.filter((v) => v !== "error").length,
    input_tokens_details: { text_tokens: 0, image_tokens: 0 },
  };
}

function createdTs(): number {
  return Math.floor(Date.now() / 1000);
}

function buildImageJsonPayload(field: ImageResponseFormat, values: string[]) {
  return {
    created: createdTs(),
    data: imageResponseData(field, values),
    usage: imageUsagePayload(values),
  };
}

async function recordImageLog(args: {
  env: Env;
  ip: string;
  model: string;
  start: number;
  keyName: string;
  status: number;
  tokenSuffix?: string;
  error: string;
}) {
  const duration = (Date.now() - args.start) / 1000;
  await addRequestLog(args.env.DB, {
    ip: args.ip,
    model: args.model,
    duration: Number(duration.toFixed(2)),
    status: args.status,
    key_name: args.keyName,
    token_suffix: args.tokenSuffix ?? "",
    error: args.error,
  });
}

function listImageFiles(form: FormData): File[] {
  return [...form.getAll("image"), ...form.getAll("image[]")].filter(
    (item): item is File => item instanceof File,
  );
}

function nonEmptyPromptOrError(prompt: string) {
  if (prompt) return null;
  return { message: "Missing 'prompt'", code: "missing_prompt" };
}

function invalidGenerationModelOrError(model: string) {
  if (model !== IMAGE_GENERATION_MODEL_ID) {
    return {
      message: `The model '${IMAGE_GENERATION_MODEL_ID}' is required for image generations.`,
      code: "model_not_supported",
    };
  }
  if (!isValidModel(model)) return { message: `Model '${model}' not supported`, code: "model_not_supported" };
  if (!isValidImageModel(model)) return { message: `Model '${model}' is not an image model`, code: "invalid_model" };
  return null;
}

function invalidEditModelOrError(model: string) {
  if (model !== IMAGE_EDIT_MODEL_ID) {
    return {
      message: `The model '${IMAGE_EDIT_MODEL_ID}' is required for image edits.`,
      code: "model_not_supported",
    };
  }
  if (!isValidModel(model)) return { message: `Model '${model}' not supported`, code: "model_not_supported" };
  if (!isValidImageModel(model)) return { message: `Model '${model}' is not an image model`, code: "invalid_model" };
  return null;
}

function baseUrlFromSettings(settingsBundle: Awaited<ReturnType<typeof getSettings>>, origin: string): string {
  return (settingsBundle.global.base_url ?? "").trim() || origin;
}

function imageCallPrompt(kind: "generation" | "edit", prompt: string): string {
  return kind === "edit" ? `Image Edit: ${prompt}` : `Image Generation: ${prompt}`;
}

function imageFormatDefault(settingsBundle: Awaited<ReturnType<typeof getSettings>>): string {
  return String(settingsBundle.global.image_mode ?? "url");
}

function parseResponseFormatOrError(raw: unknown, defaultMode: string) {
  const resolved = resolveResponseFormat(raw, defaultMode);
  if (!resolved) {
    return { error: { message: invalidResponseFormatMessage(), code: "invalid_response_format" } };
  }
  return { value: resolved };
}

function resolveImageResponseFormatByMethodOrError(
  raw: unknown,
  defaultMode: string,
  imageMethod: ReturnType<typeof resolveImageGenerationMethod>,
) {
  const missing =
    raw === undefined ||
    raw === null ||
    (typeof raw === "string" && raw.trim().length === 0);
  const normalizedDefault = String(defaultMode || "url").trim().toLowerCase();
  const effectiveDefault =
    missing &&
    imageMethod === IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL &&
    normalizedDefault === "url"
      ? "b64_json"
      : defaultMode;
  return parseResponseFormatOrError(raw, effectiveDefault);
}

const ALLOWED_REASONING_EFFORT = new Set(["none", "minimal", "low", "medium", "high", "xhigh"]);
const RESPONSE_TOOL_OUTPUT_TYPES = new Set([
  "tool_output",
  "function_call_output",
  "tool_call_output",
  "input_tool_output",
]);

function parseOptionalNumber(raw: unknown): number | undefined {
  const n = Number(raw);
  if (!Number.isFinite(n)) return undefined;
  return n;
}

function parseOptionalInteger(raw: unknown): number | undefined {
  const n = parseOptionalNumber(raw);
  if (n === undefined) return undefined;
  return Math.max(1, Math.floor(n));
}

function normalizeReasoningEffort(reasoningEffortRaw: unknown, reasoningRaw: unknown): string | undefined {
  const direct = String(reasoningEffortRaw ?? "").trim().toLowerCase();
  if (direct && ALLOWED_REASONING_EFFORT.has(direct)) return direct;

  if (reasoningRaw && typeof reasoningRaw === "object") {
    const obj = reasoningRaw as Record<string, unknown>;
    const nested = String(obj.effort ?? obj.reasoning_effort ?? "").trim().toLowerCase();
    if (nested && ALLOWED_REASONING_EFFORT.has(nested)) return nested;
  }

  return undefined;
}

function normalizeInclude(raw: unknown): string[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const include = raw
    .filter((v): v is string => typeof v === "string")
    .map((v) => v.trim())
    .filter(Boolean);
  return include.length > 0 ? include : undefined;
}

function isToolChoiceRequired(toolChoice: OpenAIToolChoice | undefined): boolean {
  if (toolChoice === "required") return true;
  if (toolChoice && typeof toolChoice === "object") {
    const forcedName = String(toolChoice.function?.name ?? "").trim();
    return String(toolChoice.type ?? "").trim() === "function" && Boolean(forcedName);
  }
  return false;
}

function hasToolCallsInChatResponse(json: Record<string, unknown>): boolean {
  const choices = Array.isArray(json.choices) ? json.choices : [];
  const first = choices[0] as Record<string, unknown> | undefined;
  const message = first?.message as Record<string, unknown> | undefined;
  return Array.isArray(message?.tool_calls) && message.tool_calls.length > 0;
}

function defaultUsage(): Record<string, unknown> {
  return {
    total_tokens: 0,
    input_tokens: 0,
    output_tokens: 0,
    input_tokens_details: { text_tokens: 0, image_tokens: 0 },
  };
}

function ensureUsage(usage: unknown): Record<string, unknown> {
  if (usage && typeof usage === "object") return usage as Record<string, unknown>;
  return defaultUsage();
}

function makeResponseId(): string {
  return `resp_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

function makeResponseMessageId(): string {
  return `msg_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

function makeResponseFunctionCallId(): string {
  return `fc_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

function coerceResponseContentItem(item: unknown): Record<string, unknown> | null {
  if (!item || typeof item !== "object") return null;
  const block = item as Record<string, unknown>;
  const blockType = String(block.type ?? "").trim();

  if (blockType === "input_text" || blockType === "text" || blockType === "output_text") {
    const text = String(block.text ?? block.content ?? "").trim();
    if (!text) return null;
    return { type: "text", text };
  }

  if (blockType === "input_image" || blockType === "image" || blockType === "image_url" || blockType === "output_image") {
    const imageUrl = block.image_url;
    let url = "";
    if (typeof imageUrl === "string") url = imageUrl;
    else if (imageUrl && typeof imageUrl === "object") {
      url = String((imageUrl as Record<string, unknown>).url ?? "");
    } else {
      url = String(block.url ?? block.image ?? "");
    }
    url = url.trim();
    if (!url) return null;
    return { type: "image_url", image_url: { url } };
  }

  if (blockType === "input_file" || blockType === "file") {
    const fileObj = block.file && typeof block.file === "object" ? (block.file as Record<string, unknown>) : {};
    const fileData = String(block.file_data ?? fileObj.file_data ?? "").trim();
    const fileId = String(block.file_id ?? fileObj.file_id ?? "").trim();
    if (!fileData && !fileId) return null;
    const payload: Record<string, unknown> = {};
    if (fileData) payload.file_data = fileData;
    if (fileId) payload.file_id = fileId;
    return { type: "file", file: payload };
  }

  if (blockType === "input_audio" || blockType === "audio") {
    const audioObj = block.audio && typeof block.audio === "object" ? (block.audio as Record<string, unknown>) : {};
    const data = String(audioObj.data ?? block.data ?? "").trim();
    if (!data) return null;
    return { type: "input_audio", input_audio: { data } };
  }

  return null;
}

function coerceResponseContent(content: unknown): unknown {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  const arr = Array.isArray(content) ? content : [content];
  const blocks = arr
    .map((item) => coerceResponseContentItem(item))
    .filter((v): v is Record<string, unknown> => Boolean(v));
  if (blocks.length > 0) return blocks;
  return String(content);
}

function coerceResponseInputToMessages(input: unknown): Array<Record<string, unknown>> {
  if (input === null || input === undefined) return [];
  if (typeof input === "string") return [{ role: "user", content: input }];

  if (!Array.isArray(input)) {
    if (typeof input === "object") {
      const obj = input as Record<string, unknown>;
      if (obj.type === "message" || ("role" in obj && "content" in obj)) {
        return [{ role: String(obj.role ?? "user"), content: coerceResponseContent(obj.content) }];
      }
      const block = coerceResponseContentItem(obj);
      if (block) return [{ role: "user", content: [block] }];
    }
    return [{ role: "user", content: String(input) }];
  }

  const out: Array<Record<string, unknown>> = [];
  let pendingBlocks: Record<string, unknown>[] = [];
  const flushPending = () => {
    if (!pendingBlocks.length) return;
    out.push({ role: "user", content: pendingBlocks });
    pendingBlocks = [];
  };

  for (const item of input) {
    if (typeof item === "string") {
      pendingBlocks.push({ type: "text", text: item });
      continue;
    }
    if (!item || typeof item !== "object") continue;
    const obj = item as Record<string, unknown>;
    const itemType = String(obj.type ?? "").trim();

    if (itemType === "message" || ("role" in obj && "content" in obj)) {
      flushPending();
      out.push({ role: String(obj.role ?? "user"), content: coerceResponseContent(obj.content) });
      continue;
    }

    if (RESPONSE_TOOL_OUTPUT_TYPES.has(itemType)) {
      flushPending();
      const callId = String(obj.call_id ?? obj.tool_call_id ?? obj.id ?? `call_${crypto.randomUUID()}`);
      out.push({
        role: "tool",
        tool_call_id: callId,
        content: String(obj.output ?? obj.content ?? ""),
      });
      continue;
    }

    const block = coerceResponseContentItem(obj);
    if (block) pendingBlocks.push(block);
  }
  flushPending();
  return out;
}

type ResponsesRequestMeta = {
  model: string;
  instructions?: string;
  maxOutputTokens?: number;
  parallelToolCalls?: boolean;
  previousResponseId?: string;
  reasoningEffort?: string;
  store?: boolean;
  temperature?: number;
  toolChoice?: OpenAIToolChoice;
  tools?: OpenAIToolDefinition[];
  topP?: number;
  truncation?: string;
  user?: string;
  metadata?: Record<string, unknown>;
};

function buildResponsesOutputFromChatMessage(message: Record<string, unknown>): Record<string, unknown>[] {
  const output: Record<string, unknown>[] = [];
  const content = message.content;
  if (typeof content === "string" && content.trim()) {
    output.push({
      id: makeResponseMessageId(),
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: content, annotations: [] }],
    });
  }

  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of toolCalls) {
    if (!call || typeof call !== "object") continue;
    const tool = call as Record<string, unknown>;
    const fn = (tool.function && typeof tool.function === "object"
      ? (tool.function as Record<string, unknown>)
      : {}) as Record<string, unknown>;
    output.push({
      id: makeResponseFunctionCallId(),
      type: "function_call",
      status: "completed",
      call_id: String(tool.id ?? `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`),
      name: String(fn.name ?? ""),
      arguments: String(fn.arguments ?? "{}"),
    });
  }
  return output;
}

function buildResponsesObjectFromChat(args: {
  chatJson: Record<string, unknown>;
  meta: ResponsesRequestMeta;
  responseId?: string;
}): Record<string, unknown> {
  const responseId = args.responseId ?? makeResponseId();
  const createdAt = Math.floor(Date.now() / 1000);
  const choices = Array.isArray(args.chatJson.choices) ? args.chatJson.choices : [];
  const first = (choices[0] as Record<string, unknown> | undefined) ?? {};
  const message = ((first.message as Record<string, unknown> | undefined) ?? {
    role: "assistant",
    content: "",
  }) as Record<string, unknown>;

  return {
    id: responseId,
    object: "response",
    created_at: createdAt,
    completed_at: createdAt,
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: args.meta.instructions ?? null,
    max_output_tokens: args.meta.maxOutputTokens ?? null,
    model: String(args.chatJson.model ?? args.meta.model),
    output: buildResponsesOutputFromChatMessage(message),
    parallel_tool_calls: args.meta.parallelToolCalls ?? true,
    previous_response_id: args.meta.previousResponseId ?? null,
    reasoning: { effort: args.meta.reasoningEffort ?? null, summary: null },
    store: args.meta.store ?? true,
    temperature: args.meta.temperature ?? 1,
    text: { format: { type: "text" } },
    tool_choice: args.meta.toolChoice ?? "auto",
    tools: args.meta.tools ?? [],
    top_p: args.meta.topP ?? 1,
    truncation: args.meta.truncation ?? "disabled",
    usage: ensureUsage(args.chatJson.usage),
    user: args.meta.user ?? null,
    metadata: args.meta.metadata ?? {},
  };
}

function createResponsesStreamFromChatSse(args: {
  chatResp: Response;
  meta: ResponsesRequestMeta;
}): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const responseId = makeResponseId();
  const createdAt = Math.floor(Date.now() / 1000);
  const outputTextParts: string[] = [];
  const toolCallsByIndex = new Map<number, { id: string; name: string; arguments: string }>();
  let usage: Record<string, unknown> = defaultUsage();
  let model = args.meta.model;

  const emit = (controller: ReadableStreamDefaultController<Uint8Array>, event: string, payload: Record<string, unknown>) => {
    controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`));
  };

  const buildInProgressResponse = () => ({
    id: responseId,
    object: "response",
    created_at: createdAt,
    completed_at: null,
    status: "in_progress",
    error: null,
    incomplete_details: null,
    instructions: args.meta.instructions ?? null,
    max_output_tokens: args.meta.maxOutputTokens ?? null,
    model,
    output: [],
    parallel_tool_calls: args.meta.parallelToolCalls ?? true,
    previous_response_id: args.meta.previousResponseId ?? null,
    reasoning: { effort: args.meta.reasoningEffort ?? null, summary: null },
    store: args.meta.store ?? true,
    temperature: args.meta.temperature ?? 1,
    text: { format: { type: "text" } },
    tool_choice: args.meta.toolChoice ?? "auto",
    tools: args.meta.tools ?? [],
    top_p: args.meta.topP ?? 1,
    truncation: args.meta.truncation ?? "disabled",
    usage: null,
    user: args.meta.user ?? null,
    metadata: args.meta.metadata ?? {},
  });

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      emit(controller, "response.created", {
        type: "response.created",
        response: buildInProgressResponse(),
      });
      emit(controller, "response.in_progress", {
        type: "response.in_progress",
        response: buildInProgressResponse(),
      });

      if (!args.chatResp.body) {
        emit(controller, "response.completed", {
          type: "response.completed",
          response: {
            ...buildInProgressResponse(),
            status: "completed",
            completed_at: Math.floor(Date.now() / 1000),
            output: [],
            usage,
          },
        });
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        return;
      }

      const reader = args.chatResp.body.getReader();
      let buffer = "";
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });
          let idx = buffer.indexOf("\n");
          while (idx >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            idx = buffer.indexOf("\n");
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (!payload) continue;
            if (payload === "[DONE]") {
              idx = buffer.indexOf("\n");
              continue;
            }
            let chunk: Record<string, unknown>;
            try {
              chunk = JSON.parse(payload) as Record<string, unknown>;
            } catch {
              continue;
            }
            if (typeof chunk.model === "string" && chunk.model.trim()) {
              model = chunk.model;
            }
            usage = ensureUsage(chunk.usage);
            const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
            const first = choices[0] as Record<string, unknown> | undefined;
            const delta = (first?.delta as Record<string, unknown> | undefined) ?? {};
            const content = delta.content;
            if (typeof content === "string" && content) {
              outputTextParts.push(content);
              emit(controller, "response.output_text.delta", {
                type: "response.output_text.delta",
                response_id: responseId,
                delta: content,
              });
            }
            const toolCalls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
            for (const call of toolCalls) {
              if (!call || typeof call !== "object") continue;
              const tc = call as Record<string, unknown>;
              const index = Number(tc.index ?? 0);
              const fn = (tc.function && typeof tc.function === "object"
                ? (tc.function as Record<string, unknown>)
                : {}) as Record<string, unknown>;
              const current = toolCallsByIndex.get(index) ?? {
                id: String(tc.id ?? `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`),
                name: String(fn.name ?? ""),
                arguments: "",
              };
              if (typeof fn.name === "string" && fn.name.trim()) current.name = fn.name.trim();
              const deltaArgs = typeof fn.arguments === "string" ? fn.arguments : "";
              if (deltaArgs) {
                current.arguments += deltaArgs;
                emit(controller, "response.function_call_arguments.delta", {
                  type: "response.function_call_arguments.delta",
                  response_id: responseId,
                  item_id: current.id,
                  output_index: index,
                  delta: deltaArgs,
                });
              }
              toolCallsByIndex.set(index, current);
            }
          }
        }
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
      }

      const chatLikeMessage: Record<string, unknown> = {
        role: "assistant",
        content: outputTextParts.join(""),
      };
      if (toolCallsByIndex.size > 0) {
        chatLikeMessage.tool_calls = Array.from(toolCallsByIndex.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([, tc], idx) => ({
            id: tc.id,
            index: idx,
            type: "function",
            function: { name: tc.name, arguments: tc.arguments },
          }));
      }

      const response = buildResponsesObjectFromChat({
        chatJson: { model, choices: [{ message: chatLikeMessage }], usage },
        meta: args.meta,
        responseId,
      });
      emit(controller, "response.completed", {
        type: "response.completed",
        response,
      });
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });
}

openAiRoutes.get("/models", async (c) => {
  const ts = Math.floor(Date.now() / 1000);
  const data = Object.entries(MODEL_CONFIG).map(([id, cfg]) => ({
    id,
    object: "model",
    created: ts,
    owned_by: "x-ai",
    display_name: cfg.display_name,
    description: cfg.description,
    raw_model_path: cfg.raw_model_path,
    default_temperature: cfg.default_temperature,
    default_max_output_tokens: cfg.default_max_output_tokens,
    supported_max_output_tokens: cfg.supported_max_output_tokens,
    default_top_p: cfg.default_top_p,
  }));
  return c.json({ object: "list", data });
});

openAiRoutes.get("/models/:modelId", async (c) => {
  const modelId = c.req.param("modelId");
  if (!isValidModel(modelId)) return c.json(openAiError(`Model '${modelId}' not found`, "model_not_found"), 404);
  const cfg = MODEL_CONFIG[modelId]!;
  const ts = Math.floor(Date.now() / 1000);
  return c.json({
    id: modelId,
    object: "model",
    created: ts,
    owned_by: "x-ai",
    display_name: cfg.display_name,
    description: cfg.description,
    raw_model_path: cfg.raw_model_path,
    default_temperature: cfg.default_temperature,
    default_max_output_tokens: cfg.default_max_output_tokens,
    supported_max_output_tokens: cfg.supported_max_output_tokens,
    default_top_p: cfg.default_top_p,
  });
});

openAiRoutes.get("/images/method", async (c) => {
  const settingsBundle = await getSettings(c.env);
  return c.json({ image_generation_method: imageGenerationMethod(settingsBundle) });
});

openAiRoutes.post("/chat/completions", async (c) => {
  const start = Date.now();
  const ip = getClientIp(c.req.raw);
  const keyName = c.get("apiAuth").name ?? "Unknown";

  const origin = new URL(c.req.url).origin;

  let requestedModel = "";
  try {
    const body = (await c.req.json()) as {
      model?: string;
      messages?: any[];
      stream?: boolean;
      reasoning_effort?: unknown;
      reasoning?: { effort?: unknown; reasoning_effort?: unknown };
      include?: unknown;
      temperature?: unknown;
      top_p?: unknown;
      max_output_tokens?: unknown;
      tools?: OpenAIToolDefinition[];
      tool_choice?: OpenAIToolChoice;
      parallel_tool_calls?: boolean;
      video_config?: {
        aspect_ratio?: string;
        video_length?: number;
        resolution?: string;
        preset?: string;
      };
    };

    requestedModel = String(body.model ?? "");
    if (!requestedModel) return c.json(openAiError("Missing 'model'", "missing_model"), 400);
    if (!Array.isArray(body.messages)) return c.json(openAiError("Missing 'messages'", "missing_messages"), 400);
    if (!isValidModel(requestedModel))
      return c.json(openAiError(`Model '${requestedModel}' not supported`, "model_not_supported"), 400);

    const settingsBundle = await getSettings(c.env);
    const cfg = MODEL_CONFIG[requestedModel]!;

    const retryCodes = Array.isArray(settingsBundle.grok.retry_status_codes)
      ? settingsBundle.grok.retry_status_codes
      : [401, 429];

    const stream = Boolean(body.stream);
    const maxRetry = 3;
    let lastErr: string | null = null;
    const debugNdjsonRequested = toBool(c.req.header("X-Grok-Debug-NDJSON"));
    const debugNdjsonEnabled =
      Boolean(c.get("apiAuth").is_admin) && debugNdjsonRequested && !stream;
    const reasoningEffort = normalizeReasoningEffort(body.reasoning_effort, body.reasoning);
    const include = normalizeInclude(body.include);
    const temperature = parseOptionalNumber(body.temperature);
    const topP = parseOptionalNumber(body.top_p);
    const maxOutputTokens = parseOptionalInteger(body.max_output_tokens);

    // === Quota check (best-effort) ===
    // - heavy: consumes both heavy + chat
    // - image model: counts as 2 images per request (grok upstream emits up to 2)
    // - video model: 1 video per request
    // - others: 1 chat per request
    const quotaKind = cfg.is_video_model ? "video" : cfg.is_image_model ? "image" : "chat";
    const quota = await enforceQuota({
      env: c.env,
      apiAuth: c.get("apiAuth"),
      model: requestedModel,
      kind: quotaKind as any,
      ...(cfg.is_image_model ? { imageCount: 2 } : {}),
    });
    if (!quota.ok) return quota.resp;

    for (let attempt = 0; attempt < maxRetry; attempt++) {
      const chosen = await selectBestToken(c.env.DB, requestedModel);
      if (!chosen) return c.json(openAiError("No available token", "NO_AVAILABLE_TOKEN"), 503);

      const jwt = chosen.token;
      const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
      const cookie = cf ? `sso-rw=${jwt};sso=${jwt};${cf}` : `sso-rw=${jwt};sso=${jwt}`;

      const tools = Array.isArray(body.tools) ? body.tools : undefined;
      const toolChoice = body.tool_choice;
      const requireToolCall = isToolChoiceRequired(toolChoice);
      const parallelToolCalls = body.parallel_tool_calls ?? true;
      const extractOptions = {
        ...(tools ? { tools } : {}),
        ...(toolChoice !== undefined ? { toolChoice } : {}),
        parallelToolCalls,
      };
      const { content, images } = extractContent(body.messages as any, extractOptions);
      const isVideoModel = Boolean(cfg.is_video_model);
      const imgInputs = isVideoModel && images.length > 1 ? images.slice(0, 1) : images;

      try {
        const uploads = await mapLimit(imgInputs, 5, (u) => uploadImage(u, cookie, settingsBundle.grok));
        const imgIds = uploads.map((u) => u.fileId).filter(Boolean);
        const imgUris = uploads.map((u) => u.fileUri).filter(Boolean);

        let postId: string | undefined;
        if (isVideoModel) {
          if (imgUris.length) {
            const post = await createPost(imgUris[0]!, cookie, settingsBundle.grok);
            postId = post.postId || undefined;
          } else {
            const post = await createMediaPost(
              { mediaType: "MEDIA_POST_TYPE_VIDEO", prompt: content },
              cookie,
              settingsBundle.grok,
            );
            postId = post.postId || undefined;
          }
        }

        const { payload, referer } = buildConversationPayload({
          requestModel: requestedModel,
          content,
          imgIds,
          imgUris,
          ...(reasoningEffort ? { reasoningEffort } : {}),
          ...(include ? { include } : {}),
          ...(temperature !== undefined ? { temperature } : {}),
          ...(topP !== undefined ? { topP } : {}),
          ...(maxOutputTokens !== undefined ? { maxOutputTokens } : {}),
          ...(postId ? { postId } : {}),
          ...(isVideoModel && body.video_config ? { videoConfig: body.video_config } : {}),
          settings: settingsBundle.grok,
        });

        const upstream = await sendConversationRequest({
          payload,
          cookie,
          settings: settingsBundle.grok,
          ...(referer ? { referer } : {}),
        });

        if (!upstream.ok) {
          const txt = await upstream.text().catch(() => "");
          lastErr = `Upstream ${upstream.status}: ${txt.slice(0, 200)}`;
          await recordTokenFailure(c.env.DB, jwt, upstream.status, txt.slice(0, 200));
          await applyCooldown(c.env.DB, jwt, upstream.status);
          if (retryCodes.includes(upstream.status) && attempt < maxRetry - 1) continue;
          break;
        }

        if (stream) {
          const sse = createOpenAiStreamFromGrokNdjson(upstream, {
            cookie,
            settings: settingsBundle.grok,
            global: settingsBundle.global,
            origin,
            requestedModel,
            ...(tools ? { tools } : {}),
            ...(toolChoice !== undefined ? { toolChoice } : {}),
            onFinish: async ({ status, duration }) => {
              await addRequestLog(c.env.DB, {
                ip,
                model: requestedModel,
                duration: Number(duration.toFixed(2)),
                status,
                key_name: keyName,
                token_suffix: jwt.slice(-6),
                error: status === 200 ? "" : "stream_error",
              });
            },
          });

          return new Response(sse, {
            status: 200,
            headers: {
              "Content-Type": "text/event-stream; charset=utf-8",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
              "Access-Control-Allow-Origin": "*",
            },
          });
        }

        const json = await parseOpenAiFromGrokNdjson(upstream, {
          cookie,
          settings: settingsBundle.grok,
          global: settingsBundle.global,
          origin,
          requestedModel,
          ...(tools ? { tools } : {}),
          ...(toolChoice !== undefined ? { toolChoice } : {}),
          ...(debugNdjsonEnabled ? { debugNdjsonSample: true } : {}),
        });
        if (requireToolCall && tools?.length && !hasToolCallsInChatResponse(json)) {
          return c.json(
            openAiError("tool_choice is 'required' but model did not produce tool_calls.", "required_tool_call_missing"),
            422,
          );
        }

        const duration = (Date.now() - start) / 1000;
        await addRequestLog(c.env.DB, {
          ip,
          model: requestedModel,
          duration: Number(duration.toFixed(2)),
          status: 200,
          key_name: keyName,
          token_suffix: jwt.slice(-6),
          error: "",
        });

        return c.json(json);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg.includes("tool_choice is 'required'")) {
          const duration = (Date.now() - start) / 1000;
          await addRequestLog(c.env.DB, {
            ip,
            model: requestedModel,
            duration: Number(duration.toFixed(2)),
            status: 422,
            key_name: keyName,
            token_suffix: jwt.slice(-6),
            error: "required_tool_call_missing",
          });
          return c.json(openAiError(msg, "required_tool_call_missing"), 422);
        }
        lastErr = msg;
        await recordTokenFailure(c.env.DB, jwt, 500, msg);
        await applyCooldown(c.env.DB, jwt, 500);
        if (attempt < maxRetry - 1) continue;
      }
    }

    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel,
      duration: Number(duration.toFixed(2)),
      status: 500,
      key_name: keyName,
      token_suffix: "",
      error: lastErr ?? "unknown_error",
    });

    return c.json(openAiError(lastErr ?? "Upstream error", "upstream_error"), 500);
  } catch (e) {
    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel || "unknown",
      duration: Number(duration.toFixed(2)),
      status: 500,
      key_name: keyName,
      token_suffix: "",
      error: e instanceof Error ? e.message : String(e),
    });
    return c.json(openAiError("Internal error", "internal_error"), 500);
  }
});

openAiRoutes.post("/responses", async (c) => {
  const start = Date.now();
  const ip = getClientIp(c.req.raw);
  const keyName = c.get("apiAuth").name ?? "Unknown";
  let requestedModel = "";
  try {
    const body = (await c.req.json()) as {
      model?: unknown;
      input?: unknown;
      instructions?: unknown;
      stream?: unknown;
      tools?: unknown;
      tool_choice?: unknown;
      parallel_tool_calls?: unknown;
      reasoning?: unknown;
      include?: unknown;
      temperature?: unknown;
      top_p?: unknown;
      max_output_tokens?: unknown;
      metadata?: unknown;
      user?: unknown;
      store?: unknown;
      previous_response_id?: unknown;
      truncation?: unknown;
    };

    requestedModel = String(body.model ?? "").trim();
    if (!requestedModel) return c.json(openAiError("Missing 'model'", "missing_model"), 400);
    if (!isValidModel(requestedModel)) {
      return c.json(openAiError(`Model '${requestedModel}' not supported`, "model_not_supported"), 400);
    }
    if (body.input === undefined || body.input === null) {
      return c.json(openAiError("Missing 'input'", "missing_input"), 400);
    }

    const messages = coerceResponseInputToMessages(body.input);
    const instructions = String(body.instructions ?? "").trim();
    if (instructions) {
      messages.unshift({ role: "system", content: instructions });
    }
    if (!messages.length) {
      return c.json(openAiError("Unable to build messages from 'input'", "invalid_input"), 400);
    }

    const tools = Array.isArray(body.tools) ? (body.tools as OpenAIToolDefinition[]) : undefined;
    const toolChoice =
      body.tool_choice === undefined ? undefined : (body.tool_choice as OpenAIToolChoice);
    const requireToolCall = isToolChoiceRequired(toolChoice);
    const parallelToolCalls =
      typeof body.parallel_tool_calls === "boolean" ? body.parallel_tool_calls : true;
    const reasoningEffort = normalizeReasoningEffort(undefined, body.reasoning);
    const include = normalizeInclude(body.include);
    const temperature = parseOptionalNumber(body.temperature);
    const topP = parseOptionalNumber(body.top_p);
    const maxOutputTokens = parseOptionalInteger(body.max_output_tokens);
    const stream = Boolean(body.stream);
    const debugNdjsonRequested = toBool(c.req.header("X-Grok-Debug-NDJSON"));
    const debugNdjsonEnabled =
      Boolean(c.get("apiAuth").is_admin) && debugNdjsonRequested && !stream;

    const chatPayload: Record<string, unknown> = {
      model: requestedModel,
      messages,
      stream,
      ...(tools ? { tools } : {}),
      ...(toolChoice !== undefined ? { tool_choice: toolChoice } : {}),
      parallel_tool_calls: parallelToolCalls,
      ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
      ...(include ? { include } : {}),
      ...(temperature !== undefined ? { temperature } : {}),
      ...(topP !== undefined ? { top_p: topP } : {}),
      ...(maxOutputTokens !== undefined ? { max_output_tokens: maxOutputTokens } : {}),
    };

    const origin = new URL(c.req.url).origin;
    const headers = new Headers();
    const authHeader = c.req.header("Authorization");
    if (authHeader) headers.set("Authorization", authHeader);
    const xApiKeyHeader = c.req.header("X-API-Key");
    if (xApiKeyHeader) headers.set("X-API-Key", xApiKeyHeader);
    if (debugNdjsonEnabled) headers.set("X-Grok-Debug-NDJSON", "1");
    headers.set("Content-Type", "application/json");
    const internalReq = new Request(`${origin}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify(chatPayload),
    });
    const chatResp = await openAiRoutes.fetch(internalReq, c.env, c.executionCtx);

    const responseMeta: ResponsesRequestMeta = {
      model: requestedModel,
      ...(instructions ? { instructions } : {}),
      ...(maxOutputTokens !== undefined ? { maxOutputTokens } : {}),
      parallelToolCalls,
      ...(typeof body.previous_response_id === "string" && body.previous_response_id
        ? { previousResponseId: body.previous_response_id }
        : {}),
      ...(reasoningEffort ? { reasoningEffort } : {}),
      ...(typeof body.store === "boolean" ? { store: body.store } : {}),
      ...(temperature !== undefined ? { temperature } : {}),
      ...(toolChoice !== undefined ? { toolChoice } : {}),
      ...(tools ? { tools } : {}),
      ...(topP !== undefined ? { topP } : {}),
      ...(typeof body.truncation === "string" && body.truncation ? { truncation: body.truncation } : {}),
      ...(typeof body.user === "string" && body.user ? { user: body.user } : {}),
      ...(body.metadata && typeof body.metadata === "object" ? { metadata: body.metadata as Record<string, unknown> } : {}),
    };

    if (!chatResp.ok) {
      const errorText = await chatResp.text().catch(() => "");
      const duration = (Date.now() - start) / 1000;
      await addRequestLog(c.env.DB, {
        ip,
        model: requestedModel,
        duration: Number(duration.toFixed(2)),
        status: chatResp.status,
        key_name: keyName,
        token_suffix: "",
        error: errorText.slice(0, 200) || "responses_upstream_error",
      });
      return new Response(errorText, {
        status: chatResp.status,
        headers: {
          "Content-Type": chatResp.headers.get("Content-Type") || "application/json; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    if (stream) {
      const sse = createResponsesStreamFromChatSse({ chatResp, meta: responseMeta });
      const duration = (Date.now() - start) / 1000;
      await addRequestLog(c.env.DB, {
        ip,
        model: requestedModel,
        duration: Number(duration.toFixed(2)),
        status: 200,
        key_name: keyName,
        token_suffix: "",
        error: "",
      });
      return new Response(sse, {
        status: 200,
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          "X-Accel-Buffering": "no",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    const chatJson = (await chatResp.json()) as Record<string, unknown>;
    if (requireToolCall && tools?.length && !hasToolCallsInChatResponse(chatJson)) {
      return c.json(
        openAiError("tool_choice is 'required' but model did not produce tool_calls.", "required_tool_call_missing"),
        422,
      );
    }
    const responseJson = buildResponsesObjectFromChat({ chatJson, meta: responseMeta });
    if (debugNdjsonEnabled && Object.prototype.hasOwnProperty.call(chatJson, "_debug_ndjson")) {
      responseJson._debug_ndjson = (chatJson as Record<string, unknown>)._debug_ndjson;
    }
    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel,
      duration: Number(duration.toFixed(2)),
      status: 200,
      key_name: keyName,
      token_suffix: "",
      error: "",
    });
    return c.json(responseJson);
  } catch (e) {
    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel || "unknown",
      duration: Number(duration.toFixed(2)),
      status: 500,
      key_name: keyName,
      token_suffix: "",
      error: e instanceof Error ? e.message : String(e),
    });
    return c.json(openAiError("Internal error", "internal_error"), 500);
  }
});

openAiRoutes.post("/images/generations", async (c) => {
  const start = Date.now();
  const ip = getClientIp(c.req.raw);
  const keyName = c.get("apiAuth").name ?? "Unknown";
  const origin = new URL(c.req.url).origin;

  let requestedModel = IMAGE_GENERATION_MODEL_ID;
  try {
    const body = (await c.req.json()) as {
      prompt?: unknown;
      model?: unknown;
      n?: unknown;
      size?: unknown;
      concurrency?: unknown;
      stream?: unknown;
      response_format?: unknown;
    };
    const prompt = parseImagePrompt(body.prompt);
    const promptErr = nonEmptyPromptOrError(prompt);
    if (promptErr) return c.json(openAiError(promptErr.message, promptErr.code), 400);

    requestedModel = parseImageModel(body.model, IMAGE_GENERATION_MODEL_ID);
    const modelErr = invalidGenerationModelOrError(requestedModel);
    if (modelErr) return c.json(openAiError(modelErr.message, modelErr.code), 400);

    const n = parseImageCount(body.n);
    const size = parseImageSize(body.size);
    const aspectRatio = resolveAspectRatio(size);
    const concurrencyParsed = parseImageConcurrencyOrError(body.concurrency);
    if ("error" in concurrencyParsed) {
      return c.json(
        openAiError(concurrencyParsed.error.message, concurrencyParsed.error.code),
        400,
      );
    }
    const concurrency = concurrencyParsed.value;
    const stream = parseImageStream(body.stream);
    if (stream && ![1, 2].includes(n)) {
      return c.json(openAiError(invalidStreamNMessage(), "invalid_stream_n"), 400);
    }

    const settingsBundle = await getSettings(c.env);
    const imageMethod = imageGenerationMethod(settingsBundle);
    const parsedResponseFormat = resolveImageResponseFormatByMethodOrError(
      body.response_format,
      imageFormatDefault(settingsBundle),
      imageMethod,
    );
    if ("error" in parsedResponseFormat) {
      return c.json(
        openAiError(parsedResponseFormat.error.message, parsedResponseFormat.error.code),
        400,
      );
    }
    const responseFormat = parsedResponseFormat.value;
    const responseField = responseFieldName(responseFormat);
    const baseUrl = baseUrlFromSettings(settingsBundle, origin);
    const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");

    const quota = await enforceQuota({
      env: c.env,
      apiAuth: c.get("apiAuth"),
      model: requestedModel,
      kind: "image",
      imageCount: n,
    });
    if (!quota.ok) return quota.resp;

    if (stream) {
      if (imageMethod === IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL) {
        const experimentalToken = await selectBestToken(c.env.DB, requestedModel);
        if (experimentalToken) {
          const experimentalCookie = buildCookie(experimentalToken.token, cf);
          const streamBody = createExperimentalImageEventStream({
            prompt: imageCallPrompt("generation", prompt),
            n,
            cookie: experimentalCookie,
            settings: settingsBundle.grok,
            responseFormat,
            responseField,
            baseUrl,
            aspectRatio,
            concurrency,
            onFinish: async ({ status, duration }) => {
              await addRequestLog(c.env.DB, {
                ip,
                model: requestedModel,
                duration: Number(duration.toFixed(2)),
                status,
                key_name: keyName,
                token_suffix: getTokenSuffix(experimentalToken.token),
                error: status === 200 ? "" : "stream_error",
              });
            },
          });
          return new Response(streamBody, { status: 200, headers: streamHeaders() });
        }
      }

      const chosen = await selectBestToken(c.env.DB, requestedModel);
      if (!chosen) {
        await recordImageLog({
          env: c.env,
          ip,
          model: requestedModel,
          start,
          keyName,
          status: 503,
          error: "NO_AVAILABLE_TOKEN",
        });
        return new Response(
          createStreamErrorImageEventStream({
            message: "No available token",
            responseField,
          }),
          { status: 200, headers: streamHeaders() },
        );
      }
      const cookie = buildCookie(chosen.token, cf);

      const upstream = await runImageStreamCall({
        requestModel: requestedModel,
        prompt: imageCallPrompt("generation", prompt),
        fileIds: [],
        cookie,
        settings: settingsBundle.grok,
      });
      if (!upstream.ok) {
        const txt = await upstream.text().catch(() => "");
        await recordTokenFailure(c.env.DB, chosen.token, upstream.status, txt.slice(0, 200));
        await applyCooldown(c.env.DB, chosen.token, upstream.status);
        await recordImageLog({
          env: c.env,
          ip,
          model: requestedModel,
          start,
          keyName,
          status: upstream.status,
          tokenSuffix: getTokenSuffix(chosen.token),
          error: txt.slice(0, 200),
        });
        return new Response(
          createStreamErrorImageEventStream({
            message: isContentModerationMessage(txt)
              ? txt.slice(0, 500)
              : `Upstream ${upstream.status}`,
            responseField,
          }),
          { status: 200, headers: streamHeaders() },
        );
      }

      const streamBody = createImageEventStream({
        upstream,
        responseFormat,
        baseUrl,
        cookie,
        settings: settingsBundle.grok,
        n,
        onFinish: async ({ status, duration }) => {
          await addRequestLog(c.env.DB, {
            ip,
            model: requestedModel,
            duration: Number(duration.toFixed(2)),
            status,
            key_name: keyName,
            token_suffix: getTokenSuffix(chosen.token),
            error: status === 200 ? "" : "stream_error",
          });
        },
      });
      return new Response(streamBody, { status: 200, headers: streamHeaders() });
    }

    if (imageMethod === IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL) {
      const experimentalToken = await selectBestToken(c.env.DB, requestedModel);
      if (experimentalToken) {
        const experimentalCookie = buildCookie(experimentalToken.token, cf);
        try {
          const urls = await collectExperimentalGenerationImages({
            prompt: imageCallPrompt("generation", prompt),
            n,
            cookie: experimentalCookie,
            settings: settingsBundle.grok,
            responseFormat,
            baseUrl,
            aspectRatio,
            concurrency,
          });
          const selected = pickImageResults(urls, n);
          await recordImageLog({
            env: c.env,
            ip,
            model: requestedModel,
            start,
            keyName,
            status: 200,
            tokenSuffix: getTokenSuffix(experimentalToken.token),
            error: "",
          });
          return c.json(buildImageJsonPayload(responseField, selected));
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          await recordTokenFailure(c.env.DB, experimentalToken.token, 500, msg.slice(0, 200));
          await applyCooldown(c.env.DB, experimentalToken.token, 500);
          console.warn("Experimental image generation failed, fallback to legacy:", msg);
        }
      }
    }

    const calls = Math.ceil(n / 2);
    const urlsNested = await mapLimit(
      Array.from({ length: calls }),
      Math.min(calls, Math.max(1, concurrency)),
      async () => {
      const chosen = await selectBestToken(c.env.DB, requestedModel);
      if (!chosen) throw new Error("No available token");
      const cookie = buildCookie(chosen.token, cf);
      try {
        return await runImageCall({
          requestModel: requestedModel,
          prompt: imageCallPrompt("generation", prompt),
          fileIds: [],
          cookie,
          settings: settingsBundle.grok,
          responseFormat,
          baseUrl,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        await recordTokenFailure(c.env.DB, chosen.token, 500, msg.slice(0, 200));
        await applyCooldown(c.env.DB, chosen.token, 500);
        throw e;
      }
    },
    );
    const urls = dedupeImages(urlsNested.flat().filter(Boolean));
    const selected = pickImageResults(urls, n);

    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel,
      start,
      keyName,
      status: 200,
      error: "",
    });

    return c.json(buildImageJsonPayload(responseField, selected));
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    if (isContentModerationMessage(message)) {
      await recordImageLog({
        env: c.env,
        ip,
        model: requestedModel || "image",
        start,
        keyName,
        status: 400,
        error: message,
      });
      return c.json(openAiError(message, "content_policy_violation"), 400);
    }
    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel || "image",
      start,
      keyName,
      status: 500,
      error: message,
    });
    return c.json(openAiError(message || "Internal error", "internal_error"), 500);
  }
});

openAiRoutes.post("/images/edits", async (c) => {
  const start = Date.now();
  const ip = getClientIp(c.req.raw);
  const keyName = c.get("apiAuth").name ?? "Unknown";
  const origin = new URL(c.req.url).origin;
  const maxImageBytes = 50 * 1024 * 1024;

  let requestedModel = IMAGE_EDIT_MODEL_ID;
  try {
    const form = await c.req.formData();
    const prompt = parseImagePrompt(form.get("prompt"));
    const promptErr = nonEmptyPromptOrError(prompt);
    if (promptErr) return c.json(openAiError(promptErr.message, promptErr.code), 400);

    requestedModel = parseImageModel(form.get("model"), IMAGE_EDIT_MODEL_ID);
    const modelErr = invalidEditModelOrError(requestedModel);
    if (modelErr) return c.json(openAiError(modelErr.message, modelErr.code), 400);

    const n = parseImageCount(form.get("n"));
    const stream = parseImageStream(form.get("stream"));
    if (stream && ![1, 2].includes(n)) {
      return c.json(openAiError(invalidStreamNMessage(), "invalid_stream_n"), 400);
    }

    const files = listImageFiles(form);
    if (!files.length) return c.json(openAiError("Image is required", "missing_image"), 400);
    if (files.length > 16) {
      return c.json(openAiError("Too many images. Maximum is 16.", "invalid_image_count"), 400);
    }

    const settingsBundle = await getSettings(c.env);
    const imageMethod = imageGenerationMethod(settingsBundle);
    const parsedResponseFormat = resolveImageResponseFormatByMethodOrError(
      form.get("response_format"),
      imageFormatDefault(settingsBundle),
      imageMethod,
    );
    if ("error" in parsedResponseFormat) {
      return c.json(
        openAiError(parsedResponseFormat.error.message, parsedResponseFormat.error.code),
        400,
      );
    }
    const responseFormat = parsedResponseFormat.value;
    const responseField = responseFieldName(responseFormat);
    const baseUrl = baseUrlFromSettings(settingsBundle, origin);

    const quota = await enforceQuota({
      env: c.env,
      apiAuth: c.get("apiAuth"),
      model: requestedModel,
      kind: "image",
      imageCount: n,
    });
    if (!quota.ok) return quota.resp;

    const chosen = await selectBestToken(c.env.DB, requestedModel);
    if (!chosen) {
      if (stream) {
        await recordImageLog({
          env: c.env,
          ip,
          model: requestedModel,
          start,
          keyName,
          status: 503,
          error: "NO_AVAILABLE_TOKEN",
        });
        return new Response(
          createStreamErrorImageEventStream({
            message: "No available token",
            responseField,
          }),
          { status: 200, headers: streamHeaders() },
        );
      }
      return c.json(openAiError("No available token", "NO_AVAILABLE_TOKEN"), 503);
    }
    const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
    const cookie = buildCookie(chosen.token, cf);

    const fileIds: string[] = [];
    const fileUris: string[] = [];
    for (const file of files) {
      const bytes = await file.arrayBuffer();
      if (bytes.byteLength <= 0) {
        return c.json(openAiError("File content is empty", "empty_file"), 400);
      }
      if (bytes.byteLength > maxImageBytes) {
        return c.json(openAiError("Image file too large. Maximum is 50MB.", "file_too_large"), 400);
      }

      const mime = parseAllowedImageMime(file);
      if (!mime) {
        return c.json(
          openAiError("Unsupported image type. Supported: png, jpg, webp.", "invalid_image_type"),
          400,
        );
      }

      const dataUrl = `data:${mime};base64,${arrayBufferToBase64(bytes)}`;
      const uploaded = await uploadImage(dataUrl, cookie, settingsBundle.grok);
      if (uploaded.fileId) fileIds.push(uploaded.fileId);
      if (uploaded.fileUri) fileUris.push(uploaded.fileUri);
    }

    if (stream) {
      if (imageMethod === IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL) {
        try {
          const upstream = await sendExperimentalImageEditRequest({
            prompt: imageCallPrompt("edit", prompt),
            fileUris,
            cookie,
            settings: settingsBundle.grok,
          });

          const streamBody = createImageEventStream({
            upstream,
            responseFormat,
            baseUrl,
            cookie,
            settings: settingsBundle.grok,
            n,
            onFinish: async ({ status, duration }) => {
              await addRequestLog(c.env.DB, {
                ip,
                model: requestedModel,
                duration: Number(duration.toFixed(2)),
                status,
                key_name: keyName,
                token_suffix: getTokenSuffix(chosen.token),
                error: status === 200 ? "" : "stream_error",
              });
            },
          });
          return new Response(streamBody, { status: 200, headers: streamHeaders() });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          await recordTokenFailure(c.env.DB, chosen.token, 500, msg.slice(0, 200));
          await applyCooldown(c.env.DB, chosen.token, 500);
          console.warn("Experimental image edit stream failed, fallback to legacy:", msg);
        }
      }

      const upstream = await runImageStreamCall({
        requestModel: requestedModel,
        prompt: imageCallPrompt("edit", prompt),
        fileIds,
        cookie,
        settings: settingsBundle.grok,
      });
      if (!upstream.ok) {
        const txt = await upstream.text().catch(() => "");
        await recordTokenFailure(c.env.DB, chosen.token, upstream.status, txt.slice(0, 200));
        await applyCooldown(c.env.DB, chosen.token, upstream.status);
        await recordImageLog({
          env: c.env,
          ip,
          model: requestedModel,
          start,
          keyName,
          status: upstream.status,
          tokenSuffix: getTokenSuffix(chosen.token),
          error: txt.slice(0, 200),
        });
        return new Response(
          createStreamErrorImageEventStream({
            message: isContentModerationMessage(txt)
              ? txt.slice(0, 500)
              : `Upstream ${upstream.status}`,
            responseField,
          }),
          { status: 200, headers: streamHeaders() },
        );
      }

      const streamBody = createImageEventStream({
        upstream,
        responseFormat,
        baseUrl,
        cookie,
        settings: settingsBundle.grok,
        n,
        onFinish: async ({ status, duration }) => {
          await addRequestLog(c.env.DB, {
            ip,
            model: requestedModel,
            duration: Number(duration.toFixed(2)),
            status,
            key_name: keyName,
            token_suffix: getTokenSuffix(chosen.token),
            error: status === 200 ? "" : "stream_error",
          });
        },
      });
      return new Response(streamBody, { status: 200, headers: streamHeaders() });
    }

    if (imageMethod === IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL) {
      try {
        const calls = Math.ceil(n / 2);
        const urlsNested = await mapLimit(Array.from({ length: calls }), 3, async () =>
          runExperimentalImageEditCall({
            prompt: imageCallPrompt("edit", prompt),
            fileUris,
            cookie,
            settings: settingsBundle.grok,
            responseFormat,
            baseUrl,
          }),
        );
        const urls = dedupeImages(urlsNested.flat().filter(Boolean));
        if (!urls.length) throw new Error("Experimental image edit returned no images");
        const selected = pickImageResults(urls, n);

        await recordImageLog({
          env: c.env,
          ip,
          model: requestedModel,
          start,
          keyName,
          status: 200,
          tokenSuffix: getTokenSuffix(chosen.token),
          error: "",
        });
        return c.json(buildImageJsonPayload(responseField, selected));
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        await recordTokenFailure(c.env.DB, chosen.token, 500, msg.slice(0, 200));
        await applyCooldown(c.env.DB, chosen.token, 500);
        console.warn("Experimental image edit failed, fallback to legacy:", msg);
      }
    }

    const calls = Math.ceil(n / 2);
    const urlsNested = await mapLimit(Array.from({ length: calls }), 3, async () => {
      return runImageCall({
        requestModel: requestedModel,
        prompt: imageCallPrompt("edit", prompt),
        fileIds,
        cookie,
        settings: settingsBundle.grok,
        responseFormat,
        baseUrl,
      });
    });
    const urls = dedupeImages(urlsNested.flat().filter(Boolean));
    const selected = pickImageResults(urls, n);

    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel,
      start,
      keyName,
      status: 200,
      tokenSuffix: getTokenSuffix(chosen.token),
      error: "",
    });

    return c.json(buildImageJsonPayload(responseField, selected));
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    if (isContentModerationMessage(message)) {
      await recordImageLog({
        env: c.env,
        ip,
        model: requestedModel || "image",
        start,
        keyName,
        status: 400,
        error: message,
      });
      return c.json(openAiError(message, "content_policy_violation"), 400);
    }
    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel || "image",
      start,
      keyName,
      status: 500,
      error: message,
    });
    return c.json(openAiError(message || "Internal error", "internal_error"), 500);
  }
});

openAiRoutes.post("/uploads/image", async (c) => {
  try {
    const form = await c.req.formData();
    const file = form.get("file");
    if (!(file instanceof File)) return c.json(openAiError("Missing file", "missing_file"), 400);

    const mime = String(file.type || "application/octet-stream");
    if (!mime.toLowerCase().startsWith("image/"))
      return c.json(openAiError(`Unsupported mime: ${mime}`, "unsupported_file"), 400);

    const bytes = await file.arrayBuffer();
    const size = bytes.byteLength;
    const maxBytes = Math.min(25 * 1024 * 1024, Math.max(1, parseIntSafe(c.env.KV_CACHE_MAX_BYTES, 25 * 1024 * 1024)));
    if (size > maxBytes) return c.json(openAiError(`File too large (${size} > ${maxBytes})`, "file_too_large"), 413);

    const ext = (() => {
      const m = mime.toLowerCase();
      if (m === "image/png") return "png";
      if (m === "image/webp") return "webp";
      if (m === "image/gif") return "gif";
      if (m === "image/jpeg" || m === "image/jpg") return "jpg";
      return "jpg";
    })();

    const name = `upload-${crypto.randomUUID()}.${ext}`;
    const kvKey = `image/${name}`;

    const tz = parseIntSafe(c.env.CACHE_RESET_TZ_OFFSET_MINUTES, 480);
    const expiresAt = nextLocalMidnightExpirationSeconds(nowMs(), tz);

    await c.env.KV_CACHE.put(kvKey, bytes, {
      expiration: expiresAt,
      metadata: { contentType: mime, size },
    });

    const now = nowMs();
    await upsertCacheRow(c.env.DB, {
      key: kvKey,
      type: "image",
      size,
      content_type: mime,
      created_at: now,
      last_access_at: now,
      expires_at: expiresAt * 1000,
    });

    return c.json({
      url: `/images/${encodeURIComponent(name)}`,
      name,
      size_bytes: size,
    });
  } catch (e) {
    return c.json(openAiError(e instanceof Error ? e.message : "Internal error", "internal_error"), 500);
  }
});

openAiRoutes.options("/*", (c) => c.body(null, 204));
