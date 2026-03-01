const WORKER_NAME = "grok2api-edge-proxy";

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function parseApiKeys(raw) {
  return (raw || "")
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);
}

function isAuthorized(request, validKeys) {
  const auth = request.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) return false;
  const key = auth.slice(7).trim();
  return validKeys.includes(key);
}

function bearerFromRequest(request) {
  const auth = request.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) return "";
  return auth.slice(7).trim();
}

function buildUpstreamUrl(base, incomingUrl) {
  const upstream = new URL(base);
  const incoming = new URL(incomingUrl);
  upstream.pathname = `${upstream.pathname.replace(/\/$/, "")}${incoming.pathname}`;
  upstream.search = incoming.search;
  return upstream.toString();
}

export default {
  async fetch(request, env, ctx) {
    const upstreamBase = (env.UPSTREAM_BASE_URL || "").trim();
    const validKeys = parseApiKeys(env.API_KEYS);
    const hasDb = !!env.DB;

    if (!upstreamBase) {
      return json({ error: "UPSTREAM_BASE_URL is not configured" }, 500);
    }

    if (new URL(request.url).pathname === "/__health") {
      return json({
        ok: true,
        worker: WORKER_NAME,
        upstream: upstreamBase,
        d1: hasDb,
      });
    }

    if (!validKeys.length && !hasDb) {
      return json({ error: "API_KEYS or D1 is not configured" }, 500);
    }

    const clientKey = bearerFromRequest(request);
    let authorized = isAuthorized(request, validKeys);
    if (!authorized && hasDb && clientKey) {
      try {
        const row = await env.DB.prepare(
          "SELECT 1 AS ok FROM api_keys WHERE key = ? AND enabled = 1 LIMIT 1"
        )
          .bind(clientKey)
          .first();
        authorized = !!(row && row.ok === 1);
      } catch (_) {
        // D1 查询失败时保持为未授权
      }
    }

    if (!authorized) {
      return json({ error: "Invalid API key" }, 401);
    }

    const targetUrl = buildUpstreamUrl(upstreamBase, request.url);

    const headers = new Headers(request.headers);
    headers.delete("host");
    if ((env.UPSTREAM_AUTH_BEARER || "").trim()) {
      headers.set("authorization", `Bearer ${env.UPSTREAM_AUTH_BEARER.trim()}`);
    }

    const init = {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "follow",
    };

    try {
      const upstreamResp = await fetch(targetUrl, init);
      if (hasDb) {
        const path = new URL(request.url).pathname;
        const method = request.method;
        const status = upstreamResp.status;
        const colo = request.cf && request.cf.colo ? request.cf.colo : "";
        const ua = request.headers.get("user-agent") || "";
        const ip = request.headers.get("cf-connecting-ip") || "";
        const keyPrefix = clientKey ? clientKey.slice(0, 8) : "";
        const createdAt = new Date().toISOString();
        const logPromise = env.DB.prepare(
          "INSERT INTO request_logs (created_at, method, path, status, key_prefix, colo, ip, user_agent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
          .bind(createdAt, method, path, status, keyPrefix, colo, ip, ua)
          .run();
        if (ctx && typeof ctx.waitUntil === "function") {
          ctx.waitUntil(logPromise);
        }
      }
      return new Response(upstreamResp.body, {
        status: upstreamResp.status,
        statusText: upstreamResp.statusText,
        headers: upstreamResp.headers,
      });
    } catch (err) {
      return json(
        {
          error: "Upstream request failed",
          detail: err instanceof Error ? err.message : String(err),
        },
        502
      );
    }
  },
};
