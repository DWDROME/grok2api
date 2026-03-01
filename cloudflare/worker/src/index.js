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

function buildUpstreamUrl(base, incomingUrl) {
  const upstream = new URL(base);
  const incoming = new URL(incomingUrl);
  upstream.pathname = `${upstream.pathname.replace(/\/$/, "")}${incoming.pathname}`;
  upstream.search = incoming.search;
  return upstream.toString();
}

export default {
  async fetch(request, env) {
    const upstreamBase = (env.UPSTREAM_BASE_URL || "").trim();
    const validKeys = parseApiKeys(env.API_KEYS);

    if (!upstreamBase) {
      return json({ error: "UPSTREAM_BASE_URL is not configured" }, 500);
    }

    if (new URL(request.url).pathname === "/__health") {
      return json({
        ok: true,
        worker: WORKER_NAME,
        upstream: upstreamBase,
      });
    }

    if (!validKeys.length) {
      return json({ error: "API_KEYS is not configured" }, 500);
    }

    if (!isAuthorized(request, validKeys)) {
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
