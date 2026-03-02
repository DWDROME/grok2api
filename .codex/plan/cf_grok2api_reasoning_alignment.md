# CF Grok2API Reasoning Alignment Plan

## Context
- Target repo: DWDROME/grok2api
- Goal: align Cloudflare Worker behavior with local grok2api for thinking/reasoning readability and responses endpoint availability.
- Baseline observed:
  - /v1/chat/completions works
  - /v1/responses returns 404
  - reasoning parameters not fully passed through
  - payload uses fixed isReasoning=false
  - non-stream usage often null

## Approved execution scope
1. Extend chat request parameter chain in Worker.
2. Inject reasoning/model overrides into upstream payload.
3. Implement /v1/responses compatibility subset.
4. Improve non-stream usage and required-tool consistency.
5. Update docs for CF behavior.
6. Validate locally and against deployed endpoint.

## Acceptance targets
- /v1/responses no longer 404.
- Chat stream can emit think-formatted chunks when upstream provides thinking tokens and show_thinking enabled.
- Non-stream usage is object, not null.
- reasoning_effort can reach upstream modelConfigOverride.
- tool_choice=required without tool_calls returns structured error.
