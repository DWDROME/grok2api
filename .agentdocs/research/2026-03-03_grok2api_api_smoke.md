# grok2api API 冒烟与参数链路核查（2026-03-03）

## 子问题

1. `reasoning_effort` 是否被接口定义、校验并传递到 Grok 请求？
2. 本地服务是否可启动并响应 OpenAI 兼容路径？
3. 为什么会出现“可访问但仍失败”的情况？

## 停止条件

- 找到请求字段定义 + 校验 + 下游透传三处代码证据。
- 完成一次本地启动与至少两条 API 冒烟请求（成功/失败各一类）。

## 代码证据

- 请求字段定义：`app/api/v1/chat.py:60`
- 合法值校验：`app/api/v1/chat.py:489`
- 透传到 `modelConfigOverride.reasoningEffort`：`app/services/grok/services/chat.py:362`
- FastAPI 主路由挂载（`/v1/*`）：`main.py:136`
- TypeScript/Workers 侧兼容 `reasoning.effort`：`src/routes/openai.ts:352`、`src/routes/openai.ts:372`
- TS 侧写入 payload 覆盖项：`src/grok/conversation.ts:332`

## 执行记录（本地）

```bash
uv run main.py
curl -i http://127.0.0.1:8000/v1/models
curl -i http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"grok-4.1-thinking","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"ultra"}'
curl -i http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"grok-4.1-thinking","messages":[{"role":"user","content":"hi"}],"stream":false,"reasoning_effort":"low"}'
```

观测结果：

- `/v1/models` 返回 `200`，路由可用。
- `reasoning_effort="ultra"` 返回 `400 invalid_reasoning_effort`（校验生效）。
- `reasoning_effort="low"` 请求通过参数校验，但返回 `429 rate_limit_exceeded`，原因为本地无可用 Grok token（非参数链路问题）。

## 质量验证记录

- `npm run typecheck`：通过
- `uv run ruff check .`：失败（仓库现存问题，非本次引入）
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q`：无测试用例（exit code 5）

## 结论

1. `reasoning_effort` 链路在 FastAPI 与 TS/Workers 两套实现中都已接入，参数本身可用。  
2. 当前“请求失败”主要由 token 池为空导致，不是 `reasoning_effort` 或路由参数映射故障。  
3. 如果要完成端到端成功响应，需要先在管理端补充可用 token（或连接已有 token 存储）。  

## 未决点

- 需补一条“带可用 token 的真实成功请求”日志，作为最终上线前验收证据。
