# Cloudflare Worker 部署说明

本仓库通过 GitHub Actions 自动发布 Worker。

## 1. 必填 GitHub Secrets

在仓库 `Settings -> Secrets and variables -> Actions` 添加：

- `CLOUDFLARE_API_TOKEN`：Cloudflare API Token（至少包含 Workers Scripts:Edit、Workers Routes:Edit、Zone:Read）
- `CLOUDFLARE_ACCOUNT_ID`：Cloudflare Account ID
- `UPSTREAM_BASE_URL`：上游 API 地址（例如 `http://你的服务器域名`）
- `WORKER_API_KEYS`：对外鉴权 key，支持多个，逗号分隔
- `UPSTREAM_AUTH_BEARER`（可选）：若填写，Worker 转发时会强制使用该上游令牌

## 2. 自动部署触发

- 推送到 `main`，且变更命中以下路径时自动部署：
  - `cloudflare/worker/**`
  - `.github/workflows/deploy-cloudflare-worker.yml`
- 也可在 Actions 页面手动触发 `Deploy Cloudflare Worker`。

## 3. 路由与域名

`cloudflare/worker/wrangler.toml` 当前配置路由为：

- `grokapi.ds-everything-ocean.xyz/*`

请确保 Cloudflare DNS 中存在对应 `grokapi` 记录且为代理状态（橙云）。
