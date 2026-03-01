CREATE TABLE IF NOT EXISTS api_keys (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL UNIQUE,
  enabled INTEGER NOT NULL DEFAULT 1,
  note TEXT DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_api_keys_enabled ON api_keys(enabled);

CREATE TABLE IF NOT EXISTS request_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  method TEXT NOT NULL,
  path TEXT NOT NULL,
  status INTEGER NOT NULL,
  key_prefix TEXT DEFAULT '',
  colo TEXT DEFAULT '',
  ip TEXT DEFAULT '',
  user_agent TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);

INSERT OR IGNORE INTO api_keys (key, enabled, note)
VALUES ('12345', 1, 'default key from WORKER_API_KEYS');
