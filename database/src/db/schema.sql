-- Minimal schema for phase-2 implementation (SQLite version).

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS user (
  user_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  username     TEXT NOT NULL UNIQUE,
  password     TEXT NOT NULL,
  phone        TEXT UNIQUE,
  email        TEXT UNIQUE,
  photo        TEXT,
  create_time  TEXT NOT NULL DEFAULT (datetime('now')),
  status       INTEGER NOT NULL DEFAULT 1,
  role         TEXT NOT NULL DEFAULT 'user' -- 'user' | 'admin'
);

CREATE TABLE IF NOT EXISTS face_feature (
  feature_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id         INTEGER NOT NULL,
  feature_vector  BLOB NOT NULL, -- fixed: 512 * float32 bytes
  create_time     TEXT NOT NULL DEFAULT (datetime('now')),
  is_active       INTEGER NOT NULL DEFAULT 1,
  image_path      TEXT,
  FOREIGN KEY(user_id) REFERENCES user(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recognition_log (
  log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id         INTEGER, -- nullable: unknown/stranger
  input_image_url TEXT,
  similarity      REAL,
  result          INTEGER NOT NULL, -- 1 success, 0 fail
  recognize_time  TEXT NOT NULL DEFAULT (datetime('now')),
  device_info     TEXT,
  FOREIGN KEY(user_id) REFERENCES user(user_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS system_config (
  config_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  config_key   TEXT NOT NULL UNIQUE,
  config_value TEXT NOT NULL,
  update_time  TEXT NOT NULL DEFAULT (datetime('now')),
  description  TEXT
);

INSERT OR IGNORE INTO system_config (config_key, config_value, description)
VALUES ('threshold', '0.65', 'cosine similarity threshold');

