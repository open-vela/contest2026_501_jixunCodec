# logs/ - AI Coding 日志目录

本目录包含 `zhangiws` 使用 Codex 桌面版完成极讯 Codec 项目期间的真实会话日志。

## 导出方式

- 来源：`$CODEX_HOME/sessions/**/rollout-*.jsonl` 中 `session_meta.cwd=E:\jixuncodec_prj` 的项目会话。
- 适配：使用官方 event/manifest schema，并按官方 Codex rollout 适配规则把 `payload` 消息、推理和工具调用摊平。
- 编码：JSONL 使用 ASCII 转义，避免 Unicode 行分隔符破坏单行结构；事件内容、顺序和 `seq` 未被删减。
- 脱敏：使用官方默认规则，并额外将 `tp-...` API Key 替换为 `tp-***REDACTED***`。
- 校验：官方 `contest-log-collector/tools/validate-log.py` 检查结果为 `15 files / 75,800 events / ALL OK`。

## 会话清单

| session | 事件数 | 日期范围 |
|---|---:|---|
| `01a089e1-c962-71f1-aee6-3b3ed2f367a2` | 49,473 | 2026-09-10 至 2026-09-18 |
| `01a0ad05-b996-7380-b60c-b2d73dbe9666` | 409 | 2026-09-17 |
| `01a0b225-bc15-71d3-ad79-d6eab0dedf08` | 22,753 | 2026-09-18 至 2026-09-20 |
| `01a0bd4f-7da4-70b2-aa55-e7b52be36863` | 42 | 2026-09-20 |
| `01a0bda5-0d2c-73c3-893d-13c1fe09dd9a` | 3,123 | 2026-09-20 |

每个会话的原始 rollout 路径、SHA256、字节数和采集时间记录在
`zhangiws/manifest.json` 的 `source_integrity` 字段中。

## 目录结构

```text
logs/
└── <github_login>/              # 你的 GitHub 用户名，一人一目录
    ├── manifest.json            # 会话清单
    └── <date>/                  # 日期 YYYY-MM-DD
        └── <tool>__<sid>.jsonl  # 一个会话一个文件（工具名与 session id 用 __ 连接）
```

- `<tool>`：`claude-code` / `opencode` / `codex` / `kiro` / `mimocode` / `cursor`
- 每个 `.jsonl` 每行一个事件，由组委会提供的日志归集工具导出，**只提交 JSONL 本身**。

导出与提交的完整步骤、字段定义见[《AI Coding 日志归集与提交手册》](https://github.com/open-vela/docs/blob/dev-ai-contest-2026/zh-cn/contest_2026/ai_coding_log_guide.md)。
