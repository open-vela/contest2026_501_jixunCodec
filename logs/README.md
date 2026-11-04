# logs/ - AI Coding 日志目录

本目录包含 `zhangiws` 使用 Codex 桌面版完成极讯 Codec 项目期间的真实会话日志。

## 2026-11-04 补充日志

本次补充以下两个指定对话，保留原有历史日志与 manifest 会话条目：

| 对话 | session | 事件数 | UTC 日期范围 |
|---|---|---:|---|
| K7移植openvela | `01a19c4f-0ac7-7160-9868-71bcc831db7d` | 7,770 | 2026-11-02 至 2026-11-04 |
| esp32双板互通 | `01a1a1d2-7e34-7753-b109-feaeedd8e346` | 2,230 | 2026-11-03 至 2026-11-04 |

- 按 `logs/zhangiws/<UTC日期>/codex__<session_id>.jsonl` 整理，共 5 个 JSONL 文件，新增 10,000 条事件；同一会话跨日期的 `seq` 连续递增。
- 使用官方 `1.0` event/manifest schema 和 Codex 消息解析规则，并补充桌面版 `function_call` / `custom_tool_call` 及对应输出的字段转换。每条记录的 `metadata.source_line` 对应原始 rollout 行号。
- 导出真实用户消息、助手消息、工具输入和输出；运行环境注入、界面上下文和重复传输事件不作为对话再次导出。原始推理记录为加密内容，未解密、未填造 `thinking` 字段。
- 使用凭据模式进行脱敏，保留事件次序与计数；源文件 SHA256、大小、修改时间和采集时间记录在 manifest 的 `source_integrity` 中。
- 校验使用官方工具仓 `open-vela/.claude` 的 `ac36068f8571b2cced14c5f23f6cebb47026d1c6` 版本。下方“导出方式”和“会话清单”记录 2026-09 的历史导出。
- 官方 `validate-log.py logs/` 校验结果：`20 files / 85,800 events / ALL OK`；其中本次新增 `5 files / 10,000 events`。

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
