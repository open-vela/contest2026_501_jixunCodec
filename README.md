# 极讯 AI Codec：端侧神经语音通信

选题方向：**AI 硬件产品创新**。在 ESP32-S3 上完成 16 kHz 语音采集、INT8 神经编解码和 Token 传输，接收板解码后通过扬声器播放；PC 只负责控制、导出与评分。

## 当前版本与硬件

| 配置 | 用途 | 应用入口 | 板级入口 | 验证范围 |
|---|---|---|---|---|
| AiBox，240×240 非触摸屏 | 2026-11-04 双板 LoRa 3k 实测与简约 UI | `app/jixun-aibox/` | `board/esp32s3-aibox/` | 原实测固件已保存；本次整理验证见下方报告 |
| ESP32-S3-BOX，480×320 | 2026-09 历史 1k/3k/6k 实现 | `app/jixun-codec/` | `board/esp32s3-box/` | 历史报告，不能代表当前 AiBox 稳定性 |

当前 AiBox 只整理 **3 kbps** 模型。发送板短按实体 **BOOT** 开始约 3 秒录音，听完两声开始音后说话，结束有一声低音；录音期间再按一次可请求提前结束。接收板开机自动监听，BOOT 切换信息显示。网页预览只用于外观参考。

源码沿用现有收发版本，本次整理未进一步裁剪模型或替换权重。当前编码首块为此前优化后的 32 通道（原导出 80），解码张量保持指定 light-A INT8 模型的值。发送/接收通过公共源码加 7 个接收侧覆盖文件生成，避免维护两份整目录。详见 [AiBox 应用说明](app/jixun-aibox/README.md)。

## K7 原生离线 LoRa 语音与屏幕 UI（2026-11-04）

新增 KICKPI K7 / RK3576 openVela 主从目标，固定 Light-A 3K 模型，通过 SX1262 完成离线入网、单调时钟对齐和持续双向语音，LVGL 界面沿用原极讯图标与布局。两板触摸、冷启动自动入网、界面发起/结束通话已验证；偶发栈标记改写与 USB 枚举问题仍未根治。

- 源码：`app/jixun-k7/`；固定基线及可检查补丁：`board/kickpi-k7/`。
- 目标清单：`contest2026_501_jixunCodec_k7.xml`。原 ESP32 清单、应用和三档码率实现保持独立。
- 构建：先按 [K7 板级说明](board/kickpi-k7/README.md) 准备独立 SDK，再执行 `scripts/build_k7.sh`，分别选择 `master` / `slave`。
- 数据与边界：[K7 验证记录](docs/kickpi-k7/VALIDATION.md)；串口工具：[host/k7](host/k7/README.md)。
- 二进制、完整 SDK、工具链和本地交接包不进入此次 Git 更新；固件保留在被忽略的 `deliverables/kickpi-k7/`。

## AiBox 构建

**当前实测 SDK 的 NuttX 基线为 `fc3f07c0871f4bdaca0d19e48bd9d8b45fe2314b`，与历史 BOX 官方基线 `dd92bcf4` 不同。** AiBox 参考补丁尚未迁移验证到官方 manifest 基线，不应把参考 SDK 构建通过写成官方 SDK 构建通过。

在一份可丢弃的、依赖完整的参考 SDK 副本上运行（包含 `nuttx/`、`apps/`、工具链及依赖）：

```bash
bash scripts/build_aibox.sh /path/to/reference-sdk-copy tx 4
bash scripts/build_aibox.sh /path/to/reference-sdk-copy rx 4
```

脚本检查 NuttX 精确基线、应用参考补丁、生成独立角色源码并编译。产物写到 SDK 副本的 `aibox-tx.*/`、`aibox-rx.*/`，不会自动烧录。**该脚本会修改所传 SDK 的配置和构建文件，因此必须传入副本。** 参考 SDK 不随本仓打包，获取与官方迁移事项见 [板级说明](board/esp32s3-aibox/README.md)。

只生成收发源码、不编译：

```bash
python3 scripts/materialize_aibox.py --role tx --output /tmp/jixun-tx
python3 scripts/materialize_aibox.py --role rx --output /tmp/jixun-rx
python3 scripts/test_aibox_materializer.py
```

输出目录必须不存在，且位于本仓外。历史 BOX 构建、控制台与烧录说明见 [历史版本说明](docs/历史BOX使用说明_20260920.md)，其中串口号和结果仅适用于当时环境。

## 实测与 AMR-NB 对比

同一 ESP32-S3 上对比 3k 与 AMR-NB MR475。PC 没有代替 ESP32 运行编解码。

| 指标 | 神经 Codec 3k | AMR-NB MR475 | 口径 |
|---|---:|---:|---|
| 语音码率 | 3.00 kbps | 4.75 kbps | 纯语音，不含无线封装；3k 少 36.84% |
| 模型音频采样率 | 16 kHz | 8 kHz | 模型输入输出；不等同于有效听感带宽 |
| 固定 3 秒单次解码耗时均值 | 8470 ms | 152.577 ms | 各 3 次，同板历史独立解码测量 |
| PESQ-NB 均值 | 2.8420 | 3.3789 | 3 段同输入，统一 8 kHz 评分 |
| STOI 均值 | 0.8752 | 0.8921 | 同上，不能作为统计等效结论 |
| 固定 3 秒样本 PESQ-WB | 2.2715 | 2.1689 | 单段补充指标；不代表均值领先 |

3k 的明确优势是低码率和原生 16 kHz 输入输出，部分样本宽带得分与信噪比更高；当前板端解码速度是主要差距。AMR 未测 LoRa 到扬声器全链路，不能拿其独立解码耗时替代端到端首音时间。完整逐段数据、失败轮次和限定条件见 [测评入口](docs/验证报告/20261104_ESP32_3k_AMRNB/README.md)。

本次 UI/提示音版本与历史音质测评固件分别记录 SHA256，**不借用历史得分证明新固件音质等价**。当前版本的已测结果、按键问题和双路导出阻塞见 [当前版本说明](docs/AiBox当前版本与限制_20261104.md)。

## 仓库结构

```text
app/jixun-aibox/          当前 AiBox 公共源码、角色覆盖、3k 权重和 UI 检查
app/jixun-codec/          保留的历史 BOX 三档实现
board/esp32s3-aibox/      当前参考 SDK 补丁、收发配置和基线说明
board/esp32s3-box/        保留的历史官方基线补丁
host/benchmarks/         板端导出数据的离线评分入口
host/                    原 BOX 控制台与回归工具
scripts/                 构建、补丁和源码生成工具
logs/                    既有真实 AI Coding 日志；本次新增会话尚待正式导出
docs/                   使用说明、量化对比、验证边界和要求对照
deliverables/           原始证据和固件附件，本地保留且不进入 Git
```

## 更新与比赛要求

[要求对照及待办](docs/仓库要求与本次更新_20261104.md)、[本次检查清单](REVIEW_CHECKLIST.md)、[验证记录](docs/整理验证记录_20261104.md)。

既有 `logs/` 会话保留，本次新增开发会话仍需按[官方日志手册](https://github.com/open-vela/docs/blob/dev-ai-contest-2026/zh-cn/contest_2026/ai_coding_log_guide.md)在完整 openvela 工作区中导出和校验。后续提交遵循[官方提交指南](https://github.com/open-vela/docs/blob/dev-ai-contest-2026/zh-cn/contest_2026/code_submission_guide.md)，公共 NuttX 改动的移植及上游 PR 单独处理。

本仓新增代码按 Apache License 2.0 提交，上游组件保留各自许可证。见 [LICENSE](LICENSE)、[NOTICE](NOTICE)。模型、测试音与字体等资产沿用原工程来源说明；公开发布前需确认相应授权，本文不新增第三方授权声明。
