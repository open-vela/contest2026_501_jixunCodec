# 极讯 AI Codec：端侧神经 Token 语音通信系统

## 一、作品简介

极讯 AI Codec 在两块 ESP32-S3-BOX V2.0 上运行 openvela/NuttX，实现端侧神经语音编解码与双板 Token 传输。发送端把 16 kHz PCM 编码为离散 FSQ Token，接收端完成分片校验、重组和神经解码后播放语音；全链路只传输 Token，不传输 PCM。

系统提供 `1 kbps`、`3 kbps`、`6 kbps` 三档码率。PC 控制台可选择码率并发起测试，板端界面统一以 `tokens/s` 展示实际 Token 速率。测试音当前为 2 秒，仅用于固定对照；链路按 500 ms 分块连续流水，并不限制语音时长。

## 二、选题方向

选题方向：**AI 硬件产品创新**。

作品把神经 Codec、INT8 量化、ESP32-S3 PIE 向量内核、双核调度、USB/Wi-Fi Token 传输、480x320 板端显示和 PC Token 观察界面组合成完整闭环。

## 三、主要实测结果

- `1K`：2 秒测试音输出 120 Token、4 个分片；`4/4` 到达，`played=1/1`，播放完成 16.852 s。
- `3K`：2 秒测试音输出 336 Token、756 B、4 个分片；`4/4` 到达，`played=1/1`，播放完成 15.506 s。
- `6K`：2 秒测试音输出 668 Token、8 个分片；`8/8` 到达，`played=1/1`，播放完成 18.467 s。
- 质量：板端真实中文样本 PESQ-WB 为 `1K=1.779`、`3K=2.181`、`6K=2.669`。
- 稳定性：三档分别在两块板上各运行 50 轮，共 300 轮，无断言、无串口超时，同码率双板指纹一致。

完整数据和原始日志说明见 `docs/验证报告/`。

## 四、目录结构

```text
app/jixun-codec/                应用源码、三档权重和板端 UI
board/esp32s3-box/              ESP32-S3-BOX 板级补丁与 openvela defconfig
host/                           PC Token 中继、控制台、烧录和稳定性测试工具
skills/jixuncodec-rate-guard/   自定义速率门禁 Skill
deliverables/                   技术报告、演示视频等本地交付物（按团队要求不进入远程仓）
docs/验证报告/                  三档码率、时延、质量和稳定性验证数据
logs/                           AI Coding 日志目录（提交前由官方工具导出）
scripts/                        构建和板级补丁应用脚本
```

## 五、构建前准备

1. 使用组委会 manifest 拉取完整 openvela 工作区。
2. 板级补丁已重放到比赛分支 `dev-ai-contest-2026` 的 NuttX 基线 `dd92bcf4`，并用 `git apply --check` 验证可干净应用。
3. 应用板级补丁，并把本项目 defconfig 放入 ESP32-S3-BOX 配置目录：

```bash
cd contest2026_501_jixunCodec
./scripts/apply_board_patch.sh /path/to/openvela-workspace
```

脚本只修改 openvela 工作区内的 `nuttx` 目录；参赛仓本身结构不会被改动。

## 六、编译

推荐直接使用 openvela 统一入口，并通过环境变量选择模型：

```bash
cd /path/to/openvela-workspace
JX_RATE=3k ./build.sh esp32s3-box:openvela distclean
JX_RATE=3k ./build.sh esp32s3-box:openvela -j8
```

支持 `JX_RATE=1k|3k|6k`。如果当前工作区使用 Make 两遍构建流程，也可以执行：

```bash
./contest2026_501_jixunCodec/scripts/build_jixun.sh /path/to/openvela-workspace 3k 8
```

产物：`nuttx/nuttx.bin`。

## 七、烧录

两块板固定角色：`COM23` 发送，`COM4` 接收。只写应用区 `0x10000`：

```bash
python -m esptool --chip esp32s3 --port COM23 --baud 921600 \
  --before default-reset --after hard-reset write-flash 0x10000 nuttx/nuttx.bin

python -m esptool --chip esp32s3 --port COM4 --baud 921600 \
  --before default-reset --after hard-reset write-flash 0x10000 nuttx/nuttx.bin
```

不要执行 `erase_flash`，不要覆盖 bootloader 和分区表。

## 八、PC 控制台

PC 侧位于 `host/`，需要 Python 3、`pyserial` 和 Windows 自带 Tk：

```bat
cd host
python -m pip install -r requirements.txt
SwitchCodec_1k.cmd
SwitchCodec_3k.cmd
SwitchCodec_6k.cmd
```

控制台负责选择码率、烧录对应固件、转发 Token 和展示分片/播放状态。PC 不参与 PCM 解码或播放。

快速稳定性回归：

```bat
python _rate_stability_smoke.py --rates 1k 3k 6k --cycles 1 --restore 3k
```

## 九、板级改动说明

`board/esp32s3-box/patches/nuttx-dev-ai-contest-2026-jixun.patch` 汇总了本作品在 NuttX 公共仓上的改动，包括：

- ESP32-S3 PSRAM、Wi-Fi、I2S、摄像头和内存布局修复；
- ST7796 LCD 初始化、SPI 片选保持和大块写屏修复；
- LCD 与 GT911 触摸共享 GPIO48 复位脚，启动顺序改为先触摸、最后初始化 LCD，避免面板被二次复位后黑屏；
- 板端 UI 自动启动和 SPI/显示资源隔离。

`board/esp32s3-box/patches/esp-hal-3rdparty-jixun.patch` 补充 ESP HAL 的自旋锁初始化编译修复。mbedTLS 的自动化命名适配不在本仓携带，由 NuttX 构建流程按官方补丁重新生成。

`configs/openvela/defconfig` 是当前可运行配置。其中的 API Key 字段已清空，提交仓不包含真实密钥。

## 十、AI Coding 日志

`logs/` 当前只保留官方格式说明，尚未放入真实会话日志。正式上传前必须在 openvela 工作区内运行组委会日志采集工具，完成会话导出和 `validate-log.py` 校验，再由人工确认后提交。

## 十一、许可证

本仓新增代码按 Apache License 2.0 提交；引入的上游组件继续遵循各自许可证。详见 `LICENSE` 和 `NOTICE`。
