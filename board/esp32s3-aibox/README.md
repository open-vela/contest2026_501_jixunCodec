# AiBox 参考板级配置

此目录用于复现当前实测环境，不覆盖历史 `esp32s3-box/` 配置。硬件虽沿用 SDK 中 `esp32s3-box` 目标名，实际接线和屏幕属于 AiBox。

## 基线与适用边界

参考 NuttX 精确提交：`fc3f07c0871f4bdaca0d19e48bd9d8b45fe2314b`。`patch_base.json` 列出 36 个来源路径，`patches/nuttx-fc3f07c-aibox.patch` 保存参考 SDK 的板级和底层变更。包含 I2S/HAL、SPI/显示、内存、串口及相关历史修复，属于参考环境快照差异，尚未拆成适合上游审阅的最小主题补丁。

历史 BOX 的官方补丁基于 `dd92bcf4`；两套补丁不可叠加套用。`scripts/apply_aibox_patch.sh` 只接受精确参考基线，已应用时通过反向检查识别，不接受未知版本。

当前参考 SDK 不随仓库提供。构建需要完整匹配的 NuttX、apps、external、frameworks、packages、vendor、tests 和 Xtensa ESP32-S3 工具链。已有工具链可通过 SDK 的 `prebuilts/` 或 PATH 使用。还需 `python3` 和与参考 NuttX 构建命令兼容的 `esptool.py`；入口将用户 `.local/bin` 加入 PATH 并检查依赖。仅凭应用目录和本补丁，不能宣称可在任意官方 SDK 完成构建。

正式比赛基线迁移待办：以官方 manifest 拉取完整独立工作区，将补丁分主题移植；消除不必要的旧修复；两角色 clean build；复核 LCD、音频、BOOT、LoRa 和分区；重做板测及哈希；按官方指南在 NuttX fork 提 PR。此次未修改官方公共仓。

## 接线与配置

| 功能 | GPIO / 参数 |
|---|---|
| BOOT | GPIO0，低电平按下 |
| 屏幕 | ST7789，240×240，非触摸 |
| LCD SPI2 | CS5 / CLK4 / MOSI2 / MISO41 |
| LCD 控制 | DC47 / RST38 / 背光42（LEDC、低有效） |
| ES8311 I2C0 | SCL14 / SDA15 |
| I2S0 | MCLK16 / BCLK9 / WS45 / DOUT8 / DIN10 |
| 扬声器 PA | GPIO46 |
| SX1262 SPI3 | CS13 / CLK17 / MOSI18 / MISO21 |
| SX1262 控制 | BUSY40 / TXEN11 / RXEN12 |

启动 UI 发送/自动接收路径默认 SF8、BW 配置值 `0x06`；不要误用驱动文件顶部旧的 SF7 注释作为当前参数。诊断环境变量可覆盖部分默认值，应随测评记录。Codec 采样率 16 kHz，I2S 配置/播放路径的硬件速率由具体初始化代码设置，不能把 defconfig 中的 24 kHz 初始值直接当作模型采样率。

`configs/tx/defconfig` 和 `configs/rx/defconfig` 来源于对应实测固件，Wi-Fi SSID、口令、API Key 等字段清空。`partition_table.csv` 为参考表，需与设备现有 bootloader/分区核对，不自动烧录。

## 构建与烧录边界

```bash
bash scripts/build_aibox.sh /path/to/reference-sdk-copy tx 4
bash scripts/build_aibox.sh /path/to/reference-sdk-copy rx 4
```

该入口修改传入 SDK，生成的新角色源码在 SDK 内独立目录。请使用副本，并确认没有指向原工程的可写源码/生成目录符号链接；工具链的只读链接可以保留。构建记录注明采用已有 SDK 副本还是全新工作区，二者不要混称。

本轮只整理与构建，未烧录。此前硬件 TX=COM12、RX=COM18；串口号仅为当时记录，不能作为将来烧录的设备识别依据。只写应用区也需要先核对设备身份和分区；本仓不提供自动擦除或自动刷机动作。

参考 SDK 的 `apps/` 下若存在旧 `jixun-codec` 实目录或 `jixun-doc_*` 备份目录，需在**SDK 副本内**先将它们完整移到 `apps/` 之外保存。NuttX 会自动包含所有顶层 Make.defs，仅改目录名不能禁用旧注册。构建入口遇到这些冲突会拒绝继续，不会替换实目录或静默选用旧源码。
