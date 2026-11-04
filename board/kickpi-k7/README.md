# KICKPI K7 / RK3576 构建与板级改动

本目录是新增的原生 K7 目标。其 RK3576 基线来自固定版本的 BeaconCat NuttX 和 PharosTech BSP，**不是** ESP32 目标使用的 `dd92bcf4` 基线。不要把本目录补丁直接用于任意版本的官方 NuttX。

## 固定来源

| 组件 | 版本 |
|---|---|
| NuttX | `BeaconCat/nuttx@31a860ad27b10dfa69be929250c17cebe9454be9` |
| BSP | `open-vela/contest2026_062_PharosTech@33edeb82c6bd6eb44b206d5299290a5ec3aac0f1` |
| apps | `open-vela/nuttx-apps@dcc6a95c3b323e533c98fde8fb209f99e24f0fdd` |
| GCC | `aarch64-none-elf-gcc 13.4.0` |
| LVGL / libuv / SpeexDSP | `9.2.1` / `1.46.0` / `1.2.1` |
| microADB | `spiriou/microADB@85cc7a90101b2ae76ee4112ba285141c8fba5012` |

详细仓库版本、下载来源与归档 SHA256 在 `sources.lock.json`。平台补丁的逐文件修改前/后 SHA256 在 `platform.lock.json`。不在此仓提交完整 SDK、工具链、启动镜像或 eMMC 备份。

## 准备独立 SDK

可用 `contest2026_501_jixunCodec_k7.xml` 拉取目标所需的固定核心源码。该清单是可选 K7 入口，原 `contest2026_501_jixunCodec.xml` 和 ESP32 工作区不变。

```bash
mkdir k7-workspace
cd k7-workspace
repo init -u https://github.com/open-vela/contest2026_501_jixunCodec.git \
  -b dev-ai-contest-2026 -m contest2026_501_jixunCodec_k7.xml
repo sync
```

此命令只有在本次改动以后获准上传后才可从远端取到新增清单；当前仅本地准备时，将待更新代码放在独立工作区的 `contest2026_501_jixunCodec/`。不要在现有同事工程中试验。

安装 Linux x86_64 的宿主工具（Python3、make、git、patch、zip/unzip、宿主 C/C++ 编译器），并准备 `genromfs`、kconfiglib 的 `olddefconfig` 等入口及上述 GCC。可复用已校验交接 SDK 的 `hosttools/`、`prebuilts/`。工具链、宿主依赖不包含在此 Git 代码更新中。

```bash
cd contest2026_501_jixunCodec
python3 scripts/fetch_k7_dependencies.py --cache ../k7-cache
python3 scripts/prepare_k7_workspace.py .. --role master --cache ../k7-cache
```

后一个命令默认只预检查。遇到未知文件摘要、部分应用补丁、占用的源码链接或非预期启动脚本会停止。依赖包也按 SHA256 校验；预检查使用临时副本验证依赖补丁，不修改 SDK。

## 编译主板与从板

在项目根目录执行，按环境设置 `K7_TOOLCHAIN_DIR`（GCC 的 bin 目录）、`K7_HOSTTOOLS_DIR`（含 bin、lib 等的宿主工具目录）。默认查找工作区的 `prebuilts/gcc/linux-x86_64/aarch64-none-elf/bin` 与工作区旁的 `hosttools/`。

```bash
bash scripts/build_k7.sh .. master ../k7-cache 2
bash scripts/build_k7.sh .. slave  ../k7-cache 2
```

脚本应用已预检查的补丁，建立 BSP/应用链接，安装 `configs/jixun_k7_ui`，按角色生成启动脚本并通过统一 `build.sh` 编译。重复构建认可脚本自己生成的 defconfig 摘要，未知修改仍会被拒绝。**同一 SDK 按顺序编译两个角色，不要并行切换 rcS。**

产物在本仓 `deliverables/kickpi-k7/rebuilt/{master,slave}/`，该目录已忽略。构建不会刷写板卡。字节摘要可能因版本信息、构建时间和源码路径变化而不同；已实装版本的摘要另见验证文档。

## 补丁内容

- `nuttx-jixun-k7.patch`：跨集群 CPU 索引映射、ES8388 S16LE 能力、FocalTech 格式修复。
- `bsp-jixun-k7.patch`：八核拓扑与启动、显示/触摸驱动、启动槽重试兼容及封装脚本。
- `apps-jixun-k7.patch`：ALSA 队列延时查询兼容、离线依赖保留、LVGL 宏适配、ADB 编译/日志兼容。
- `microadb-jixun-k7.patch`、`libuv-tls.patch`：固定依赖上的原生兼容修改。libuv 先应用 SDK 自带的 `0001`，再应用 TLS 头补丁。
- `rcS.base`：不自动启动 ADB 的串口/存储启动基底；构建时追加主/从角色和 UI 启动。

原板的显示为 FF8756 系列 720×1280，触摸为 FocalTech（I2C 0x38、实读 ID 0x87），不是 SDK 示例的 ILI9881D/GT911。软件旋转为 1280×720；DSI 视频状态应为3。原图标和布局已恢复并实际验收。

## 接线与业务参数

| 信号 | 引脚 |
|---|---|
| SPI4 SCK/MOSI/MISO | GPIO4_B0/B1/B2，AF9 |
| CS / BUSY / RESET | GPIO1_D0 / GPIO3_B0 / GPIO3_C6 |
| RXEN / OE / TXEN | GPIO2_D7 / GPIO3_C5 / SX1262 DIO2 |

433.175 MHz、配置22 dBm；接入SF8/BW500 kHz，语音SF7/BW500 kHz、CR4/5；96 ms周期、从板槽偏移48 ms。16 kHz、3K编解码；默认喇叭80、麦克风增益75。入网与对时只走LoRa，无需IP网络。

## 更新边界

当前设备使用 eMMC A/B；p3 bootctrl、p4 NuttX A、p5 NuttX B，p6保留原Linux rootfs。测试版A已确认启动成功，B保留此前语音基线。不要直接套用SDK默认AMP/整盘写入脚本，也不要覆盖BootROM/SPL、boot0/1或原分区表。

本次只整理源码，不安装新系统或重新刷写设备。N-Boot、DDR、ATF、OP-TEE仍使用原厂预编译载荷；本仓补丁不等于包含这些载荷的完整可重建源码。当前USB枚举问题仍未根治，保持板卡USB数据线断开，通过DEBUG串口调试。其余已知问题见[验证报告](../../docs/kickpi-k7/VALIDATION.md)。
