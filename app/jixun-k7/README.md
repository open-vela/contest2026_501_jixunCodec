# K7 原生离线 LoRa 语音与触屏控制台

本目标运行于 RK3576/openVela，使用 Light-A 3K 编解码、SX1262、ES8388 和 LVGL。与 `app/jixun-codec/` 的 ESP32 三档码率目标分别构建，不替换原 ESP32 代码。

- `k7_pair_main.c`：节点生命周期、单一射频所有者、接入/通话切换。
- `access_sync.h`：JR v2 接入、注册、单调时钟对齐与时隙锚点协商。
- `k7_voice_main.c`、`call_mode.h`：编码、时隙收发、解码、播放和计数。
- `A_fixed/`：当前模型配置、完整 INT8 权重、NEON 与并行内核。此目标固定 `SQM_RATE_3K`。
- `rf/`：SX1262 和 RK3576 原生 SPI/GPIO 适配。
- `k7_ui_main.c`、`ui/console_views.inc`：LVGL 显示/触摸和七页业务界面。
- `ui/`：已生成的原极讯图标、中文字体 C 资源及 Noto 许可；普通构建无需重新生成。
- `speex/`：随应用携带的 SpeexDSP，保留原版权和 `COPYING`。

主/从使用同一份代码与权重，启动脚本分别调用 `k7_pair node a` / `k7_pair node b`，然后 `k7_ui &`。按一次“开始通话”后持续通话，任意一端结束后恢复入网待机。

构建与固定 SDK 基线见 [板级说明](../../board/kickpi-k7/README.md)。实装固件来源和本次整理的注释与行尾规范化记录见 `SOURCE_PROVENANCE.json`；历史上板验证与新目录构建验证分别报告，不将新编译产物冒充已实装版本。
