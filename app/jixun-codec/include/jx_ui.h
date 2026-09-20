/****************************************************************************
 * apps/jixun-codec/include/jx_ui.h
 *
 * 极讯 Codec 单板 UI —— ST7796 480x320 仪表盘
 *
 * 设计约束（重要）：显示不能抢编解码的资源。
 *   1. 屏走 SPI2 @64MHz，编解码吃 PSRAM 带宽和 CPU —— 总线不重叠；
 *   2. 暂存缓冲只用内部 SRAM（malloc 小缓冲），绝不碰 PSRAM；
 *   3. 编解码期间 jx_ui_set_busy(1)，所有绘制入口直接返回，占用为 0；
 *   4. 只重画变化的小块，不做周期全屏刷新。
 * 字库从 SD 卡读：/mnt/sdcard/jixun_font.bin（PC 侧 _fontgen.py 生成）。
 ****************************************************************************/

#ifndef __JX_UI_H
#define __JX_UI_H

/* 初始化：取 LCD、分配暂存、挂 SD、载字库。
 * 返回 0 成功；<0 失败（屏不可用）。字库缺失不算失败，g_font_ok 会报 0。 */
int  jx_ui_init(void);

/* 屏与字库状态 */
int  jx_ui_ready(void);
int  jx_ui_font_ok(void);
const char *jx_ui_font_path(void);

/* 编解码闸门：1 = 忙，UI 一律不绘制 */
void jx_ui_set_busy(int busy);
int  jx_ui_busy(void);

/* 整屏重绘（含顶栏/底栏 + 指定页）。n 越界会被夹到合法范围。 */
int  jx_ui_page(int n);
int  jx_ui_page_count(void);
int  jx_ui_page_get(void);

/* 顶栏现场信息（链路/时钟），可随时调用；忙时不画 */
void jx_ui_set_link(int up, int rssi_bars);

/* 日志页：追加一行（环形缓冲，最多保留 8 行） */
void jx_ui_log(const char *s);

/* 话音页波形：画一段 PCM（int16），忙时丢弃 */
void jx_ui_wave(const int16_t *pcm, int n);

/* 话音页上的两组实时数值（码率 kbps / 令牌·帧） */
void jx_ui_stat(const char *rate_kbps, const char *tokinfo);

/* 话音页“发送语音”按钮的回调。回调应快速返回，不要在里面做录音/编码。 */
typedef void (*jx_ui_send_cb_t)(void);
void jx_ui_set_send_cb(jx_ui_send_cb_t cb);

/* 话音页状态文字，例如“发送中... / 发送完成 / 未设置对端”。 */
void jx_ui_set_status(const char *s);

/* UI 侧保存的对端地址，供发送任务读取。 */
void jx_ui_set_peer(const char *ip);
const char *jx_ui_peer_get(void);

/* 自检：逐项打印耗时（屏、满屏填充、文字、波形） */
int  jx_ui_selftest(void);

/* 显存回环自检：putarea 写图案 → getrun 读回 → 逐像素比对。
 * 用来区分「驱动写不进去」和「背光/面板不亮」——屏幕黑的时候先跑这个。 */
int  jx_ui_rdtest(void);

/* 触摸诊断：阻塞读取 /dev/input0 若干秒，打印坐标和命中的按钮页。 */
int  jx_ui_touch_test(int seconds);

#endif /* __JX_UI_H */
