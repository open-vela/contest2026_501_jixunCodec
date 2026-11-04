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

/* Screen rotation: -1 = 90 deg counter-clockwise (left), 0 = off,
 * 1 = 90 deg clockwise (right).  Takes effect immediately. */
void jx_ui_set_rot(int rot);
int  jx_ui_get_rot(void);

/* BOOT (GPIO0) polling gate: 0 makes the UI thread ignore the key so a
 * PTT command can own it. */
void jx_ui_set_boot_enable(int on);

/* BOOT (GPIO0) onsite diagnostic: current level (0 = pressed) and gate state.
 * trigger_send runs the exact BOOT callback so the link can be checked from
 * the console without a finger on the key. */
int  jx_ui_boot_level(void);
int  jx_ui_boot_enabled(void);
void jx_ui_trigger_send(void);

/* 1.3 AI BOX 没有公开按键原理图：自动扫描安全脚，任意脚稳定翻转即
 * 视为按键。jx_ui_key_gpio() 返回已锁定的脚（-1 = 还没识别到），
 * jx_ui_key_set_gpio(N) 可现场指定，jx_ui_key_map() 打印全部候选脚。 */
int  jx_ui_key_gpio(void);
void jx_ui_key_set_gpio(int gpio);
void jx_ui_key_map(void);

/* Poll BOOT for N seconds and print every edge (field check for the key). */
void jx_ui_boot_watch(int secs);

/* Read-only scan of candidate pads: prints any pad whose input level
 * changes while the key is pressed. Does not reconfigure any pad. */
void jx_ui_bootdiag(void);
void jx_ui_allscan(int secs);
void jx_ui_pin_scan(int secs);

/* Self-test: drive GPIO0 low for N ms to emulate a BOOT press.
 * Safe on this board (10k pull-up + key to GND). */
void jx_ui_pull_boot(int ms);

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

/* 话音页通信统计：收包数 + RSSI、已发包数。
 * 屏幕直接显示，不用看串口就知道这一句收了多少包、信号多强。 */
void jx_ui_set_rx(int pkt, int rssi);
void jx_ui_set_tx(int pkt);

/* 把屏幕当前真正持有的状态字符串/页面/按键脚打印到串口，
 * 用于不读图也能确认“REC / TX DONE / RX N 包”确实写进了 UI。 */
void jx_ui_status_dump(void);

/* UI 侧保存的对端地址，供发送任务读取。 */
void jx_ui_set_peer(const char *ip);
const char *jx_ui_peer_get(void);

/* 自检：逐项打印耗时（屏、满屏填充、文字、波形） */
int  jx_ui_selftest(void);

/* 显存回环自检：putarea 写图案 → getrun 读回 → 逐像素比对。
 * 用来区分「驱动写不进去」和「背光/面板不亮」——屏幕黑的时候先跑这个。 */
int  jx_ui_rdtest(void);

/* Read back the panel and report where the four logical corners land. */
int  jx_ui_rotcheck(void);

/* 触摸诊断：阻塞读取 /dev/input0 若干秒，打印坐标和命中的按钮页。 */
int  jx_ui_touch_test(int seconds);

#endif /* __JX_UI_H */
