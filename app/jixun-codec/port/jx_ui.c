/****************************************************************************
 * apps/jixun-codec/port/jx_ui.c
 *
 * 极讯 Codec 单板 UI —— ST7796 480x320 仪表盘
 *
 * 资源纪律（这是本文件所有设计取舍的理由）：
 *   1. 屏走 SPI2 @64MHz（NuttX st7789 驱动，PIO 模式），编解码吃 PSRAM
 *      带宽 + 双核 CPU —— 两者总线不重叠，所以显示本身不抢编解码的带宽；
 *   2. 唯一的暂存 B->scr 是 480x16 RGB565 = 15KB，只从内部 SRAM 分配，
 *      绝不落 PSRAM（落了就直接抢编解码的 52~85MB/s）；
 *   3. 编解码期间 jx_ui_set_busy(1)，所有绘制入口第一行就返回，占用为 0；
 *   4. 大块用 putarea 一次发（一次 setarea + 一次 FIFO 传输），不做逐行 putrun。
 *
 * 实测成本（64MHz / PIO）：满屏 480x320 = 307KB ≈ 38ms（只在切页时做），
 * 一个数值卡片 120x32 ≈ 7.7KB ≈ 1ms，文字一行 ≈ 0.5ms。
 ****************************************************************************/

#include <nuttx/config.h>
#include <nuttx/board.h>
#include <nuttx/lcd/lcd.h>

#include <sys/stat.h>
#include <sys/mount.h>
#include <sys/ioctl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>

#include <nuttx/input/touchscreen.h>

#include "jx_ui.h"
#include "jx_font_bin.h"   /* flash 内置字库（生成见 _fontgen.py） */

extern const char *jx_rate_name(void);

/****************************************************************************
 * 常量
 ****************************************************************************/

#define UI_W       480
#define UI_H       320
#define BAND_H     16                    /* 暂存高度（= 字高） */
#define SCR_PIX    (UI_W * BAND_H)

#define RGB(r,g,b) ((uint16_t)((((r) & 0xf8u) << 8) | (((g) & 0xfcu) << 3) | ((b) >> 3)))

/* 配色（取自 UI 稿的 CSS 变量） */
#define C_BG      RGB(0x05,0x0b,0x0f)
#define C_PANEL   RGB(0x0b,0x16,0x1d)
#define C_PANEL2  RGB(0x08,0x13,0x1a)
#define C_LINE    RGB(0x17,0x26,0x2e)
#define C_LINE2   RGB(0x1b,0x2c,0x35)
#define C_TXT     RGB(0xc8,0xe4,0xec)
#define C_HI      RGB(0xe8,0xf6,0xfa)
#define C_DIM     RGB(0x7c,0x98,0xa4)
#define C_DIM2    RGB(0x5a,0x74,0x81)
#define C_CY      RGB(0x22,0xd3,0xee)
#define C_OK      RGB(0x4a,0xde,0x80)
#define C_WARN    RGB(0xef,0x9f,0x27)
#define C_BAD     RGB(0xef,0x44,0x44)
#define C_BTN     RGB(0x0d,0x1a,0x22)
#define C_BTNL    RGB(0x1e,0x32,0x3d)
#define C_TOPBG   RGB(0x08,0x12,0x1a)

/* 三段式布局 */
#define TOP_H     24
#define BODY_Y    TOP_H
#define BODY_H    200
#define BOT_Y     (TOP_H + BODY_H)       /* 224 */
#define BOT_H     (UI_H - BOT_Y)         /* 96  */

#define NBTN      6
#define BTN_W     73
#define BTN_GAP   5
#define BTN_X0    8
#define BTN_Y     228
#define BTN_H     84

#define NPAGE     5

/* 话音页内置的“发送语音”按钮 */
#define SEND_X    350
#define SEND_Y    (BODY_Y + 98)
#define SEND_W    120
#define SEND_H    84

/* 触摸输入 */
#define TOUCH_PATH  "/dev/input0"
#define TOUCH_MAXPOINT 8

/****************************************************************************
 * 状态
 ****************************************************************************/


/* 字库 */
#define FONT_PATH "/mnt/sdcard/jixun_font.bin"

/* 所有稍大的缓冲集中成一块，运行期一次 malloc。
 * 理由：dram0_0_seg（内部 DRAM 数据段）只剩约 200 字节，静态放置会直接
 * 链接失败 —— 第一版就是这么溢出 1200 字节的。 */
#define WAVE_N  224
#define LOGN    8
#define LOGL    64

typedef struct
{
  char    log[LOGN][LOGL];
  char    stat_rate[8];
  char    stat_tok[16];
  int16_t wave[WAVE_N];
  int     log_n;
  int     log_head;
  int     wave_n;
  volatile int dirty;

  /* 下面这些原来都是文件作用域静态量。内部 DRAM 只剩约 8 字节，这 72
   * 字节的 .bss/.data 直接把链接撑爆（溢出 64 字节），所以全部搬到堆上。 */
  struct lcd_dev_s *lcd;
  uint16_t         *scr;
  const uint8_t    *font;
  int (*pa)(FAR struct lcd_dev_s *, fb_coord_t, fb_coord_t,
            fb_coord_t, fb_coord_t, FAR const uint8_t *, fb_coord_t);
  int (*pr)(FAR struct lcd_dev_s *, fb_coord_t, fb_coord_t,
            FAR const uint8_t *, size_t);
  int (*gr)(FAR struct lcd_dev_s *, fb_coord_t, fb_coord_t,
            FAR uint8_t *, size_t);
  uint32_t fsz;
  uint32_t fdata;
  int      gw, gh, fc;
  int      ready;
  int      bpp;
  int      busy;
  int      page;
  int      link_up;
  int      rssi;
  int      font_ok;
  int      font_src;      /* 0 = flash 内置，1 = SD 卡覆盖 */
  int      touch_started;
  int      send_latched;
  int      send_pressed;
  char     voice_msg[40];
  char     peer[32];
  jx_ui_send_cb_t send_cb;
  pthread_mutex_t draw_lock;
  int      draw_lock_ok;
} ui_big_t;

static ui_big_t *B;

/****************************************************************************
 * 小工具
 ****************************************************************************/

static double now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static uint32_t rd32(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t rd16(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

/* UTF-8 解码；返回下一个字节位置 */
static const char *u8next(const char *s, uint32_t *cp)
{
  unsigned char c = (unsigned char)*s++;

  if (c < 0x80u) { *cp = c; return s; }

  if ((c & 0xe0u) == 0xc0u)
    {
      *cp = ((uint32_t)(c & 0x1fu) << 6) | ((unsigned char)*s & 0x3fu);
      return s + 1;
    }

  if ((c & 0xf0u) == 0xe0u)
    {
      *cp = ((uint32_t)(c & 0x0fu) << 12) |
            ((uint32_t)((unsigned char)s[0] & 0x3fu) << 6) |
            ((uint32_t)((unsigned char)s[1] & 0x3fu));
      return s + 2;
    }

  if ((c & 0xf8u) == 0xf0u)
    {
      *cp = ((uint32_t)(c & 0x07u) << 18) |
            ((uint32_t)((unsigned char)s[0] & 0x3fu) << 12) |
            ((uint32_t)((unsigned char)s[1] & 0x3fu) << 6) |
            ((uint32_t)((unsigned char)s[2] & 0x3fu));
      return s + 3;
    }

  *cp = '?';
  return s;
}

static int str_chars(const char *s)
{
  int n = 0;
  uint32_t cp;

  while (*s != '\0') { s = u8next(s, &cp); n++; }
  return n;
}

/****************************************************************************
 * 字库
 ****************************************************************************/

/* 文件格式（小端）：
 *   0  : "JXFN"
 *   4  : u16 字宽   6: u16 字高
 *   8  : u32 字形数
 *   12 : u32 位图区偏移
 *   16 : 索引表[字形数]，每项 u32 码点 + u32 位图偏移（相对位图区），按码点升序
 *   位图：((宽+7)/8)*高 字节/字，行优先，高位在左
 */

static int glyph_find(uint32_t cp, uint32_t *off)
{
  int lo = 0, hi = B->fc - 1;

  while (lo <= hi)
    {
      int mid = (lo + hi) >> 1;
      const uint8_t *e = B->font + 16 + (size_t)mid * 8;
      uint32_t v = rd32(e);

      if (v == cp) { *off = rd32(e + 4); return 1; }
      if (v < cp)  { lo = mid + 1; } else { hi = mid - 1; }
    }

  return 0;
}

/* 可选路径：读 SD 卡上的字库。成功 0 并把 B->font 指过去，失败 -1。 */
static int font_load_sd(void)
{
  struct stat st;
  uint8_t *buf;
  int fd;

  fd = open(FONT_PATH, O_RDONLY);
  if (fd < 0)
    {
      /* 没有自动挂载器，自己挂一次 */
      mkdir("/mnt", 0777);
      mkdir("/mnt/sdcard", 0777);
      mount("/dev/mmcsd0", "/mnt/sdcard", "vfat", 0, NULL);
      fd = open(FONT_PATH, O_RDONLY);
    }

  if (fd < 0)
    {
      printf("[ui] SD 字库打不开 %s: %d\n", FONT_PATH, errno);
      return -1;
    }

  if (fstat(fd, &st) != 0 || st.st_size < 24)
    {
      printf("[ui] SD 字库文件异常\n");
      close(fd);
      return -1;
    }

  buf = (uint8_t *)malloc((size_t)st.st_size);
  if (buf == NULL)
    {
      printf("[ui] SD 字库缓冲 %d 字节分配失败\n", (int)st.st_size);
      close(fd);
      return -1;
    }

  {
    ssize_t got = read(fd, buf, (size_t)st.st_size);
    close(fd);

    if (got != (ssize_t)st.st_size)
      {
        printf("[ui] SD 字库只读到 %d/%d\n", (int)got, (int)st.st_size);
        free(buf);
        return -1;
      }
  }

  B->font     = buf;
  B->fsz      = (uint32_t)st.st_size;
  B->font_src = 1;
  return 0;
}

/* 载入并校验字库。
 * 默认用烧进 flash 的那份（const 数组 → 只占 flash，内部 DRAM 零成本，
 * 也不依赖 SD 卡是否插着）；JX_UI_SD=1 时优先用 SD 上的同名文件。 */
static int font_load(void)
{
  const char *e = getenv("JX_UI_SD");

  if (e != NULL && e[0] == (char)49)
    {
      if (font_load_sd() != 0)
        {
          printf("[ui] SD 字库不可用，回退 flash 内置字库\n");
        }
    }

  if (B->font == NULL)
    {
      B->font     = g_jx_font;
      B->fsz      = JX_FONT_LEN;
      B->font_src = 0;
    }

  if (B->fsz < 24 || memcmp(B->font, "JXFN", 4) != 0)
    {
      printf("[ui] 字库头无效\n");
      B->font = NULL;
      return -1;
    }

  B->gw    = (int)rd16(B->font + 4);
  B->gh    = (int)rd16(B->font + 6);
  B->fc    = (int)rd32(B->font + 8);
  B->fdata = rd32(B->font + 12);

  if (B->gw <= 0 || B->gw > 32 || B->gh <= 0 || B->gh > BAND_H ||
      B->fc <= 0 || 16u + (uint32_t)B->fc * 8u > B->fsz || B->fdata > B->fsz)
    {
      printf("[ui] 字库参数越界 gw=%d gh=%d n=%d\n", B->gw, B->gh, B->fc);
      B->font = NULL;
      return -1;
    }

  B->font_ok = 1;
  printf("[ui] 字库就绪: %d 字形 %dx%d, 来源 %s, %u 字节\n",
         B->fc, B->gw, B->gh, B->font_src ? "SD" : "flash", (unsigned)B->fsz);
  return 0;
}

/****************************************************************************
 * 绘制原语
 ****************************************************************************/

/* 把 B->scr 前 w*h 个像素送到 (x,y)。putarea 一次发完，比逐行 putrun 省
 * 掉 h 次 setarea（每次 ~30us），满屏能省 10ms 量级。 */
static void flush(int x, int y, int w, int h)
{
  if (w <= 0 || h <= 0) { return; }

  if (B->pa != NULL)
    {
      B->pa(B->lcd, (fb_coord_t)y, (fb_coord_t)(y + h - 1),
                   (fb_coord_t)x, (fb_coord_t)(x + w - 1),
                   (FAR uint8_t *)B->scr, (fb_coord_t)(w * 2));
    }
  else
    {
      for (int r = 0; r < h; r++)
        {
          B->pr(B->lcd, (fb_coord_t)(y + r), (fb_coord_t)x,
                      (FAR uint8_t *)(B->scr + (size_t)r * w), (size_t)w);
        }
    }
}

static void fill_rect(int x, int y, int w, int h, uint16_t c)
{
  if (B == NULL || !B->ready) { return; }

  if (x < 0) { w += x; x = 0; }
  if (y < 0) { h += y; y = 0; }
  if (x + w > UI_W) { w = UI_W - x; }
  if (y + h > UI_H) { h = UI_H - y; }
  if (w <= 0 || h <= 0) { return; }

  while (h > 0)
    {
      int bh = (h > BAND_H) ? BAND_H : h;
      int n  = w * bh;

      for (int i = 0; i < n; i++) { B->scr[i] = c; }
      flush(x, y, w, bh);
      y += bh;
      h -= bh;
    }
}

static void frame_rect(int x, int y, int w, int h, uint16_t c)
{
  fill_rect(x, y, w, 1, c);
  fill_rect(x, y + h - 1, w, 1, c);
  fill_rect(x, y, 1, h, c);
  fill_rect(x + w - 1, y, 1, h, c);
}

/* 点阵字库每个格子固定 16px，但 ASCII 字形实际只占 8~11px。
 * 这里按字形墨水边界做比例推进；中文仍保持 16px。 */
static int glyph_metrics(uint32_t cp, uint32_t *off, int *left, int *adv)
{
  int stride;
  int lo, hi;

  *left = 0;
  *adv  = B->gw;

  if (cp == 0x20u)
    {
      *adv = B->gw / 3;
      return 0;
    }

  if (!glyph_find(cp, off))
    {
      return 0;
    }

  if (cp >= 0x80u)
    {
      return 1;                 /* 中文/全角字形保持原节奏 */
    }

  stride = (B->gw + 7) / 8;
  lo = B->gw;
  hi = -1;

  {
    const uint8_t *bm = B->font + B->fdata + *off;

    for (int r = 0; r < B->gh; r++)
      {
        const uint8_t *src = bm + (size_t)r * stride;

        for (int cx = 0; cx < B->gw; cx++)
          {
            if ((src[cx >> 3] & (0x80u >> (cx & 7))) != 0)
              {
                if (cx < lo) { lo = cx; }
                if (cx > hi) { hi = cx; }
              }
          }
      }
  }

  if (hi >= lo)
    {
      *left = lo;
      *adv  = hi - lo + 1 + 3;
      if (*adv > B->gw) { *adv = B->gw; }
    }

  return 1;
}

static int text_width(const char *s, int maxw)
{
  int w = 0;
  uint32_t cp;

  if (B == NULL || s == NULL) { return 0; }

  while (*s != '\0')
    {
      s = u8next(s, &cp);
      uint32_t off;
      int left, a;

      glyph_metrics(cp, &off, &left, &a);
      if (w + a > maxw) { break; }
      w += a;
    }

  return w;
}

/* 单倍文字（16x16）。y 到 y+B->gh-1 必须落在屏内。 */
static void text1(int x, int y, const char *s, uint16_t fg, uint16_t bg)
{
  int stride, room, totalw, pen = 0;
  uint32_t cp;

  if (B == NULL || !B->ready || !B->font_ok || s == NULL) { return; }
  if (x < 0) { x = 0; }
  if (y < 0 || y + B->gh > UI_H || x >= UI_W) { return; }

  stride = (B->gw + 7) / 8;
  room   = UI_W - x;
  totalw = text_width(s, room);
  if (totalw <= 0) { return; }

  {
    int pix = totalw * B->gh;
    for (int i = 0; i < pix; i++) { B->scr[i] = bg; }
  }

  while (*s != '\0' && pen < totalw)
    {
      uint32_t off;
      int left;
      int a;
      int ok;

      s = u8next(s, &cp);
      ok = glyph_metrics(cp, &off, &left, &a);
      if (pen + a > totalw) { break; }

      if (ok)
        {
          const uint8_t *bm = B->font + B->fdata + off;
          int dx0 = pen - left;

          for (int r = 0; r < B->gh; r++)
            {
              const uint8_t *src = bm + (size_t)r * stride;
              uint16_t      *dst = B->scr + (size_t)r * (size_t)totalw;

              for (int cx = 0; cx < B->gw; cx++)
                {
                  int dx = dx0 + cx;

                  if (dx >= 0 && dx < totalw)
                    {
                      if ((src[cx >> 3] & (0x80u >> (cx & 7))) != 0)
                        {
                          dst[dx] = fg;
                        }
                    }
                }
            }
        }

      pen += a;
    }

  flush(x, y, totalw, B->gh);
}

/* 双倍文字（32x32），给大号数值用。分两条带各刷一次。 */
static void text2(int x, int y, const char *s, uint16_t fg, uint16_t bg)
{
  int stride, room, maxc, total;
  uint32_t cp;

  if (B == NULL || !B->ready || !B->font_ok || s == NULL) { return; }
  if (x < 0) { x = 0; }
  if (y < 0 || y + B->gh * 2 > UI_H || x >= UI_W) { return; }

  stride = (B->gw + 7) / 8;
  room   = UI_W - x;
  maxc   = room / (B->gw * 2);
  if (maxc <= 0) { return; }

  total = str_chars(s);
  if (total > maxc) { total = maxc; }
  if (total <= 0) { return; }

  for (int half = 0; half < 2; half++)
    {
      int r0 = half * (B->gh / 2);
      int r1 = r0 + B->gh / 2;
      /* 带里要放 B->gh 行（每条源行展开成 2 行），不是 B->gh/2 行 */
      int pix = total * B->gw * 2 * B->gh;
      int n = 0;
      const char *p = s;

      for (int i = 0; i < pix; i++) { B->scr[i] = bg; }

      while (*p != '\0' && n < total)
        {
          uint32_t off;

          p = u8next(p, &cp);
          if (glyph_find(cp, &off))
            {
              const uint8_t *bm = B->font + B->fdata + off;

              for (int r = r0; r < r1; r++)
                {
                  const uint8_t *src = bm + (size_t)r * stride;
                  int orow = (r - r0) * 2;

                  for (int rep = 0; rep < 2; rep++)
                    {
                      uint16_t *dst = B->scr +
                                      (size_t)(orow + rep) * (size_t)(total * B->gw * 2) +
                                      (size_t)n * B->gw * 2;

                      for (int cx = 0; cx < B->gw; cx++)
                        {
                          uint16_t v = (src[cx >> 3] & (0x80u >> (cx & 7))) ? fg : bg;
                          dst[cx * 2] = v;
                          dst[cx * 2 + 1] = v;
                        }
                    }
                }
            }

          n++;
        }

      flush(x, y + half * (B->gh), total * B->gw * 2, B->gh);
    }
}

/* 文字居中：cx 是中心 x */
static void text_c(int cx, int y, const char *s, uint16_t fg, uint16_t bg)
{
  int w = text_width(s, UI_W);
  text1(cx - w / 2, y, s, fg, bg);
}

/****************************************************************************
 * 顶栏 / 底栏
 ****************************************************************************/

static void draw_top(void)
{
  int x = 10;
  char buf[16];
  time_t t;
  struct tm tmv;

  fill_rect(0, 0, UI_W, TOP_H, C_TOPBG);
  fill_rect(0, TOP_H - 1, UI_W, 1, C_LINE);

  /* 菱形 logo（用两个三角拼，避免依赖字库里的符号） */
  fill_rect(x + 3, 6, 4, 2, C_CY);
  fill_rect(x + 2, 8, 6, 2, C_CY);
  fill_rect(x + 3, 10, 4, 2, C_CY);
  fill_rect(x + 1, 12, 8, 2, C_CY);
  x += 12;

  {
    int tw = text_width("极讯 AI CODEC", UI_W);

    text1(x, 4, "极讯 AI CODEC", C_CY, C_TOPBG);
    text1(x + tw + 12, 4, "openvela", C_DIM, C_TOPBG);
  }

  /* 右侧：链路点 + RSSI + 时钟 */
  {
    int time_w, link_w, time_x, bars_x, link_x;

    t = time(NULL);
    localtime_r(&t, &tmv);
    snprintf(buf, sizeof(buf), "%02d:%02d", tmv.tm_hour, tmv.tm_min);
    time_w = text_width(buf, UI_W);
    time_x = UI_W - 8 - time_w;
    text1(time_x, 4, buf, C_DIM, C_TOPBG);

    bars_x = time_x - 8 - 15;
    for (int i = 0; i < B->rssi; i++)
      {
        fill_rect(bars_x + i * 4, 14 - i * 2, 3, 2 + i * 2, C_OK);
      }

    link_w = text_width(B->link_up ? "链路通" : "链路断", UI_W);
    link_x = bars_x - 8 - link_w;
    text1(link_x, 4, B->link_up ? "链路通" : "链路断", C_DIM, C_TOPBG);

    fill_rect(link_x - 12, 9, 6, 6, B->link_up ? C_OK : C_BAD);
  }
}

static const char *const g_btn_label[NBTN] =
{
  "话音", "图像", "误码", "日志", "码率", "设置"
};

static void draw_icon(int cx, int cy, int idx, uint16_t col)
{
  switch (idx)
    {
      case 0:  /* 麦克风 */
        fill_rect(cx - 2, cy - 8, 5, 9, col);
        fill_rect(cx - 5, cy - 3, 11, 2, col);
        fill_rect(cx - 5, cy - 1, 2, 3, col);
        fill_rect(cx + 4, cy - 1, 2, 3, col);
        fill_rect(cx - 5, cy + 2, 11, 2, col);
        fill_rect(cx, cy + 4, 1, 4, col);
        break;

      case 1:  /* 相机 */
        fill_rect(cx - 8, cy - 5, 17, 11, col);
        fill_rect(cx - 5, cy - 8, 10, 3, col);
        fill_rect(cx - 2, cy - 1, 5, 4, col);
        break;

      case 2:  /* 波形 */
        fill_rect(cx - 8, cy + 1, 4, 2, col);
        fill_rect(cx - 4, cy - 3, 4, 2, col);
        fill_rect(cx, cy - 7, 4, 2, col);
        fill_rect(cx + 4, cy - 3, 4, 2, col);
        fill_rect(cx - 4, cy + 4, 4, 2, col);
        fill_rect(cx, cy + 4, 4, 2, col);
        fill_rect(cx + 4, cy + 4, 4, 2, col);
        break;

      case 3:  /* 文档 */
        frame_rect(cx - 6, cy - 8, 13, 16, col);
        fill_rect(cx - 4, cy - 4, 9, 1, col);
        fill_rect(cx - 4, cy, 9, 1, col);
        fill_rect(cx - 4, cy + 4, 6, 1, col);
        break;

      case 4:  /* 柱状 */
        fill_rect(cx - 7, cy + 1, 4, 7, col);
        fill_rect(cx - 1, cy - 4, 4, 12, col);
        fill_rect(cx + 5, cy - 1, 4, 9, col);
        break;

      default: /* 齿轮 */
        frame_rect(cx - 5, cy - 5, 11, 11, col);
        fill_rect(cx - 1, cy - 1, 3, 3, col);
        fill_rect(cx - 1, cy - 9, 3, 3, col);
        fill_rect(cx - 1, cy + 7, 3, 3, col);
        fill_rect(cx - 9, cy - 1, 3, 3, col);
        fill_rect(cx + 7, cy - 1, 3, 3, col);
        break;
    }
}

static void draw_bottom(void)
{
  for (int i = 0; i < NBTN; i++)
    {
      int x = BTN_X0 + i * (BTN_W + BTN_GAP);
      int act = (i == B->page);
      int dis = (i == 5);                 /* 设置页未接入 */
      uint16_t bd = act ? C_CY : (dis ? C_LINE : C_BTNL);
      uint16_t fc = act ? C_CY : (dis ? C_DIM2 : C_TXT);
      uint16_t bg = act ? RGB(0x0c,0x2b,0x33) : (dis ? RGB(0x0b,0x12,0x18) : C_BTN);

      fill_rect(x, BTN_Y, BTN_W, BTN_H, bg);
      frame_rect(x, BTN_Y, BTN_W, BTN_H, bd);
      draw_icon(x + BTN_W / 2, BTN_Y + 28, i, dis ? C_DIM2 : (act ? C_CY : RGB(0x5f,0x8b,0x9b)));
      text_c(x + BTN_W / 2, BTN_Y + 46, g_btn_label[i], fc, bg);
      fill_rect(x + (BTN_W - 34) / 2, BTN_Y + BTN_H - 10, 34, 3,
                act ? C_CY : C_BTNL);
    }
}

/* 返回被点击的按钮页号；未命中或“设置”页返回 -1。 */
static int ui_hit_button(int x, int y)
{
  int i;
  int bx;

  if (x < BTN_X0 || y < BTN_Y || y >= BTN_Y + BTN_H)
    {
      return -1;
    }

  i = (x - BTN_X0) / (BTN_W + BTN_GAP);
  if (i < 0 || i >= NBTN || i >= NPAGE)
    {
      return -1;
    }

  bx = BTN_X0 + i * (BTN_W + BTN_GAP);
  if (x >= bx + BTN_W)
    {
      return -1;
    }

  return i;
}

static int ui_hit_send(int x, int y)
{
  return x >= SEND_X && x < SEND_X + SEND_W &&
         y >= SEND_Y && y < SEND_Y + SEND_H;
}

/****************************************************************************
 * 卡片 / 波形
 ****************************************************************************/

static void tile(int x, int y, int w, int h, const char *k,
                 const char *v, const char *u, uint16_t vcol, int big)
{
  int ty = y + (big ? 6 : 8);

  fill_rect(x, y, w, h, C_PANEL);
  frame_rect(x, y, w, h, C_LINE2);

  text1(x + 9, ty, k, C_DIM, C_PANEL);

  if (big)
    {
      text2(x + 9, ty + 18, v, vcol, C_PANEL);
    }
  else
    {
      text1(x + 9, ty + 18, v, vcol, C_PANEL);
    }

  if (u != NULL && *u != '\0')
    {
      int vw = big ? str_chars(v) * B->gw * 2 : text_width(v, UI_W);

      text1(x + 11 + vw, ty + 18 + (big ? 16 : 0), u, C_DIM2, C_PANEL);
    }
}

/* 波形：把 g_wave 画到 (x,y,w,h) 的框里。
 * 按 16 行一条带组织，整条带一次 putarea —— 逐列 fill_rect 会发 460 次
 * setarea（每次 ~30us），那样光命令开销就 14ms；分带只要 h/16 次。 */
static void draw_wave(int x, int y, int w, int h)
{
  int n = B->wave_n;
  int mid;

  if (B == NULL || !B->ready) { return; }
  if (n <= 0 || w <= 0 || h <= 0 || w > UI_W) { return; }

  mid = h / 2;

  for (int b = 0; b < h; b += BAND_H)
    {
      int bh = (h - b > BAND_H) ? BAND_H : (h - b);
      int npix = w * bh;

      for (int i = 0; i < npix; i++) { B->scr[i] = C_PANEL2; }

      for (int c = 0; c < w; c++)
        {
          int i0 = (int)((long long)c * n / w);
          int i1 = (int)((long long)(c + 1) * n / w);
          int lo = 32767, hi = -32768;
          int y0, y1;

          if (i1 <= i0) { i1 = i0 + 1; }
          if (i1 > n)   { i1 = n; }

          for (int i = i0; i < i1; i++)
            {
              int v = B->wave[i];
              if (v < lo) { lo = v; }
              if (v > hi) { hi = v; }
            }

          y0 = mid - (int)((long long)hi * mid / 32768);
          y1 = mid - (int)((long long)lo * mid / 32768);
          if (y1 < y0) { int t = y0; y0 = y1; y1 = t; }
          if (y1 - y0 < 1) { y1 = y0 + 1; }

          /* 裁到本带内 */
          if (y0 < b) { y0 = b; }
          if (y1 > b + bh) { y1 = b + bh; }

          for (int r = y0; r < y1; r++)
            {
              B->scr[(size_t)(r - b) * w + c] = C_CY;
            }
        }

      flush(x, y + b, w, bh);
    }

  fill_rect(x, y + mid, w, 1, RGB(0x14,0x28,0x32));
  frame_rect(x, y, w, h, C_LINE);
}

/****************************************************************************
 * 页面
 ****************************************************************************/

static void body_clear(void)
{
  fill_rect(0, BODY_Y, UI_W, BODY_H, C_BG);
}

static void page_voice(void)
{
  int tw = (UI_W - 20 - 12) / 3;

  text1(10, BODY_Y + 6, "VOICE LINK", C_CY, C_BG);
  text1(10, BODY_Y + 20, "话音传输", C_HI, C_BG);

  tile(10, BODY_Y + 40, tw, 52, "码率", B->stat_rate, "kbps", C_HI, 0);
  tile(10 + tw + 6, BODY_Y + 40, tw, 52, "tokens", B->stat_tok, "", C_HI, 0);
  tile(10 + 2 * (tw + 6), BODY_Y + 40, tw, 52, "状态",
       B->busy ? "编码中" : "就绪", "", B->busy ? C_WARN : C_OK, 0);

  draw_wave(10, BODY_Y + 98, 330, 84);

  {
    uint16_t sbg = B->send_pressed ? RGB(0x12,0x3b,0x46) : RGB(0x0c,0x2b,0x33);

    fill_rect(SEND_X, SEND_Y, SEND_W, SEND_H, sbg);
    frame_rect(SEND_X, SEND_Y, SEND_W, SEND_H, C_CY);
    draw_icon(SEND_X + SEND_W / 2, SEND_Y + 26, 0, C_CY);
    text_c(SEND_X + SEND_W / 2, SEND_Y + 50, "发送语音", C_HI, sbg);
  }

  text1(10, BODY_Y + 186, B->busy ? "编解码进行中 · 界面暂停刷新"
                                 : B->voice_msg, C_DIM, C_BG);
}

static void page_reserved(const char *tag, const char *title, const char *note)
{
  text1(10, BODY_Y + 6, tag, C_CY, C_BG);
  text1(10, BODY_Y + 20, title, C_HI, C_BG);

  fill_rect(10, BODY_Y + 44, UI_W - 20, 148, C_PANEL2);
  frame_rect(10, BODY_Y + 44, UI_W - 20, 148, C_LINE);

  fill_rect(22, BODY_Y + 56, 74, 20, C_BTN);
  frame_rect(22, BODY_Y + 56, 74, 20, C_LINE2);
  text1(28, BODY_Y + 58, "预留", C_WARN, C_BTN);

  text_c(UI_W / 2, BODY_Y + 104, note, C_DIM, C_PANEL2);
  text_c(UI_W / 2, BODY_Y + 128, "本固件无对应数据源", C_DIM2, C_PANEL2);
}

static void page_log(void)
{
  text1(10, BODY_Y + 6, "SYSTEM LOG", C_CY, C_BG);
  text1(10, BODY_Y + 20, "运行日志", C_HI, C_BG);

  fill_rect(10, BODY_Y + 42, UI_W - 20, 154, C_PANEL2);
  frame_rect(10, BODY_Y + 42, UI_W - 20, 154, C_LINE);

  for (int i = 0; i < LOGN; i++)
    {
      int idx = (B->log_head - B->log_n + i + LOGN * 2) % LOGN;

      if (i >= B->log_n) { break; }
      text1(18, BODY_Y + 48 + i * 18, B->log[idx], RGB(0x8f,0xae,0xb9), C_PANEL2);
    }
}

static void page_rate(void)
{
  static const char *const nm[4] = { "0.2kbps", "1.2kbps", "2.4kbps", "12kbps" };
  static const char *const nt[4] =
  {
    "极限压缩", "可懂语音", "卫星电话标准", "接近常规通话"
  };
  static const int pw[4] = { 100, 34, 20, 8 };

  text1(10, BODY_Y + 6, "RATE ADAPTATION", C_CY, C_BG);
  text1(10, BODY_Y + 20, "码率自适应", C_HI, C_BG);

  for (int i = 0; i < 4; i++)
    {
      int y = BODY_Y + 48 + i * 34;
      int bw = UI_W - 20 - 20;

      text1(10, y, nm[i], C_DIM, C_BG);
      fill_rect(20, y + 18, bw, 12, C_PANEL2);
      frame_rect(20, y + 18, bw, 12, C_LINE);
      fill_rect(20, y + 18, bw * pw[i] / 100, 12, C_CY);
      text1(28, y + 18, nt[i], C_HI, C_CY);
    }
}

static void draw_page(int n)
{
  body_clear();

  switch (n)
    {
      case 0: page_voice(); break;
      case 1: page_reserved("IMAGE CODEC", "图像回传 · 生成式超压缩",
                            "图像语义编解码"); break;
      case 2: page_reserved("CHANNEL", "误码对抗 · 语义容错",
                            "信道误码仿真"); break;
      case 3: page_log(); break;
      default: page_rate(); break;
    }
}

/****************************************************************************
 * 公共 API
 ****************************************************************************/

#ifdef CONFIG_INPUT_TOUCHSCREEN
static int jx_ui_touch_task(int argc, char *argv[])
{
  uint8_t buf[SIZEOF_TOUCH_SAMPLE_S(TOUCH_MAXPOINT)];
  uint8_t maxpoint = TOUCH_MAXPOINT;
  int fd;

  (void)argc;
  (void)argv;

  fd = open(TOUCH_PATH, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    {
      printf("[ui] 触摸节点打不开 %s: %d\n", TOUCH_PATH, errno);
      if (B != NULL) { B->touch_started = 0; }
      return 1;
    }

  if (ioctl(fd, TSIOC_GETMAXPOINTS, &maxpoint) < 0 ||
      maxpoint == 0 || maxpoint > TOUCH_MAXPOINT)
    {
      printf("[ui] 触摸 maxpoint 无效: %d\n", maxpoint);
      maxpoint = 5;
    }

  printf("[ui] 触摸线程启动: %s maxpoint=%d\n", TOUCH_PATH, maxpoint);

  for (;;)
    {
      struct touch_sample_s *sample = (struct touch_sample_s *)buf;
      ssize_t n;
      int i;

      if (B != NULL && B->dirty)
        {
          B->dirty = 0;
          jx_ui_page(B->page);
        }

      n = read(fd, buf, SIZEOF_TOUCH_SAMPLE_S(maxpoint));
      if (n < 0)
        {
          if (errno == EINTR || errno == EAGAIN)
            {
              usleep(50000);
              continue;
            }
          printf("[ui] 触摸读取失败: %d\n", errno);
          usleep(100000);
          continue;
        }

      if (n < (ssize_t)SIZEOF_TOUCH_SAMPLE_S(1))
        {
          continue;
        }

      if (sample->npoints > maxpoint) { sample->npoints = maxpoint; }
      for (i = 0; i < sample->npoints; i++)
        {
          struct touch_point_s *p = &sample->point[i];
          int idx;

          if ((p->flags & TOUCH_UP) != 0)
            {
              B->send_latched = 0;
              continue;
            }

          if ((p->flags & (TOUCH_DOWN | TOUCH_POS_VALID)) !=
              (TOUCH_DOWN | TOUCH_POS_VALID))
            {
              continue;
            }

          if (B->page == 0 && ui_hit_send(p->x, p->y))
            {
              if (!B->send_latched)
                {
                  B->send_latched = 1;
                  if (B->send_cb != NULL)
                    {
                      B->send_cb();
                    }
                  else
                    {
                      jx_ui_set_status("PEER NOT SET");
                    }

                  jx_ui_page(0);
                }

              continue;
            }

          idx = ui_hit_button(p->x, p->y);
          if (idx >= 0 && idx != jx_ui_page_get())
            {
              jx_ui_page(idx);
            }
        }
    }
}

static int jx_ui_touch_start(void)
{
  int pid;

  if (B == NULL || B->touch_started) { return 0; }

  B->touch_started = 1;
  pid = task_create("jixun-touch", 150, 8192, jx_ui_touch_task, NULL);
  if (pid < 0)
    {
      B->touch_started = 0;
      printf("[ui] 触摸任务创建失败: %d\n", pid);
      return -1;
    }

  return 0;
}
#else
static int jx_ui_touch_start(void) { return 0; }
#endif

int jx_ui_touch_test(int seconds)
{
#ifdef CONFIG_INPUT_TOUCHSCREEN
  uint8_t buf[SIZEOF_TOUCH_SAMPLE_S(TOUCH_MAXPOINT)];
  uint8_t maxpoint = TOUCH_MAXPOINT;
  struct pollfd pfd;
  double t0;
  int fd;

  if (seconds < 1)  { seconds = 1; }
  if (seconds > 60) { seconds = 60; }

  fd = open(TOUCH_PATH, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    {
      printf("[ui] 触摸节点打不开 %s: %d\n", TOUCH_PATH, errno);
      return -1;
    }

  if (ioctl(fd, TSIOC_GETMAXPOINTS, &maxpoint) < 0 ||
      maxpoint == 0 || maxpoint > TOUCH_MAXPOINT)
    {
      printf("[ui] 触摸 maxpoint 无效: %d\n", maxpoint);
      maxpoint = 5;
    }

  pfd.fd = fd;
  pfd.events = POLLIN;
  pfd.revents = 0;
  t0 = now_ms();

  printf("[ui] 触摸诊断 %d 秒: 请在屏幕上点按按钮\n", seconds);

  while (now_ms() - t0 < (double)seconds * 1000.0)
    {
      int pr = poll(&pfd, 1, 250);

      if (pr < 0)
        {
          if (errno == EINTR) { continue; }
          printf("[ui] touch poll 失败: %d\n", errno);
          close(fd);
          return -1;
        }

      if (pr > 0 && (pfd.revents & POLLIN) != 0)
        {
          struct touch_sample_s *sample = (struct touch_sample_s *)buf;
          ssize_t n = read(fd, buf, SIZEOF_TOUCH_SAMPLE_S(maxpoint));
          int i;

          if (n < (ssize_t)SIZEOF_TOUCH_SAMPLE_S(1))
            {
              continue;
            }

          if (sample->npoints > maxpoint) { sample->npoints = maxpoint; }
          for (i = 0; i < sample->npoints; i++)
            {
              struct touch_point_s *p = &sample->point[i];
              int idx = -1;

              if ((p->flags & TOUCH_POS_VALID) != 0)
                {
                  idx = ui_hit_button(p->x, p->y);
                }

              printf("[ui] event n=%d p%d flags=0x%02x x=%d y=%d btn=%d\n",
                     sample->npoints, i, p->flags, p->x, p->y, idx);
            }
        }
    }

  close(fd);
  printf("[ui] 触摸诊断结束\n");
  return 0;
#else
  (void)seconds;
  printf("[ui] 固件未启用 CONFIG_INPUT_TOUCHSCREEN\n");
  return -1;
#endif
}

/* 下面这些小接口在 UI 还没 init（B == NULL）时必须安全 —— bench / loop 在
 * 编解码前后会无条件调 jx_ui_set_busy()，没有判空就是空指针崩溃（实测踩过）。 */
int jx_ui_ready(void)     { return (B != NULL) ? B->ready   : 0; }
int jx_ui_font_ok(void)   { return (B != NULL) ? B->font_ok : 0; }
const char *jx_ui_font_path(void) { return FONT_PATH; }
int jx_ui_page_count(void) { return NPAGE; }
int jx_ui_page_get(void)   { return (B != NULL) ? B->page : 0; }

void jx_ui_set_busy(int busy)
{
  if (B != NULL) { B->busy = busy ? 1 : 0; }
}

int jx_ui_busy(void)
{
  return (B != NULL) ? B->busy : 0;
}

void jx_ui_set_link(int up, int rssi_bars)
{
  if (B == NULL) { return; }

  B->link_up = up ? 1 : 0;
  B->rssi = (rssi_bars < 0) ? 0 : (rssi_bars > 4 ? 4 : rssi_bars);
}

void jx_ui_log(const char *s)
{
  int i;

  if (B == NULL || s == NULL) { return; }

  for (i = 0; i < LOGL - 1 && s[i] != '\0'; i++)
    {
      B->log[B->log_head][i] = s[i];
    }
  B->log[B->log_head][i] = '\0';

  B->log_head = (B->log_head + 1) % LOGN;
  if (B->log_n < LOGN) { B->log_n++; }
  B->dirty = 1;
}

void jx_ui_wave(const int16_t *pcm, int n)
{
  if (B == NULL || pcm == NULL || n <= 0) { return; }

  if (n > WAVE_N)
    {
      int step = n / WAVE_N;

      for (int i = 0; i < WAVE_N; i++) { B->wave[i] = pcm[i * step]; }
      B->wave_n = WAVE_N;
    }
  else
    {
      for (int i = 0; i < n; i++) { B->wave[i] = pcm[i]; }
      B->wave_n = n;
    }
  B->dirty = 1;
}

void jx_ui_stat(const char *rate_kbps, const char *tokinfo)
{
  if (B == NULL) { return; }

  if (rate_kbps != NULL) { snprintf(B->stat_rate, sizeof(B->stat_rate), "%s", rate_kbps); }
  if (tokinfo  != NULL)  { snprintf(B->stat_tok,  sizeof(B->stat_tok),  "%s", tokinfo);  }
  B->dirty = 1;
}

void jx_ui_set_send_cb(jx_ui_send_cb_t cb)
{
  if (B != NULL) { B->send_cb = cb; }
}

void jx_ui_set_status(const char *s)
{
  if (B == NULL || s == NULL) { return; }
  snprintf(B->voice_msg, sizeof(B->voice_msg), "%s", s);
  B->dirty = 1;
}

void jx_ui_set_peer(const char *ip)
{
  if (B == NULL || ip == NULL) { return; }
  snprintf(B->peer, sizeof(B->peer), "%s", ip);
  B->dirty = 1;
}

const char *jx_ui_peer_get(void)
{
  return (B != NULL) ? B->peer : "";
}

int jx_ui_page(int n)
{
  if (B == NULL || !B->ready) { return -1; }
  if (B->busy)   { return -2; }

  if (B->draw_lock_ok)
    {
      pthread_mutex_lock(&B->draw_lock);
      if (B->busy)
        {
          pthread_mutex_unlock(&B->draw_lock);
          return -2;
        }
    }

  if (n < 0)       { n = 0; }
  if (n >= NPAGE)  { n = NPAGE - 1; }

  B->page = n;
  draw_top();
  draw_page(n);
  draw_bottom();

  if (B->draw_lock_ok)
    {
      pthread_mutex_unlock(&B->draw_lock);
    }
  return 0;
}

int jx_ui_init(void)
{
  /* 状态块先建：内部 DRAM 放不下这些字段，它们全在堆上 */
  if (B == NULL)
    {
      B = (ui_big_t *)malloc(sizeof(ui_big_t));
      if (B == NULL)
        {
          printf("[ui] 状态块 %d 字节分配失败\n", (int)sizeof(ui_big_t));
          return -1;
        }

      memset(B, 0, sizeof(ui_big_t));
      snprintf(B->stat_rate, sizeof(B->stat_rate), "3.0");
      snprintf(B->stat_tok, sizeof(B->stat_tok), "--");
      snprintf(B->voice_msg, sizeof(B->voice_msg), "READY  SEND");
      B->link_up = 1;
      B->rssi    = 4;

      if (pthread_mutex_init(&B->draw_lock, NULL) == 0)
        {
          B->draw_lock_ok = 1;
        }
      else
        {
          printf("[ui] 绘制锁初始化失败\n");
        }
    }

  if (B->ready) { return 0; }

  B->lcd = board_lcd_getdev(0);
  if (B->lcd == NULL)
    {
      printf("[ui] board_lcd_getdev(0) 为空\n");
      return -1;
    }

  {
    struct lcd_planeinfo_s pi;

    if (B->lcd->getplaneinfo(B->lcd, 0, &pi) != 0 || pi.bpp != 16)
      {
        printf("[ui] getplaneinfo 失败或 bpp=%d（本 UI 只支持 RGB565）\n", pi.bpp);
        return -1;
      }

    B->pa  = pi.putarea;
    B->pr  = pi.putrun;
    B->gr  = pi.getrun;
    B->bpp = pi.bpp;
  }

  B->scr = (uint16_t *)malloc(sizeof(uint16_t) * SCR_PIX);
  if (B->scr == NULL)
    {
      printf("[ui] 暂存 %d 字节分配失败\n", (int)(sizeof(uint16_t) * SCR_PIX));
      return -1;
    }

  B->ready = 1;
  printf("[ui] 屏就绪 480x320 RGB565, 暂存 %p (%d 字节)\n",
         (void *)B->scr, (int)(sizeof(uint16_t) * SCR_PIX));
  printf("[ui] putarea=%s\n", B->pa ? "有" : "无(退回 putrun)");

  (void)font_load();
#ifdef CONFIG_INPUT_TOUCHSCREEN
  (void)jx_ui_touch_start();
#endif
  return 0;
}

int jx_ui_selftest(void)
{
  double t0, t1;
  char buf[64];

  if (jx_ui_init() != 0) { return -1; }

  t0 = now_ms();
  fill_rect(0, 0, UI_W, UI_H, C_BG);
  t1 = now_ms();
  printf("[ui] 满屏填充      : %7.1f ms (%d 字节)\n",
         t1 - t0, UI_W * UI_H * 2);

  t0 = now_ms();
  text1(10, 40, "极讯 AI CODEC 自检 0123456789 kbps ms", C_TXT, C_BG);
  t1 = now_ms();
  printf("[ui] 单行文字      : %7.1f ms\n", t1 - t0);

  t0 = now_ms();
  text2(10, 80, "3.01", C_HI, C_BG);
  t1 = now_ms();
  printf("[ui] 大号数字      : %7.1f ms\n", t1 - t0);

  jx_ui_wave(B->wave, B->wave_n);
  t0 = now_ms();
  draw_wave(10, 130, UI_W - 20, 84);
  t1 = now_ms();
  printf("[ui] 波形 460x84   : %7.1f ms\n", t1 - t0);

  t0 = now_ms();
  draw_top();
  draw_bottom();
  t1 = now_ms();
  printf("[ui] 顶栏+底栏     : %7.1f ms\n", t1 - t0);

  snprintf(buf, sizeof(buf), "[UI] self test ok, font=%s",
           B->font_ok ? "SD" : "缺失");
  jx_ui_log(buf);
  return 0;
}

/****************************************************************************
 * 显存回环自检
 *
 * 屏幕黑有两种完全不同的原因：
 *   (a) putarea/putrun 根本没写进面板（驱动 CS / 时序问题）；
 *   (b) 数据写进去了，但背光或 panel 没亮。
 * 写一段已知图案再读回来比对就能把两者分开：读回来一致 -> 写通路是通的，
 * 往背光和 panel 初始化方向查；不一致 -> 驱动写不进去。
 *
 * 起因：st7789 驱动里 st7789_fill（初始化走的那条）和 st7789_setarea
 * （putarea 走的那条）对 CS 的处理是矛盾的 —— 前者注释写
 * "CASET+RASET+RAMWR+data 必须在同一次 CS 拉低里完成"，后者却在 RASET 和
 * CASET 之间把 CS 拉高了。本测试就是为了验证这一点。
 ****************************************************************************/

int jx_ui_rdtest(void)
{
  enum { RX = 20, RY = 40, RW = 64, RH = 4 };
  uint16_t wr[RW * RH];
  uint16_t rd[RW * RH];
  const uint16_t pat[RH] =
  {
    RGB(0xff,0x00,0x00), RGB(0x00,0xff,0x00),
    RGB(0x00,0x00,0xff), RGB(0xff,0xff,0xff)
  };
  int bad = 0;

  if (jx_ui_init() != 0) { return -1; }

  for (int r = 0; r < RH; r++)
    {
      for (int c = 0; c < RW; c++) { wr[r * RW + c] = pat[r]; }
    }

  memset(rd, 0, sizeof(rd));

  if (B->pa != NULL)
    {
      B->pa(B->lcd, (fb_coord_t)RY, (fb_coord_t)(RY + RH - 1),
            (fb_coord_t)RX, (fb_coord_t)(RX + RW - 1),
            (FAR uint8_t *)wr, (fb_coord_t)(RW * 2));
    }
  else
    {
      for (int r = 0; r < RH; r++)
        {
          B->pr(B->lcd, (fb_coord_t)(RY + r), (fb_coord_t)RX,
                (FAR uint8_t *)&wr[r * RW], (size_t)RW);
        }
    }

  if (B->gr == NULL)
    {
      printf("[ui] 驱动没有 getrun，无法回环验证（CONFIG_LCD_NOGETRUN?）\n");
      return -1;
    }

  for (int r = 0; r < RH; r++)
    {
      int n = B->gr(B->lcd, (fb_coord_t)(RY + r), (fb_coord_t)RX,
                    (FAR uint8_t *)&rd[r * RW], (size_t)RW);
      int rowbad = 0;

      for (int c = 0; c < RW; c++)
        {
          if (rd[r * RW + c] != wr[r * RW + c]) { rowbad++; bad++; }
        }

      printf("[ui] 行%d rc=%d 写入=0x%04x 读回[0]=0x%04x [32]=0x%04x 不符=%d/%d\n",
             r, n, pat[r], rd[r * RW], rd[r * RW + 32], rowbad, RW);
    }

  if (bad == 0)
    {
      printf("[ui] ==> 写通路正常（%d 像素全一致）：黑屏不是写不进去，"
             "请查背光与 panel 初始化\n", RW * RH);
    }
  else
    {
      printf("[ui] ==> 写通路异常（%d/%d 像素不符）：putarea/putrun 没真正写进面板\n",
             bad, RW * RH);
    }

  jx_ui_log(bad == 0 ? "[UI] GRAM 回环 OK" : "[UI] GRAM 回环异常");
  return (bad == 0) ? 0 : 1;
}
