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

#include "esp32s3_gpio.h"

#include "jx_ui.h"
#include "jx_simple_role.h"
#include "jx_font_bin.h"   /* flash 内置字库（生成见 _fontgen.py） */

extern const char *jx_rate_name(void);

extern void jx_ui_send_action(void);

/****************************************************************************
 * 常量
 ****************************************************************************/

#define UI_W       240
#define UI_H       240
#define BAND_H     16                    /* 暂存高度（= 字高） */
#define SCR_PIX    (UI_W * BAND_H)

#define RGB(r,g,b) ((uint16_t)((((r) & 0xf8u) << 8) | (((g) & 0xfcu) << 3) | ((b) >> 3)))

/* 配色（取自 UI 稿的 CSS 变量） */
#define C_BG      RGB(0xff,0xff,0xff)
#define C_PANEL   RGB(0xe8,0xe8,0xe8)
#define C_PANEL2  RGB(0xf0,0xf0,0xf0)
#define C_LINE    RGB(0x80,0x80,0x80)
#define C_LINE2   RGB(0x90,0x90,0x90)
#define C_TXT     RGB(0x00,0x00,0x00)
#define C_HI      RGB(0x00,0x00,0x00)
#define C_DIM     RGB(0x40,0x40,0x40)
#define C_DIM2    RGB(0x50,0x50,0x50)
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
#define BODY_H    150
#define BOT_Y     (TOP_H + BODY_H)       /* 224 */
#define BOT_H     (UI_H - BOT_Y)         /* 96  */

#define NBTN      3
#define BTN_W     72
#define BTN_GAP   4
#define BTN_X0    6
#define BTN_Y     (BOT_Y + 4)
#define BTN_H     58

#define NPAGE     2

/* 话音页内置的“发送语音”按钮 */
#define SEND_X    150
#define SEND_Y    (BODY_Y + 92)
#define SEND_W    80
#define SEND_H    52

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
  int      touch_ok;
  int      page;
  int      rot;
  int      link_up;
  int      rssi;
  int      font_ok;
  int      font_src;      /* 0 = flash 内置，1 = SD 卡覆盖 */
  int      touch_started;
  int      send_latched;
  int      send_pressed;
  char     voice_msg[40];
  char     peer[32];
  int      rx_pkt;
  int      rx_rssi;
  int      tx_pkt;
  jx_ui_send_cb_t send_cb;
  pthread_mutex_t draw_lock;
  int      draw_lock_ok;
  int      compact_drawn;
  int      compact_page;
  int      compact_phase;
  int      compact_rx;
  int      compact_tx;
  int      compact_rssi;
  double   compact_time;
} ui_big_t;

static ui_big_t *B;

/* BOOT (GPIO0) poller gate, see jx_ui_set_boot_enable(). */
static volatile int g_ui_boot_enable = 1;
/* BX_BUSY_PROGRESS: 发送/编码期间让状态行每秒走一次，
 * 用户看到 ENC 5s/6s 就知道没卡死。 */
static double g_ui_busy_t0;
static int    g_ui_busy_sec = -1;
/* 2026-10-29 诊断：把按键任务内部状态镜像出来，供 jixun ui status 查看。 */
static volatile int g_dbg_base;
static volatile int g_dbg_lc;
static volatile int g_dbg_armed;
static volatile int g_dbg_cool;
/* 上一次画到状态行的 BOOT 电平。放在文件级，set_boot_enable(1) 时可以
 * 清成 -1，强制下一轮重画，避免发送结束后界面永久停在 KEY DOWN!。 */
static int g_ui_boot_last_shown = -1;

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
static void rev_row(uint16_t *p, int n)
{
  int i = 0;
  int j = n - 1;

  while (i < j)
    {
      uint16_t t = p[i];
      p[i] = p[j];
      p[j] = t;
      i++;
      j--;
    }
}

static void flush(int x, int y, int w, int h)
{
  int r;

  if (w <= 0 || h <= 0) { return; }

  /* 2026-10-29: 240x240 square panel; the unit is held horizontally, so the
   * image has to be folded 90 degrees to the left.  The rotation happens at
   * the pixel exit: one logical row becomes one physical column, so no extra
   * transpose buffer is needed.  JX_UI_ROT=0 disables, 1 = clockwise. */

  if (B->rot != 0)
    {
      for (r = 0; r < h; r++)
        {
          uint16_t *row = B->scr + (size_t)r * (size_t)w;
          int py;
          int px;

          if (B->rot < 0)
            {
              py = UI_W - x - w;          /* counter-clockwise (left) */
              px = y + r;
              rev_row(row, w);
              if (B->rot == -2) { px = UI_W - 1 - px; }
            }
          else
            {
              py = x;                     /* clockwise (right) */
              px = UI_H - 1 - (y + r);
              if (B->rot == 2) { px = UI_H - 1 - px; }
            }

          if (py < 0 || py + w > UI_H || px < 0 || px >= UI_W)
            {
              continue;
            }

          if (B->pa != NULL)
            {
              B->pa(B->lcd, (fb_coord_t)py, (fb_coord_t)(py + w - 1),
                    (fb_coord_t)px, (fb_coord_t)px,
                    (FAR uint8_t *)row, (fb_coord_t)2);
            }
          else if (B->pr != NULL)
            {
              for (int k = 0; k < w; k++)
                {
                  B->pr(B->lcd, (fb_coord_t)(py + k), (fb_coord_t)px,
                        (FAR uint8_t *)(row + k), 1);
                }
            }
        }

      return;
    }

  if (B->pa != NULL)
    {
      B->pa(B->lcd, (fb_coord_t)y, (fb_coord_t)(y + h - 1),
                   (fb_coord_t)x, (fb_coord_t)(x + w - 1),
                   (FAR uint8_t *)B->scr, (fb_coord_t)(w * 2));
    }
  else
    {
      for (r = 0; r < h; r++)
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

#include "jx_simple_ui.inc"

static void draw_top(void)
{
  simple_state_t s = { JX_SIMPLE_RX, B->voice_msg, B->rx_pkt, B->rx_rssi, B->tx_pkt };
  simple_header(&s, B->page == 1);
}

static const char *const g_btn_label[NBTN] =
{
  "话音", "图像", "误码"
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
  /* Actions are inside the main 240x240 scene. */
}

/* 返回被点击的按钮页号；未命中或“设置”页返回 -1。 */
static int ui_hit_button(int x, int y)
{
  (void)x; (void)y;
  return -1;
}

static int ui_hit_send(int x, int y)
{
  (void)x; (void)y;
  return 0;
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

static volatile int g_boot_saw_low;

static void page_voice(void)
{
  simple_state_t s = { JX_SIMPLE_RX, B->voice_msg, B->rx_pkt, B->rx_rssi, B->tx_pkt };
  simple_main(&s);
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
  simple_state_t s = { JX_SIMPLE_RX, B->voice_msg, B->rx_pkt, B->rx_rssi, B->tx_pkt };
  if (n == 1) { simple_info(&s); } else { simple_main(&s); }
}

/****************************************************************************
 * 公共 API
 ****************************************************************************/

#ifdef CONFIG_INPUT_TOUCHSCREEN
static int jx_ui_touch_task(int argc, char *argv[])
{
  (void)argc; (void)argv;
  B->touch_ok = 0;
  B->dirty = 1;
  printf("[ui] non-touch display refresh ready; BOOT is the physical control\n");
  for (;;)
    {
      if (B->dirty && !B->busy)
        {
          B->dirty = 0;
          jx_ui_page(B->page);
        }
      usleep(20000);
    }
}

#define JX_GPIO_IN0_REG 0x6000403cul
#define JX_GPIO_IN1_REG 0x60004040ul

/* 2026-10-29：ESP32-S3 的 GPIO32..48 在 IN1 寄存器里，之前用
 * 32 位移位 (1ul << 35) 会溢出成 bit3，把 GPIO35（PSRAM 脚）
 * 误当成按键，真正的键反而漏掉。这里按 0..31 / 32..48 分开读。
 *
 * 候选脚是穷举完板级占用后剩下的全部真空白脚：
 *   0/1/14/15  —— 通用空脚
 *   3          —— 触摸 INT（这块板没触摸芯片，空着）
 *   39         —— 摄像头 XMCLK（没摄像头，空着）
 *   46         —— input-only 空脚
 * LCD(2/4/5/6/7/38/41/42/47/48)、I2S(8/9/10/16/45)、
 * LoRa(11/12/13/17/18/21/40)、UART(43/44)、USB(19/20)、
 * Flash/PSRAM(26..37) 全部排除，绝不碰。 */
/* 2026-10-29 修正：GPIO14/15 是 I2C0 的 SCL/SDA（.config:
 * CONFIG_ESP32S3_I2C0_SCLPIN=14 / SDAPIN=15），ES8311 就挂在这条总线上。
 * 之前把 15 上的 I2C 数据跳变当成按键，pull 低 15 等于拉死 I2C，
 * 这就是"按一次之后偶发卡死"的来源。14/15 永久排除。
 *
 * 1.54 官方源码 main.ino 里 `#define BOOT_BTN 0`，1.3 同系列，
 * 首选 GPIO0。其余候选只做兜底，全部要求稳定低 60 ms。 */
#define JX_BOOT_PREFERRED 0

static const uint8_t g_boot_pads[] =
{
  0, 1, 3, 39
};

#define BOOT_MAX_PADS ((int)(sizeof(g_boot_pads) / sizeof(g_boot_pads[0])))

#define g_jx_ui_tx_busy ((B != NULL) ? B->busy : 0)

static volatile int g_boot_gpio = JX_BOOT_PREFERRED; /* -1 = 自动识别 */
static volatile int g_boot_found = -1;   /* 已识别到的脚 */
static int g_boot_npad = BOOT_MAX_PADS;

static int jx_pad_read(int gpio)
{
  volatile uint32_t *in0 = (volatile uint32_t *)JX_GPIO_IN0_REG;
  volatile uint32_t *in1 = (volatile uint32_t *)JX_GPIO_IN1_REG;

  if (gpio < 32)
    {
      return (*in0 >> gpio) & 1;
    }

  return (*in1 >> (gpio - 32)) & 1;
}

int jx_ui_key_gpio(void)
{
  return (g_boot_found >= 0) ? g_boot_found : g_boot_gpio;
}

void jx_ui_key_set_gpio(int gpio)
{
  if (gpio < 0)
    {
      g_boot_gpio = -1;
      g_boot_found = -1;
      printf("[ui] KEY -> auto scan\n");
      return;
    }

  g_boot_gpio = gpio;
  g_boot_found = gpio;
  esp32s3_configgpio(gpio, INPUT_PULLUP);
  printf("[ui] KEY -> GPIO%d (level=%d)\n", gpio, jx_pad_read(gpio));
}

void jx_ui_key_map(void)
{
  int i;

  printf("[ui] KEY map (%d pads):\n", g_boot_npad);
  for (i = 0; i < g_boot_npad; i++)
    {
      int g = g_boot_pads[i];

      esp32s3_configgpio(g, INPUT_PULLUP);
      printf("  GPIO%-2d level=%d%s\n", g, jx_pad_read(g),
             (g == g_boot_found) ? "  <== locked" : "");
    }
}

static int jx_ui_bootkey_task(int argc, char *argv[])
{
  /* 2026-10-29：原来这 4 个数组是栈上局部变量，任务栈只有 4096，
   * 实测运行中 base[0] 会从 1 变成 0（被别的东西覆写），一旦变 0
   * 就永远不满足 base==1 的触发条件，表现就是「按 BOOT 没反应」。
   * 挪到静态区，任务只创建一次，不存在重入问题。 */
  static int base[BOOT_MAX_PADS];
  static int lowcnt[BOOT_MAX_PADS];
  static int armed[BOOT_MAX_PADS];
  static int last[BOOT_MAX_PADS];
  int cool = 0;
  int show_levels = 1;
  double t_start;
  int tick = 0;
  int i;

  (void)argc;
  (void)argv;

  /* 只开上拉、必须关掉下拉。NuttX 的 INPUT_PULLUP 实测会把 PD 也打开
   * （bootdiag 读到 IO_MUX0=0x1b00，bit8 PD=1），上下拉同时开等于把脚
   * 钉在 1.65V，外部按键若串了电阻就拉不到 0。这里手动清 PD。 */
  for (i = 0; i < g_boot_npad; i++)
    {
      esp32s3_configgpio(g_boot_pads[i], INPUT_PULLUP);
    }

  {
    /* GPIO0 的 IO_MUX 固定是 0x60009004，只动这一只，别碰别的脚。 */
    volatile uint32_t *mux0 = (volatile uint32_t *)0x60009004ul;

    *mux0 = (*mux0 & ~(1u << 8)) | (1u << 9);
  }

  usleep(100000);
  t_start = now_ms();

  for (i = 0; i < g_boot_npad; i++)
    {
      base[i]   = jx_pad_read(g_boot_pads[i]);
      lowcnt[i] = 0;
      armed[i]  = 1;
      last[i]   = base[i];
    }

  printf("[ui] KEY task ready, %d input pads base:", g_boot_npad);
  for (i = 0; i < g_boot_npad; i++)
    {
      printf(" %d=%d", g_boot_pads[i], base[i]);
    }

  printf(" (auto), gate=%d\n", g_ui_boot_enable);

  for (;;)
    {
      int fired = -1;

      /* State is driven by codec events, not an invented elapsed ENC phase. */

      /* 别的地方（PTT/扫描/驱动）会再调 esp32s3_configgpio 把 PD 打开。
       * 2026-10-29：从每 5 秒改成每 20 ms 一次，杜绝按键落在守护窗口里
       * 被 PD 拖住的可能。只动 GPIO0 这一只脚的 IO_MUX，成本可忽略。 */
      {
        volatile uint32_t *mux0 = (volatile uint32_t *)0x60009004ul;

        *mux0 = (*mux0 & ~(1u << 8)) | (1u << 9);
      }

      /* 第一次按键之前，把 7 个候选脚的电平直接写在屏幕状态行。
       * 用户按 BOOT 时哪个数字从 1 变 0，就是真实按键脚；
       * 同时这个变化也会触发录音，不需要再看串口。 */
      /* 2026-10-29：电平显示改成常驻，不再 10 秒后消失。
       * 用户按 BOOT 时屏幕上的 1 会立刻变 0，按没按到一眼就能看出来；
       * 录音/发送期间不覆盖主状态。 */
      if (g_ui_boot_enable && !g_jx_ui_tx_busy)
        {
          int lv = jx_pad_read(JX_BOOT_PREFERRED);

          /* 每次 20 ms 循环只读一个寄存器，只有电平变化才写状态，
           * 不会拖慢任务，也不会把 LCD 刷爆。 */
          if (lv == 0) { g_boot_saw_low = 1; }

          if (lv != g_ui_boot_last_shown)
            {
              char sbuf[32];

              g_ui_boot_last_shown = lv;
              snprintf(sbuf, sizeof(sbuf), "K0=%d %s", lv,
                       lv ? "PRESS BOOT" : "KEY DOWN!");
              /* GPIO diagnostics stay in the log; never overwrite voice state. */
              printf("[ui] BOOT level %d -> %s (msg='%s')\n", lv,
                     lv ? "released" : "PRESSED", B->voice_msg);
            }
        }

      for (i = 0; i < g_boot_npad; i++)
        {
          int g = g_boot_pads[i];
          int lvl;
          int is_target;

          /* 锁定脚优先：g_boot_found 是实测确认过的脚，
           * 未确认时用 g_boot_gpio（默认 15）。其余脚只记边沿。 */
          /* 首选脚 15 先认；如果 15 一直没动静，任何候选脚稳定低
           * 60 ms 都可以触发。这块板没有公开原理图，宁可多覆盖。 */
          is_target = 1;

          lvl = jx_pad_read(g);

          /* 任何边沿都打印：真实按键脚如果因为基线/上拉问题没触发，
           * 也能从这一行直接看出是哪个脚在动。 */
          if (lvl != last[i])
            {
              printf("[ui] KEY edge gpio=%d %d->%d\n", g, last[i], lvl);
              last[i] = lvl;
            }

          if (!is_target)
            {
              continue;
            }

          /* 2026-10-29：原来要求 base[i] == 1 才计数。可 base 是任务启动
           * 瞬间采的，如果那一刻 GPIO0 恰好是低（USB 复位、按键残留），
           * base[0] 就永远是 0，按键再也触发不了 —— 这正是「有时按 BOOT
           * 有反应、有时完全没反应」的根因。改成只看当前电平 + armed
           * 边沿：稳定低 40 ms 就触发，松开后重新武装。 */
          if (lvl == 0)
            {
              if (lowcnt[i] < 100) { lowcnt[i]++; }
            }
          else
            {
              lowcnt[i] = 0;
            }

          if (armed[i] && lowcnt[i] >= 2 && lvl == 0)
            {
              fired = g;
              armed[i] = 0;
              break;
            }

          if (lvl == 1)
            {
              armed[i] = 1;
            }
        }

      if (cool > 0)
        {
          cool--;
        }

      if (fired >= 0 && g_ui_boot_enable && cool == 0)
        {
          /* 200 ms 防抖。原来 100 个 20 ms 周期 = 2 s，用户说完再按
           * 第二下会被直接吞掉，无法实现“再按一次结束录音”。 */
          cool = 10;
          show_levels = 0;
          g_boot_found = fired;
          printf("[ui] KEY gpio=%d pressed -> record (cb=%p)\n",
                 fired, B != NULL ? (void *)B->send_cb : (void *)0);

          if (JX_SIMPLE_RX)
            {
              if (B != NULL && !B->busy) { jx_ui_page(B->page == 0 ? 1 : 0); }
            }
          else if (B != NULL && B->send_cb != NULL)
            {
              B->send_cb();
            }
          else
            {
              jx_ui_send_action();
            }
        }

      g_dbg_base  = base[0];
      g_dbg_lc    = lowcnt[0];
      g_dbg_armed = armed[0];
      g_dbg_cool  = cool;
      usleep(20000);
    }

  return 0;
}

static void jx_ui_bootkey_start(void)
{
  static int started;
  int pid;

  if (started || B == NULL)
    {
      return;
    }

  started = 1;
  pid = task_create("jixun-boot", 150, 4096, jx_ui_bootkey_task, NULL);
  if (pid < 0)
    {
      started = 0;
      printf("[ui] BOOT task create fail: %d\n", pid);
    }
}

static int jx_ui_touch_start(void)
{
  int pid;

  if (B == NULL || B->touch_started) { return 0; }

  B->touch_started = 1;
  pid = task_create("jixun-display", 90, 8192, jx_ui_touch_task, NULL);
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
  (void)seconds;
  printf("[ui] non-touch hardware: use the physical BOOT key\n");
  return -ENODEV;
}

/* 下面这些小接口在 UI 还没 init（B == NULL）时必须安全 —— bench / loop 在
 * 编解码前后会无条件调 jx_ui_set_busy()，没有判空就是空指针崩溃（实测踩过）。 */
int jx_ui_ready(void)     { return (B != NULL) ? B->ready   : 0; }
int jx_ui_font_ok(void)   { return (B != NULL) ? B->font_ok : 0; }
const char *jx_ui_font_path(void) { return FONT_PATH; }
int jx_ui_page_count(void) { return NPAGE; }
int jx_ui_page_get(void)   { return (B != NULL) ? B->page : 0; }

void jx_ui_set_rot(int rot)
{
  if (B == NULL) { return; }

  if (rot > 2)  { rot = 2; }
  if (rot < -2) { rot = -2; }
  if (rot == B->rot) { return; }

  B->rot = rot;
  printf("[ui] rot -> %d\n", rot);
  B->compact_drawn = 0;
  if (B->ready) { jx_ui_page(B->page); }
}

int jx_ui_get_rot(void) { return (B != NULL) ? B->rot : 0; }

/* Global gate for the BOOT key poller.  The PTT command takes GPIO0 over
 * and must turn the UI path off first. */

void jx_ui_set_boot_enable(int on)
{
  g_ui_boot_enable = on;
  if (!on)
    {
      g_ui_busy_t0  = now_ms();
      g_ui_busy_sec = -1;
    }
  if (on)
    {
      /* 发送期间状态行被冻结在 KEY DOWN!，恢复时必须强制重画一次。 */
      g_ui_boot_last_shown = -1;
    }
  printf("[ui] boot key %s\n", on ? "enabled" : "disabled (PTT owns GPIO0)");
}

int jx_ui_boot_enabled(void)
{
  return g_ui_boot_enable;
}

int jx_ui_boot_level(void)
{
  int g = (g_boot_found >= 0) ? g_boot_found : JX_BOOT_PREFERRED;

  return jx_pad_read(g) ? 1 : 0;
}

void jx_ui_trigger_send(void)
{
  if (B != NULL && B->send_cb != NULL)
    {
      B->send_cb();
    }
}

void jx_ui_pull_boot(int ms)
{
  if (ms < 100) { ms = 100; }
  if (ms > 3000) { ms = 3000; }

  printf("[ui] pull GPIO%d low for %d ms (BOOT emulation)\n",
         JX_BOOT_PREFERRED, ms);
  esp32s3_configgpio(JX_BOOT_PREFERRED, OUTPUT);
  esp32s3_gpiowrite(JX_BOOT_PREFERRED, false);
  usleep(ms * 1000);
  esp32s3_configgpio(JX_BOOT_PREFERRED, INPUT_PULLUP);
  printf("[ui] GPIO%d released, level=%d\n", JX_BOOT_PREFERRED,
         esp32s3_gpioread(JX_BOOT_PREFERRED) ? 1 : 0);
}


/* 2026-10-29: 全脚只读扫描。不配置任何脚、不碰外设，只读 GPIO_IN0/IN1
 * 原始寄存器，找出「按下时真的会变低」的脚。
 *
 * 目的：这块 1.3 AI Box 没有公开原理图，BOOT 可能不在 GPIO0/15 上，
 * 也可能挂在 IO 扩展芯片后面。先把所有会动的脚抓出来再决定。 */
#define JX_GPIO_EN0_REG 0x60004020ul
#define JX_GPIO_EN1_REG 0x60004024ul


/* 2026-10-29: GPIO0 的 IO_MUX / GPIO 寄存器一次性打印，判断 BOOT 脚
 * 到底有没有被配成「GPIO 输入 + 上拉」。只读，不改任何东西。 */
void jx_ui_bootdiag(void)
{
  volatile uint32_t *iomux = (volatile uint32_t *)0x60009004ul; /* GPIO0 */
  volatile uint32_t *in0   = (volatile uint32_t *)JX_GPIO_IN0_REG;
  volatile uint32_t *en0   = (volatile uint32_t *)0x60004020ul;
  uint32_t m = *iomux;
  uint32_t a = *in0;
  uint32_t e = *en0;

  printf("[bootdiag] IO_MUX0=%08lx IE=%lu PU=%lu PD=%lu DRV=%lu SEL=%lu\n",
         (unsigned long)m,
         (unsigned long)((m >> 11) & 1), (unsigned long)((m >> 9) & 1),
         (unsigned long)((m >> 8) & 1), (unsigned long)((m >> 12) & 7),
         (unsigned long)((m >> 15) & 1));
  printf("[bootdiag] GPIO_EN0=%08lx bit0=%lu (1=output)\n",
         (unsigned long)e, (unsigned long)(e & 1));
  printf("[bootdiag] GPIO_IN0=%08lx bit0=%lu\n",
         (unsigned long)a, (unsigned long)(a & 1));
  printf("[bootdiag] bit0..3 = %lu%lu%lu%lu\n",
         (unsigned long)((a >> 3) & 1), (unsigned long)((a >> 2) & 1),
         (unsigned long)((a >> 1) & 1), (unsigned long)(a & 1));
}

void jx_ui_allscan(int secs)
{
  volatile uint32_t *in0  = (volatile uint32_t *)JX_GPIO_IN0_REG;
  volatile uint32_t *in1  = (volatile uint32_t *)JX_GPIO_IN1_REG;
  volatile uint32_t *en0  = (volatile uint32_t *)JX_GPIO_EN0_REG;
  volatile uint32_t *en1  = (volatile uint32_t *)JX_GPIO_EN1_REG;
  int lowcnt[49];
  int reported[49];
  uint32_t preva;
  uint32_t prevb;
  int n;
  int k;
  int g;

  if (secs < 1)   { secs = 1; }
  if (secs > 600) { secs = 600; }
  n = secs * 20;

  memset(lowcnt, 0, sizeof(lowcnt));
  memset(reported, 0, sizeof(reported));

  preva = *in0;
  prevb = *in1;

  printf("[ui] ALLSCAN %d s  en0=%08lx en1=%08lx\n",
         secs, (unsigned long)*en0, (unsigned long)*en1);
  printf("[ui] ALLSCAN base in0=%08lx in1=%08lx  (press BOOT now)\n",
         (unsigned long)preva, (unsigned long)prevb);

  for (k = 0; k < n; k++)
    {
      uint32_t a = *in0;
      uint32_t b = *in1;

      for (g = 0; g <= 48; g++)
        {
          int lvl = (g < 32) ? (int)((a >> g) & 1) : (int)((b >> (g - 32)) & 1);

          /* 任何位翻转都打一行，高频抖动的脚会自己暴露出来。 */
          {
            int was = (g < 32) ? (int)((preva >> g) & 1)
                               : (int)((prevb >> (g - 32)) & 1);
            if (lvl != was)
              {
                printf("[ui] ALL edge gpio=%d %d->%d t=%.2f\n",
                       g, was, lvl, (double)k * 0.05);
              }
          }

          if (lvl == 0) { if (lowcnt[g] < 1000) { lowcnt[g]++; } }
          else          { lowcnt[g] = 0; }

          /* 稳定低 100 ms 才报，滤掉毛刺。46 是 input-only 天生低，跳过。 */
          if (g != 46 && lowcnt[g] == 2 && reported[g] == 0)
            {
              reported[g] = 1;
              printf("[ui] ALL gpio=%d LOW stable t=%.2f\n", g,
                     (double)k * 0.05);
            }

          if (lvl == 1 && reported[g] != 0)
            {
              reported[g] = 0;
              printf("[ui] ALL gpio=%d released t=%.2f\n", g,
                     (double)k * 0.05);
            }
        }

      preva = a;
      prevb = b;
      usleep(50000);
    }

  printf("[ui] ALLSCAN done\n");
}

void jx_ui_pin_scan(int secs)
{
  uint32_t lowcnt[BOOT_MAX_PADS];
  uint32_t reported[BOOT_MAX_PADS];
  int n;
  int i;
  int k;

  if (secs < 1)   { secs = 1; }
  if (secs > 600) { secs = 600; }
  n = secs * 20;

  memset(lowcnt, 0, sizeof(lowcnt));
  memset(reported, 0, sizeof(reported));

  for (i = 0; i < g_boot_npad; i++)
    {
      esp32s3_configgpio(g_boot_pads[i], INPUT_PULLUP);
    }

  usleep(50000);
  printf("[ui] pin scan %d s (stable-low only), press the key now\n", secs);

  for (k = 0; k < n; k++)
    {
      for (i = 0; i < g_boot_npad; i++)
        {
          int g = g_boot_pads[i];

          if (jx_pad_read(g) == 0)
            {
              if (lowcnt[i] < 1000)
                {
                  lowcnt[i]++;
                }

              if (lowcnt[i] >= 3 && reported[i] == 0)
                {
                  reported[i] = 1;
                  printf("[ui] pin%d LOW stable t=%.2fs\n", g, (double)k * 0.05);
                }
            }
          else
            {
              if (reported[i] != 0)
                {
                  printf("[ui] pin%d released t=%.2fs\n", g, (double)k * 0.05);
                  reported[i] = 0;
                }

              lowcnt[i] = 0;
            }
        }

      usleep(50000);
    }

  printf("[ui] pin scan done\n");
}

void jx_ui_boot_watch(int secs)
{
  int last = -1;
  int n;
  int i;

  if (secs < 1)  { secs = 1; }
  if (secs > 600) { secs = 600; }
  n = secs * 20;

  printf("[ui] BOOT watch %d s, press the key now\n", secs);

  for (i = 0; i < n; i++)
    {
      int lvl = jx_ui_boot_level();

      if (lvl != last)
        {
          printf("[ui] BOOT level=%d (%s) t=%.2fs\n",
                 lvl, lvl ? "released" : "PRESSED", (double)i * 0.05);
          last = lvl;
        }

      usleep(50000);
    }

  printf("[ui] BOOT watch done, last=%d\n", last);
}


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
  /* Radio packets may continue arriving after playback starts. Preserve
   * the actual playback stage; counts/RSSI still update independently. */
  if (JX_SIMPLE_RX && strstr(B->voice_msg, "PLAY") != NULL &&
      strncmp(s, "RX ", 3) == 0 && strstr(s, "DONE") == NULL &&
      strstr(s, "WAIT") == NULL && strstr(s, "FAIL") == NULL) { return; }
  if (strcmp(B->voice_msg, s) == 0) { return; }
  snprintf(B->voice_msg, sizeof(B->voice_msg), "%s", s);
  B->dirty = 1;
}

void jx_ui_set_rx(int pkt, int rssi)
{
  if (B == NULL) { return; }
  B->rx_pkt  = pkt;
  B->rx_rssi = rssi;
  B->dirty   = 1;
}

void jx_ui_set_tx(int pkt)
{
  if (B == NULL) { return; }
  B->tx_pkt = pkt;
  B->dirty  = 1;
}

void jx_ui_status_dump(void)
{
  if (B == NULL)
    {
      printf("[ui] status: UI 未初始化 (B=NULL)\n");
      return;
    }

  printf("[ui] status: page=%d ready=%d busy=%d touch_ok=%d dirty=%d "
         "key_gpio=%d key_level=%d msg=\"%s\" peer=\"%s\" link=%d rssi=%d "
         "gate=%d base0=%d lc0=%d armed0=%d cool=%d "
         "rx_pkt=%d rx_rssi=%d tx_pkt=%d\n",
         B->page, B->ready, B->busy, B->touch_ok, B->dirty,
         jx_ui_key_gpio(), jx_ui_boot_level(),
         B->voice_msg, B->peer, B->link_up, B->rssi,
         g_ui_boot_enable, g_dbg_base, g_dbg_lc, g_dbg_armed, g_dbg_cool,
         B->rx_pkt, B->rx_rssi, B->tx_pkt);
  printf("[ui_screen] role=%s input=BOOT touch=disabled drawn=%d phase=%d page=%d\n",
         JX_SIMPLE_RX ? "RX" : "TX", B->compact_drawn, B->compact_phase, B->compact_page);
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
  int changed;
  double now;
  simple_state_t s;
  enum simple_phase phase;
  if (B == NULL || !B->ready) { return -1; }
  if (B->busy) { B->dirty = 1; return -2; }
  if (B->draw_lock_ok) { pthread_mutex_lock(&B->draw_lock); }
  if (B->busy)
    {
      B->dirty = 1;
      if (B->draw_lock_ok) { pthread_mutex_unlock(&B->draw_lock); }
      return -2;
    }
  n = n == 0 ? 0 : 1;
  s = (simple_state_t){ JX_SIMPLE_RX, B->voice_msg, B->rx_pkt, B->rx_rssi, B->tx_pkt };
  phase = simple_phase_for(&s);
  now = now_ms();
  changed = !B->compact_drawn || n != B->compact_page || (int)phase != B->compact_phase;
  if (!changed && n == 1 &&
      (s.rx_pkt != B->compact_rx || s.tx_pkt != B->compact_tx || s.rx_rssi != B->compact_rssi))
    {
      if (now - B->compact_time >= 1000.0) { changed = 1; }
      else { B->dirty = 1; }
    }
  B->page = n;
  if (changed)
    {
      if (!B->compact_drawn || n != B->compact_page) { simple_header(&s, n); }
      if (n) { simple_info(&s); } else { simple_main(&s); }
      B->compact_drawn = 1;
      B->compact_page = n;
      B->compact_phase = (int)phase;
      B->compact_rx = s.rx_pkt; B->compact_tx = s.tx_pkt; B->compact_rssi = s.rx_rssi;
      B->compact_time = now;
    }
  if (B->draw_lock_ok) { pthread_mutex_unlock(&B->draw_lock); }
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
      snprintf(B->voice_msg, sizeof(B->voice_msg), JX_SIMPLE_RX ? "RX STOPPED" : "READY");
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

  jx_ui_bootkey_start();

  if (B->ready) { return 0; }

  {
    const char *rv = getenv("JX_UI_ROT");

    B->rot = (rv != NULL && rv[0] != '\0') ? atoi(rv) : 2;
    if (B->rot > 2)  { B->rot = 2; }
    if (B->rot < -2) { B->rot = -2; }
    printf("[ui] rot=%d (%s)\n", B->rot,
           B->rot == 0 ? "off" : (B->rot < 0 ? "ccw90" : "cw90"));
  }

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
  printf("[ui] 屏就绪 240x240 RGB565, 暂存 %p (%d 字节)\n",
         (void *)B->scr, (int)(sizeof(uint16_t) * SCR_PIX));
  printf("[ui] putarea=%s\n", B->pa ? "有" : "无(退回 putrun)");

  /* Power-on panel proof: one full white frame.  If this is not visible the
   * problem is the panel/backlight, not the UI drawing code.
   */

  fill_rect(0, 0, UI_W, UI_H, S_BG);
  nxsig_usleep(600 * 1000);

  (void)font_load();
  B->compact_drawn = 0;
  B->dirty = 1;
#ifdef CONFIG_INPUT_TOUCHSCREEN
  (void)jx_ui_touch_start();
#endif
  extern int ai_voice_prepare_record_cue(void);
  (void)ai_voice_prepare_record_cue();
  /* Fixed receiver role must listen after power-on without an NSH command. */
  { extern void jx_ui_receiver_autostart(void); jx_ui_receiver_autostart(); }
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
  text1(10, 40, "光头强一号 自检 0123456789 kbps ms", C_TXT, C_BG);
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

/* 2026-10-29: read back the panel and report where the four logical
 * corners actually land.  Used to verify the 90-degree rotation without
 * looking at the screen. */

int jx_ui_rotcheck(void)
{
  static const uint16_t col[4] =
  {
    RGB(0xff, 0xff, 0xff), RGB(0xff, 0x00, 0x00),
    RGB(0x00, 0xff, 0x00), RGB(0x00, 0x00, 0xff)
  };
  static const char *nm[4] = { "TL", "TR", "BL", "BR" };
  static const int x0[4] = { 4, 0, 4, 0 };
  static const int y0[4] = { 4, 4, 0, 0 };
  int lx[4];
  int ly[4];
  long sx[4] = { 0, 0, 0, 0 };
  long sy[4] = { 0, 0, 0, 0 };
  long cnt[4] = { 0, 0, 0, 0 };
  uint16_t *line;
  int i;

  if (jx_ui_init() != 0) { return -1; }
  if (B->gr == NULL)
    {
      printf("[rotc] no getrun\n");
      return -1;
    }

  for (i = 0; i < 4; i++)
    {
      lx[i] = (i == 1 || i == 3) ? UI_W - 12 : x0[i];
      ly[i] = (i == 2 || i == 3) ? UI_H - 12 : y0[i];
    }

  fill_rect(0, 0, UI_W, UI_H, RGB(0x00, 0x00, 0x00));
  for (i = 0; i < 4; i++)
    {
      fill_rect(lx[i], ly[i], 8, 8, col[i]);
    }

  line = (uint16_t *)malloc(sizeof(uint16_t) * UI_W);
  if (line == NULL) { return -1; }

  for (int py = 0; py < UI_H; py++)
    {
      if (B->gr(B->lcd, (fb_coord_t)py, 0, (FAR uint8_t *)line,
                (size_t)UI_W) != 0)
        {
          printf("[rotc] getrun failed at row %d\n", py);
          free(line);
          return -1;
        }

      for (int px = 0; px < UI_W; px++)
        {
          for (i = 0; i < 4; i++)
            {
              if (line[px] == col[i])
                {
                  sx[i]  += px;
                  sy[i]  += py;
                  cnt[i] += 1;
                }
            }
        }
    }

  printf("[rotc] rot=%d panel=%dx%d\n", B->rot, UI_W, UI_H);
  for (i = 0; i < 4; i++)
    {
      if (cnt[i] == 0)
        {
          printf("[rotc] %s logical(%d,%d) NOT FOUND\n", nm[i], lx[i], ly[i]);
        }
      else
        {
          printf("[rotc] %s logical(%d,%d) -> phys(%ld,%ld) n=%ld\n",
                 nm[i], lx[i], ly[i],
                 sx[i] / cnt[i], sy[i] / cnt[i], cnt[i]);
        }
    }

  free(line);
  return 0;
}

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
