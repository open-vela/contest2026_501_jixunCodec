
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "jx_font_bin.h"
#define UI_W 240
#define UI_H 240
#define RGB(r,g,b) ((uint16_t)((((r)&0xf8u)<<8)|(((g)&0xfcu)<<3)|((b)>>3)))
static uint16_t frame[240*240];
typedef struct {
  const uint8_t *font; uint16_t *scr; int fc,gw,gh,ready,font_ok; uint32_t fdata;
} host_t;
static host_t host;
static host_t *B=&host;
static uint16_t scratch[240*16];
static void flush(int x,int y,int w,int h) {
  assert(x>=0 && y>=0 && w>0 && h>0 && x+w<=240 && y+h<=240 && w*h<=240*16);
  for(int yy=0;yy<h;yy++) memcpy(frame+(y+yy)*240+x,B->scr+yy*w,w*2);
}
static void fill_rect(int x,int y,int w,int h,uint16_t c) {
  assert(x>=0 && y>=0 && w>=0 && h>=0 && x+w<=240 && y+h<=240);
  for(int yy=y;yy<y+h;yy++) for(int xx=x;xx<x+w;xx++)frame[yy*240+xx]=c;
}
static void __attribute__((unused)) frame_rect(int x,int y,int w,int h,uint16_t c) {
  fill_rect(x,y,w,1,c);fill_rect(x,y+h-1,w,1,c);
  fill_rect(x,y,1,h,c);fill_rect(x+w-1,y,1,h,c);
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

static void text_c(int cx, int y, const char *s, uint16_t fg, uint16_t bg)
{
  int w = text_width(s, UI_W);
  text1(cx - w / 2, y, s, fg, bg);
}

#include "jx_simple_ui.inc"
static void phase_tests(void) {
  struct { const char *raw; enum simple_phase want; } checks[] = {
    {"READY",SIMPLE_IDLE},{"K0=1 PRESS BOOT",SIMPLE_IDLE},
    {"REC PREP 2s",SIMPLE_PREP},{"RECORDING",SIMPLE_RECORD},
    {"ENCODE TX",SIMPLE_SEND},{"STOP REQ",SIMPLE_SEND},
    {"TX DONE",SIMPLE_SENT},{"TX FAILED",SIMPLE_ERROR},
    {"RX WAIT",SIMPLE_WAIT},{"RX 24 pkt",SIMPLE_RX},
    {"RX DECODE",SIMPLE_DECODE},{"PLAYING",SIMPLE_PLAY},
    {"RX DONE",SIMPLE_DONE},{"RX FAILED",SIMPLE_ERROR},
    {"RX STOPPED",SIMPLE_STOP},{"PEER NOT SET",SIMPLE_ERROR}
  };
  for(unsigned i=0;i<sizeof(checks)/sizeof(checks[0]);i++) {
    simple_state_t s={0,checks[i].raw,0,0,0};assert(simple_phase_for(&s)==checks[i].want);
  }
  simple_state_t rx={1,"K0=1 PRESS BOOT",0,0,0};assert(simple_phase_for(&rx)==SIMPLE_WAIT);
}
int main(int argc,char **argv) {
  assert(argc==5);
  host.font=g_jx_font;host.scr=scratch;host.gw=rd16(g_jx_font+4);host.gh=rd16(g_jx_font+6);
  host.fc=rd32(g_jx_font+8);host.fdata=rd32(g_jx_font+12);host.ready=host.font_ok=1;
  phase_tests();
  simple_state_t s={atoi(argv[1]),argv[2],0,0,0};
  simple_render(&s,atoi(argv[3]));
  FILE *f=fopen(argv[4],"wb");assert(f);fprintf(f,"P6\n240 240\n255\n");
  for(int i=0;i<240*240;i++) {
    uint16_t v=frame[i];unsigned char p[3]={((v>>11)&31)*255/31,((v>>5)&63)*255/63,(v&31)*255/31};
    assert(fwrite(p,1,3,f)==3);
  }
  assert(fclose(f)==0);
  printf("native_phase_tests=17 bounds=pass role=%d phase=%d page=%s\n",s.rx_role,(int)simple_phase_for(&s),argv[3]);
  return 0;
}
