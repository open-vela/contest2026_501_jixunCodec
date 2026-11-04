/* ============================================================================
 * jx_amax.c — 见 include/jx_amax.h
 * ========================================================================== */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "jx_amax.h"

unsigned *jx_amax_cur = NULL;
unsigned long jx_amax_prod_pos = 0;
unsigned long jx_amax_prod_call = 0;
unsigned long jx_amax_tot_pos = 0;
unsigned long jx_amax_hit_pos = 0;
unsigned long jx_amax_hit_elem = 0;
unsigned long jx_amax_calls = 0;
unsigned long jx_amax_rej = 0;
unsigned long jx_amax_rj[6] = {0,0,0,0,0,0};   /* off/nv/p/t/c/v */

#define JX_AMAX_NBUF 2
static unsigned *g_buf[JX_AMAX_NBUF];
static int       g_cap = 0;
static int       g_next = 0;     /* 下一片要写的缓冲 */
static const float *g_p = NULL;
static int       g_C = 0, g_T = 0;
static int       g_valid = 0;    /* 已发布且未被作废 */

static int g_on = -1;
static int jx_amax_on(void)
{
  if (g_on < 0)
    {
      /* 2026-09-17 第四十五轮：默认改成 **关**。
       * 板端 A/B（同固件交替，3 轮）：开 3620 / 关 3600 ms，净亏 20 ms。
       * 原因：Pass B 本来吃到 Pass A 的 L1 预热，取消 A 后 B 变冷，
       * 省下的（22.5-30.3+15.7）≈ 7.9 周期/元素对不上生产者 epilogue
       * 里那一次读改写的 ~18 周期/元素。详见
       * _REALTIME_ROUND45_P0_20260917.md。JX_AMAX=1 可打开做对照。 */
      const char *e = getenv("JX_AMAX");
      g_on = (e != NULL && atoi(e) != 0);   /* 默认关 */
    }
  return g_on;
}

static int g_vn = -1;
static int jx_amax_vn(void)
{
  if (g_vn < 0)
    {
      const char *e = getenv("JX_AMAXV");
      g_vn = (e != NULL) ? atoi(e) : 4;
      if (g_vn < 0) { g_vn = 0; }
      if (g_vn > 16) { g_vn = 16; }
    }
  return g_vn;
}

static int g_dbg = -1;
static int jx_amax_dbg(void)
{
  if (g_dbg < 0) { const char *e = getenv("JX_AMAXD"); g_dbg = (e != NULL && atoi(e) != 0); }
  return g_dbg;
}

void jx_amax_arm(const float *dst, int C, int T)
{
  if (!jx_amax_on() || dst == NULL || C <= 0 || T <= 0) { jx_amax_kill(); return; }
  if (g_buf[0] == NULL || T > g_cap)
    {
      int cap = T + 64;
      unsigned *b0 = (unsigned *)malloc(sizeof(unsigned) * (size_t)cap);
      unsigned *b1 = (unsigned *)malloc(sizeof(unsigned) * (size_t)cap);
      if (b0 == NULL || b1 == NULL)
        {
          free(b0); free(b1);
          g_buf[0] = g_buf[1] = NULL; g_cap = 0;
          jx_amax_kill(); return;
        }
      free(g_buf[0]); free(g_buf[1]);
      g_buf[0] = b0; g_buf[1] = b1; g_cap = cap;
    }
  unsigned *b = g_buf[g_next];
  memset(b, 0, sizeof(unsigned) * (size_t)T);
  g_p = dst; g_C = C; g_T = T;
  g_valid = 0;              /* 写完 pub() 才作数 */
  jx_amax_cur = b;
  jx_amax_prod_call++;
}

void jx_amax_pub(void)
{
  if (jx_amax_cur != NULL)
    {
      g_valid = 1;
      jx_amax_prod_pos += (unsigned long)g_T;
      g_next ^= 1;
    }
  jx_amax_cur = NULL;
}

void jx_amax_kill(void)
{
  g_valid = 0;
  jx_amax_cur = NULL;
}

const unsigned *jx_amax_probe(const float *src, int C, int T, int *have)
{
  if (have != NULL) { *have = 0; }
  jx_amax_calls++;
  jx_amax_tot_pos += (unsigned long)((T > 0) ? T : 0);
  if (!jx_amax_on() || src == NULL || T <= 0) { jx_amax_rej++; jx_amax_rj[0]++; return NULL; }
  if (!g_valid) { jx_amax_rej++; jx_amax_rj[1]++; return NULL; }
  if (g_p != src) { jx_amax_rej++; jx_amax_rj[2]++;
    if (jx_amax_dbg()) { printf("  [amaxd] 指针不同 src=%p pub=%p C=%d/%d T=%d/%d\n",
                               (const void *)src, (const void *)g_p, C, g_C, T, g_T); }
    return NULL; }
  if (g_T != T || g_C > C || g_C < 1) { jx_amax_rej++; jx_amax_rj[3]++;
    if (jx_amax_dbg()) { printf("  [amaxd] 形状不同 src=%p pub=%p C=%d/%d T=%d/%d\n",
                               (const void *)src, (const void *)g_p, C, g_C, T, g_T); }
    return NULL; }

  const unsigned *b = g_buf[g_next ^ 1];
  const int cov = g_C;   /* 只复核边车**真正覆盖**的那几条通道：
                          * 81 = 80 + 1 拼接时第 81 条是别人写的 */
  int vn = jx_amax_vn();
  for (int k = 0; k < vn; k++)
    {
      int t = (int)(((long long)T * (long long)(2 * k + 1)) / (long long)(2 * vn));
      if (t >= T) { t = T - 1; }
      unsigned m = 0u;
      for (int ci = 0; ci < cov; ci++)
        {
          unsigned u;
          float v = src[(size_t)ci * (size_t)T + (size_t)t];
          __builtin_memcpy(&u, &v, sizeof(u));
          u &= 0x7fffffffu;
          if (u > m) { m = u; }
        }
      if (m != b[t]) { jx_amax_rej++; jx_amax_rj[5]++;
        if (jx_amax_dbg()) { printf("  [amaxd] 复核不过 src=%p C=%d T=%d t=%d 算=%08x 边车=%08x\n",
                                   (const void *)src, C, T, t, m, b[t]); }
        return NULL; }
    }

  if (have != NULL) { *have = g_C; }
  jx_amax_hit_pos += (unsigned long)T;
  jx_amax_hit_elem += (unsigned long)T * (unsigned long)g_C;
  return b;
}

void jx_amax_dump(void)
{
  double cov = (jx_amax_tot_pos > 0)
             ? (100.0 * (double)jx_amax_hit_pos / (double)jx_amax_tot_pos) : 0.0;
  printf("  [amax] 生产 %lu 位置/%lu 次 | 消费 probe %lu 次 (拒 %lu) 覆盖 %lu/%lu 位置 = %.1f%%"
         "  免读 %lu (ci,pos) 对\n",
         jx_amax_prod_pos, jx_amax_prod_call, jx_amax_calls, jx_amax_rej,
         jx_amax_hit_pos, jx_amax_tot_pos, cov, jx_amax_hit_elem);
  printf("  [amax] 拒因 off/nv/指针/形状/复核 = %lu/%lu/%lu/%lu/%lu\n",
         jx_amax_rj[0], jx_amax_rj[1], jx_amax_rj[2], jx_amax_rj[3], jx_amax_rj[5]);
}
