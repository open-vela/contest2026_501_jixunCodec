/* ============================================================================
 * jx_p0probe.c  -  2026-09-15 第四十轮
 *
 * P0（生产者直出 int8 + 逐行 scale）可行性探针。
 *
 * 问题：量化站点实测 41.9 周期/元素（[cf] l2：3.15M 元素 = 132 Mcyc = 550ms）、
 *       40.9（[i8mm] prologue）、51.1（[k1s] PassA+PassB）。
 *       这 40 多周期里，"读 PSRAM 浮点"占多少、"浮点依赖链"占多少？
 *       —— 这个比例决定 int8+scale 值多少钱：
 *         现在：生产者写 float(4B) → 消费者 读 float + 变换 + 求 max → 再量化 → 写 int8
 *         P0  ：生产者(值已在寄存器/行缓冲) 变换+max+量化 → 写 int8(1B)；消费者只读 int8
 *
 * 形状取真实值：行 = 128 通道（l2 的 in 实测有 64/128/256/352），501 行，
 * REP 16 → 每次 1.03M 元素。行缓冲走栈（任务栈在内部 SRAM）。
 * 变体：
 *   [q1] 现状：读 PSRAM float → 变换+max（行写 SRAM）→ 量化 → 写 PSRAM int8
 *   [q2] 同 q1，但输入行常驻内部 SRAM（去掉那趟 PSRAM 读）
 *   [q3] 只做第二趟（max 已知）：读 PSRAM float → ×inv → 取整 → 钳位 → 写 int8
 *   [q4] P0 新消费者：只读 PSRAM int8 行
 *   [q5] P0 新生产者：值已在 SRAM 行 → 变换+max+量化 → 写 PSRAM int8
 *   [q6] 纯写 PSRAM float 行（旧生产者存储成本）
 *   [q7] 纯写 PSRAM int8 行（新生产者存储成本）
 *   [q8] 纯读 PSRAM float 行
 * P0 净收益 ≈ ([q1] + [q6]) − ([q5] + [q4])。
 *
 * 单独一个编译单元：jx_main.c 里 jx_qacc_dot8_s16 的 asm 在寄存器压力变化时
 * 会报 "cannot find a register in class 'RL_REGS'"（已踩过一次），放这里隔离。
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "jx_fastmath.h"

static volatile float g_p0_sink;

static double p0_now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* ============================================================================
 * 第四十一轮：[s*] snake1d 现场分解探针
 *
 * 板上 profile：snake1d = 930 ms / 1,026,048 元素 = 217.5 周期/元素。
 * 同一份多项式的孤立微基准 [1c] 只要 32 周期/次，而且 1/2/4/8 链完全不改善
 * —— 那 32 周期是"指令吞吐地板"，不是延迟墙。两者差 6.8 倍，差额只可能来自
 *   ① 访存（PSRAM 实测 0.34 字节/周期）  ② 慢路径回退分支  ③ 循环/取指结构。
 * 本探针跑真实形状（352 通道 × 512 行，四路展开，与 snake_apply 内层同构），
 * 把上面三项逐个拿掉做对照，看每一项值多少周期/元素。
 * 缓冲区全部走堆：jx_main.c 内部 DRAM 已满，本文件不许新增静态数组。
 * ========================================================================== */
static inline unsigned jx_sp_cc(void)
{
  unsigned c;
  __asm__ __volatile__("rsr %0, ccount" : "=r"(c));
  return c;
}

/* 寄存器内 4 路快路径多项式：纯吞吐地板，零访存（对照 [1c] 的 32 周期/次） */
static float jx_sp_reg(long rep)
{
  const jx_sq4 K = jx_sq4_load();
  float x0 = 0.30f, x1 = 0.31f, x2 = 0.32f, x3 = 0.33f;
  for (long r = 0; r < rep; r++)
    {
      float w0 = x0 * x0, w1 = x1 * x1, w2 = x2 * x2, w3 = x3 * x3;
      x0 = (((K.k0 * w0 + K.k1) * w0 + K.k2) * w0 + K.k3) * w0;
      x1 = (((K.k0 * w1 + K.k1) * w1 + K.k2) * w1 + K.k3) * w1;
      x2 = (((K.k0 * w2 + K.k1) * w2 + K.k2) * w2 + K.k3) * w2;
      x3 = (((K.k0 * w3 + K.k1) * w3 + K.k2) * w3 + K.k3) * w3;
    }
  return x0 + x1 + x2 + x3;
}

/* [s1] 现状：与 nn_ops.c snake_apply 内层逐字同构（含慢路径回退） */
static void jx_sp_full(float *x, const float *a, const float *inv, int rows, int C)
{
  const jx_sq4 K = jx_sq4_load();
  for (int i = 0; i < rows; i++)
    {
      float *p = x + (size_t)i * C;
      int c = 0;
      for (; c + 4 <= C; c += 4)
        {
          float v0 = p[c], v1 = p[c + 1], v2 = p[c + 2], v3 = p[c + 3];
          float s0 = jx_sinsq4(a[c]     * v0, K);
          float s1 = jx_sinsq4(a[c + 1] * v1, K);
          float s2 = jx_sinsq4(a[c + 2] * v2, K);
          float s3 = jx_sinsq4(a[c + 3] * v3, K);
          p[c]     = v0 + inv[c]     * s0;
          p[c + 1] = v1 + inv[c + 1] * s1;
          p[c + 2] = v2 + inv[c + 2] * s2;
          p[c + 3] = v3 + inv[c + 3] * s3;
        }
      for (; c < C; c++) { float v = p[c]; p[c] = v + inv[c] * jx_sinsq4(a[c] * v, K); }
    }
}

/* [s2] 去掉慢路径分支（恒快路径多项式），其余完全相同 -> 量"分支"值多少 */
static void jx_sp_nobr(float *x, const float *a, const float *inv, int rows, int C)
{
  const jx_sq4 K = jx_sq4_load();
  for (int i = 0; i < rows; i++)
    {
      float *p = x + (size_t)i * C;
      int c = 0;
      for (; c + 4 <= C; c += 4)
        {
          float v0 = p[c], v1 = p[c + 1], v2 = p[c + 2], v3 = p[c + 3];
          float w0 = a[c] * v0, w1 = a[c + 1] * v1, w2 = a[c + 2] * v2, w3 = a[c + 3] * v3;
          w0 *= w0; w1 *= w1; w2 *= w2; w3 *= w3;
          float s0 = (((K.k0 * w0 + K.k1) * w0 + K.k2) * w0 + K.k3) * w0;
          float s1 = (((K.k0 * w1 + K.k1) * w1 + K.k2) * w1 + K.k3) * w1;
          float s2 = (((K.k0 * w2 + K.k1) * w2 + K.k2) * w2 + K.k3) * w2;
          float s3 = (((K.k0 * w3 + K.k1) * w3 + K.k2) * w3 + K.k3) * w3;
          p[c]     = v0 + inv[c]     * s0;
          p[c + 1] = v1 + inv[c + 1] * s1;
          p[c + 2] = v2 + inv[c + 2] * s2;
          p[c + 3] = v3 + inv[c + 3] * s3;
        }
      for (; c < C; c++)
        {
          float v = p[c];
          float w = a[c] * v; w *= w;
          p[c] = v + inv[c] * ((((K.k0 * w + K.k1) * w + K.k2) * w + K.k3) * w);
        }
    }
}

/* [s9]/[s10]：生产路径 snake1d_copy 内层的两种写法，dst != src，形状与真实一致。
 * s9  = 现在生产用的 jx_sinsq4_x4（4 路指针式）
 * s10 = 改成 4 次独立 jx_sinsq4（与 [s1] 的算式/次序完全相同）
 * 两者数值逐位一致，只是表达式拆分方式不同 => 差多少就是纯调度损失。 */
static void jx_sp_x4(float *d, const float *s, const float *a, const float *inv,
                     int rows, int C)
{
  const jx_sq4 K = jx_sq4_load();
  for (int i = 0; i < rows; i++)
    {
      const float *sr = s + (size_t)i * C;
      float *p = d + (size_t)i * C;
      int c = 0;
      for (; c + 4 <= C; c += 4)
        {
          float v0 = sr[c], v1 = sr[c + 1], v2 = sr[c + 2], v3 = sr[c + 3];
          float z0, z1, z2, z3;
          jx_sinsq4_x4(a[c] * v0, a[c + 1] * v1, a[c + 2] * v2, a[c + 3] * v3,
                       K, &z0, &z1, &z2, &z3);
          p[c]     = v0 + inv[c]     * z0;
          p[c + 1] = v1 + inv[c + 1] * z1;
          p[c + 2] = v2 + inv[c + 2] * z2;
          p[c + 3] = v3 + inv[c + 3] * z3;
        }
      for (; c < C; c++) { float v = sr[c]; p[c] = v + inv[c] * jx_sinsq4(a[c] * v, K); }
    }
}

static void jx_sp_4c(float *d, const float *s, const float *a, const float *inv,
                     int rows, int C)
{
  const jx_sq4 K = jx_sq4_load();
  for (int i = 0; i < rows; i++)
    {
      const float *sr = s + (size_t)i * C;
      float *p = d + (size_t)i * C;
      int c = 0;
      for (; c + 4 <= C; c += 4)
        {
          float v0 = sr[c], v1 = sr[c + 1], v2 = sr[c + 2], v3 = sr[c + 3];
          p[c]     = v0 + inv[c]     * jx_sinsq4(a[c]     * v0, K);
          p[c + 1] = v1 + inv[c + 1] * jx_sinsq4(a[c + 1] * v1, K);
          p[c + 2] = v2 + inv[c + 2] * jx_sinsq4(a[c + 2] * v2, K);
          p[c + 3] = v3 + inv[c + 3] * jx_sinsq4(a[c + 3] * v3, K);
        }
      for (; c < C; c++) { float v = sr[c]; p[c] = v + inv[c] * jx_sinsq4(a[c] * v, K); }
    }
}

/* [s3] 只读 + 算，不回写 -> 与 [s1] 之差 = 写 PSRAM 的成本 */
static float jx_sp_read(const float *x, const float *a, int rows, int C)
{
  const jx_sq4 K = jx_sq4_load();
  float acc = 0.0f;
  for (int i = 0; i < rows; i++)
    {
      const float *p = x + (size_t)i * C;
      int c = 0;
      for (; c + 4 <= C; c += 4)
        {
          float v0 = p[c], v1 = p[c + 1], v2 = p[c + 2], v3 = p[c + 3];
          acc += jx_sinsq4(a[c] * v0, K) + jx_sinsq4(a[c + 1] * v1, K)
               + jx_sinsq4(a[c + 2] * v2, K) + jx_sinsq4(a[c + 3] * v3, K);
        }
      for (; c < C; c++) { acc += jx_sinsq4(a[c] * p[c], K); }
    }
  return acc;
}

/* [s4] 纯读地板 / [s5] 纯写地板 */
static float jx_sp_sum(const float *x, int rows, int C)
{
  float acc = 0.0f;
  for (int i = 0; i < rows; i++)
    {
      const float *p = x + (size_t)i * C;
      for (int c = 0; c < C; c++) { acc += p[c]; }
    }
  return acc;
}

static void jx_sp_wr(float *x, int rows, int C, float v)
{
  for (int i = 0; i < rows; i++)
    {
      float *p = x + (size_t)i * C;
      for (int c = 0; c < C; c++) { p[c] = v; }
    }
}

void jx_snake_probe(void)
{
  const int C = 352, ROWS = 512, REP = 4;
  const int C2 = 352, ROWS2 = 16;
  const size_t n  = (size_t)ROWS * (size_t)C;
  const size_t n2 = (size_t)ROWS2 * (size_t)C2;
  float *fx = (float *)malloc(n * sizeof(float));
  float *fd = (float *)malloc(n * sizeof(float));
  float *sx = (float *)malloc(n2 * sizeof(float));
  float *fa = (float *)malloc((size_t)C * sizeof(float));
  float *fi = (float *)malloc((size_t)C * sizeof(float));
  if (fx == NULL || fd == NULL || sx == NULL || fa == NULL || fi == NULL)
    { printf("  [snake] alloc fail\n"); free(fx); free(fd); free(sx); free(fa); free(fi); return; }
  for (int c = 0; c < C; c++)
    {
      fa[c] = 0.5f + (float)(c % 16) * 0.1f;
      fi[c] = 1.0f / fa[c];
    }
  for (size_t i = 0; i < n; i++)  { fx[i] = (float)((int)(i % 9973) % 200) * 0.01f - 1.0f; }
  for (size_t i = 0; i < n2; i++) { sx[i] = fx[i]; }
  printf("=== [snake] 现场分解（%d 通道 x %d 行, REP %d, 四路展开） ===\n", C, ROWS, REP);

  {
    const int RC = 2000000;
    unsigned c0 = jx_sp_cc();
    float s = jx_sp_reg(RC);
    unsigned c1 = jx_sp_cc();
    printf("  %-32s: %7.2f Mcyc  %6.2f 周期/次（零访存, 4 路交错）\n",
           "[s0] 纯多项式寄存器吞吐", (double)(c1 - c0) / 1e6,
           (double)(c1 - c0) / ((double)RC * 4.0));
    g_p0_sink = s;
  }

#define JX_SP_RUN(TAG, CALL, N)                                        \
  do {                                                                 \
    unsigned c0 = jx_sp_cc();                                          \
    for (int r = 0; r < (REP); r++) { CALL; }                          \
    unsigned c1 = jx_sp_cc();                                          \
    printf("  %-32s: %7.2f Mcyc  %6.2f 周期/元素\n", TAG,              \
           (double)(c1 - c0) / 1e6,                                    \
           (double)(c1 - c0) / ((double)(REP) * (double)(N)));         \
  } while (0)

  JX_SP_RUN("[s1] 现状 PSRAM 就地读改",  jx_sp_full(fx, fa, fi, ROWS, C), n);
  JX_SP_RUN("[s2] 去掉慢路径分支",       jx_sp_nobr(fx, fa, fi, ROWS, C), n);
  JX_SP_RUN("[s3] 只读+算, 不回写",      g_p0_sink = jx_sp_read(fx, fa, ROWS, C), n);
  JX_SP_RUN("[s4] 纯读地板",             g_p0_sink = jx_sp_sum(fx, ROWS, C), n);
  JX_SP_RUN("[s5] 纯写地板",             jx_sp_wr(fx, ROWS, C, 1.0f), n);
  JX_SP_RUN("[s6] 同 s1 但数据在内部 SRAM", jx_sp_full(sx, fa, fi, ROWS2, C2), n2);
  JX_SP_RUN("[s7] 同 s2 但数据在内部 SRAM", jx_sp_nobr(sx, fa, fi, ROWS2, C2), n2);
  JX_SP_RUN("[s8] 纯读地板(内部 SRAM)",  g_p0_sink = jx_sp_sum(sx, ROWS2, C2), n2);
  JX_SP_RUN("[s9]  src->dst x4 指针式(生产)", jx_sp_x4(fd, fx, fa, fi, ROWS, C), n);
  JX_SP_RUN("[s10] src->dst 4x jx_sinsq4    ", jx_sp_4c(fd, fx, fa, fi, ROWS, C), n);
  JX_SP_RUN("[s11] 原地 x4 指针式           ", jx_sp_x4(fx, fx, fa, fi, ROWS, C), n);
#undef JX_SP_RUN
  free(fx); free(fd); free(sx); free(fa); free(fi);
}

void jx_p0_probe(void)
{
  const double SYSCLK = 240.0e6;
  const int C = 128, Tt = 501, REP = 16;
  const size_t n = (size_t)Tt * (size_t)C;
  /* 内部 DRAM 只剩 ~0.8KB，禁止新增静态数组：
   * 行缓冲走栈（任务栈在内部 SRAM），gam/bet 走堆（真实代码里它们就是 PSRAM 权重）。 */
  float       rowx[128], rowv[128];
  signed char qs[128];
  float       *fx  = (float *)malloc(n * sizeof(float));
  signed char *qx  = (signed char *)malloc(n);
  float       *gam = (float *)malloc((size_t)C * sizeof(float));
  float       *bet = (float *)malloc((size_t)C * sizeof(float));
  if (fx == NULL || qx == NULL || gam == NULL || bet == NULL)
    { printf("  [p0] alloc fail\n"); free(fx); free(qx); free(gam); free(bet); return; }
  for (int c = 0; c < C; c++)
    {
      gam[c] = 0.9f + (float)(c % 7) * 0.02f;
      bet[c] = (float)(c % 5) * 0.01f;
      rowx[c] = (float)((c * 13) % 100) * 0.02f - 1.0f;
      qs[c] = (signed char)((c % 251) - 125);
    }
  for (size_t i = 0; i < n; i++)
    {
      fx[i] = (float)((int)(i % 9973) % 100) * 0.02f - 1.0f;
      qx[i] = (signed char)((int)(i % 251) - 125);
    }
  const float nx = 0.9f;
  printf("=== P0 量化站点分解探针 (行=%d 通道, %d 行 = %d 元素/趟, REP %d) ===\n",
         C, Tt, (int)n, REP);

#define P0_RUN(TAG, BLOCK)                                                     \
  do {                                                                         \
    double t0 = p0_now_ms();                                                   \
    for (int rep = 0; rep < REP; rep++) { BLOCK }                              \
    double ms = p0_now_ms() - t0;                                              \
    g_p0_sink = fx[n / 2] + (float)qx[n / 2] + (float)qs[0];                   \
    printf("  %-30s: %8.1f ms  %6.1f 周期/元素\n", TAG, ms,                    \
           ms * 1e-3 * SYSCLK / ((double)REP * (double)n));                    \
  } while (0)

  /* [q1] 现状：读 PSRAM float → 变换+max（行写 SRAM）→ 量化 → 写 PSRAM int8 */
  P0_RUN("[q1] 现状 读float+变换+max+量化",
    {
      for (int t = 0; t < Tt; t++)
        {
          const float *x = fx + (size_t)t * C;
          signed char *q = qx + (size_t)t * C;
          float mx = 0.0f;
          for (int i = 0; i < C; i++)
            {
              float v = gam[i] * (x[i] * nx) + bet[i] + x[i];
              rowv[i] = v;
              float a = fabsf(v); if (a > mx) { mx = a; }
            }
          float inv = 127.0f / mx;
          for (int i = 0; i < C; i++)
            {
              float v = rowv[i] * inv;
              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
              if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
              q[i] = (signed char)iv;
            }
        }
    });

  /* [q2] 输入行常驻内部 SRAM。nxr 带上 t 的微扰，防止 GCC 把整段提到循环外。 */
  P0_RUN("[q2] 同q1 但输入在SRAM",
    {
      for (int t = 0; t < Tt; t++)
        {
          const float *x = rowx;
          signed char *q = qx + (size_t)t * C;
          const float nxr = nx * (1.0f + (float)(t & 1) * 1e-12f);
          float mx = 0.0f;
          for (int i = 0; i < C; i++)
            {
              float v = gam[i] * (x[i] * nxr) + bet[i] + x[i];
              rowv[i] = v;
              float a = fabsf(v); if (a > mx) { mx = a; }
            }
          float inv = 127.0f / mx;
          for (int i = 0; i < C; i++)
            {
              float v = rowv[i] * inv;
              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
              if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
              q[i] = (signed char)iv;
            }
        }
    });

  /* [q3] 只做第二趟（max 已知 = 1.0） */
  P0_RUN("[q3] 只第二趟 max已知",
    {
      for (int t = 0; t < Tt; t++)
        {
          const float *x = fx + (size_t)t * C;
          signed char *q = qx + (size_t)t * C;
          const float inv = 127.0f;
          for (int i = 0; i < C; i++)
            {
              float v = x[i] * inv;
              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
              if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
              q[i] = (signed char)iv;
            }
        }
    });

  /* [q4] P0 新消费者：只读 PSRAM int8 行 */
  P0_RUN("[q4] P0消费者 只读int8",
    {
      int32_t a = 0;
      for (int t = 0; t < Tt; t++)
        {
          const signed char *xq = qx + (size_t)t * C;
          for (int i = 0; i < C; i++) { a += xq[i]; }
        }
      if (a == 0x7fffffff) { qs[0] = 1; }
    });

  /* [q5] P0 新生产者：值已在 SRAM 行 → 变换+max+量化 → 写 PSRAM int8 */
  P0_RUN("[q5] P0生产者 变换+max+量化+写",
    {
      for (int t = 0; t < Tt; t++)
        {
          const float *x = rowx;
          signed char *q = qx + (size_t)t * C;
          const float nxr = nx * (1.0f + (float)(t & 1) * 1e-12f);
          float mx = 0.0f;
          for (int i = 0; i < C; i++)
            {
              float v = gam[i] * (x[i] * nxr) + bet[i] + x[i];
              rowv[i] = v;
              float a = fabsf(v); if (a > mx) { mx = a; }
            }
          float inv = 127.0f / mx;
          for (int i = 0; i < C; i++)
            {
              float v = rowv[i] * inv;
              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
              if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
              q[i] = (signed char)iv;
            }
        }
    });

  /* [q6] 纯写 PSRAM float 行（旧生产者存储成本） */
  P0_RUN("[q6] 纯写float行(4B)",
    {
      for (int t = 0; t < Tt; t++)
        {
          float *d = fx + (size_t)t * C;
          for (int i = 0; i < C; i++) { d[i] = rowx[i]; }
        }
    });

  /* [q7] 纯写 PSRAM int8 行（新生产者存储成本） */
  P0_RUN("[q7] 纯写int8行(1B)",
    {
      for (int t = 0; t < Tt; t++)
        {
          signed char *d = qx + (size_t)t * C;
          for (int i = 0; i < C; i++) { d[i] = qs[i]; }
        }
    });

  /* [q8] 纯读 PSRAM float 行 */
  P0_RUN("[q8] 纯读float行(4B)",
    {
      float a = 0.0f;
      for (int t = 0; t < Tt; t++)
        {
          const float *x = fx + (size_t)t * C;
          for (int i = 0; i < C; i++) { a += x[i]; }
        }
      if (a == 1e30f) { qs[0] = 1; }
    });


  /* [q9] 只第二趟 + 无分支钳位（三目） */

  P0_RUN("[q9] 2nd pass 无分支钳位",

    {

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *q = qx + (size_t)t * C;

          const float inv = 127.0f;

          for (int i = 0; i < C; i++)

            {

              float v = x[i] * inv;

              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));

              iv = (iv > 127) ? 127 : ((iv < -127) ? -127 : iv);

              q[i] = (signed char)iv;

            }

        }

    });



  /* [q10] 4 路展开 + 无分支钳位 */

  P0_RUN("[q10] 2nd pass 4路+无分支钳位",

    {

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *q = qx + (size_t)t * C;

          const float inv = 127.0f;

          int i = 0;

          for (; i + 4 <= C; i += 4)

            {

              float v0 = x[i] * inv; float v1 = x[i + 1] * inv;

              float v2 = x[i + 2] * inv; float v3 = x[i + 3] * inv;

              int a0 = (int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));

              int a1 = (int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));

              int a2 = (int)(v2 + (v2 >= 0.0f ? 0.5f : -0.5f));

              int a3 = (int)(v3 + (v3 >= 0.0f ? 0.5f : -0.5f));

              a0 = (a0 > 127) ? 127 : ((a0 < -127) ? -127 : a0);

              a1 = (a1 > 127) ? 127 : ((a1 < -127) ? -127 : a1);

              a2 = (a2 > 127) ? 127 : ((a2 < -127) ? -127 : a2);

              a3 = (a3 > 127) ? 127 : ((a3 < -127) ? -127 : a3);

              q[i] = (signed char)a0; q[i + 1] = (signed char)a1;

              q[i + 2] = (signed char)a2; q[i + 3] = (signed char)a3;

            }

          for (; i < C; i++)

            {

              float v = x[i] * inv;

              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));

              iv = (iv > 127) ? 127 : ((iv < -127) ? -127 : iv);

              q[i] = (signed char)iv;

            }

        }

    });



  /* [q11] 4 路展开 + 分支钳位（现写法对照） */

  P0_RUN("[q11] 2nd pass 4路+分支钳位",

    {

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *q = qx + (size_t)t * C;

          const float inv = 127.0f;

          int i = 0;

          for (; i + 4 <= C; i += 4)

            {

              float v0 = x[i] * inv; float v1 = x[i + 1] * inv;

              float v2 = x[i + 2] * inv; float v3 = x[i + 3] * inv;

              int a0 = (int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));

              int a1 = (int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));

              int a2 = (int)(v2 + (v2 >= 0.0f ? 0.5f : -0.5f));

              int a3 = (int)(v3 + (v3 >= 0.0f ? 0.5f : -0.5f));

              if (a0 > 127) { a0 = 127; } else if (a0 < -127) { a0 = -127; }

              if (a1 > 127) { a1 = 127; } else if (a1 < -127) { a1 = -127; }

              if (a2 > 127) { a2 = 127; } else if (a2 < -127) { a2 = -127; }

              if (a3 > 127) { a3 = 127; } else if (a3 < -127) { a3 = -127; }

              q[i] = (signed char)a0; q[i + 1] = (signed char)a1;

              q[i + 2] = (signed char)a2; q[i + 3] = (signed char)a3;

            }

          for (; i < C; i++)

            {

              float v = x[i] * inv;

              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));

              if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }

              q[i] = (signed char)iv;

            }

        }

    });



  /* [q12] 4 路展开 + 无钳位（上限参考） */

  P0_RUN("[q12] 2nd pass 4路+无钳位(上限)",

    {

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *q = qx + (size_t)t * C;

          const float inv = 127.0f;

          int i = 0;

          for (; i + 4 <= C; i += 4)

            {

              float v0 = x[i] * inv; float v1 = x[i + 1] * inv;

              float v2 = x[i + 2] * inv; float v3 = x[i + 3] * inv;

              q[i]     = (signed char)(int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));

              q[i + 1] = (signed char)(int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));

              q[i + 2] = (signed char)(int)(v2 + (v2 >= 0.0f ? 0.5f : -0.5f));

              q[i + 3] = (signed char)(int)(v3 + (v3 >= 0.0f ? 0.5f : -0.5f));

            }

          for (; i < C; i++)

            {

              float v = x[i] * inv;

              q[i] = (signed char)(int)(v + (v >= 0.0f ? 0.5f : -0.5f));

            }

        }

    });



  /* [q13] 4 路展开 + 无钳位 + 截断（舍入成本参考） */

  P0_RUN("[q13] 2nd pass 4路+截断无舍入",

    {

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *q = qx + (size_t)t * C;

          const float inv = 127.0f;

          int i = 0;

          for (; i + 4 <= C; i += 4)

            {

              q[i]     = (signed char)(int)(x[i]     * inv);

              q[i + 1] = (signed char)(int)(x[i + 1] * inv);

              q[i + 2] = (signed char)(int)(x[i + 2] * inv);

              q[i + 3] = (signed char)(int)(x[i + 3] * inv);

            }

          for (; i < C; i++) { q[i] = (signed char)(int)(x[i] * inv); }

        }

    });





  /* [q14] 4 路展开 + 跨步字节写（stride=32，模拟 im2col s_tnq[j*K16]） */

  P0_RUN("[q14] 4路+跨步字节写 stride32",

    {

      signed char *qb = (signed char *)fx;

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *qd = qb + (size_t)(t % 64) * 32;

          const float inv = 127.0f;

          int i = 0;

          for (; i + 4 <= C && i + 3 < 64; i += 4)

            {

              float v0 = x[i] * inv; float v1 = x[i + 1] * inv;

              float v2 = x[i + 2] * inv; float v3 = x[i + 3] * inv;

              int a0 = (int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));

              int a1 = (int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));

              int a2 = (int)(v2 + (v2 >= 0.0f ? 0.5f : -0.5f));

              int a3 = (int)(v3 + (v3 >= 0.0f ? 0.5f : -0.5f));

              a0 = (a0 > 127) ? 127 : ((a0 < -127) ? -127 : a0);

              a1 = (a1 > 127) ? 127 : ((a1 < -127) ? -127 : a1);

              a2 = (a2 > 127) ? 127 : ((a2 < -127) ? -127 : a2);

              a3 = (a3 > 127) ? 127 : ((a3 < -127) ? -127 : a3);

              qd[(size_t)i * 32] = (signed char)a0; qd[(size_t)(i + 1) * 32] = (signed char)a1;

              qd[(size_t)(i + 2) * 32] = (signed char)a2; qd[(size_t)(i + 3) * 32] = (signed char)a3;

            }

          for (; i < C && i < 64; i++)

            {

              float v = x[i] * inv;

              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));

              iv = (iv > 127) ? 127 : ((iv < -127) ? -127 : iv);

              qd[(size_t)i * 32] = (signed char)iv;

            }

        }

    });



  /* [q15] 4 路展开 + 跨步字节写（stride=272） */

  P0_RUN("[q15] 4路+跨步字节写 stride272",

    {

      signed char *qb = (signed char *)fx;

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *qd = qb + (size_t)(t % 64) * 272;

          const float inv = 127.0f;

          int i = 0;

          for (; i + 4 <= C && i + 3 < 64; i += 4)

            {

              float v0 = x[i] * inv; float v1 = x[i + 1] * inv;

              float v2 = x[i + 2] * inv; float v3 = x[i + 3] * inv;

              int a0 = (int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));

              int a1 = (int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));

              int a2 = (int)(v2 + (v2 >= 0.0f ? 0.5f : -0.5f));

              int a3 = (int)(v3 + (v3 >= 0.0f ? 0.5f : -0.5f));

              a0 = (a0 > 127) ? 127 : ((a0 < -127) ? -127 : a0);

              a1 = (a1 > 127) ? 127 : ((a1 < -127) ? -127 : a1);

              a2 = (a2 > 127) ? 127 : ((a2 < -127) ? -127 : a2);

              a3 = (a3 > 127) ? 127 : ((a3 < -127) ? -127 : a3);

              qd[(size_t)i * 272] = (signed char)a0; qd[(size_t)(i + 1) * 272] = (signed char)a1;

              qd[(size_t)(i + 2) * 272] = (signed char)a2; qd[(size_t)(i + 3) * 272] = (signed char)a3;

            }

          for (; i < C && i < 64; i++)

            {

              float v = x[i] * inv;

              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));

              iv = (iv > 127) ? 127 : ((iv < -127) ? -127 : iv);

              qd[(size_t)i * 272] = (signed char)iv;

            }

        }

    });



  /* [q16] 4 路展开 + 连续字节写（对照 q14/q15） */

  P0_RUN("[q16] 4路+连续字节写",

    {

      signed char *qb = (signed char *)fx;

      for (int t = 0; t < Tt; t++)

        {

          const float *x = fx + (size_t)t * C;

          signed char *qd = qb + (size_t)(t % 64) * 64;

          const float inv = 127.0f;

          int i = 0;

          for (; i + 4 <= C && i + 3 < 64; i += 4)

            {

              float v0 = x[i] * inv; float v1 = x[i + 1] * inv;

              float v2 = x[i + 2] * inv; float v3 = x[i + 3] * inv;

              int a0 = (int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));

              int a1 = (int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));

              int a2 = (int)(v2 + (v2 >= 0.0f ? 0.5f : -0.5f));

              int a3 = (int)(v3 + (v3 >= 0.0f ? 0.5f : -0.5f));

              a0 = (a0 > 127) ? 127 : ((a0 < -127) ? -127 : a0);

              a1 = (a1 > 127) ? 127 : ((a1 < -127) ? -127 : a1);

              a2 = (a2 > 127) ? 127 : ((a2 < -127) ? -127 : a2);

              a3 = (a3 > 127) ? 127 : ((a3 < -127) ? -127 : a3);

              qd[i] = (signed char)a0; qd[i + 1] = (signed char)a1;

              qd[i + 2] = (signed char)a2; qd[i + 3] = (signed char)a3;

            }

          for (; i < C && i < 64; i++)

            {

              float v = x[i] * inv;

              int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));

              iv = (iv > 127) ? 127 : ((iv < -127) ? -127 : iv);

              qd[i] = (signed char)iv;

            }

        }

    });



#undef P0_RUN
  free(fx); free(qx); free(gam); free(bet);
  jx_snake_probe();
}

/* ============================================================================
 * r64：[epi] l1 epilogue（jx_linear_i8_rows 的 4 路 snake 收尾）现场分解
 *
 * 板端 profile：epilogue 251.9 Mcyc / 3,496,266 元素 = 72.0 周期/元素。
 * 反汇编（objdump jx_linear_i8_rows，loop a13,12bc）数下来热循环是
 *   127 条指令 / 4 元素 = 31.8 条/元素，其中 mov.s 16 条、栈溢出往返 15 条。
 * IPC 只有 0.44 —— 必须知道"是指令太多，还是在等延迟"。
 * 本探针把真实循环逐项拿掉做对照，直接读出每一部分值多少周期/元素。
 *
 *   形状：256 通道 x 512 行 x 8 趟 = 1,048,576 元素/变体（与真实内层同构）
 *   行缓冲走堆（首次触碰后常驻 L1），模拟真实的 g_l1row（内部 SRAM）
 *   变体：
 *     [e0] 现状 4 路（dequant + sin^2 + ssr + 写行缓冲）
 *     [e1] 去掉 sin^2（v 直接写）
 *     [e2] 去掉 ssr 累加
 *     [e3] 去掉 dequant 的两个数组读（v = (float)acc）
 *     [e4] 2 路
 *     [e5] 8 路
 *     [e6] 单路
 *     [e7] 4 路 + 通道参数合并成一条数组（一个基址指针）
 *     [e8] 同 e0 但输出写 PSRAM 大缓冲（量"行缓冲 vs PSRAM 输出"）
 * ========================================================================== */
typedef struct {
  const int32_t *accb;
  const float *swc, *sbb, *sna, *sni, *pack;
  float *out, *ssr;
  int rows, cn;
} jx_epi_ctx;

static volatile float g_epi_sink;

/* helper：4 路 sin^2（与 jx_sinsq4_x4 同一份算式） */
#define JX_EPI_SNAKE4(K, IN0, IN1, IN2, IN3, Z0, Z1, Z2, Z3) \
  jx_sinsq4_x4((IN0), (IN1), (IN2), (IN3), (K), &(Z0), &(Z1), &(Z2), &(Z3))

static void jx_epi_e0(jx_epi_ctx *c0)
{
  const int CN = c0->cn, ROWS = c0->rows;
  const jx_sq4 K = jx_sq4_load();
  const int32_t *accb = c0->accb;
  const float *swc = c0->swc, *sbb = c0->sbb, *sna = c0->sna, *sni = c0->sni;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      float ssr_acc = ssr[r];
      int i = 0;
      for (; i + 4 <= CN; i += 4)
        {
          float v0 = (float)accb[i]     * (s * swc[i])     + sbb[i];
          float v1 = (float)accb[i + 1] * (s * swc[i + 1]) + sbb[i + 1];
          float v2 = (float)accb[i + 2] * (s * swc[i + 2]) + sbb[i + 2];
          float v3 = (float)accb[i + 3] * (s * swc[i + 3]) + sbb[i + 3];
          float z0, z1, z2, z3;
          JX_EPI_SNAKE4(K, sna[i] * v0, sna[i + 1] * v1, sna[i + 2] * v2, sna[i + 3] * v3,
                        z0, z1, z2, z3);
          v0 = v0 + sni[i]     * z0;
          v1 = v1 + sni[i + 1] * z1;
          v2 = v2 + sni[i + 2] * z2;
          v3 = v3 + sni[i + 3] * z3;
          out[i]     = v0; ssr_acc += v0 * v0;
          out[i + 1] = v1; ssr_acc += v1 * v1;
          out[i + 2] = v2; ssr_acc += v2 * v2;
          out[i + 3] = v3; ssr_acc += v3 * v3;
        }
      ssr[r] = ssr_acc;
    }
}

static void jx_epi_e1(jx_epi_ctx *c0)   /* 去掉 sin^2 */
{
  const int CN = c0->cn, ROWS = c0->rows;
  const int32_t *accb = c0->accb;
  const float *swc = c0->swc, *sbb = c0->sbb, *sna = c0->sna, *sni = c0->sni;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      float ssr_acc = ssr[r];
      int i = 0;
      for (; i + 4 <= CN; i += 4)
        {
          float v0 = (float)accb[i]     * (s * swc[i])     + sbb[i];
          float v1 = (float)accb[i + 1] * (s * swc[i + 1]) + sbb[i + 1];
          float v2 = (float)accb[i + 2] * (s * swc[i + 2]) + sbb[i + 2];
          float v3 = (float)accb[i + 3] * (s * swc[i + 3]) + sbb[i + 3];
          v0 = v0 + sni[i]     * (sna[i]     * v0);
          v1 = v1 + sni[i + 1] * (sna[i + 1] * v1);
          v2 = v2 + sni[i + 2] * (sna[i + 2] * v2);
          v3 = v3 + sni[i + 3] * (sna[i + 3] * v3);
          out[i]     = v0; ssr_acc += v0 * v0;
          out[i + 1] = v1; ssr_acc += v1 * v1;
          out[i + 2] = v2; ssr_acc += v2 * v2;
          out[i + 3] = v3; ssr_acc += v3 * v3;
        }
      ssr[r] = ssr_acc;
    }
}

static void jx_epi_e2(jx_epi_ctx *c0)   /* 去掉 ssr 累加 */
{
  const int CN = c0->cn, ROWS = c0->rows;
  const jx_sq4 K = jx_sq4_load();
  const int32_t *accb = c0->accb;
  const float *swc = c0->swc, *sbb = c0->sbb, *sna = c0->sna, *sni = c0->sni;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      int i = 0;
      for (; i + 4 <= CN; i += 4)
        {
          float v0 = (float)accb[i]     * (s * swc[i])     + sbb[i];
          float v1 = (float)accb[i + 1] * (s * swc[i + 1]) + sbb[i + 1];
          float v2 = (float)accb[i + 2] * (s * swc[i + 2]) + sbb[i + 2];
          float v3 = (float)accb[i + 3] * (s * swc[i + 3]) + sbb[i + 3];
          float z0, z1, z2, z3;
          JX_EPI_SNAKE4(K, sna[i] * v0, sna[i + 1] * v1, sna[i + 2] * v2, sna[i + 3] * v3,
                        z0, z1, z2, z3);
          out[i]     = v0 + sni[i]     * z0;
          out[i + 1] = v1 + sni[i + 1] * z1;
          out[i + 2] = v2 + sni[i + 2] * z2;
          out[i + 3] = v3 + sni[i + 3] * z3;
        }
      ssr[r] = s;
    }
}

static void jx_epi_e3(jx_epi_ctx *c0)   /* 去掉 dequant 的两条数组读 */
{
  const int CN = c0->cn, ROWS = c0->rows;
  const jx_sq4 K = jx_sq4_load();
  const int32_t *accb = c0->accb;
  const float *swc = c0->swc, *sna = c0->sna, *sni = c0->sni;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      float ssr_acc = ssr[r];
      int i = 0;
      for (; i + 4 <= CN; i += 4)
        {
          float v0 = (float)accb[i]     * swc[i];
          float v1 = (float)accb[i + 1] * swc[i + 1];
          float v2 = (float)accb[i + 2] * swc[i + 2];
          float v3 = (float)accb[i + 3] * swc[i + 3];
          float z0, z1, z2, z3;
          JX_EPI_SNAKE4(K, sna[i] * v0, sna[i + 1] * v1, sna[i + 2] * v2, sna[i + 3] * v3,
                        z0, z1, z2, z3);
          v0 = v0 + sni[i]     * z0;
          v1 = v1 + sni[i + 1] * z1;
          v2 = v2 + sni[i + 2] * z2;
          v3 = v3 + sni[i + 3] * z3;
          out[i]     = v0; ssr_acc += v0 * v0;
          out[i + 1] = v1; ssr_acc += v1 * v1;
          out[i + 2] = v2; ssr_acc += v2 * v2;
          out[i + 3] = v3; ssr_acc += v3 * v3;
        }
      ssr[r] = ssr_acc;
    }
}

static void jx_epi_e2w(jx_epi_ctx *c0)   /* 2 路 */
{
  const int CN = c0->cn, ROWS = c0->rows;
  const jx_sq4 K = jx_sq4_load();
  const int32_t *accb = c0->accb;
  const float *swc = c0->swc, *sbb = c0->sbb, *sna = c0->sna, *sni = c0->sni;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      float ssr_acc = ssr[r];
      int i = 0;
      for (; i + 2 <= CN; i += 2)
        {
          float v0 = (float)accb[i]     * (s * swc[i])     + sbb[i];
          float v1 = (float)accb[i + 1] * (s * swc[i + 1]) + sbb[i + 1];
          float z0 = jx_sinsq4(sna[i] * v0, K);
          float z1 = jx_sinsq4(sna[i + 1] * v1, K);
          v0 = v0 + sni[i]     * z0;
          v1 = v1 + sni[i + 1] * z1;
          out[i]     = v0; ssr_acc += v0 * v0;
          out[i + 1] = v1; ssr_acc += v1 * v1;
        }
      ssr[r] = ssr_acc;
    }
}

static void jx_epi_e8w(jx_epi_ctx *c0)   /* 8 路 */
{
  const int CN = c0->cn, ROWS = c0->rows;
  const jx_sq4 K = jx_sq4_load();
  const int32_t *accb = c0->accb;
  const float *swc = c0->swc, *sbb = c0->sbb, *sna = c0->sna, *sni = c0->sni;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      float ssr_acc = ssr[r];
      int i = 0;
      for (; i + 8 <= CN; i += 8)
        {
          float v0 = (float)accb[i]     * (s * swc[i])     + sbb[i];
          float v1 = (float)accb[i + 1] * (s * swc[i + 1]) + sbb[i + 1];
          float v2 = (float)accb[i + 2] * (s * swc[i + 2]) + sbb[i + 2];
          float v3 = (float)accb[i + 3] * (s * swc[i + 3]) + sbb[i + 3];
          float v4 = (float)accb[i + 4] * (s * swc[i + 4]) + sbb[i + 4];
          float v5 = (float)accb[i + 5] * (s * swc[i + 5]) + sbb[i + 5];
          float v6 = (float)accb[i + 6] * (s * swc[i + 6]) + sbb[i + 6];
          float v7 = (float)accb[i + 7] * (s * swc[i + 7]) + sbb[i + 7];
          float z0, z1, z2, z3, z4, z5, z6, z7;
          JX_EPI_SNAKE4(K, sna[i] * v0, sna[i + 1] * v1, sna[i + 2] * v2, sna[i + 3] * v3,
                        z0, z1, z2, z3);
          JX_EPI_SNAKE4(K, sna[i + 4] * v4, sna[i + 5] * v5, sna[i + 6] * v6, sna[i + 7] * v7,
                        z4, z5, z6, z7);
          v0 = v0 + sni[i]     * z0; v1 = v1 + sni[i + 1] * z1;
          v2 = v2 + sni[i + 2] * z2; v3 = v3 + sni[i + 3] * z3;
          v4 = v4 + sni[i + 4] * z4; v5 = v5 + sni[i + 5] * z5;
          v6 = v6 + sni[i + 6] * z6; v7 = v7 + sni[i + 7] * z7;
          out[i]     = v0; ssr_acc += v0 * v0;
          out[i + 1] = v1; ssr_acc += v1 * v1;
          out[i + 2] = v2; ssr_acc += v2 * v2;
          out[i + 3] = v3; ssr_acc += v3 * v3;
          out[i + 4] = v4; ssr_acc += v4 * v4;
          out[i + 5] = v5; ssr_acc += v5 * v5;
          out[i + 6] = v6; ssr_acc += v6 * v6;
          out[i + 7] = v7; ssr_acc += v7 * v7;
        }
      ssr[r] = ssr_acc;
    }
}

static void jx_epi_e1w(jx_epi_ctx *c0)   /* 单路 */
{
  const int CN = c0->cn, ROWS = c0->rows;
  const jx_sq4 K = jx_sq4_load();
  const int32_t *accb = c0->accb;
  const float *swc = c0->swc, *sbb = c0->sbb, *sna = c0->sna, *sni = c0->sni;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      float ssr_acc = ssr[r];
      for (int i = 0; i < CN; i++)
        {
          float v = (float)accb[i] * (s * swc[i]) + sbb[i];
          v = v + sni[i] * jx_sinsq4(sna[i] * v, K);
          out[i] = v; ssr_acc += v * v;
        }
      ssr[r] = ssr_acc;
    }
}

static void jx_epi_epack(jx_epi_ctx *c0)  /* 4 路 + 参数合并成一条数组 */
{
  const int CN = c0->cn, ROWS = c0->rows;
  const jx_sq4 K = jx_sq4_load();
  const int32_t *accb = c0->accb;
  const float *p = c0->pack;
  float *out = c0->out, *ssr = c0->ssr;
  for (int r = 0; r < ROWS; r++)
    {
      float s = 1.0f + (float)(r & 7) * 0.017f;
      float ssr_acc = ssr[r];
      int i = 0;
      for (; i + 4 <= CN; i += 4)
        {
          const float *p0 = p + (size_t)i * 4;
          float v0 = (float)accb[i]     * (s * p0[0]) + p0[1];
          float v1 = (float)accb[i + 1] * (s * p0[4]) + p0[5];
          float v2 = (float)accb[i + 2] * (s * p0[8]) + p0[9];
          float v3 = (float)accb[i + 3] * (s * p0[12]) + p0[13];
          float z0, z1, z2, z3;
          JX_EPI_SNAKE4(K, p0[2] * v0, p0[6] * v1, p0[10] * v2, p0[14] * v3,
                        z0, z1, z2, z3);
          v0 = v0 + p0[3] * z0;
          v1 = v1 + p0[7] * z1;
          v2 = v2 + p0[11] * z2;
          v3 = v3 + p0[15] * z3;
          out[i]     = v0; ssr_acc += v0 * v0;
          out[i + 1] = v1; ssr_acc += v1 * v1;
          out[i + 2] = v2; ssr_acc += v2 * v2;
          out[i + 3] = v3; ssr_acc += v3 * v3;
        }
      ssr[r] = ssr_acc;
    }
}

void jx_epi_probe(void)
{
  const int ROWS = 512, CN = 256, REP = 8;
  const size_t N = (size_t)ROWS * (size_t)CN;
  const double SYSCLK = 240.0e6;
  jx_sq_kt_ensure();
  int32_t *accb = (int32_t *)malloc((size_t)CN * sizeof(int32_t));
  float *swc = (float *)malloc((size_t)CN * sizeof(float));
  float *sbb = (float *)malloc((size_t)CN * sizeof(float));
  float *sna = (float *)malloc((size_t)CN * sizeof(float));
  float *sni = (float *)malloc((size_t)CN * sizeof(float));
  float *pack = (float *)malloc((size_t)CN * 4 * sizeof(float));
  float *rowbuf = (float *)malloc((size_t)CN * sizeof(float));
  float *bigout = (float *)malloc(N * sizeof(float));
  float *ssr = (float *)malloc((size_t)ROWS * sizeof(float));
  if (accb == NULL || swc == NULL || sbb == NULL || sna == NULL || sni == NULL ||
      pack == NULL || rowbuf == NULL || bigout == NULL || ssr == NULL)
    {
      printf("  [epi] alloc fail\n");
      return;
    }
  for (int i = 0; i < CN; i++)
    {
      accb[i] = (int32_t)(((i * 2654435761u) >> 13) % 20001) - 10000;
      swc[i]  = 0.0007f + (float)(i % 17) * 0.00013f;
      sbb[i]  = (float)(i % 11) * 0.011f - 0.055f;
      sna[i]  = 0.5f + (float)(i % 16) * 0.09f;
      sni[i]  = 1.0f / sna[i];
      pack[i * 4 + 0] = swc[i]; pack[i * 4 + 1] = sbb[i];
      pack[i * 4 + 2] = sna[i]; pack[i * 4 + 3] = sni[i];
    }
  printf("=== [epi] l1 epilogue 分解（%d 通道 x %d 行 = %d 元素/趟, REP %d） ===\n",
         CN, ROWS, (int)N, REP);
  {
    jx_epi_ctx c0;
    c0.accb = accb; c0.swc = swc; c0.sbb = sbb; c0.sna = sna; c0.sni = sni;
    c0.pack = pack; c0.ssr = ssr; c0.rows = ROWS; c0.cn = CN;

#define JX_EPI_RUN(TAG, FN, OUTP)                                              \
    do {                                                                        \
      double best = 1e30;                                                       \
      c0.out = (OUTP);                                                          \
      for (int rep = 0; rep < 3; rep++)                                         \
        {                                                                       \
          for (int q = 0; q < ROWS; q++) { ssr[q] = 0.0f; }                     \
          unsigned c0a = jx_sp_cc();                                            \
          for (int rr = 0; rr < REP; rr++) { FN(&c0); }                          \
          unsigned c0b = jx_sp_cc();                                            \
          double cy = (double)(c0b - c0a);                                      \
          if (cy < best) { best = cy; }                                          \
        }                                                                       \
      g_epi_sink += ssr[0] + ((OUTP) != NULL ? (OUTP)[0] : 0.0f);               \
      printf("  %-34s: %8.2f Mcyc  %6.2f 周期/元素\n", (TAG), best / 1e6,      \
             best / ((double)REP * (double)N));                                 \
    } while (0)

    JX_EPI_RUN("[e0] 现状 4 路完整",        jx_epi_e0,    rowbuf);
    JX_EPI_RUN("[e1] 去掉 sin^2",           jx_epi_e1,    rowbuf);
    JX_EPI_RUN("[e2] 去掉 ssr 累加",        jx_epi_e2,    rowbuf);
    JX_EPI_RUN("[e3] 去掉 swc/sbb 读",      jx_epi_e3,    rowbuf);
    JX_EPI_RUN("[e4] 2 路",                 jx_epi_e2w,   rowbuf);
    JX_EPI_RUN("[e5] 8 路",                 jx_epi_e8w,   rowbuf);
    JX_EPI_RUN("[e6] 单路",                 jx_epi_e1w,   rowbuf);
    JX_EPI_RUN("[e7] 4 路 + 参数合并",      jx_epi_epack, rowbuf);
    JX_EPI_RUN("[e8] 现状 4 路, 写 PSRAM",  jx_epi_e0,    bigout);
#undef JX_EPI_RUN
  }
  free(accb); free(swc); free(sbb); free(sna); free(sni);
  free(pack); free(rowbuf); free(bigout); free(ssr);
}

/* ============================================================================
 * r82：FPU 吞吐 / 延迟判定微基准
 *
 * 问题：所有"浮点逐元素"热循环都稳定在 1.6~4 周期/条浮点指令上。
 *       到底是 (a) 吞吐受限 —— FPU 每 N 周期才能吃完一条，ILP 再高也没用；
 *       还是 (b) 延迟受限 —— 单条延迟 L 周期，只要独立链够多就能接近 1 条/周期？
 *       这个判定决定后面所有算子的改法：
 *         (a) 成立  -> 只能减少浮点指令条数（换整数/查表/SIMD），ILP 无用
 *         (b) 成立  -> 提高 ILP（寄存器够的话）就能白拿收益
 *
 * 手法：同一指令、不同独立链数 nch，指令用 inline asm 固定，避免 GCC
 *       重排/常量折叠。读出 cycles /（rep * nch）= 该指令的有效周期/条。
 *       nch=1 时读数 = max(延迟, 循环开销)；nch=6/8 时 = max(吞吐*nch, 延迟)。
 * ========================================================================== */
#define JXFP_MADD(a, b, c) __asm__ __volatile__("madd.s %0, %1, %2" : "+f"(a) : "f"(b), "f"(c))
#define JXFP_MUL(a, b)     __asm__ __volatile__("mul.s %0, %0, %1"  : "+f"(a) : "f"(b))
#define JXFP_ADD(a, b)     __asm__ __volatile__("add.s %0, %0, %1"  : "+f"(a) : "f"(b))

static volatile float g_fp_sink;

static float jx_fptp_madd1(long rep)
{
  float a = 1.0f, s = 1e-9f, k = 1.0000001f;
  for (long r = 0; r < rep; r++) { JXFP_MADD(a, s, k); }
  return a;
}

static float jx_fptp_madd6(long rep)
{
  float a0 = 1.0f, a1 = 1.0f, a2 = 1.0f, a3 = 1.0f, a4 = 1.0f, a5 = 1.0f;
  float s0 = 1e-9f, s1 = 2e-9f, s2 = 3e-9f, s3 = 4e-9f, s4 = 5e-9f, s5 = 6e-9f;
  float k = 1.0000001f;
  for (long r = 0; r < rep; r++)
    {
      JXFP_MADD(a0, s0, k); JXFP_MADD(a1, s1, k); JXFP_MADD(a2, s2, k);
      JXFP_MADD(a3, s3, k); JXFP_MADD(a4, s4, k); JXFP_MADD(a5, s5, k);
    }
  return a0 + a1 + a2 + a3 + a4 + a5;
}

static float jx_fptp_mul1(long rep)
{
  float a = 1.0000001f, k = 1.0000001f;
  for (long r = 0; r < rep; r++) { JXFP_MUL(a, k); }
  return a;
}

static float jx_fptp_mul8(long rep)
{
  float a0 = 1.0000001f, a1 = 1.0000002f, a2 = 1.0000003f, a3 = 1.0000004f;
  float a4 = 1.0000005f, a5 = 1.0000006f, a6 = 1.0000007f, a7 = 1.0000008f;
  float k = 1.0000001f;
  for (long r = 0; r < rep; r++)
    {
      JXFP_MUL(a0, k); JXFP_MUL(a1, k); JXFP_MUL(a2, k); JXFP_MUL(a3, k);
      JXFP_MUL(a4, k); JXFP_MUL(a5, k); JXFP_MUL(a6, k); JXFP_MUL(a7, k);
    }
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

static float jx_fptp_add1(long rep)
{
  float a = 1.0f, k = 1e-9f;
  for (long r = 0; r < rep; r++) { JXFP_ADD(a, k); }
  return a;
}

static float jx_fptp_add8(long rep)
{
  float a0 = 1.0f, a1 = 1.0f, a2 = 1.0f, a3 = 1.0f;
  float a4 = 1.0f, a5 = 1.0f, a6 = 1.0f, a7 = 1.0f;
  float k = 1e-9f;
  for (long r = 0; r < rep; r++)
    {
      JXFP_ADD(a0, k); JXFP_ADD(a1, k); JXFP_ADD(a2, k); JXFP_ADD(a3, k);
      JXFP_ADD(a4, k); JXFP_ADD(a5, k); JXFP_ADD(a6, k); JXFP_ADD(a7, k);
    }
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

static float jx_fptp_nop(long rep)
{
  float a = 1.0f;
  for (long r = 0; r < rep; r++) { __asm__ __volatile__("" : "+f"(a)); }
  return a;
}

void jx_fptp_probe(void)
{
  const long REP = 400000;
  unsigned c0, c1;
  float s;

  printf("==== r82 FPU 吞吐/延迟判定（每读数 = 周期/条）====\n");

  c0 = jx_sp_cc(); s = jx_fptp_nop(REP); c1 = jx_sp_cc();
  double nop = (double)(c1 - c0) / (double)REP;
  g_fp_sink = s;
  printf("  [f0] 空循环开销        : %6.2f 周期/次\n", nop);

  c0 = jx_sp_cc(); s = jx_fptp_madd1(REP); c1 = jx_sp_cc();
  printf("  [f1] madd.s  1 链      : %6.2f 周期/条\n", (double)(c1 - c0) / (double)REP);
  g_fp_sink = s;

  c0 = jx_sp_cc(); s = jx_fptp_madd6(REP); c1 = jx_sp_cc();
  printf("  [f2] madd.s  6 链      : %6.2f 周期/条\n", (double)(c1 - c0) / (double)(REP * 6));
  g_fp_sink = s;

  c0 = jx_sp_cc(); s = jx_fptp_mul1(REP); c1 = jx_sp_cc();
  printf("  [f3] mul.s   1 链      : %6.2f 周期/条\n", (double)(c1 - c0) / (double)REP);
  g_fp_sink = s;

  c0 = jx_sp_cc(); s = jx_fptp_mul8(REP); c1 = jx_sp_cc();
  printf("  [f4] mul.s   8 链      : %6.2f 周期/条\n", (double)(c1 - c0) / (double)(REP * 8));
  g_fp_sink = s;

  c0 = jx_sp_cc(); s = jx_fptp_add1(REP); c1 = jx_sp_cc();
  printf("  [f5] add.s   1 链      : %6.2f 周期/条\n", (double)(c1 - c0) / (double)REP);
  g_fp_sink = s;

  c0 = jx_sp_cc(); s = jx_fptp_add8(REP); c1 = jx_sp_cc();
  printf("  [f6] add.s   8 链      : %6.2f 周期/条\n", (double)(c1 - c0) / (double)(REP * 8));
  g_fp_sink = s;

  printf("==== 判读：nch=1 是延迟，nch=6/8 是吞吐 ====\n");
}

/* ============================================================================
 * r83：把循环开销摊薄的版本（每轮 32 条运算，循环开销只占 1/32）
 * r82 的读数里每轮只有 6~8 条运算，而"空循环"本身要 4.14 周期/次，
 * 无法区分「FPU 吞吐」和「循环开销」。这里每轮塞 32 条，开销被摊到 ~0.13/条。
 * ========================================================================== */
#define JXINT_ADD(a, b) __asm__ __volatile__("add %0, %0, %1" : "+r"(a) : "r"(b))

static float jx_fptp2_nop(long rep)
{
  float a0=1.f,a1=1.f,a2=1.f,a3=1.f,a4=1.f,a5=1.f,a6=1.f,a7=1.f;
  for (long r = 0; r < rep; r++)
    {
      for (int q = 0; q < 4; q++)
        {
          __asm__ __volatile__("" : "+f"(a0)); __asm__ __volatile__("" : "+f"(a1));
          __asm__ __volatile__("" : "+f"(a2)); __asm__ __volatile__("" : "+f"(a3));
          __asm__ __volatile__("" : "+f"(a4)); __asm__ __volatile__("" : "+f"(a5));
          __asm__ __volatile__("" : "+f"(a6)); __asm__ __volatile__("" : "+f"(a7));
        }
    }
  return a0+a1+a2+a3+a4+a5+a6+a7;
}

static int jx_fptp2_iadd(long rep)
{
  int a0=0,a1=1,a2=2,a3=3,a4=4,a5=5,a6=6,a7=7; int k = 1;
  for (long r = 0; r < rep; r++)
    {
      for (int q = 0; q < 4; q++)
        {
          JXINT_ADD(a0,k); JXINT_ADD(a1,k); JXINT_ADD(a2,k); JXINT_ADD(a3,k);
          JXINT_ADD(a4,k); JXINT_ADD(a5,k); JXINT_ADD(a6,k); JXINT_ADD(a7,k);
        }
    }
  return a0+a1+a2+a3+a4+a5+a6+a7;
}

static float jx_fptp2_madd1(long rep)
{
  float a = 1.0f, s = 1e-9f, k = 1.0000001f;
  for (long r = 0; r < rep; r++)
    {
      for (int q = 0; q < 8; q++) { JXFP_MADD(a, s, k); }
    }
  return a;
}

static float jx_fptp2_madd8(long rep)
{
  float a0=1.f,a1=1.f,a2=1.f,a3=1.f,a4=1.f,a5=1.f,a6=1.f,a7=1.f;
  float s0=1e-9f,s1=2e-9f,s2=3e-9f,s3=4e-9f; float k = 1.0000001f;
  for (long r = 0; r < rep; r++)
    {
      for (int q = 0; q < 4; q++)
        {
          JXFP_MADD(a0,s0,k); JXFP_MADD(a1,s1,k); JXFP_MADD(a2,s2,k); JXFP_MADD(a3,s3,k);
          JXFP_MADD(a4,s0,k); JXFP_MADD(a5,s1,k); JXFP_MADD(a6,s2,k); JXFP_MADD(a7,s3,k);
        }
    }
  return a0+a1+a2+a3+a4+a5+a6+a7;
}

static float jx_fptp2_mul8(long rep)
{
  float a0=1.0000001f,a1=1.0000002f,a2=1.0000003f,a3=1.0000004f;
  float a4=1.0000005f,a5=1.0000006f,a6=1.0000007f,a7=1.0000008f;
  float k = 1.0000001f;
  for (long r = 0; r < rep; r++)
    {
      for (int q = 0; q < 4; q++)
        {
          JXFP_MUL(a0,k); JXFP_MUL(a1,k); JXFP_MUL(a2,k); JXFP_MUL(a3,k);
          JXFP_MUL(a4,k); JXFP_MUL(a5,k); JXFP_MUL(a6,k); JXFP_MUL(a7,k);
        }
    }
  return a0+a1+a2+a3+a4+a5+a6+a7;
}

static float jx_fptp2_add8(long rep)
{
  float a0=1.0f,a1=1.0f,a2=1.0f,a3=1.0f,a4=1.0f,a5=1.0f,a6=1.0f,a7=1.0f;
  float k = 1e-9f;
  for (long r = 0; r < rep; r++)
    {
      for (int q = 0; q < 4; q++)
        {
          JXFP_ADD(a0,k); JXFP_ADD(a1,k); JXFP_ADD(a2,k); JXFP_ADD(a3,k);
          JXFP_ADD(a4,k); JXFP_ADD(a5,k); JXFP_ADD(a6,k); JXFP_ADD(a7,k);
        }
    }
  return a0+a1+a2+a3+a4+a5+a6+a7;
}

/* 混合：4 madd + 4 mul + 4 add（模拟真实 epilogue 的指令配比） */
static float jx_fptp2_mix(long rep)
{
  float a0=1.f,a1=1.f,a2=1.f,a3=1.f,a4=1.0000001f,a5=1.0000001f,a6=1.0000001f,a7=1.0000001f;
  float s0=1e-9f,s1=2e-9f,s2=3e-9f,s3=4e-9f; float k = 1.0000001f;
  for (long r = 0; r < rep; r++)
    {
      for (int q = 0; q < 4; q++)
        {
          JXFP_MADD(a0,s0,k); JXFP_MADD(a1,s1,k); JXFP_MADD(a2,s2,k); JXFP_MADD(a3,s3,k);
          JXFP_MUL(a4,k);     JXFP_MUL(a5,k);     JXFP_MUL(a6,k);     JXFP_MUL(a7,k);
          JXFP_ADD(a0,k);     JXFP_ADD(a1,k);     JXFP_ADD(a2,k);     JXFP_ADD(a3,k);
        }
    }
  return a0+a1+a2+a3+a4+a5+a6+a7;
}

void jx_fptp2_probe(void)
{
  const long REP = 200000;
  unsigned c0, c1; float s; int si;

  printf("==== r83 FPU/整数 吞吐（每轮 32 条，开销已摊薄）====\n");

  c0 = jx_sp_cc(); s = jx_fptp2_nop(REP); c1 = jx_sp_cc();
  g_fp_sink = s;
  printf("  [g0] nop      x32/轮 : %6.3f 周期/条\n", (double)(c1-c0)/(double)(REP*32));

  c0 = jx_sp_cc(); si = jx_fptp2_iadd(REP); c1 = jx_sp_cc();
  printf("  [g1] int add  x32/轮 : %6.3f 周期/条\n", (double)(c1-c0)/(double)(REP*32));
  g_fp_sink = (float)si;

  c0 = jx_sp_cc(); s = jx_fptp2_madd1(REP); c1 = jx_sp_cc();
  g_fp_sink = s;
  printf("  [g2] madd.s  1 链 x8 : %6.3f 周期/条（= 链延迟）\n", (double)(c1-c0)/(double)(REP*8));

  c0 = jx_sp_cc(); s = jx_fptp2_madd8(REP); c1 = jx_sp_cc();
  g_fp_sink = s;
  printf("  [g3] madd.s  8 链 x32: %6.3f 周期/条（= 吞吐）\n", (double)(c1-c0)/(double)(REP*32));

  c0 = jx_sp_cc(); s = jx_fptp2_mul8(REP); c1 = jx_sp_cc();
  g_fp_sink = s;
  printf("  [g4] mul.s   8 链 x32: %6.3f 周期/条\n", (double)(c1-c0)/(double)(REP*32));

  c0 = jx_sp_cc(); s = jx_fptp2_add8(REP); c1 = jx_sp_cc();
  g_fp_sink = s;
  printf("  [g5] add.s   8 链 x32: %6.3f 周期/条\n", (double)(c1-c0)/(double)(REP*32));

  c0 = jx_sp_cc(); s = jx_fptp2_mix(REP); c1 = jx_sp_cc();
  g_fp_sink = s;
  printf("  [g6] 4madd+4mul+4add : %6.3f 周期/条\n", (double)(c1-c0)/(double)(REP*48));

  printf("==== 判读：整数 1.0 是单发射地板；FP 若 >1.6 说明 FPU 是慢吞吐单元 ====\n");
}

/* ============================================================================
 * r84：存储层次探测 —— PSRAM/D-cache/内部 SRAM 到底值多少周期
 *
 * 背景：r83 已证明 FPU 是 1.03 周期/条全流水、整数 1.0 周期/条，
 *       也就是「时间 ≈ 指令条数」。可是现场热循环普遍是 2~3 周期/指令，
 *       说明真正吃掉时间的是**访存停顿**。
 * 本探针直接量：
 *   [m1] 8KB 堆数组 第 1 遍（冷）
 *   [m2] 8KB 堆数组 第 2 遍起（D-cache 命中？）
 *   [m3] 4MB 堆数组 顺序扫（流式带宽）
 *   [m4] 栈上数组反复扫（内部 SRAM 参照）
 *   [m5] 8KB 堆数组按 4KB 跨步访问（组冲突）
 *   [m6] 4MB 堆数组 纯顺序写
 * ========================================================================== */
static double jx_mb_cyc_per_el(float *p, size_t n)
{
  unsigned c0 = jx_sp_cc();
  float acc = 0.0f;
  for (size_t i = 0; i < n; i++) { acc += p[i]; }
  unsigned c1 = jx_sp_cc();
  g_fp_sink = acc;
  return (double)(c1 - c0) / (double)n;
}

static double jx_mb_cyc_per_el_w(float *p, size_t n, float v)
{
  unsigned c0 = jx_sp_cc();
  for (size_t i = 0; i < n; i++) { p[i] = v; }
  unsigned c1 = jx_sp_cc();
  return (double)(c1 - c0) / (double)n;
}

void jx_cache_probe(void)
{
  const size_t NS = 2048;              /* 8 KB */
  const size_t NB = 1024u * 1024u;     /* 4 MB */
  float *sm = (float *)malloc(NS * 4);
  float *bg = (float *)malloc(NB * 4);
  float stk[1024];                     /* 4 KB，任务栈在内部 SRAM */
  if (sm == NULL || bg == NULL) { printf("  malloc 失败\n"); if (sm) free(sm); if (bg) free(bg); return; }
  for (size_t i = 0; i < NS; i++) { sm[i] = (float)(i & 255); }
  for (size_t i = 0; i < NB; i++) { bg[i] = 1.0f; }
  for (int i = 0; i < 1024; i++) { stk[i] = (float)(i & 255); }

  printf("==== r84 存储层次（周期/元素，元素=4 字节 float）====\n");
  printf("  堆地址 small=%p  big=%p\n", (void *)sm, (void *)bg);

  { double r = jx_mb_cyc_per_el(sm, NS);
    printf("  [m1] 8KB 堆 第1遍(冷)     : %6.2f\n", r); }
  { double best = 1e9;
    for (int rp = 0; rp < 8; rp++) { double r = jx_mb_cyc_per_el(sm, NS); if (r < best) { best = r; } }
    printf("  [m2] 8KB 堆 重复扫(最热)  : %6.2f   <- 若 >3 说明 D-cache 对 PSRAM 无效\n", best); }
  { double r = jx_mb_cyc_per_el(bg, NB);
    printf("  [m3] 4MB 堆 顺序读        : %6.2f\n", r); }
  { double best = 1e9;
    for (int rp = 0; rp < 8; rp++) { double r = jx_mb_cyc_per_el(stk, 1024); if (r < best) { best = r; } }
    printf("  [m4] 4KB 栈 重复扫        : %6.2f   <- 内部 SRAM 参照\n", best); }
  { unsigned c0 = jx_sp_cc(); float acc = 0.0f;
    for (int rp = 0; rp < 64; rp++) { for (size_t i = 0; i < 512; i++) { acc += sm[i * 8]; } }
    unsigned c1 = jx_sp_cc(); g_fp_sink = acc;
    printf("  [m5] 8KB 堆 4KB 跨步      : %6.2f\n", (double)(c1 - c0) / (double)(64 * 512)); }
  { double r = jx_mb_cyc_per_el_w(bg, NB, 2.0f);
    printf("  [m6] 4MB 堆 顺序写        : %6.2f\n", r); }

  free(sm); free(bg);
  printf("==== 判读：内部 SRAM 约 1；PSRAM 顺序约 2~3 正常，~9 则是带宽/延迟墙 ====\n");
}

/* r86：D-cache 是否对 PSRAM 生效 —— 关键判定。
 * 手法：4MB 的 PSRAM 缓冲，反复只扫它开头 8KB。若 D-cache 生效，
 * 第二次起应该和内部 SRAM 一样快；若每次都要回 PSRAM 取，说明缓存形同虚设。 */
void jx_cache2_probe(void)
{
  const size_t NB = 1024u * 1024u;      /* 4 MB */
  float *bg = (float *)malloc(NB * 4);
  float stk[1024];
  if (bg == NULL) { printf("  malloc 失败\n"); return; }
  for (size_t i = 0; i < NB; i++) { bg[i] = (float)(i & 255); }
  for (int i = 0; i < 1024; i++) { stk[i] = (float)(i & 255); }

  printf("==== r86 缓存判定 ====\n");
  printf("  big=%p\n", (void *)bg);
  {
    /* 第 1 遍：冷 */
    double r = jx_mb_cyc_per_el(bg, 2048);
    printf("  [n1] PSRAM 头 8KB 第1遍(冷)   : %6.2f\n", r);
  }
  {
    double best = 1e9;
    for (int rp = 0; rp < 16; rp++) { double r = jx_mb_cyc_per_el(bg, 2048); if (r < best) { best = r; } }
    printf("  [n2] PSRAM 头 8KB 重复扫(热)  : %6.2f  <- 若 ~= [n4] 说明缓存生效\n", best);
  }
  {
    double r = jx_mb_cyc_per_el(bg, NB);
    printf("  [n3] PSRAM 4MB 顺序读         : %6.2f\n", r);
  }
  {
    double best = 1e9;
    for (int rp = 0; rp < 16; rp++) { double r = jx_mb_cyc_per_el(stk, 1024); if (r < best) { best = r; } }
    printf("  [n4] 内部 SRAM 4KB 重复扫     : %6.2f\n", best);
  }
  {
    /* 32KB 重复扫（超过 L1 一半，看缓存容量） */
    double best = 1e9;
    for (int rp = 0; rp < 8; rp++) { double r = jx_mb_cyc_per_el(bg, 8192); if (r < best) { best = r; } }
    printf("  [n5] PSRAM 头 32KB 重复扫     : %6.2f\n", best);
  }
  free(bg);
  printf("==== 判读：n2 ≈ n4 说明 D-cache 对 PSRAM 生效；n2 ≈ n3 说明完全不生效 ====\n");
}
