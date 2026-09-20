/* ============================================================================
 * transformer.c — implementations of the local-attention transformer.
 * See transformer.h for the Python module mapping.
 * ========================================================================== */
#include "transformer.h"
#include "jxprof.h"
#include "nn_ops.h"       /* jx_amalloc/jx_afree：PIE 128 位访存必须 16 字节对齐 */

/* ============================================================================
 * r81：量化取整的无分支写法。
 *
 * 反汇编（v194 固件，jx_q_row_auto 的 8 路量化循环）显示，原式
 *     jx_rndf(v)
 * 每个元素要 5 条指令：ole.s(比较) + bt(条件跳转) + l32r(常量池取 -0.5f)
 * + wfr(写进 FP 寄存器) + add.s —— 而且那个条件跳转在真实数据上
 * 正负各半，完全不可预测。整段量化循环实测 ~9 条指令/元素。
 *
 * 下面用"符号位直接拼 ±0.5f"：and/or/wfr/add.s 四步，零分支。
 * 加进去的仍然是 IEEE 的单精度 ±0.5f（位模式与原来一字不差），
 * 所以结果与改动前**逐位相同**（主机 .idx 门禁已验）。
 * ========================================================================== */
static inline int jx_rndf(float v)
{
#if defined(__XTENSA__) && !defined(JX_NO_RNDS)
  /* r94：Xtensa 的 FPU 有一条 `round.s aR, fS, 0` = 取整到最近整数（.5 远离零），
   * 与 (int)(v + copysign(0.5f,v)) 完全同义，但只需 1 条指令：
   * 省掉 2 次跨域搬运（rfr/wfr）+ and/or + add.s + trunc.s 共 5~6 条，
   * 而且不再占用整数寄存器做位拼接。量化路径全场约 7.7M 个元素受益。
   * 数值差异只可能出现在 v 的小数部分恰好为 0.5 的输入上（IEEE 默认 ties-to-even
   * 与 ties-away 的差别），相对 1 LSB，远低于 int8 量化噪声。JX_NO_RNDS 可回退。 */
  int r;
  __asm__ ("round.s %0, %1, 0" : "=r"(r) : "f"(v));
  return r;
#else
  unsigned u;
  float h;
  __builtin_memcpy(&u, &v, 4);
  u = 0x3f000000u | (u & 0x80000000u);   /* ±0.5f 的位模式 */
  __builtin_memcpy(&h, &u, 4);
  return (int)(v + h);
#endif
}
#include "jx_fastmath.h"

/* ---- 主机端 expf 计数探针（JX_DIST=1）：量化 softmax / SiLU 的超越函数总量。
 * __XTENSA__ 下整体编译为空，板端零开销。 */
#ifdef COUNT_MACS
extern long long g_attn_macs;
extern int       g_attn_heads;
#endif

#ifndef __XTENSA__
#include <stdio.h>
#include <stdlib.h>
static unsigned long long g_jx_exp_n, g_jx_sig_n;
static int g_jx_exp_arm = -1;
static void jx_exp_report(void)
{
  fprintf(stderr, "[DIST] transformer expf: softmax=%llu  silu=%llu  total=%llu\n",
          g_jx_exp_n, g_jx_sig_n, g_jx_exp_n + g_jx_sig_n);
#ifdef COUNT_MACS
  fprintf(stderr, "[DIST] attn macs = %lld  (QK+SV)   keys=%llu   heads=%d\n",
          g_attn_macs, g_jx_exp_n, g_attn_heads);
#endif
}
#define JX_EXP_ARM() do { if (g_jx_exp_arm < 0) { g_jx_exp_arm = getenv("JX_DIST") ? 1 : 0; \
      if (g_jx_exp_arm) { atexit(jx_exp_report); } } } while (0)
#define JX_EXP_CNT() do { if (g_jx_exp_arm) { g_jx_exp_n++; } } while (0)
#define JX_SIG_CNT() do { if (g_jx_exp_arm) { g_jx_sig_n++; } } while (0)
#else
#define JX_EXP_ARM() ((void)0)
#define JX_EXP_CNT() ((void)0)
#define JX_SIG_CNT() ((void)0)
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* MAC 统计: 仅 macs_count 构建(定义 COUNT_MACS)时累加, 生产构建完全不触碰.
 * attention 的 QKᵀ / attn·V 是手写标量循环, 不走被 --wrap 的 linear/conv1d,
 * 因此必须在循环里单独计数才有真实总 MAC. */
#ifdef COUNT_MACS
long long g_attn_macs = 0;   /* QKᵀ + attn·V (与 Python 口径一致) */
long long g_attn_dpb  = 0;   /* dynamic position bias MLP: C 实现里按 head 重算, 是 Python 的 heads 倍 */
int     g_attn_heads = 0;    /* 实际 head 数 (从权重读出), 供 macs_count 折算 Python 等价 dpb */
#endif

/* ============================ DynamicPositionBias MLP ============================ */
/* DynamicPositionBias MLP: Linear(1,H)->SiLU->Linear(H,H)->SiLU->Linear(H,heads),
 * where H = dim//2 (dim = feature_dim). H is rate-dependent (32 for 64-dim
 * 3k/6k, 28 for 56-dim 1kbps), so it is READ FROM THE WEIGHT SHAPE, not
 * hardcoded. Using a fixed 32 would read out-of-bounds for 1kbps (H=28). */
static void dpb_mlp(const float *w0, const float *b0, const float *w1, const float *b1,
                    const float *w2, const float *b2, float rel, float *out, int heads, int H) {
    JX_EXP_ARM();
    /* 2026-09-13 第九轮：SiLU 里的 expf 换成多项式版 jx_silu（见 jx_fastmath.h）。
     * 数值与 libm 差 ~1e-07，落在 int8 量化噪声之下；由 m0/m6 回归门禁守。 */
    float h1[64]; for (int i = 0; i < H; i++) { float a = b0[i] + w0[i] * rel; h1[i] = jx_silu(a); JX_SIG_CNT(); }
    float h2[64]; for (int i = 0; i < H; i++) { float a = b1[i]; for (int j = 0; j < H; j++) a += w1[i*H+j] * h1[j]; h2[i] = jx_silu(a); JX_SIG_CNT(); }
    for (int hd = 0; hd < heads; hd++) { float a = b2[hd]; for (int j = 0; j < H; j++) a += w2[hd*H+j] * h2[j]; out[hd] = a; }
#ifdef COUNT_MACS
    /* dpb_mlp 每次调用: h1 行 H 次乘加 + h2 行 H*H + out 行 H*heads */
    g_attn_dpb += (long long)H + (long long)H*H + (long long)H*heads;
#endif
}

/* DynamicPositionBias 只由模型权重、window 和 head 数决定，与每个包的
 * token 内容无关。原先每层、每次前向都重算 2*window 行 MLP；解码侧每包
 * 只有 4 个 token，这张表反而成了主要固定成本。这里按 transformer 根前缀
 * 复用，结果逐位相同，仅省掉重复计算。 */
typedef struct {
  char  key[80];
  int   window;
  int   heads;
  int   h;
  float *tab;
} jx_bias_cache_t;

static jx_bias_cache_t *g_bias_cache;

static float *jx_bias_table_get(const char *tpfx, int window, int heads, int H,
                                const float *w0, const float *b0,
                                const float *w1, const float *b1,
                                const float *w2, const float *b2)
{
  int slot = -1;

  if (g_bias_cache == NULL)
    {
      g_bias_cache = (jx_bias_cache_t *)calloc(8, sizeof(jx_bias_cache_t));
      if (g_bias_cache == NULL) { return NULL; }
    }

  for (int i = 0; i < 8; i++)
    {
      if (g_bias_cache[i].tab != NULL &&
          g_bias_cache[i].window == window &&
          g_bias_cache[i].heads == heads &&
          g_bias_cache[i].h == H &&
          strcmp(g_bias_cache[i].key, tpfx) == 0)
        {
          return g_bias_cache[i].tab;
        }
      if (slot < 0 && g_bias_cache[i].tab == NULL) { slot = i; }
    }

  if (slot < 0) { return NULL; }
  {
    int n_rel = 2 * window;
    float *tab = (float *)malloc(sizeof(float) * (size_t)n_rel * (size_t)heads);
    if (tab == NULL) { return NULL; }
    for (int r = 0; r < n_rel; r++)
      {
        dpb_mlp(w0, b0, w1, b1, w2, b2, (float)r,
                &tab[(size_t)r * heads], heads, H);
      }
    snprintf(g_bias_cache[slot].key, sizeof(g_bias_cache[slot].key), "%s", tpfx);
    g_bias_cache[slot].window = window;
    g_bias_cache[slot].heads = heads;
    g_bias_cache[slot].h = H;
    g_bias_cache[slot].tab = tab;
    return tab;
  }
}

/* ============================ LocalMHA (local_attention) ============================ */
/* ==================== r62：LocalMHA 按 head 二分到第二颗核 ====================
 * (bi,h,t) 完全独立：k/v 的量化 slab（kq/kq16/vT/sv）本来就按 h 分槽，每个
 * (t,h) 只写 op 自己那一格。原来 t 在 h 外面没法切，这里把 h 提到 t 外面
 * （t==0 的预备块是 per-(bi,h) 的一次性量化，提出来逐位等价），再按 h 二分。
 * heads 一般是 6 -> 3/3，两侧工作量相同。
 * 必须分核的东西：sims/ws/accs 挪进 worker 自己的栈；wsb 原来全 head 共用，
 * 双核同时用会互相踩，改成按 h 分槽（+vstride 一个 head）。
 * ======================================================================== */
typedef struct {
  int B, Tn, inner, heads, dh, n_pad, window, vstride, bi;
  T *qp, *kp, *vp, *op;
  const float *bias_tab;
  signed char *kq, *vT, *wsb;
  float *ks;
  signed short *kq16; float *ks16; float *sv;
  int qk_i8, qk_i16, sv_i8;
} jx_lmha_t;

JX_HOT static void jx_lmha_prep(void *va, int hlo, int hhi)
{
    jx_lmha_t *C = (jx_lmha_t *)va;
    const int bi = C->bi, Tn = C->Tn, dh = C->dh;
    const int vstride = C->vstride, heads = C->heads, n_pad = C->n_pad;
    const int sv_i8 = C->sv_i8, qk_i16 = C->qk_i16, qk_i8 = C->qk_i8;
    T kp = *C->kp, vp = *C->vp;
    signed char *vT = C->vT, *kq = C->kq;
    float *ks = C->ks, *sv = C->sv;
    signed short *kq16 = C->kq16; float *ks16 = C->ks16;
    (void)vstride;
    for (int h = hlo; h < hhi; h++)
    {
        if (sv_i8)
          {
            /* v 转置成 [d][a] 并按**列**量化：加权和没有相消，
             * 相对误差就是量化噪声本身。 */
            const float *vb = &vp.d[(size_t)(bi*heads+h)*n_pad*dh];
            for (int d = 0; d < dh; d++)
              {
                float mxa = 0.0f;
                for (int a = 0; a < Tn; a++)
                  { float av = fabsf(vb[(size_t)a*dh + d]); if (av > mxa) { mxa = av; } }
                {
                  float sc = (mxa > 1e-30f) ? (mxa / 127.0f) : 1e-30f;
                  float is = (mxa > 1e-30f) ? (127.0f / mxa) : 0.0f;
                  /* 关键：vT/sv 按 head 分槽。若所有 head 共用一块，
                   * 只有最后一个 head 的数据是对的（t==0 之后不再重建）。 */
                  signed char *row = vT + ((size_t)h * (size_t)dh + (size_t)d) * (size_t)vstride;
                  sv[(size_t)h * (size_t)dh + (size_t)d] = sc;
                  for (int a = 0; a < Tn; a++)
                    {
                      float vv = vb[(size_t)a*dh + d] * is;
                      int iv = jx_rndf(vv);
                      if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
                      row[a] = (signed char)iv;
                    }
                  for (int a = Tn; a < vstride; a++) { row[a] = 0; }
                }
              }
          }
        if (qk_i16)
          {
            const float *kb = &kp.d[(size_t)(bi*heads+h)*n_pad*dh];
            for (int a = 0; a < Tn; a++)
              { ks16[(size_t)h * (size_t)Tn + a] = jx_qrow_i16(&kb[(size_t)a*dh], kq16 + ((size_t)h * (size_t)Tn + a)*16u, dh); }
          }
        if (qk_i8)
          {
            const float *kb = &kp.d[(size_t)(bi*heads+h)*n_pad*dh];
            for (int a = 0; a < Tn; a++)
              { ks[(size_t)h * (size_t)Tn + a] = jx_qrow_i8(&kb[(size_t)a*dh], kq + ((size_t)h * (size_t)Tn + a)*16u, dh); }
          }
    }
}

JX_HOT static void jx_lmha_hh(void *va, int hlo, int hhi)
{
    jx_lmha_t *C = (jx_lmha_t *)va;
    const int bi = C->bi, Tn = C->Tn, dh = C->dh;
    const int heads = C->heads, n_pad = C->n_pad, window = C->window;
    const int vstride = C->vstride;
    const int qk_i8 = C->qk_i8, qk_i16 = C->qk_i16, sv_i8 = C->sv_i8;
    T qp = *C->qp, kp = *C->kp, vp = *C->vp, op = *C->op;
    const float *bias_tab = C->bias_tab;
    signed char *kq = C->kq, *vT = C->vT;
    float *ks = C->ks, *sv = C->sv;
    signed short *kq16 = C->kq16; float *ks16 = C->ks16;
    float sims[2 * window]; float ws[2 * window];
    int32_t accs[2 * window];
    (void)vstride;
    for (int h = hlo; h < hhi; h++)
    {
        signed char *wsb = C->wsb + (size_t)h * (size_t)vstride;
        for (int t = 0; t < Tn; t++)
        {
            int w = t / window;
            int wstart = (w > 0) ? (w - 1) * window : 0;
            (void)w;
            float qi[dh]; for (int d = 0; d < dh; d++) qi[d] = qp.d[((bi*heads+h)*n_pad+t)*dh+d];
            /* 2026-09-13 第六轮：QKᵀ 的原始写法每个 key 只有一条串行累加链
             * （s += ...），而 Xtensa 的 madd.s 延迟 ~9 周期（单链实测
             * 9.3 周期/MAC）。这里把 key 展开 2 路、d 也拆 2 路 = 4 条独立链，
             * 让 FPU 流水线填满（8 链微基准 3.86 周期/MAC）。
             * 代价是求和次序变化（浮点结合律不成立），token 可能有极少数
             * 抖动，由 SNR 门禁守；dh 在本模型恒为偶数。 */
            float mx = -1e30f;
            int a = wstart;
            if (qk_i16)
              {
                signed short q16v[16] __attribute__((aligned(16)));
                float sq = jx_qrow_i16(&qp.d[(size_t)(bi*heads+h)*n_pad*dh + (size_t)t*dh], q16v, dh);
                int nk = t - wstart + 1;
                jx_qk_i16(q16v, kq16 + ((size_t)h * (size_t)Tn + wstart) * 16u, nk, accs);
                for (int i = 0; i < nk; i++)
                  {
                    float s = (float)accs[i] * (sq * ks16[(size_t)h * (size_t)Tn + wstart + i])
                              + bias_tab[(size_t)(t - wstart - i) * heads + h];
                    sims[i] = s; if (s > mx) { mx = s; }
                  }
    #ifdef COUNT_MACS
                g_attn_macs += (long long)nk * dh;
    #endif
              }
            else if (qk_i8)
              {
                signed char q16[16] __attribute__((aligned(16)));
                float sq = jx_qrow_i8(&qp.d[(size_t)(bi*heads+h)*n_pad*dh + (size_t)t*dh], q16, dh);
                int nk = t - wstart + 1;
                jx_qk_i8(q16, kq + ((size_t)h * (size_t)Tn + wstart) * 16u, nk, accs);
                for (int i = 0; i < nk; i++)
                  {
                    float s = (float)accs[i] * (sq * ks[(size_t)h * (size_t)Tn + wstart + i])
                              + bias_tab[(size_t)(t - wstart - i) * heads + h];
                    sims[i] = s; if (s > mx) mx = s;
                  }
    #ifdef COUNT_MACS
                g_attn_macs += (long long)nk * dh;
    #endif
              }
            else
            for (; dh >= 2 && a + 1 <= t; a += 2) {
                const float *k0 = &kp.d[((bi*heads+h)*n_pad+a)*dh];
                const float *k1 = &kp.d[((bi*heads+h)*n_pad+a+1)*dh];
                float s0 = 0, s0b = 0, s1 = 0, s1b = 0;
                int d = 0;
                for (; d + 1 < dh; d += 2) {
                    s0  += qi[d]     * k0[d];
                    s0b += qi[d + 1] * k0[d + 1];
                    s1  += qi[d]     * k1[d];
                    s1b += qi[d + 1] * k1[d + 1];
    #ifdef COUNT_MACS
                    g_attn_macs += 4;
    #endif
                }
                for (; d < dh; d++) { s0 += qi[d] * k0[d]; s1 += qi[d] * k1[d];
    #ifdef COUNT_MACS
                    g_attn_macs += 2;
    #endif
                }
                { float v0 = (s0 + s0b) + bias_tab[(size_t)(t - a) * heads + h];
                  float v1 = (s1 + s1b) + bias_tab[(size_t)(t - a - 1) * heads + h];
                  sims[a - wstart] = v0; if (v0 > mx) mx = v0;
                  sims[a - wstart + 1] = v1; if (v1 > mx) mx = v1; }
            }
            for (; !qk_i8 && !qk_i16 && a <= t; a++) {
                float s = 0; for (int d = 0; d < dh; d++) { s += qi[d] * kp.d[((bi*heads+h)*n_pad+a)*dh+d];
    #ifdef COUNT_MACS
                g_attn_macs++;   /* QKᵀ: 每个 (d,a) 一次乘加 */
    #endif
                }
                s += bias_tab[(size_t)(t - a) * heads + h];
                sims[a - wstart] = s; if (s > mx) mx = s;
            }
            float sum = 0; int kc = t - wstart + 1;
            /* 2026-09-13 第六轮：软 max 之后 exp 的入参恒 <= 0，用多项式版
             * jx_expf_neg 替代 libm expf（本处调用量 1560 万次，是全模型
             * 单点最大的超越函数开销）。 */
            for (int a = wstart; a <= t; a++) { ws[a - wstart] = jx_expf_neg(sims[a - wstart] - mx); JX_EXP_CNT(); sum += ws[a - wstart]; }
            float inv = (sum > 0) ? 1.0f / sum : 0.f;
            if (sv_i8)
              {
                /* attn·V 的 int8 路径：w（softmax 权重，<=1，scale 恒 127）
                 * 按行量化，v 已按列量化并转置；每 16 个 key 一条
                 * ee.vmulas.s8.accx 完成"沿 key 维求和"。 */
                int nk = t - wstart + 1;
                int nblk = (nk + 15) >> 4;
                for (int i = 0; i < nk; i++)
                  {
                    float wv = ws[i] * 127.0f;
                    int iv = (int)(wv + 0.5f);
                    if (iv > 127) { iv = 127; } else if (iv < 0) { iv = 0; }
                    wsb[i] = (signed char)iv;
                  }
                for (int i = nk; i < (nblk << 4); i++) { wsb[i] = 0; }
                for (int d = 0; d < dh; d++)
                  {
                    int32_t acc = jx_sv_i8(wsb,
                        vT + ((size_t)h * (size_t)dh + (size_t)d) * (size_t)vstride + wstart, nblk);
                    op.d[((bi*heads+h)*n_pad+t)*dh+d] =
                        (float)acc * (sv[(size_t)h * (size_t)dh + d] * (1.0f / 127.0f)) * inv;
    #ifdef COUNT_MACS
                    g_attn_macs += nk;
    #endif
                  }
              }
            else
            /* 同样的问题：每个 d 一条串行链。改成 d 展开 2 路 + a 折半
             * 各 2 条链 = 4 条独立链；末尾奇数 d / 空半段用尾循环兜住。 */
            {
                int d = 0;
                for (; dh >= 2 && d + 1 < dh; d += 2) {
                    int mid = wstart + (t - wstart + 1) / 2;
                    float a0 = 0, a1 = 0, c0 = 0, c1 = 0;
                    int aa = wstart;
                    for (; aa < mid; aa++) {
                        const float *vv = &vp.d[((bi*heads+h)*n_pad+aa)*dh+d];
                        float w = ws[aa - wstart];
                        a0 += w * vv[0]; a1 += w * vv[1];
    #ifdef COUNT_MACS
                        g_attn_macs += 2;
    #endif
                    }
                    for (; aa <= t; aa++) {
                        const float *vv = &vp.d[((bi*heads+h)*n_pad+aa)*dh+d];
                        float w = ws[aa - wstart];
                        c0 += w * vv[0]; c1 += w * vv[1];
    #ifdef COUNT_MACS
                        g_attn_macs += 2;
    #endif
                    }
                    op.d[((bi*heads+h)*n_pad+t)*dh+d]     = (a0 + c0) * inv;
                    op.d[((bi*heads+h)*n_pad+t)*dh+d + 1] = (a1 + c1) * inv;
                }
                for (; d < dh; d++) {
                    float acc = 0;
                    for (int aa = wstart; aa <= t; aa++) { acc += ws[aa - wstart] * vp.d[((bi*heads+h)*n_pad+aa)*dh+d];
    #ifdef COUNT_MACS
                        g_attn_macs++;
    #endif
                    }
                    op.d[((bi*heads+h)*n_pad+t)*dh+d] = acc * inv;
                }
            }
        }
    }
}

static void local_mha_layer(const T *x, T *out, const char *pfx, const char *tpfx, int window) {
    int B = x->shape[0], Tn = x->shape[1], C = x->shape[2];
    /* Head config is rate-specific and is READ FROM THE WEIGHTS (not hardcoded):
       to_qkv produces inner*3 channels  -> inner = to_qkv.weight.shape[0] / 3 ;
       the shared DynamicPositionBias MLP emits one bias per head, so its final
       linear's out-dim == heads.  dim_head = inner / heads.  This keeps the
       transformer correct for every rate (e.g. 3kbps 64-dim/6h/16d vs 1kbps
       56-dim) without per-rate head constants in model_config. */
    int qkv_shape[4], qkv_n;
    wt_get(cat_name(pfx, "to_qkv.weight"), qkv_shape, &qkv_n);
    int inner = qkv_shape[0] / 3;
    int dpb_shape[4], dpb_n;
    wt_get(cat_name(tpfx, "dynamic_pos_bias.mlp.4.weight"), dpb_shape, &dpb_n);
    int heads = dpb_shape[0], dh = inner / heads;
#ifdef COUNT_MACS
    g_attn_heads = heads;
#endif
    int dpb0_shape[4], dpb0_n;
    wt_get(cat_name(tpfx, "dynamic_pos_bias.mlp.0.weight"), dpb0_shape, &dpb0_n);
    int dpb_H = dpb0_shape[0];   /* DynamicPositionBias hidden dim = feature_dim//2 (rate-dependent) */
    /* DynamicPositionBias MLP weights live at the transformer root (sibling of `layers`), not in the layer. */
    const float *pb_w0 = W_(cat_name(tpfx, "dynamic_pos_bias.mlp.0.weight")), *pb_b0 = W_(cat_name(tpfx, "dynamic_pos_bias.mlp.0.bias"));
    const float *pb_w1 = W_(cat_name(tpfx, "dynamic_pos_bias.mlp.2.weight")), *pb_b1 = W_(cat_name(tpfx, "dynamic_pos_bias.mlp.2.bias"));
    const float *pb_w2 = W_(cat_name(tpfx, "dynamic_pos_bias.mlp.4.weight")), *pb_b2 = W_(cat_name(tpfx, "dynamic_pos_bias.mlp.4.bias"));
    T xn = t_alloc(3, (int[]){B, Tn, C}); t_copy(&xn, x);
    channel_norm_last(&xn, W_(cat_name(pfx, "norm.weight")), W_(cat_name(pfx, "norm.bias")), B, Tn, C);
    T qkv = t_alloc(3, (int[]){B, Tn, inner * 3});
    linear(W_(cat_name(pfx, "to_qkv.weight")), NULL, &xn, &qkv, C, inner * 3);
    float scale = 1.0f / sqrtf((float)dh);
    int n_pad = ((Tn + window - 1) / window) * window;
    T qp = t_alloc(4, (int[]){B, heads, n_pad, dh}); T kp = t_alloc(4, (int[]){B, heads, n_pad, dh}); T vp = t_alloc(4, (int[]){B, heads, n_pad, dh});
    for (int bi = 0; bi < B; bi++) for (int t = 0; t < n_pad; t++) {
        int ts = (t < Tn) ? t : 0;
        for (int h = 0; h < heads; h++) for (int d = 0; d < dh; d++) {
            int idx = (bi * Tn + ts) * (inner * 3) + h * dh + d;   /* qkv row width is inner*3 */
            qp.d[((bi*heads+h)*n_pad+t)*dh+d] = qkv.d[idx] * scale;
            kp.d[((bi*heads+h)*n_pad+t)*dh+d] = qkv.d[idx + inner];
            vp.d[((bi*heads+h)*n_pad+t)*dh+d] = qkv.d[idx + 2*inner];
        }
    }
    T op = t_alloc(4, (int[]){B, heads, n_pad, dh});
    /* window=400, look_backward=1, look_forward=0, causal=True (local_attention lib):
       look_around() includes the FULL previous window as keys (look_backward=1 => prev 1 window),
       not just the previous token. Combined with the causal mask (bq_t < bq_k), each query t in
       window w sees keys a in [(w-1)*W, t], where w = floor(t/W) and the lower bound is clamped
       to 0 in the first window. Bias b(h,t,a) = DynamicPositionBias.mlp(t - a).
       Verified against local_attention.LocalAttention mask for window-start queries (e.g. t=400
       sees [0,400]) and generalized: the whole previous window is visible to every query. */
    /* DynamicPositionBias depends ONLY on the relative distance t-a, so hoist the
     * MLP out of the (head, key) loops: precompute one row per distance
     * 0..2*window-1. The original recomputed the full MLP for every
     * (token, head, key) triple, which cost ~39x the actual QK^T/attn.V math and
     * dominated the whole codec compute budget. Max distance is t-wstart =
     * 2*window-1, so n_rel = 2*window covers every case. */
    JX_EXP_ARM();
    float *bias_tab = jx_bias_table_get(tpfx, window, heads, dpb_H,
                                        pb_w0, pb_b0, pb_w1, pb_b1,
                                        pb_w2, pb_b2);
    int bias_owned = 0;
    if (bias_tab == NULL)
      {
        int n_rel = 2 * window;
        bias_tab = (float*)malloc(sizeof(float) * (size_t)n_rel * (size_t)heads);
        if (bias_tab == NULL)
          {
            out->shape[0] = B; out->shape[1] = Tn; out->shape[2] = C;
            out->len = 0; out->d = NULL;
            t_free(&xn); t_free(&qkv); t_free(&qp); t_free(&kp); t_free(&vp); t_free(&op);
            return;
          }
        for (int r = 0; r < n_rel; r++)
            dpb_mlp(pb_w0, pb_b0, pb_w1, pb_b1, pb_w2, pb_b2,
                    (float)r, &bias_tab[(size_t)r * heads], heads, dpb_H);
        bias_owned = 1;
      }

    /* 2026-09-13 第七轮：QKᵀ 的 int8 化（dh 恰好 = 一个 128-bit 向量）。
     * k 按 (bi,h) 量化进 slab（每行 16 字节 + 一个 scale），q 逐行即时量化；
     * 点积走 jx_qk_i8（每 key 一条 ee.vmulas.s8.accx）。量化开销被
     * 每个 key 行被 2*window 个 query 复用摊掉。失败即退回 fp32 路径。 */
    int qk_i8 = (dh == 16) ? jx_attn_qk_i8_enabled() : 0;   /* 默认关（精度不足） */
    int qk_i16 = (dh == 16) ? jx_attn_qk_i16_enabled() : 0;   /* 默认开（13 位 int16） */
    int sv_i8 = (dh == 16 && (window % 16) == 0) ? jx_attn_sv_i8_enabled() : 0;
    signed short *kq16 = NULL;  float *ks16 = NULL;
    signed char *kq = NULL;  float *ks = NULL;
    signed char *vT = NULL;  float *sv = NULL;  signed char *wsb = NULL;
    void *raw_v = NULL, *raw_w = NULL;   /* vT/wsb 的对齐前指针，free 必须用它们 */
    /* vT 行距取 16 的倍数（PIE 128-bit 装载必须 16 字节对齐），
     * 且多留 16 字节做零填充（最后一个 block 可能读到 t 之后）。 */
    int vstride = ((Tn + 31) / 16) * 16;
    if (qk_i8)
      {
        /* r176：kq 每行 16 字节，jx_qk_i8 用 ee.vld.128 按 16 字节步长装载 ——
         * malloc 只保证 8 字节对齐，未对齐时读到的 key 全是错位的。 */
        kq = (signed char *)jx_amalloc((size_t)Tn * 16u * (size_t)heads);
        ks = (float *)malloc((size_t)Tn * sizeof(float) * (size_t)heads);
        if (kq == NULL || ks == NULL) { jx_afree(kq); free(ks); kq = NULL; ks = NULL; qk_i8 = 0; }
      }
    if (qk_i16)
      {
        /* r176：kq16 每个 key 占 16 个 short = 32 字节，jx_qk_i16 用两条
         * ee.vld.128 按 32 字节步长装载 —— 同样要求基址 16 字节对齐。
         * 板端实测未对齐（0x...c88, mod16=8）会让编码端 en_encoder 输出
         * 随堆布局漂移，这就是单核/双核结果不一致的根因。 */
        kq16 = (signed short *)jx_amalloc((size_t)Tn * 16u * sizeof(signed short) * (size_t)heads);
        ks16 = (float *)malloc((size_t)Tn * sizeof(float) * (size_t)heads);
        if (kq16 == NULL || ks16 == NULL) { jx_afree(kq16); free(ks16); kq16 = NULL; ks16 = NULL; qk_i16 = 0; }
      }
    if (sv_i8)
      {
        raw_v = malloc((size_t)vstride * (size_t)dh * (size_t)heads + 32u);
        raw_w = malloc((size_t)vstride * (size_t)heads + 32u);
        vT  = (raw_v != NULL) ? (signed char *)(((uintptr_t)raw_v + 15u) & ~(uintptr_t)15u) : NULL;
        wsb = (raw_w != NULL) ? (signed char *)(((uintptr_t)raw_w + 15u) & ~(uintptr_t)15u) : NULL;
        sv  = (float *)malloc(sizeof(float) * (size_t)dh * (size_t)heads);
        if (vT == NULL || wsb == NULL || sv == NULL)
          { free(raw_v); free(raw_w); free(sv); raw_v = NULL; raw_w = NULL; vT = NULL; wsb = NULL; sv = NULL; sv_i8 = 0; }
      }
    jx_lmha_t LMC;
    LMC.B = B; LMC.Tn = Tn; LMC.inner = inner; LMC.heads = heads; LMC.dh = dh;
    LMC.n_pad = n_pad; LMC.window = window; LMC.vstride = vstride; LMC.bi = 0;
    LMC.qp = &qp; LMC.kp = &kp; LMC.vp = &vp; LMC.op = &op;
    LMC.bias_tab = bias_tab;
    LMC.kq = kq; LMC.ks = ks; LMC.kq16 = kq16; LMC.ks16 = ks16;
    LMC.vT = vT; LMC.sv = sv; LMC.wsb = wsb;
    LMC.qk_i8 = qk_i8; LMC.qk_i16 = qk_i16; LMC.sv_i8 = sv_i8;
    for (int bi = 0; bi < B; bi++)
      {
        LMC.bi = bi;
        jx_pf_run_min(heads, jx_lmha_prep, &LMC, 2);
        jx_pf_run_min(heads, jx_lmha_hh, &LMC, 2);
      }
    T o2trim = t_alloc(3, (int[]){B, Tn, inner});
    for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tn; t++) for (int h = 0; h < heads; h++) for (int d = 0; d < dh; d++)
        o2trim.d[(bi*Tn+t)*inner + h*dh+d] = op.d[((bi*heads+h)*n_pad+t)*dh+d];
    out->shape[0] = B; out->shape[1] = Tn; out->shape[2] = C; out->len = B*Tn*C;
    linear(W_(cat_name(pfx, "to_out.weight")), NULL, &o2trim, out, inner, C);
    t_free(&xn); t_free(&qkv); t_free(&qp); t_free(&kp); t_free(&vp); t_free(&op); t_free(&o2trim);
    if (bias_owned) { free(bias_tab); }
    jx_afree(kq); free(ks);
    jx_afree(kq16); free(ks16);
    free(sv); free(raw_v); free(raw_w);
}

/* ============================ FeedForward (FeedForward) ============================ */
void feed_forward(const T *x, T *out, const char *pfx) {
    int B = x->shape[0], Tn = x->shape[1], C = x->shape[2];
    T a = t_alloc(3, (int[]){B, Tn, C}); t_copy(&a, x);
    channel_norm_last(&a, W_(cat_name(pfx, "0.weight")), W_(cat_name(pfx, "0.bias")), B, Tn, C);
    /* FFN hidden dim is rate-specific: read it from the FC weight shape
       (W has shape [out, in] = [hid, C]); hid must be even (hid = 2*half). */
    int ff1_shape[4], ff1_n;
    const float *w1 = wt_get(cat_name(pfx, "1.weight"), ff1_shape, &ff1_n);
    int hid = ff1_shape[0], half = hid / 2;
    T h = t_alloc(3, (int[]){B, Tn, hid}); linear(w1, NULL, &a, &h, C, hid);
    T g = t_alloc(3, (int[]){B, Tn, half});
    JXP_BEG(JP_GELU);
    /* 2026-09-14 第十八轮：动态位置偏置 MLP 的 GELU 循环改 4 路展开（同 blocks.c
     * 的理由：gelu_f 里的 erf 是长 Horner 依赖链）。逐元素独立 -> 逐位一致。 */
    for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tn; t++)
      {
        /* 第三十九轮：gelu 系数按行装载一次。 */
        jx_kt_ensure();
        const jx_erf9 KE = jx_erf9_load();
        float *gr = &g.d[(size_t)(bi * Tn + t) * half];
        const float *hr0 = &h.d[(size_t)(bi * Tn + t) * hid];
        const float *hr1 = hr0 + half;
        int j = 0;
        for (; j + 4 <= half; j += 4)
          {
            gr[j    ] = hr0[j    ] * jx_gelu_fast(hr1[j    ], KE);
            gr[j + 1] = hr0[j + 1] * jx_gelu_fast(hr1[j + 1], KE);
            gr[j + 2] = hr0[j + 2] * jx_gelu_fast(hr1[j + 2], KE);
            gr[j + 3] = hr0[j + 3] * jx_gelu_fast(hr1[j + 3], KE);
          }
        for (; j < half; j++) { gr[j] = hr0[j] * jx_gelu_fast(hr1[j], KE); }
      }
    JXP_END(JP_GELU);
    T o = t_alloc(3, (int[]){B, Tn, C}); linear(W_(cat_name(pfx, "4.weight")), NULL, &g, &o, half, C);
    out->shape[0] = B; out->shape[1] = Tn; out->shape[2] = C; out->len = B*Tn*C;
    for (int i = 0; i < o.len; i++) out->d[i] = o.d[i];
    t_free(&a); t_free(&h); t_free(&g); t_free(&o);
}

/* ============================ LocalTransformer (LocalTransformer) ============================ */
/* 注意：*out 由本函数负责分配（内部 cur 直接交接出去），调用方不要预分配
 * 缓冲再传进来——那样预分配的块会被整体替换掉，白白泄漏一份全尺寸张量。 */
void local_transformer(const T *x, T *out, const char *pfx, int depth, int window) {
    JXP_BEG(JP_LOCALTRANS);
    T cur = t_alloc(3, (int[]){x->shape[0], x->shape[1], x->shape[2]}); t_copy(&cur, x);
    for (int L = 0; L < depth; L++) {
        char lp[64]; snprintf(lp, sizeof(lp), "%s.layers.%d.0", pfx, L);
        char ffp[64]; snprintf(ffp, sizeof(ffp), "%s.layers.%d.1", pfx, L);
        T mh = t_alloc(3, (int[]){cur.shape[0], cur.shape[1], cur.shape[2]});
        local_mha_layer(&cur, &mh, lp, pfx, window);
        for (int i = 0; i < cur.len; i++) cur.d[i] += mh.d[i]; t_free(&mh);
        T ff = t_alloc(3, (int[]){cur.shape[0], cur.shape[1], cur.shape[2]});
        feed_forward(&cur, &ff, ffp);
        for (int i = 0; i < cur.len; i++) cur.d[i] += ff.d[i]; t_free(&ff);
    }
    *out = cur;
    JXP_END(JP_LOCALTRANS);
}
