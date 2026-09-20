/* ============================================================================
 * blocks.c — implementations of l3ac/modules.py building blocks.
 * See blocks.h for the Python class mapping.
 * ========================================================================== */
#include "blocks.h"
extern void jx_stage_h(const char *name, const void *p, size_t nbytes);   /* r166 诊断 */

#include "jxprof.h"
#include "jx_percore.h"
#include "jx_dcp.h"        /* DCache 硬件预载 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* 转置 rows x cols -> cols x rows（16x16 分块）。
 *
 * 2026-09-13 第14轮：原来是一维逐元素搬，写侧步长 = rows*4 字节
 * （conv_unit 的 Tt=2117 -> 8468B），每次写都落在不同的 cache line，
 * 且 rows*4/32 对 4-way 的 set 数取模后集中撞在同一小组里，PSRAM
 * 伪命中把一次纯搬运拖到 310+ ms。分块后读写各自只在 16 行内活动。
 *
 * 每个目标元素只写一次、写入的值与原来完全相同（只改循环顺序），
 * 因此结果逐位一致。 */
static int jx_tk(void)
{
  const char *e = getenv("JX_TK");
  int v = (e != NULL) ? atoi(e) : 64;
  if (v < 1) { v = 1; }
  if (v > 256) { v = 256; }
  return v;
}

static void jx_transpose_ct(const float *src, float *dst, int rows, int cols) {
    const int TK = jx_tk();
    for (int r0 = 0; r0 < rows; r0 += TK) {
        int r1 = r0 + TK; if (r1 > rows) { r1 = rows; }
        for (int c0 = 0; c0 < cols; c0 += TK) {
            int c1 = c0 + TK; if (c1 > cols) { c1 = cols; }
            for (int r = r0; r < r1; r++) {
                const float *s = src + (size_t)r * (size_t)cols;
                for (int c = c0; c < c1; c++) {
                    dst[(size_t)c * (size_t)rows + r] = s[c];
                }
            }
        }
    }
}


/* ==================== r69：分块拷贝 / 累加的并行化 ====================
 * first_block / conv_unit / conv_unit_add / legacy_unit_add 都用"时间分块 +
 * halo"把一个 [C,T] 张量切成 [C,w] 的小块喂给 core。包在 core 外面的
 * gather / scatter / save / 逐元素累加循环是纯搬运，原来全在单核上跑、
 * 而且完全没有被任何计时桶覆盖。这里按通道切给双核，并把耗时记进 JP_MISC。
 * 每个元素写入的值与原来逐项一致 -> 结果逐位不变。 */
typedef struct {
    const T *src; float *dst; const float *save;
    int C, Tt, bi, a, w, saved_from, H;
    int split_t;   /* r121：C==1 的单通道整段搬运，按时间维切给两核 */
} jx_glue_g_t;

/* 把 x[bi][c][a .. a+w) 收进连续缓冲 dst[c][0 .. w)。saved_from >= 0 时，
 * [saved_from, saved_from+H) 这一段取 save 里的"加之前"的原值。 */
JX_HOT static void jx_glue_gather(void *va, int clo, int chi)
{
    jx_glue_g_t *g = (jx_glue_g_t *)va;
    const int C = g->C, Tt = g->Tt, a = g->a, w = g->w, H = g->H;
    const int sf = g->saved_from;
    const float *base = &g->src->d[(size_t)g->bi * C * Tt];
    /* r121：first_block 的 gather 只有一个通道（C==1），原来只能传 n=1
     * 强制单核串行跑完整段 memcpy。这里允许按**时间维**切分：clo/chi 解释为
     * [a, a+w) 内的偏移区间。搬运的字节与逐元素结果完全相同。 */
    if (g->split_t)
      {
        const float *xd = base;
        float *dst = g->dst;
        for (int t = a + clo; t < a + chi; t++) { dst[t - a] = xd[t]; }
        return;
      }
    for (int c = clo; c < chi; c++)
      {
        if (c + 1 < chi) { jx_dcp_pf_ew(base + (size_t)(c + 1) * Tt + a, (unsigned)w * 4u); }
        const float *xd = base + (size_t)c * Tt;
        float *dst = &g->dst[(size_t)c * w];
        if (sf < 0)
          {
            memcpy(dst, xd + a, (size_t)w * sizeof(float));
          }
        else
          {
            const float *sv = g->save + (size_t)c * H;
            for (int t = a; t < a + w; t++)
              {
                dst[t - a] = (t >= sf && t < sf + H) ? sv[t - sf] : xd[t];
              }
          }
      }
}

typedef struct { float *dst; const float *src; int C, Tt, bi, t0, n, off, stride, add; } jx_glue_s_t;

/* 连续缓冲 src[c][off .. off+n) -> x[bi][c][t0 .. t0+n)，add!=0 时累加。 */
JX_HOT static void jx_glue_scatter(void *va, int clo, int chi)
{
    jx_glue_s_t *g = (jx_glue_s_t *)va;
    const int C = g->C, Tt = g->Tt, t0 = g->t0, n = g->n, off = g->off;
    float *base = &g->dst[(size_t)g->bi * C * Tt];
    for (int c = clo; c < chi; c++)
      {
        if (c + 1 < chi)
          { jx_dcp_pf_ew(&g->src[(size_t)(c + 1) * (size_t)g->stride + off], (unsigned)n * 4u); }
        float *dp = base + (size_t)c * Tt + t0;
        const float *sp = &g->src[(size_t)c * (size_t)g->stride + off];
        if (g->add)
          { for (int t = 0; t < n; t++) { dp[t] += sp[t]; } }
        else
          { memcpy(dp, sp, (size_t)n * sizeof(float)); }
      }
}

typedef struct { const float *src; float *save; int C, Tt, bi, ns, n, H; } jx_glue_v_t;

/* x[bi][c][ns .. ns+n) -> save[c][0 .. n)  （n <= H） */
JX_HOT static void jx_glue_save(void *va, int clo, int chi)
{
    jx_glue_v_t *g = (jx_glue_v_t *)va;
    const float *base = &g->src[(size_t)g->bi * g->C * g->Tt];
    for (int c = clo; c < chi; c++)
      {
        memcpy(g->save + (size_t)c * g->H,
               base + (size_t)c * g->Tt + g->ns, (size_t)g->n * sizeof(float));
      }
}

/* ============================ shared trend extraction ============================ */
/* max_pool1d(|x|, kernel, stride=1, padding) clamped at edges, then
 * avg_pool1d(..., count_include_pad=True). Used by V3FirstBlock and
 * EnhanceBlock to derive the "trend" branch of each module. */
typedef struct { const float *x; float *y; int T_in, p, kernel; } jx_tp_t;

/* r68: split channels across the two cores. Each core owns its own m/dq
 * scratch (the original shared one). Channels are fully independent, so the
 * result is bit-identical to the serial version. */
static void jx_tp_range(void *va, int clo, int chi) {
    jx_tp_t *cc = (jx_tp_t *)va;
    const float *x = cc->x; float *y = cc->y;
    const int T_in = cc->T_in, p = cc->p, kernel = cc->kernel;
    /* r71：原来每次调用都 malloc/free 两块 T_in 大小的 scratch；trend_pool 在
     * 一次编解码里要进 28 次（first_block 5 条分支 x4 批 + enhance），每次都走
     * 一趟堆。改成按核常驻、按需增长。缓冲只在本函数内使用、每次重写，
     * 与原来的行为完全一致。 */
    static float *s_m [JX_NCORE];
    static int   *s_dq[JX_NCORE];
    static int    s_cap[JX_NCORE];
    int core = jx_core_id();
    if (s_cap[core] < T_in)
      {
        float *nm = (float*)realloc(s_m [core], (size_t)T_in * sizeof(float));
        int   *nd = (int  *)realloc(s_dq[core], (size_t)T_in * sizeof(int));
        if (nm == NULL || nd == NULL) { return; }
        s_m[core] = nm; s_dq[core] = nd; s_cap[core] = T_in;
      }
    float *m  = s_m [core];
    int   *dq = s_dq[core];
    for (int c = clo; c < chi; c++) {
        const float *xc = &x[c * T_in]; float *yc = &y[c * T_in];
        /* max_pool1d(|x|, kernel, stride=1, padding=p): 单调队列 O(T)。
         * 边缘 clamp 时重复的边界值不会改变最大值，因此等价于在
         * [max(0,t-p), min(T-1,t+p)] 上求最大值。比较顺序不影响 max 结果。 */
        int head = 0, tail = 0;
        int prev_r = -1;
        for (int t = 0; t < T_in; t++) {
            int l = t - p; if (l < 0) l = 0;
            int r = t + p; if (r >= T_in) r = T_in - 1;
            if (r > prev_r) {
                float av = fabsf(xc[r]);
                while (tail > head) {
                    int idx = dq[tail - 1];
                    if (fabsf(xc[idx]) <= av) { tail--; } else { break; }
                }
                dq[tail++] = r;
                prev_r = r;
            }
            while (head < tail && dq[head] < l) { head++; }
            m[t] = fabsf(xc[dq[head]]);
        }
        /* avg_pool1d(m, kernel, stride=1, padding=p, count_include_pad=True):
         * 滑动和 O(T)。加法次序与原嵌套循环不同，浮点结果会有 1~2 ulp 差异，
         * 但平均池化输出只做趋势项，后接卷积/权重；该误差远小于 int8 量化噪声。 */
        {
          float s = 0.0f;
          for (int t = 0; t < T_in; t++) {
            if (t == 0) {
              int e = p; if (e >= T_in) e = T_in - 1;
              for (int pos = 0; pos <= e; pos++) { s += m[pos]; }
            } else {
              int add = t + p;
              int rem = t - p - 1;
              if (add < T_in) { s += m[add]; }
              if (rem >= 0) { s -= m[rem]; }
            }
            yc[t] = s / (float)kernel;
          }
        }
    }
}

static void trend_pool(const float *x, float *y, int Ci, int T_in, int kernel) {
    if (kernel <= 1) { for (int c = 0; c < Ci; c++) memcpy(&y[c * T_in], &x[c * T_in], (size_t)T_in * sizeof(float)); return; }
    JXP_BEG(JP_TRENDPOOL);
    jx_tp_t cc; cc.x = x; cc.y = y; cc.T_in = T_in; cc.p = kernel / 2; cc.kernel = kernel;
    jx_pf_run_min(Ci, jx_tp_range, &cc, 2);
    JXP_END(JP_TRENDPOOL);
}

/* one trend-pool branch conv (used by both first_block and enhance_block) */
static void first_block_branch(const float *x, float *y, int Ci, int Co, int k, int dil, int pad, int T_in, const char *wname) {
    const float *W = W_(cat_name(wname, "weight")); const float *b = W_(cat_name(wname, "bias"));
    T tmp = {.d=(float*)x, .ndim=3, .shape={1,Ci,T_in}, .len=Ci*T_in};
    /* 2026-09-13 第九轮：原来先把结果写进临时张量再 memcpy 到 y，等于把
     * 每个输出元素多搬一遍（读一次 + 写一次）。conv1d_d 只用到 shape[0..2]
     * 与 len，这里直接在 y 上建视图即可：写入位置与次序完全不变，数值逐位
     * 相同，但省掉整块往返。len 恰好等于 need，不会走 realloc 分支。 */
    T yout = {.d=(float*)y, .ndim=3, .shape={1,Co,T_in}, .len=Co*T_in};
    conv1d_d(W, b, &tmp, &yout, Ci, Co, k, 1, pad, 1, T_in, dil);
}

/* r81：first_block 的 5 条 trend-pool 分支抽成可重入 tb（原来整段单核）。
 * 5 条分支互不相干：各写 h 里自己那 4 行，各自 malloc 一块 trend-pool 临时
 * 缓冲（块内分配，天然不会两核撞同一块）。 */
typedef struct { const float *x; float *h; int B, Tt, each; } jx_fbr_t;
typedef struct { const float *x; float *h; const char *pfx; int Tt; } jx_enh_t;

/* r81：enhance_block 的 4 条分支同样抽 tb（各写 h 里自己那一行）。 */
JX_HOT static void jx_enh_range(void *va, int lo, int hi)
{
  jx_enh_t *c = (jx_enh_t *)va;
  const int ps[4] = {1,3,5,9}; const int ks[4] = {7,7,7,7}; const int dr[4] = {1,2,3,5};
  for (int br = lo; br < hi; br++)
    {
      float *tp = (float *)malloc(sizeof(float) * (size_t)c->Tt);
      if (tp == NULL) { continue; }
      trend_pool(c->x, tp, 1, c->Tt, ps[br]);
      char wn[64]; snprintf(wn, sizeof(wn), "%s.blocks.%d.1", c->pfx, br);
      first_block_branch(tp, &c->h[(size_t)br * c->Tt], 1, 1,
                         ks[br], dr[br], (ks[br]-1)*dr[br]/2, c->Tt, wn);
      free(tp);
    }
}

/* r121：原来每个工作项都 malloc/free 一次 Tt 个 float（Tt=3808 时 15KB），
 * 一个 (bi,br) 组合一次，双核还会同时打 PSRAM 分配器。改成**本核复用缓冲**：
 * 只在需要变大时重新分配，用完不释放。缓冲区内容每次都被 trend_pool 全量
 * 覆写，不依赖初值，结果与逐项分配完全相同。 */
static float *g_fbr_tp[JX_NCORE];
static int    g_fbr_tpn[JX_NCORE];

JX_HOT static void jx_fbr_range(void *va, int lo, int hi)
{
  jx_fbr_t *c = (jx_fbr_t *)va;
  extern unsigned long jx_pb[16];
  unsigned long _pbc;
  { unsigned _t; __asm__ __volatile__("rsr %0, ccount" : "=r"(_t)); _pbc = _t; }
  const int ks[5] = {7,7,7,7,7}; const int ps[5] = {1,5,11,21,45}; const int dr[5] = {1,1,1,1,1};
  const int me = jx_core_id();
  if (g_fbr_tpn[me] < c->Tt)
    {
      float *nd = (float *)realloc(g_fbr_tp[me], sizeof(float) * (size_t)c->Tt);
      if (nd == NULL) { return; }
      g_fbr_tp[me] = nd; g_fbr_tpn[me] = c->Tt;
    }
  float *tp = g_fbr_tp[me];
  if (tp == NULL) { return; }
  for (int u = lo; u < hi; u++)
    {
      int bi = u / 5, br = u - bi * 5;
      float *hd = &c->h[(size_t)bi * 20 * c->Tt];
      trend_pool(&c->x[(size_t)bi * c->Tt], tp, 1, c->Tt, ps[br]);
      jx_pb[8]++;
      { unsigned _t; __asm__ __volatile__("rsr %0, ccount" : "=r"(_t));
        jx_pb[9] += (unsigned long)(unsigned)(_t - (unsigned)_pbc); _pbc = _t; }
      char wn[64]; snprintf(wn, sizeof(wn), "encoder.blocks.0.blocks.%d.1", br);
      first_block_branch(tp, &hd[(size_t)br * c->each * c->Tt], 1, c->each,
                         ks[br], dr[br], (ks[br]-1)*dr[br]/2, c->Tt, wn);
      { unsigned _t; __asm__ __volatile__("rsr %0, ccount" : "=r"(_t));
        jx_pb[10] += (unsigned long)(unsigned)(_t - (unsigned)_pbc); _pbc = _t; }
    }
}


/* r49：把 p1 重建为「l1 + snake」的浮点结果（融合量化路径失效时的退路）。
 * 与原来的 jx_linear_i8_ex(CU 融合) 走同一条路径，数值逐位相同。 */
static void cu_rebuild_p1(const float *w1, const float *b1, const T *c1, T *p1,
                          const float *alp, float *ssr, int rows, int C, int dim4,
                          int B, int Tt) {
    if (ssr != NULL)
      {
        jx_linear_fuse_t fb;
        fb.snake_alpha = alp; fb.ss_rows = ssr;
        fb.grn_gamma = NULL;  fb.grn_beta = NULL; fb.grn_nx = 0.0f;
        fb.in_norm_w = NULL; fb.in_norm_b = NULL;
        fb.q_gamma = NULL; fb.q_beta = NULL;
        fb.out_q = NULL; fb.out_sc = NULL; fb.out_qdone = NULL;
        fb.in_q = NULL; fb.in_sc = NULL;
        if (jx_linear_i8_ex(w1, b1, c1->d, p1->d, rows, C, dim4, &fb) == 0) { return; }
      }
    linear(w1, b1, c1, p1, C, dim4);
    snake1d(p1, alp, B, Tt, dim4, 1);
}


/* ============================================================================
 * r81：逐元素循环的并行切分。
 * 下面这几个循环原来的写法都是"整段跑在调用核上"，而它们写的下标互不相交，
 * 二分之后两核各写各的一半，数值逐位相同（每元素算式一字不改）。
 * 同步走既有的 jx_pf_run_min（JX_PF=0 / 主机回归自动退化为串行）。
 * ========================================================================== */
typedef struct { const float *a; const float *b; float *y; int n; int mode; } jx_ew_t;

JX_HOT static void jx_ew_range(void *va, int lo, int hi)
{
  jx_ew_t *c = (jx_ew_t *)va;
  /* r112：分块 + 跨块预载。元素之间互不相关，结果逐位不变。 */
  if (c->mode == 0)                       /* y = x + m*x */
    {
      for (int i = lo; i < hi; )
        {
          int e = i + 4096; if (e > hi) { e = hi; }
          if (e < hi) { jx_dcp_pf_ew(c->a + e, (unsigned)JX_DCP_TILE); }
          for (; i < e; i++) { float v = c->a[i]; c->y[i] = v + c->b[i] * v; }
        }
    }
  else                                    /* y += a   （y 与 a 不同缓冲） */
    {
      for (int i = lo; i < hi; )
        {
          int e = i + 4096; if (e > hi) { e = hi; }
          if (e < hi) { jx_dcp_pf_ew(c->a + e, (unsigned)JX_DCP_TILE); }
          for (; i < e; i++) { c->y[i] += c->a[i]; }
        }
    }
}

/* ============================ ConvUnit (modules.ConvUnit) ============================ */
static void conv_unit_core(const T *x, T *y, const char *pfx) {
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2]; int dim4 = 4 * C;
    T c1 = t_alloc(3, (int[]){B, Tt, C});
    JXP_BEG(JP_CUDW);
    jx_conv1d_dw_tc(W_(cat_name(pfx, "dw_conv.weight")), W_(cat_name(pfx, "dw_conv.bias")), x, &c1, C, 7, 1, 3, Tt);
    JXP_END(JP_CUDW);
    jx_stage_h("cu.dw", c1.d, (size_t)c1.len * sizeof(float));
    JXP_BEG(JP_CUNORM);
    const float *cnw = W_(cat_name(pfx, "norm.weight"));
    const float *cnb = W_(cat_name(pfx, "norm.bias"));
    const char *cn_env = getenv("JX_CUNORM");
    int norm_fused = (B == 1 && C <= 128 &&
                      (cn_env == NULL || cn_env[0] != (char)48));
    if (!norm_fused) { channel_norm_last(&c1, cnw, cnb, B, Tt, C); }
    T p1 = t_alloc(3, (int[]){B, Tt, dim4});
    const float *w1 = W_(cat_name(pfx, "pw_conv1.weight"));
    const float *b1 = W_(cat_name(pfx, "pw_conv1.bias"));
    const float *alp = W_(cat_name(pfx, "act.alpha"));
    const float *gg  = W_(cat_name(pfx, "grn.gamma"));
    const float *gb  = W_(cat_name(pfx, "grn.beta"));
    const int rows = B * Tt;
    /* 2026-09-13 第六轮：snake+GRN 折进 l1/l2 的 epilogue/量化 prologue，
     * p1 大张量从「3 读 3 写」降到「1 读 1 写」（PSRAM ~55MB/s 是硬约束）。
     * 全部表达式与分立调用逐项一致，int8 路径结果逐位相同；任一环节不
     * 满足 int8 条件就退回分立调用（stage 0/1/2 记录已完成的阶段）。 */
    int stage = 0;
    float cu_nx = 0.0f;
    int qok = 0;
    float *ssr = NULL;
    float *f2s = NULL;
    if (B == 1)
      {
        ssr = (float *)calloc((size_t)rows, sizeof(float));
        f2s = (float *)calloc((size_t)rows, sizeof(float));
        if (ssr != NULL && f2s != NULL)
          {
            int qd = 0;
            jx_linear_fuse_t f1;
            f1.snake_alpha = alp; f1.ss_rows = ssr;
            f1.grn_gamma = NULL;   f1.grn_beta = NULL; f1.grn_nx = 0.0f;
            f1.in_norm_w = norm_fused ? cnw : NULL;
            f1.in_norm_b = norm_fused ? cnb : NULL;
            /* r49：让 l1 的 epilogue 直接把 l2 的量化输入做出来。量化结果写进
             * p1 的内存（p1 每元素 4 字节，装 1 字节/元素绰绰有余）；融合生效
             * 时 p1 不再作为浮点张量使用。q_gamma/q_beta 只给 epilogue 用，
             * grn_gamma 保持 NULL —— 那是 l2 prologue 的口子，两者不能混。 */
            f1.q_gamma = gg; f1.q_beta = gb;
            f1.out_q = (signed char *)p1.d; f1.out_sc = f2s;
            f1.out_qdone = &qd;
            f1.in_q = NULL; f1.in_sc = NULL;
            if (jx_linear_i8_ex(w1, b1, c1.d, p1.d, rows, C, dim4, &f1) == 0)
              {
                float ss = 0.0f;
                float g;
                for (int r = 0; r < rows; r++) { ss += ssr[r]; }
                g = sqrtf(ss);
                cu_nx = g / (g + XNN_EPS);
                if (getenv("JX_LINFO")) printf("[cu] C=%d dim4=%d rows=%d g=%.6g nx=%.9g q=%d\n", C, dim4, rows, (double)g, (double)cu_nx, qd);
                jx_mac_linear += (double)rows * dim4 * C; jx_n_linear++;
                stage = 1;
                if (qd != 0 && cu_nx == 1.0f) { qok = 1; }
                else if (qd != 0)
                  {
                    /* nx != 1.0f：融合所依赖的 GRN 简式不成立（实测从未出现），
                     * 必须把 p1 重建成浮点结果，退回原路径。 */
                    if (norm_fused) { channel_norm_last(&c1, cnw, cnb, B, Tt, C); norm_fused = 0; }
                    cu_rebuild_p1(w1, b1, &c1, &p1, alp, ssr, rows, C, dim4, B, Tt);
                  }
              }
          }
      }
    jx_stage_h("cu.l1", p1.d, (size_t)p1.len * sizeof(float));
    if (stage == 0)
      {
        if (norm_fused) { channel_norm_last(&c1, cnw, cnb, B, Tt, C); norm_fused = 0; }
        linear(w1, b1, &c1, &p1, C, dim4);
        snake1d(&p1, alp, B, Tt, dim4, 1);
        grn_last(&p1, gg, gb, B, Tt, dim4);
      }
    JXP_END(JP_CUNORM);
    int p2_ok = 0;
    T p2;
    JXP_BEG(JP_CUOUT);
    {
      const float *w2 = W_(cat_name(pfx, "pw_conv2.weight"));
      const float *b2 = W_(cat_name(pfx, "pw_conv2.bias"));
      int done2 = 0;
      jx_linear_fuse_t f2;
      const jx_linear_fuse_t *fuse2 = NULL;
      if (stage == 1)
        {
          f2.snake_alpha = NULL; f2.ss_rows = NULL;
          f2.grn_gamma = qok ? NULL : gg;
          f2.grn_beta  = qok ? NULL : gb;
          f2.grn_nx = cu_nx;
          f2.q_gamma = NULL; f2.q_beta = NULL;
          f2.out_q = NULL; f2.out_sc = NULL; f2.out_qdone = NULL;
          f2.in_norm_w = NULL; f2.in_norm_b = NULL;
          /* r49：in_q 非空 = 输入已由 l1 量化好，l2 跳过整个 prologue */
          f2.in_q  = qok ? (const signed char *)p1.d : NULL;
          f2.in_sc = qok ? f2s : NULL;
          fuse2 = &f2;
        }
      if (jx_linear_i8_cf(w2, b2, p1.d, y->d, rows, dim4, C, fuse2) == 0)
        {
          jx_mac_linear += (double)rows * C * dim4; jx_n_linear++;
          done2 = 1;
        }
      else if (stage == 1)
        {
          /* 融合路径下 p1 里是量化数据，先重建成浮点再补 GRN */
          if (norm_fused) { channel_norm_last(&c1, cnw, cnb, B, Tt, C); norm_fused = 0; }
          if (qok) { cu_rebuild_p1(w1, b1, &c1, &p1, alp, ssr, rows, C, dim4, B, Tt); qok = 0; }
          /* l1+snake 已融合进 p1，只差 GRN：补一次原样 GRN 再走旧 l2 */
          grn_last(&p1, gg, gb, B, Tt, dim4);
        }
      if (done2 == 0)
        {
          p2 = t_alloc(3, (int[]){B, Tt, C});
          p2_ok = 1;
          linear(w2, b2, &p1, &p2, dim4, C);
          JXP_BEG(JP_PERMUTE);
          for (int bi = 0; bi < B; bi++)
            jx_transpose_ct(p2.d + (size_t)bi * Tt * C, y->d + (size_t)bi * C * Tt, Tt, C);
          JXP_END(JP_PERMUTE);
        }
    }
    JXP_END(JP_CUOUT);
    jx_stage_h("cu.out", y->d, (size_t)y->len * sizeof(float));
    y->shape[0] = B; y->shape[1] = C; y->shape[2] = Tt; y->len = B*C*Tt;
    t_free(&c1); t_free(&p1); if (p2_ok) { t_free(&p2); }
    free(ssr); free(f2s);
}

/* ============================ V3FirstBlock (modules.V3FirstBlock) ============================ */
static int first_block_core(const T *x, T *y) {
    int B = x->shape[0], Tt = x->shape[2]; int each = 4;
    int fbkeep = 80;   /* 3k is compacted to 32 by _compact_first_block.py */
    {
        int sh[4], n;
        if (wt_get("encoder.blocks.0.conv_1.weight", sh, &n) != NULL && sh[0] > 0)
          {
            fbkeep = sh[0];
          }
    }
    const int ks[5] = {7,7,7,7,7}; const int ps[5] = {1,5,11,21,45}; const int dr[5] = {1,1,1,1,1};
    T h = t_alloc(3, (int[]){B, 20, Tt}); if (h.d == NULL) { return 1; }
    JXP_BEG(JP_FBR);
    {   /* r81：5*B 个单元二分给两核（原来整段单核） */
        jx_fbr_t fc; fc.x = x->d; fc.h = h.d; fc.B = B; fc.Tt = Tt; fc.each = each;
        jx_pf_run_min(B * 5, jx_fbr_range, &fc, 2);
    }
    JXP_END(JP_FBR);
    jx_stage_h("fb.h", h.d, (size_t)h.len * sizeof(float));
    T hc = t_alloc(3, (int[]){B, fbkeep + 1, Tt}); if (hc.d == NULL) { t_free(&h); return 1; }
    /* r180：B==1 时 h1 完全不被使用（h1v 直接指向 hc），跳过分配
     * 省掉 80*Tt*4 字节（3 秒音频 1.32 MB），修复 loop3 OOM。 */
    T h1v;
    h1v.ndim = 3; h1v.shape[0] = B; h1v.shape[1] = fbkeep; h1v.shape[2] = Tt;
    h1v.len = fbkeep * Tt;
    if (B == 1)
      {
        h1v.d = hc.d;
      }
    else
      {
        h1v = t_alloc(3, (int[]){B, 80, Tt});
      }
    JXP_BEG(JP_FBC1);
    /* 2026-09-14 第三十三轮：gelu 折进 conv_1 的 epilogue（见 nn_ops.h 的
     * conv1d_gelu）。原来这里把 80 x Tt 的 h1 整体「读一遍 + 写一遍」
     * （24 次调用共 ~36 MB PSRAM 往返），实测这一项就要 1.6 s。
     * 折进去之后数值逐位相同（同一个输入、同一个 gelu_f），
     * 只是不再额外过一遍内存。JX_GELFUSE=0 回退到原来的两段式。 */
    conv1d_gelu(W_("encoder.blocks.0.conv_1.weight"), W_("encoder.blocks.0.conv_1.bias"), &h, &h1v, 20, fbkeep, 1, 1, 0, 1, Tt);
    JXP_END(JP_FBC1);
    JXP_BEG(JP_GELU);
    /* 2026-09-14 第十八轮：4 路展开。gelu_f = 0.5x(1+erf(x/sqrt2))，erf 是
     * 12 级 Horner，单链延迟 ~100 周期（[1a] FPU 单链 9 周期/op）。原来一次
     * 只喂一条链，FPU 全在等延迟。4 个元素互不依赖，展开后 4 条链交错。
     * 逐元素独立、算式不变 -> 逐位一致。 */
    if (jx_conv_post_used != 1)
    {
      /* 第三十九轮：9 个 gelu 系数装载一次（原来每个元素 l32r+wfr 重载）。 */
      jx_kt_ensure();
      const jx_erf9 KE = jx_erf9_load();
      float *pd = h1v.d;
      int i = 0;
      for (; i + 4 <= h1v.len; i += 4)
        {
          pd[i    ] = jx_gelu_fast(pd[i    ], KE);
          pd[i + 1] = jx_gelu_fast(pd[i + 1], KE);
          pd[i + 2] = jx_gelu_fast(pd[i + 2], KE);
          pd[i + 3] = jx_gelu_fast(pd[i + 3], KE);
        }
      for (; i < h1v.len; i++) { pd[i] = jx_gelu_fast(pd[i], KE); }
    }
    JXP_END(JP_GELU);
    jx_stage_h("fb.h1", h1v.d, (size_t)h1v.len * sizeof(float));
    /* 2026-09-12 #7：这里原来是
     *     for (t) for (c) hc[(bi*81+c)*Tt+t] = h1[(bi*80+c)*Tt+t];
     * —— 每个元素都跳 Tt*4 字节（窗口 Tt=1152 时是 4608 字节），
     * 读+写双向难以命中缓存，而它本质上只是"把 80 行原样搬过去"
     * 再接上第 81 行。改成逐行连续 memcpy，读写都是顺序流，语义一模一样。 */
    JXP_BEG(JP_FBCAT);
    if (B == 1)
      {
        /* conv_1 已直接写进 hc[0..80*Tt)，这里只补第 81 行 */
        memcpy(&hc.d[(size_t)fbkeep * Tt], x->d, (size_t)Tt * sizeof(float));
      }
    else
      {
        for (int bi = 0; bi < B; bi++) {
            memcpy(&hc.d[(size_t)bi * (fbkeep + 1) * Tt], &h1v.d[(size_t)bi * fbkeep * Tt],
                   (size_t)fbkeep * Tt * sizeof(float));
            memcpy(&hc.d[((size_t)bi * (fbkeep + 1) + fbkeep) * Tt], &x->d[(size_t)bi * Tt],
                   (size_t)Tt * sizeof(float));
        }
      }
    JXP_END(JP_FBCAT);
    jx_stage_h("fb.hc", hc.d, (size_t)hc.len * sizeof(float));
    JXP_BEG(JP_FBC2);
    /* DEBUG FB_DBG: dump BaseBlock output h (20ch) and post-cat hc (81ch) */
    if (getenv("FB_DBG")) {
        FILE *fh = fopen("fb_h.bin", "wb"); if (fh) { fwrite(h.d, 4, h.len, fh); fclose(fh); }
        FILE *fhc = fopen("fb_hc.bin", "wb"); if (fhc) { fwrite(hc.d, 4, hc.len, fhc); fclose(fhc); }
        FILE *fhb = fopen("fb_h1.bin", "wb"); if (fhb) { fwrite(h1v.d, 4, h1v.len, fhb); fclose(fhb); }
    }
    conv1d(W_("encoder.blocks.0.conv_2.weight"), W_("encoder.blocks.0.conv_2.bias"), &hc, y, fbkeep + 1, ENC_DIMS[0], 1, 1, 0, 1, Tt);
    JXP_END(JP_FBC2);
    t_free(&h); if (B != 1) { t_free(&h1v); } t_free(&hc); return 0;
}

/* ============================ EnhanceBlock (modules.EnhanceBlock) ============================ */
static int chunk_width(void);   /* 定义在文件下方的“时间分块执行”一节 */

/* 允许 y == x：4 条 trend-pool 分支都已经先算完，最后那个逐元素
 * y = x + merged*x 只读自己那一个元素，就地写完全等价。
 * 解码器各 stage 靠这一点省掉一整块 [dim,T]（3 秒音频 2.7MB）。 */
void enhance_block(const T *x, T *y, const char *pfx, int dim) {
    JXP_BEG(JP_ENHANCE);
    /* EnhanceBlock(dim): 4 trendpool branches (xi=x[:,0:1]) -> cat(4ch) ->
       InstanceNorm(4) -> Conv1d(4->dim) = merge_layer; then y = x + y*x ; out[B,dim,Tt] */
    int B = x->shape[0], Tt = x->shape[2];
    const int ps[4] = {1,3,5,9}; const int ks[4] = {7,7,7,7}; const int dr[4] = {1,2,3,5};
    T h = t_alloc(3, (int[]){B, 4, Tt});
    {   /* r81：4 条分支二分给两核 */
        jx_enh_t ec; ec.x = &x->d[0 * Tt]; ec.h = h.d; ec.pfx = pfx; ec.Tt = Tt;
        jx_pf_run_min(4, jx_enh_range, &ec, 2);
    }
    const float *w = W_(cat_name(pfx, "merge_layer.0.weight"));
    const float *b = W_(cat_name(pfx, "merge_layer.0.bias"));
    /* nn.InstanceNorm1d(4, affine=True): no running stats exported -> compute
       per-sample-per-channel stats over T (eps=1e-5, the PyTorch default).
       Normalize h in place, then feed it straight into merge_layer.1. */
    instance_norm1d(&h, w, b, B, 4, Tt, 1e-5f);
    /* merge_layer.1 是 4->dim 的 1x1 卷积（逐点、无 pad、无 stride），
     * 所以可以按时间分块、直接把结果算进 y，省掉一整块 [B,dim,Tt]。
     * 每块的取样范围恰好是 [t0,t1) 自己，块与块之间没有任何重叠，
     * 数值与整段调用逐位一致（这是 blocks.c 里既有的分块套路）。
     * 块宽沿用 JX_CHUNK，0 表示整段。 */
    const float *mw = W_(cat_name(pfx, "merge_layer.1.weight"));
    const float *mb = W_(cat_name(pfx, "merge_layer.1.bias"));
    int WW = chunk_width();
    y->shape[0] = B; y->shape[1] = dim; y->shape[2] = Tt; y->len = B*dim*Tt;
    if (WW <= 0 || Tt <= WW)
      {
        T merged = t_alloc(3, (int[]){B, dim, 0});
        conv1d(mw, mb, &h, &merged, 4, dim, 1, 1, 0, 1, Tt);
        /* y = x + merged * x   （r81：二分并行） */
        { jx_ew_t ew; ew.a = x->d; ew.b = merged.d; ew.y = y->d;
          ew.n = B * dim * Tt; ew.mode = 0;
          jx_pf_run_min(ew.n, jx_ew_range, &ew, 2); }
        t_free(&merged);
      }
    else
      {
        for (int bi = 0; bi < B; bi++)
          for (int t0 = 0; t0 < Tt; t0 += WW)
            {
              int t1 = t0 + WW; if (t1 > Tt) { t1 = Tt; }
              int w = t1 - t0;
              T hc = t_alloc(3, (int[]){1, 4, w});
              T mc = t_alloc(3, (int[]){1, dim, w});
              if (hc.d == NULL || mc.d == NULL)
                {
                  t_free(&hc); t_free(&mc);
                  jx_conv_skip++;
                  printf("[ENH] 分块缓冲分配失败，跳过本次 enhance\n");
                  t_free(&h);
                  return;
                }
              for (int c = 0; c < 4; c++)
                  memcpy(&hc.d[(size_t)c * w], &h.d[((size_t)bi * 4 + c) * Tt + t0], (size_t)w * sizeof(float));
              conv1d(mw, mb, &hc, &mc, 4, dim, 1, 1, 0, 1, w);
              for (int c = 0; c < dim; c++)
                {
                  const float *xp = &x->d[((size_t)bi * dim + c) * Tt + t0];
                  float       *yp = &y->d[((size_t)bi * dim + c) * Tt + t0];
                  const float *mp = &mc.d[(size_t)c * w];
                  for (int t = 0; t < w; t++) { yp[t] = xp[t] + mp[t] * xp[t]; }
                }
              t_free(&hc); t_free(&mc);
            }
      }
    t_free(&h);
    JXP_END(JP_ENHANCE);
}

/* ============================ LegacyUnit (modules.LegacyUnit) ============================ */
/* 本体：Snake -> Conv(k=7,p=3*dil) -> Snake -> Conv(k=1)。
 * 时间感受野 = 3*dil（dil<=9 -> 27），其余逐点。分块入口见文末 legacy_unit_add()。 */
static void legacy_unit_core(const T *x, T *y, const char *pfx, int dil) {
    /* LegacyUnit(dim): Snake -> Conv(k=7,p=3*dil) -> Snake -> Conv(k=1) ; residual add inside caller.
       dilation comes from ResidualLegacyUnit(dilation=1/3/9); padding = (k-1)*dil//2 = 3*dil. */
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
    /* 2026-09-14 第三十三轮：原来是 t_copy + snake1d 两趟整张量往返，
     * 合成一趟 snake1d_copy（数值逐位相同）。 */
    T s0 = t_alloc(3, (int[]){B, C, Tt});
    snake1d_copy(&s0, x, W_(cat_name(pfx, "block.0.alpha")));
    /* 2026-09-13 第九轮：这里原来有两次纯搬运，合计每调用 4 个张量往返
     * （T=16032 时约 4MB，实测 ~90ms/次）：
     *   1) conv1d_d 写进 c1 后再 t_copy 到 s1 —— conv1d_d 已经设好
     *      shape[0..2]/len，snake1d 直接吃 c1 就行，复制毫无必要；
     *   2) 末层 conv1d 写进 c2 后再逐元素抄进 y —— 直接让 conv1d 写 y
     *      即可（y->len 恰好等于 need，不会触发 realloc）。
     * 两者都不改任何数值。 */
    T c1 = t_alloc(3, (int[]){B, C, 0});
    /* 第三十三轮：block.2 的 snake 折进 conv1d_d 的 epilogue（省掉一趟整张量往返）。 */
    conv1d_d_snake(W_(cat_name(pfx, "block.1.weight")), W_(cat_name(pfx, "block.1.bias")),
                   &s0, &c1, C, C, 7, 1, 3 * dil, 1, Tt, dil,
                   W_(cat_name(pfx, "block.2.alpha")));
    conv1d(W_(cat_name(pfx, "block.3.weight")), W_(cat_name(pfx, "block.3.bias")), &c1, y, C, C, 1, 1, 0, 1, Tt);
    y->shape[0] = B; y->shape[1] = C; y->shape[2] = Tt; y->len = B*C*Tt;
    t_free(&s0); t_free(&c1);
}

void legacy_unit(const T *x, T *y, const char *pfx, int dil) {
    JXP_BEG(JP_LEGACY);
    legacy_unit_core(x, y, pfx, dil);
    JXP_END(JP_LEGACY);
}

/* ============================ 时间分块执行 ============================ */
/* 板端 PSRAM 只有 8MB：first_block 在全速率上要同时展开 20/80/81 通道的
 * 张量（3 秒音频约 38MB），conv_unit 也要展开 10*C 通道，整段展开放不下。
 * 这两个算子在时间维度上都是有限感受野（first_block：trend_pool p<=22
 * 加 conv k=7/p=3，共 25 样点；conv_unit：dw_conv k=7/p=3，3 样点），
 * 其余算子逐样点、与时间无关。因此把时间轴切块、每块左右各带一段 halo：
 *   - 中间块只保留 [t0,t1)，halo 区结果丢弃，与整段计算逐位一致；
 *   - 首块/末块的起点/终点就是真实信号边界，各算子 padding 规则不变。
 * 块宽可用环境变量 JX_CHUNK 覆盖，0 表示关闭（主机对照用）。 */

#define JX_FB_HALO 64   /* first_block 时间感受野 25，留足余量 */
#define JX_CU_HALO 8    /* conv_unit   时间感受野 3 */

static int chunk_width(void)
{
    const char *s = getenv("JX_CHUNK");
    if (s) { int v = atoi(s); return v > 0 ? v : 0; }
    return JX_CHUNK_DEFAULT;
}

void first_block(const T *x, T *y) {
    JXP_BEG(JP_FIRSTBLOCK);
    int B = x->shape[0], Tt = x->shape[2]; const int Co = ENC_DIMS[0];
    int W = chunk_width();
    y->ndim = 3; y->shape[0] = B; y->shape[1] = Co; y->shape[2] = Tt;
    y->len = B * Co * Tt;
    if (W <= 0 || Tt <= W) {
        if (first_block_core(x, y) != 0) {
            jx_conv_skip++;
            printf("[FB] first_block_core OOM，跳过\n");
            memset(y->d, 0, (size_t)y->len * sizeof(float));
        }
        JXP_END(JP_FIRSTBLOCK); return;
    }
    for (int bi = 0; bi < B; bi++) {
        for (int t0 = 0; t0 < Tt; t0 += W) {
            int t1 = t0 + W; if (t1 > Tt) t1 = Tt;
            int a = t0 - JX_FB_HALO; if (a < 0) a = 0;
            int e = t1 + JX_FB_HALO; if (e > Tt) e = Tt;
            int w = e - a;
            T xs = t_alloc(3, (int[]){1, 1, w});
            {
                JXP_BEG(JP_MISC);
                jx_glue_g_t gg; gg.src = x; gg.dst = xs.d; gg.save = NULL;
                gg.C = 1; gg.Tt = Tt; gg.bi = bi; gg.a = a; gg.w = w;
                gg.saved_from = -1; gg.H = 0; gg.split_t = 1;
                jx_pf_run_min(w, jx_glue_gather, &gg, 64);
                JXP_END(JP_MISC);
            }
            T ys = t_alloc(3, (int[]){1, Co, w});
            if (first_block_core(&xs, &ys) != 0) { jx_conv_skip++; printf("[FB] chunk OOM t0=%d\n", t0); memset(ys.d, 0, (size_t)ys.len * sizeof(float)); }
            {
                JXP_BEG(JP_MISC);
                jx_glue_s_t gs; gs.dst = y->d; gs.src = ys.d; gs.C = Co; gs.Tt = Tt;
                gs.bi = bi; gs.t0 = t0; gs.n = t1 - t0; gs.off = t0 - a; gs.stride = w; gs.add = 0;
                jx_pf_run_min(Co, jx_glue_scatter, &gs, 2);
                JXP_END(JP_MISC);
            }
            t_free(&xs); t_free(&ys);
        }
    }
    JXP_END(JP_FIRSTBLOCK);
}

void conv_unit(const T *x, T *y, const char *pfx) {
    JXP_BEG(JP_CONVUNIT);
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
    int W = chunk_width();
    y->ndim = 3; y->shape[0] = B; y->shape[1] = C; y->shape[2] = Tt;
    y->len = B * C * Tt;
    if (W <= 0 || Tt <= W) { conv_unit_core(x, y, pfx); JXP_END(JP_CONVUNIT); return; }
    for (int bi = 0; bi < B; bi++) {
        for (int t0 = 0; t0 < Tt; t0 += W) {
            int t1 = t0 + W; if (t1 > Tt) t1 = Tt;
            int a = t0 - JX_CU_HALO; if (a < 0) a = 0;
            int e = t1 + JX_CU_HALO; if (e > Tt) e = Tt;
            int w = e - a;
            T xs = t_alloc(3, (int[]){1, C, w});
            {
                JXP_BEG(JP_MISC);
                jx_glue_g_t gg; gg.src = x; gg.dst = xs.d; gg.save = NULL;
                gg.C = C; gg.Tt = Tt; gg.bi = bi; gg.a = a; gg.w = w;
                gg.saved_from = -1; gg.H = 0; gg.split_t = 0;
                jx_pf_run_min(C, jx_glue_gather, &gg, 2);
                JXP_END(JP_MISC);
            }
            T ys = t_alloc(3, (int[]){1, C, w});
            conv_unit_core(&xs, &ys, pfx);
            {
                JXP_BEG(JP_MISC);
                jx_glue_s_t gs; gs.dst = y->d; gs.src = ys.d; gs.C = C; gs.Tt = Tt;
                gs.bi = bi; gs.t0 = t0; gs.n = t1 - t0; gs.off = t0 - a; gs.stride = w; gs.add = 0;
                jx_pf_run_min(C, jx_glue_scatter, &gs, 2);
                JXP_END(JP_MISC);
            }
            t_free(&xs); t_free(&ys);
        }
    }
    JXP_END(JP_CONVUNIT);
}

void conv_unit_add(T *x, const char *pfx) {
    JXP_BEG(JP_CONVUNIT);
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
    int W = chunk_width();
    float *save = NULL;
    if (W > JX_CU_HALO && Tt > W) save = (float*)malloc((size_t)C * JX_CU_HALO * sizeof(float));
    if (save == NULL) {
        T cu = t_alloc(3, (int[]){B, C, Tt});
        if (cu.d == NULL)
          {
            jx_conv_skip++;
            printf("[CU] 整段缓冲分配失败，跳过本次 conv_unit\n");
            JXP_END(JP_CONVUNIT);
            return;
          }
        conv_unit_core(x, &cu, pfx);
        for (int i = 0; i < x->len; i++) x->d[i] += cu.d[i];
        t_free(&cu);
        JXP_END(JP_CONVUNIT);
        return;
    }
    /* 就地累加时，本块写回 x 的区间 [t0,t1) 恰好覆盖了下一块要当左 halo 的
     * [t1-H, t1)：所以写之前先把这 H 个样点“加之前”的值留一份，
     * 下一块读 halo 时用它而不是已被覆盖的 x。 */
    for (int bi = 0; bi < B; bi++) {
        int save_from = -1;
        for (int t0 = 0; t0 < Tt; t0 += W) {
            int t1 = t0 + W; if (t1 > Tt) t1 = Tt;
            int a = t0 - JX_CU_HALO; if (a < 0) a = 0;
            int e = t1 + JX_CU_HALO; if (e > Tt) e = Tt;
            int w = e - a;
            T xs = t_alloc(3, (int[]){1, C, w});
            {
                JXP_BEG(JP_MISC);
                jx_glue_g_t gg; gg.src = x; gg.dst = xs.d; gg.save = save;
                gg.C = C; gg.Tt = Tt; gg.bi = bi; gg.a = a; gg.w = w;
                gg.saved_from = save_from; gg.H = JX_CU_HALO; gg.split_t = 0;
                jx_pf_run_min(C, jx_glue_gather, &gg, 2);
                JXP_END(JP_MISC);
            }
            T ys = t_alloc(3, (int[]){1, C, w});
            conv_unit_core(&xs, &ys, pfx);
            int ns = t1 - JX_CU_HALO; if (ns < t0) ns = t0;
            {
                JXP_BEG(JP_MISC);
                jx_glue_v_t gv; gv.src = x->d; gv.save = save; gv.C = C; gv.Tt = Tt;
                gv.bi = bi; gv.ns = ns; gv.n = t1 - ns; gv.H = JX_CU_HALO;
                jx_pf_run_min(C, jx_glue_save, &gv, 2);
                JXP_END(JP_MISC);
            }
            save_from = ns;
            {
                JXP_BEG(JP_MISC);
                jx_glue_s_t gs; gs.dst = x->d; gs.src = ys.d; gs.C = C; gs.Tt = Tt;
                gs.bi = bi; gs.t0 = t0; gs.n = t1 - t0; gs.off = t0 - a; gs.stride = w; gs.add = 1;
                jx_pf_run_min(C, jx_glue_scatter, &gs, 2);
                JXP_END(JP_MISC);
            }
            t_free(&xs); t_free(&ys);
        }
    }
    free(save);
    JXP_END(JP_CONVUNIT);
}

/* ============================ LegacyUnit 时间分块累加 ============================ */
/* 解码器末段 decoder.blocks.13 有 3 个 ResidualLegacyUnit。整段展开时，cur / lu /
 * legacy_unit 内部的 s0 与 c1 四块 [B,16,T] 会同时活着：2 秒音频 T=32064 时每块
 * 2.05MB、合计 8.2MB，已经超过 PSRAM 堆总量（~8.5MB）。结果是 conv1d_d 的
 * realloc 拿不到 2MB 连续块 -> 卷积被跳过 -> 整段输出作废（板端实测 3 次 [CONV]）。
 *
 * LegacyUnit 的时间感受野只有 3*dil <= 27 个样点（两端 Snake 与末尾 k=1 卷积都是
 * 逐点算子），所以按时间切块、每块左右各带 H = 3*dil 样点 halo 即可：
 *   - 中间块只把 [t0,t1) 写回，halo 区结果直接丢弃；
 *   - 分块内部 conv1d_d 的零填充只影响距块边界 <= pad 的输出，正好落在丢弃区；
 * 于是与整段计算逐位一致（这是 blocks.c 里 first_block / conv_unit 早就在用的做法）。
 *
 * 同时省掉一整块 [B,16,T]：结果就地累加进 x，不再单独分配 lu 再逐元素相加。
 * 块宽沿用 JX_CHUNK（默认 2048）；JX_CHUNK=0 关闭分块，退回整段路径（主机对照用）。 */
#define JX_LU_HALO 32

/* 整段路径：只额外要 2 块 [B,C,Tt]。s0 的内容在最后一步之前就用完了，
 * 直接把它那块缓冲改当 1x1 卷积的输出，省掉第 3 块。
 * 返回 0 = 成功（x 已就地累加），1 = 内存不够（x 一个字节都没动）。 */
static int legacy_unit_add_full(T *x, const char *pfx, int dil)
{
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
    T s0 = t_alloc(3, (int[]){B, C, Tt});
    if (s0.d == NULL) { return 1; }
    snake1d_copy(&s0, x, W_(cat_name(pfx, "block.0.alpha")));
    T c1 = t_alloc(3, (int[]){B, C, Tt});
    if (c1.d == NULL) { t_free(&s0); return 1; }
    conv1d_d_snake(W_(cat_name(pfx, "block.1.weight")), W_(cat_name(pfx, "block.1.bias")),
                   &s0, &c1, C, C, 7, 1, 3 * dil, 1, Tt, dil,
                   W_(cat_name(pfx, "block.2.alpha")));
    T out = s0; s0.d = NULL; s0.len = 0;      /* 所有权交接，t_free(&s0) 变成空操作 */
    conv1d(W_(cat_name(pfx, "block.3.weight")), W_(cat_name(pfx, "block.3.bias")),
           &c1, &out, C, C, 1, 1, 0, 1, Tt);
    {
        JXP_BEG(JP_MISC);
        { jx_ew_t ew; ew.a = out.d; ew.b = NULL; ew.y = x->d;
          ew.n = x->len; ew.mode = 1;
          jx_pf_run_min(ew.n, jx_ew_range, &ew, 2); }
        JXP_END(JP_MISC);
    }
    t_free(&out); t_free(&c1);
    return 0;
}

/* 分块路径：每块额外只要 2 个小缓冲（约为整段的 W/Tt）。 */
static void legacy_unit_add_chunked(T *x, const char *pfx, int dil, int W, int H)
{
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
    /* 就地在 x 上累加时，本块写回 [t0,t1)，而下一块要拿 [t1-H, t1) 当左 halo ——
     * 那 H 个样点已经被本块改过。所以写之前先把“加之前”的原值留一份，下一块读
     * halo 时用它。做法与 conv_unit_add() 完全一致。 */
    float *save = NULL;
    if (W > H && H > 0) { save = (float*)malloc((size_t)C * H * sizeof(float)); }
    if (save == NULL)
      {
        /* 连 2KB 的保存区都拿不到：只能退回整段，失败就报出来，别静默出错 */
        if (legacy_unit_add_full(x, pfx, dil) != 0)
          {
            jx_conv_skip++;
            printf("[LU] 内存不足，跳过本次 legacy 残差\n");
          }
        return;
      }
    for (int bi = 0; bi < B; bi++)
      {
        int saved_from = -1;
        for (int t0 = 0; t0 < Tt; t0 += W)
          {
            int t1 = t0 + W; if (t1 > Tt) { t1 = Tt; }
            int a  = t0 - H; if (a < 0)   { a  = 0;  }
            int e  = t1 + H; if (e > Tt)  { e  = Tt; }
            int w  = e - a;
            T xs = t_alloc(3, (int[]){1, C, w});
            T ys = t_alloc(3, (int[]){1, C, w});
            if (xs.d == NULL || ys.d == NULL)
              {
                t_free(&xs); t_free(&ys); free(save);
                jx_conv_skip++;
                printf("[LU] 分块缓冲分配失败，跳过本次 legacy 残差\n");
                return;
              }
            {
                JXP_BEG(JP_MISC);
                jx_glue_g_t gg; gg.src = x; gg.dst = xs.d; gg.save = save;
                gg.C = C; gg.Tt = Tt; gg.bi = bi; gg.a = a; gg.w = w;
                gg.saved_from = saved_from; gg.H = H; gg.split_t = 0;
                jx_pf_run_min(C, jx_glue_gather, &gg, 2);
                JXP_END(JP_MISC);
            }
            legacy_unit_core(&xs, &ys, pfx, dil);
            if (t1 < Tt)
              {
                int ns = t1 - H; if (ns < t0) { ns = t0; }
                {
                    JXP_BEG(JP_MISC);
                    jx_glue_v_t gv; gv.src = x->d; gv.save = save; gv.C = C; gv.Tt = Tt;
                    gv.bi = bi; gv.ns = ns; gv.n = t1 - ns; gv.H = H;
                    jx_pf_run_min(C, jx_glue_save, &gv, 2);
                    JXP_END(JP_MISC);
                }
                saved_from = ns;
              }
            {
                JXP_BEG(JP_MISC);
                jx_glue_s_t gs; gs.dst = x->d; gs.src = ys.d; gs.C = C; gs.Tt = Tt;
                gs.bi = bi; gs.t0 = t0; gs.n = t1 - t0; gs.off = t0 - a; gs.stride = w; gs.add = 1;
                jx_pf_run_min(C, jx_glue_scatter, &gs, 2);
                JXP_END(JP_MISC);
            }
            t_free(&xs); t_free(&ys);
          }
      }
    free(save);
}

/* 整段路径要额外 2 块 [B,C,Tt]；超过这个预算就直接分块，免得先撞一次 OOM。 */
#define JX_LU_FULL_DEF_MB 5
static long long jx_lu_full_bytes(void)
{
    const char *e = getenv("JX_LUFULL");
    int mb = (e != NULL) ? atoi(e) : JX_LU_FULL_DEF_MB;
    if (mb < 0) { mb = 0; }
    return (long long)mb * 1024 * 1024;
}

void legacy_unit_add(T *x, const char *pfx, int dil)
{
    JXP_BEG(JP_LEGACY);
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
    int W = chunk_width();
    int H = 3 * dil;
    if (H > JX_LU_HALO) { H = JX_LU_HALO; }
    int can_chunk = (W > 0 && Tt > W);
    long long extra  = 2LL * (long long)B * (long long)C * (long long)Tt * 4LL;
    int use_full = (!can_chunk) || (extra <= jx_lu_full_bytes());
    if (use_full && legacy_unit_add_full(x, pfx, dil) == 0)
      {
        JXP_END(JP_LEGACY);
        return;
      }
    if (!can_chunk)
      {
        jx_conv_skip++;
        printf("[LU] 内存不足且分块已关闭，跳过本次 legacy 残差\n");
        JXP_END(JP_LEGACY);
        return;
      }
    legacy_unit_add_chunked(x, pfx, dil, W, H);
    JXP_END(JP_LEGACY);
}
