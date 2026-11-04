/* ============================================================================
 * stream_codec.c — 因果流式编解码装配（编码端 / 解码端两个独立状态机）。
 *
 * 模块层级、block 索引、权重名与 Python 参考（l3ac/stream_codec.py）完全一致：
 *
 *   encoder (StreamEncoder)  blocks[0..8]
 *     0              StreamV3FirstBlock                 (5 条 trendpool 支路)
 *     1,3,5,7        Sequential(StreamResidualUnit)     (Stage)
 *     2,4,6          Sequential(CausalConv1d(k=stride,s=stride), ChannelNorm)
 *     8              CausalConv1d(88->64, k=3)
 *   en_encoder (LocalEncoder)   local_trans depth=1, W=128
 *   quantizer (LearnableFSQ)
 *   en_decoder (LocalDecoder)   local_trans depth=1, W=128
 *   decoder (StreamDecoder)   blocks[0..13]
 *     0              CausalConv1d(64->88, k=3)
 *     1,4,7,10       Sequential(StreamResidualUnit)
 *     2,5,8,11       StreamEnhanceBlock
 *     3,6,9,12       Sequential(CausalConv1d(k=1), CausalUpsample, ChannelNorm)
 *     13             LastBlock( 3x StreamResidualLegacyUnit + Snake + Conv(k=7) + Tanh )
 *
 * 多分支延迟补偿（必须，否则 cat/残差两侧错位）：
 *   V3FirstBlock 支路延迟 {3,7,13,23,47}   max=47；直连 x 也右移 47
 *   EnhanceBlock 支路延迟 {3,8,13,23}      max=23；x + y*x 两侧都右移 23
 *   StreamResidual                        跳连右移 unit.delay
 * ========================================================================== */
#include "stream_codec.h"
#include "quantizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

STapCb g_stap = NULL;
void  *g_stap_ud = NULL;

/* arena 容量（float 数）。流式每步仅处理固定 ~50 帧，单步真实峰值已用
 * -DDSP_PROFILE 实测：3k enc 2.22M / dec 3.26M，6k enc 1.14M / dec 1.87M。
 * 取 4M（16MB）覆盖两种码率并留 ~22% 裕量；原 8M 每状态多预留 ~19MB，
 * 在 ESP32-S3（SRAM 数百 KB + PSRAM ≤8MB）上会直接 OOM。
 * 注意：arena 是 bump 分配器，一步内所有临时缓冲同时存活，故峰值即单步活内存；
 * 3k 解码端 ~13MB 需 ≥16MB PSRAM，否则需进一步做 arena 复用（见 memory）。
 *
 * 1k 需单独放大：hop=288 -> 每 chunk 50 帧 = 14400 样本（3k 4800 的 3 倍），
 * 且 decoder_depths=[2,2,1,1] 比 3k/6k 多两个 StreamResidual 级。
 * 1k 实测峰值（-DDSP_PROFILE, chunk=50 帧=14400 样本）：
 *   enc = 6.18M floats (24.2 MB)、dec = 6.03M floats (23.6 MB)
 *   -> 取 8M（32 MB/状态，约 30% 裕量）。
 * ⚠️ 1k 单状态就要 ~24 MB，远超 ESP32-S3 的 PSRAM(≤8MB)。真要上板必须减小
 *    chunk（CHUNK_FRAMES 50 -> 10~15，arena 峰值近似线性下降），或做 arena 复用。
 *    因果流式保证「整段 ≡ 分块」，减小 chunk 不影响输出正确性。 */
#if defined(SQM_RATE_1K)
#define ARENA_FLOATS ((size_t)8 * 1024 * 1024)
#else
#define ARENA_FLOATS ((size_t)4 * 1024 * 1024)
#endif

/* 编码器结构的速率无关常量（与 stream_codec.py 的字面量一一对应） */
#define FB_POOL   {1, 5, 11, 21, 45}          /* StreamV3FirstBlock pool kernels */
#define FB_DELAY  {3, 7, 13, 23, 47}          /* 各支路延迟 = (pool-1) + 3 */
#define FB_MAXDELAY 47
#define FB_NBR 5
#define ENH_POOL  {1, 3, 5, 9}
#define ENH_DIL   {1, 2, 3, 5}                /* conv_dilation = pool//2 + 1 */
#define ENH_DELAY {3, 8, 13, 23}
#define ENH_MAXDELAY 23
#define ENH_NBR 4
#define CU_KERNEL 7                           /* ConvUnit.dw_conv kernel */
#define CU_DELAY  3                           /* ConvUnit delay = (7-1)*1//2 */
#define FN_ALPHA  0.02f                       /* FrameNorm1d 默认参数 */
#define FN_WINDOW 400
#define FN_EPS    1e-5f
#define LG_DIL    {1, 3, 9}                   /* LegacyUnit dilation */

/* ---- 结构容量上限：按最"宽"的码率（1k）取，3k/6k 只使用前几个槽位 ----
 *   encoder  residual 数 = ENC_STAGES+1（1k:5 / 3k,6k:4）
 *            down     数 = ENC_STAGES  （1k:4 / 3k,6k:3）
 *   decoder  stage 内 StreamResidual 数由 DEC_DEPTHS 决定，
 *            总数 = sum(DEC_DEPTHS[0..DEC_STAGES-1])：1k = 2+2+1+1 = 6，3k/6k = 4 */
#define MAX_ENC_RES  5
#define MAX_ENC_DOWN 4
#define MAX_DEC_RES  8

/* decoder 第 st 个 stage 的第一个 StreamResidual 在线性 dw/resd 数组中的下标 */
static int dec_res_base(int st) {
    int b = 0;
    for (int i = 0; i < st; i++) b += DEC_DEPTHS[i];
    return b;
}

/* ---- 小工具：构造 T 视图 ---- */
static T mk_tc(float *d, int rows, int C) {
    T t; t.d = d; t.ndim = 3; t.shape[0] = 1; t.shape[1] = rows; t.shape[2] = C;
    t.len = rows * C; return t;
}
static T mk_cv(float *d, int C, int Tt) {
    T t; t.d = d; t.ndim = 3; t.shape[0] = 1; t.shape[1] = C; t.shape[2] = Tt;
    t.len = C * Tt; return t;
}
static inline void tap(const char *name, const float *d, int C, int Tt, int ch_last) {
    if (g_stap) g_stap(g_stap_ud, name, d, C, Tt, ch_last);
}
static void cpy(float *dst, const float *src, size_t n) { memcpy(dst, src, n * sizeof(float)); }

/* ---- 量化分派：优先 weight-only INT8（int8 权重 + 每通道 scale），否则回退 float。
 *      所有被这些算子消费的 Conv/Linear 权重都已被 PTQ 量化，故正常情况下走 int8 路径；
 *      ChannelNorm/Snake/GRN 等非 Conv/Linear 权重不会被传入，仍由各自算子用 float W_。 ---- */
static int sconv_step_d(SHist *h, const char *wname, const char *bname,
                        int Ci, int Co, int k, int stride, int dil, int groups,
                        const float *x, int n_in, float *y) {
    const int8_t *Wi; const float *sc;
    Wi = wt_get_i8(wname, &sc, NULL, NULL, NULL);
    if (Wi) return sconv_step_q(h, Wi, sc, W_(bname), Ci, Co, k, stride, dil, groups, x, n_in, y);
    return sconv_step(h, W_(wname), W_(bname), Ci, Co, k, stride, dil, groups, x, n_in, y);
}
static void linear_d(const char *wname, const char *bname, const T *x, T *y, int in, int out) {
    const int8_t *Wi; const float *sc;
    Wi = wt_get_i8(wname, &sc, NULL, NULL, NULL);
    if (Wi) { linear_q(Wi, sc, bname ? W_(bname) : NULL, x, y, in, out); return; }
    linear(W_(wname), bname ? W_(bname) : NULL, x, y, in, out);
}
static int supsample_step_d(SHist *h, const char *swname, const char *sbname,
                            int C, int stride, int k, const float *x, int n_in, float *y) {
    const int8_t *Wi; const float *sc;
    Wi = wt_get_i8(swname, &sc, NULL, NULL, NULL);
    if (Wi) return supsample_step_q(h, C, stride, k, Wi, sc, sbname ? W_(sbname) : NULL, x, n_in, y);
    return supsample_step(h, C, stride, k, W_(swname), sbname ? W_(sbname) : NULL, x, n_in, y);
}

/* ==========================================================================
 * 编码端状态
 * ========================================================================== */
struct SEncState {
    SHist tp[FB_NBR];        /* 5 条支路的 CausalTrendPool 历史 */
    SHist cvh[FB_NBR];       /* 5 条支路的因果卷积历史 (k=7, lp=6) */
    SHist brd[FB_NBR];       /* 支路对齐延迟线 Δ = 47 - d_i */
    SHist xd;                /* 直连 x 的延迟线 (47) */
    SHist c1h, c2h;          /* conv_1(20->80,k=1) / conv_2(81->16,k=1) */
    SHist dw[MAX_ENC_RES], resd[MAX_ENC_RES];
                             /* blocks 1/3/5/7/9 的 ConvUnit.dw_conv + 残差延迟
                              * （ENC_STAGES+1 个；末一个 = 末尾 Stage，见 Python blocks[2*ENC_STAGES+1]） */
    SHist down[MAX_ENC_DOWN];/* blocks 2/4/6/8 的下采样卷积 (k=stride, lp=0)（ENC_STAGES 个） */
    SHist fin;               /* 末尾 1x1->feature_dim 卷积 (k=3, lp=2) */
    SLocalTrans lt;          /* en_encoder */
    SArena A;
};

/* ==========================================================================
 * 解码端状态
 * ========================================================================== */
typedef struct {
    SHist tp[ENH_NBR], cvh[ENH_NBR], brd[ENH_NBR];
    SHist fn;                /* merge_layer.0 = FrameNorm1d(4) */
    SHist mc;                /* merge_layer.1 = Conv1d(4->dim, k=1) */
    SHist xd;                /* EnhanceBlock.delay = 23 */
} SEnhState;

struct SDecState {
    SLocalTrans lt;          /* en_decoder (depth=1) */
    SHist c0;                /* blocks 0 (k=3, lp=2) */
    SHist dw[MAX_DEC_RES], resd[MAX_DEC_RES];
                             /* 各 stage 的 StreamResidual，按 DEC_DEPTHS 线性展开
                              * （1k: [2,2,1,1] -> 6 个；3k/6k: 全 1 -> 4 个） */
    SEnhState enh[4];        /* blocks 2/5/8/11 */
    SHist upc[4];            /* up_layer 0 = 1x1 conv */
    SHist ups[4];            /* up_layer 1 = CausalUpsample */
    SHist ldw[3], lone[3], lrd[3];   /* blocks 13: 3 个 LegacyUnit */
    SHist oconv;             /* blocks 13.block.2 (k=7, lp=6) */
    SArena A;
};

/* ==========================================================================
 * 生命周期
 * ========================================================================== */
void senc_reset(SEncState *s) {
    static const int pk[FB_NBR] = FB_POOL;
    static const int bd[FB_NBR] = FB_DELAY;
    for (int i = 0; i < FB_NBR; i++) {
        int H = (pk[i] <= 1) ? 1 : 2 * (pk[i] - 1);
        shist_reset(&s->tp[i], 1, 0, H);
        shist_reset(&s->cvh[i], 1, 6, 24);                 /* k=7 d=1 -> K=6 */
        int d = FB_MAXDELAY - bd[i];
        shist_reset(&s->brd[i], 4, d, d > 0 ? d * 2 : 1);
    }
    shist_reset(&s->xd, 1, FB_MAXDELAY, FB_MAXDELAY * 2);
    shist_reset(&s->c1h, 20, 0, 4);
    shist_reset(&s->c2h, 81, 0, 4);
    for (int i = 0; i < ENC_STAGES + 1; i++) {
        shist_reset(&s->dw[i], ENC_DIMS[i], 6, 24);
        shist_reset(&s->resd[i], ENC_DIMS[i], CU_DELAY, 16);
    }
    for (int i = 0; i < ENC_STAGES; i++) shist_reset(&s->down[i], ENC_DIMS[i], 0, 4);
    shist_reset(&s->fin, ENC_DIMS[ENC_STAGES], 2, 8);
    slt_reset(&s->lt);
}

SEncState *senc_new(void) {
    SEncState *s = (SEncState*)calloc(1, sizeof(SEncState));
    if (!s) { fprintf(stderr, "senc_new: OOM\n"); exit(1); }
    sa_init(&s->A, ARENA_FLOATS);
    senc_reset(s);
    slt_init(&s->lt, "en_encoder.local_trans", 1, EN_WINDOW);
    return s;
}

void senc_free(SEncState *s) {
    if (!s) return;
    for (int i = 0; i < FB_NBR; i++) { shist_free(&s->tp[i]); shist_free(&s->cvh[i]); shist_free(&s->brd[i]); }
    shist_free(&s->xd); shist_free(&s->c1h); shist_free(&s->c2h);
    for (int i = 0; i < ENC_STAGES + 1; i++) { shist_free(&s->dw[i]); shist_free(&s->resd[i]); }
    for (int i = 0; i < ENC_STAGES; i++) shist_free(&s->down[i]);
    shist_free(&s->fin);
    slt_free(&s->lt);
    sa_free(&s->A);
    free(s);
}

/* 返回本状态 arena 的历史峰值（float 数）。用于部署时把 ARENA_FLOATS 设到
 * 真实峰值 + 裕量，避免为流式每步仅数十帧的规模预留 32MB。 */
size_t senc_peak(const SEncState *s) { return s->A.max_off; }
size_t sdec_peak(const SDecState *s) { return s->A.max_off; }

void sdec_reset(SDecState *s) {
    static const int pk[ENH_NBR] = ENH_POOL;
    static const int dl[ENH_NBR] = ENH_DIL;
    static const int bd[ENH_NBR] = ENH_DELAY;
    static const int lgd[3] = LG_DIL;
    shist_reset(&s->c0, FEATURE_DIM, 2, 8);
    {
        /* 按 DEC_DEPTHS 展开：第 st 个 stage 有 DEC_DEPTHS[st] 个 StreamResidual，
         * 通道数都是 DEC_DIMS[st]（up_layer 在 stage 之后才改变通道） */
        int idx = 0;
        for (int st = 0; st < DEC_STAGES; st++)
            for (int j = 0; j < DEC_DEPTHS[st]; j++) {
                shist_reset(&s->dw[idx], DEC_DIMS[st], 6, 24);
                shist_reset(&s->resd[idx], DEC_DIMS[st], CU_DELAY, 16);
                idx++;
            }
    }
    for (int st = 0; st < 4; st++) {
        for (int br = 0; br < ENH_NBR; br++) {
            int H = (pk[br] <= 1) ? 1 : 2 * (pk[br] - 1);
            shist_reset(&s->enh[st].tp[br], 1, 0, H);
            shist_reset(&s->enh[st].cvh[br], 1, 6 * dl[br], 12 * dl[br] + 8);
            int d = ENH_MAXDELAY - bd[br];
            shist_reset(&s->enh[st].brd[br], 1, d, d > 0 ? d * 2 : 1);
        }
        shist_reset(&s->enh[st].fn, 4, FN_WINDOW - 1, FN_WINDOW + 8);
        shist_reset(&s->enh[st].mc, 4, 0, 4);
        shist_reset(&s->enh[st].xd, DEC_DIMS[st], ENH_MAXDELAY, ENH_MAXDELAY * 2);
        int nxt = (st + 1 < DEC_STAGES) ? DEC_DIMS[st + 1] : DEC_DIMS[DEC_STAGES];
        shist_reset(&s->upc[st], DEC_DIMS[st], 0, 4);
        int S = DEC_STRIDES[st], k = 2 * S + 1;
        shist_reset(&s->ups[st], nxt, (k - 1 + S - 1) / S, 2 * ((k - 1 + S - 1) / S) + 4);
    }
    for (int r = 0; r < 3; r++) {
        shist_reset(&s->ldw[r], DEC_DIMS[DEC_STAGES], 6 * lgd[r], 12 * lgd[r] + 8);
        shist_reset(&s->lone[r], DEC_DIMS[DEC_STAGES], 0, 4);
        shist_reset(&s->lrd[r], DEC_DIMS[DEC_STAGES], 3 * lgd[r], 6 * lgd[r] + 4);
    }
    shist_reset(&s->oconv, DEC_DIMS[DEC_STAGES], 6, 16);
    slt_reset(&s->lt);
}

SDecState *sdec_new(void) {
    SDecState *s = (SDecState*)calloc(1, sizeof(SDecState));
    if (!s) { fprintf(stderr, "sdec_new: OOM\n"); exit(1); }
    sa_init(&s->A, ARENA_FLOATS);
    sdec_reset(s);
    slt_init(&s->lt, "en_decoder.local_trans", 1, EN_WINDOW);
    return s;
}

void sdec_free(SDecState *s) {
    if (!s) return;
    shist_free(&s->c0);
    {
        int nres = 0;
        for (int st = 0; st < DEC_STAGES; st++) nres += DEC_DEPTHS[st];
        for (int i = 0; i < nres; i++) { shist_free(&s->dw[i]); shist_free(&s->resd[i]); }
    }
    for (int st = 0; st < 4; st++) {
        for (int br = 0; br < ENH_NBR; br++) { shist_free(&s->enh[st].tp[br]); shist_free(&s->enh[st].cvh[br]); shist_free(&s->enh[st].brd[br]); }
        shist_free(&s->enh[st].fn); shist_free(&s->enh[st].mc); shist_free(&s->enh[st].xd);
        shist_free(&s->upc[st]); shist_free(&s->ups[st]);
    }
    for (int r = 0; r < 3; r++) { shist_free(&s->ldw[r]); shist_free(&s->lone[r]); shist_free(&s->lrd[r]); }
    shist_free(&s->oconv);
    slt_free(&s->lt);
    sa_free(&s->A);
    free(s);
}

/* ==========================================================================
 * 编码端：blocks[0] StreamV3FirstBlock
 * ========================================================================== */
static float *enc_first_block(SEncState *s, const float *x, int n) {
    static const int pk[FB_NBR] = FB_POOL;
    static const int bd[FB_NBR] = FB_DELAY;
    SArena *A = &s->A;

    float *h   = sa_alloc(A, (size_t)20 * n);        /* 5 支路 x 4ch，CV */
    float *tpv = sa_alloc(A, (size_t)n);
    float *cvo = sa_alloc(A, (size_t)4 * n);
    for (int br = 0; br < FB_NBR; br++) {
        strendpool_step(&s->tp[br], pk[br], 1, x, n, tpv);
        char wn[96]; snprintf(wn, sizeof(wn), "encoder.blocks.0.blocks.%d.1", br);
        sconv_step_d(&s->cvh[br], cat_name(wn, "weight"), cat_name(wn, "bias"),
                     1, 4, CU_KERNEL, 1, 1, 1, tpv, n, cvo);
        int d = FB_MAXDELAY - bd[br];
        if (d > 0) sdelay_step(&s->brd[br], d, 4, cvo, n, &h[(size_t)br * 4 * n]);
        else       cpy(&h[(size_t)br * 4 * n], cvo, (size_t)4 * n);
    }
    /* conv_1: 20 -> 80 (1x1, 天然因果) */
    float *h1 = sa_alloc(A, (size_t)80 * n);
    sconv_step_d(&s->c1h, "encoder.blocks.0.conv_1.weight", "encoder.blocks.0.conv_1.bias",
                 20, 80, 1, 1, 1, 1, h, n, h1);
    gelu_apply(h1,80*n);
    /* cat([h1, causal_delay(x, 47)]) -> 81ch */
    float *hc = sa_alloc(A, (size_t)81 * n);
    cpy(hc, h1, (size_t)80 * n);
    sdelay_step(&s->xd, FB_MAXDELAY, 1, x, n, &hc[(size_t)80 * n]);
    float *out = sa_alloc(A, (size_t)ENC_DIMS[0] * n);
    sconv_step_d(&s->c2h, "encoder.blocks.0.conv_2.weight", "encoder.blocks.0.conv_2.bias",
                 81, ENC_DIMS[0], 1, 1, 1, 1, hc, n, out);
    return out;
}

/* ==========================================================================
 * 共享：StreamResidual(StreamConvUnit) —— 编码/解码端都用
 *   side = ConvUnit(x) ; base = delay(x, unit.delay) ; out = base + side
 * ========================================================================== */
static float *convunit_residual(SArena *A, SHist *dwh, SHist *resh, const char *pn,
                                const float *cur, int C, int n) {
    float *dw = sa_alloc(A, (size_t)C * n);
    sconv_step_d(dwh, cat_name(pn, "dw_conv.weight"), cat_name(pn, "dw_conv.bias"),
                 C, C, CU_KERNEL, 1, 1, C, cur, n, dw);
    /* -> TC */
    float *tc = sa_alloc(A, (size_t)n * C);
    for (int t = 0; t < n; t++) for (int c = 0; c < C; c++) tc[(size_t)t * C + c] = dw[(size_t)c * n + t];
    T tct = mk_tc(tc, n, C);
    channel_norm_last(&tct, W_(cat_name(pn, "norm.weight")), W_(cat_name(pn, "norm.bias")), 1, n, C);
    float *p1 = sa_alloc(A, (size_t)n * 4 * C);
    T p1t = mk_tc(p1, n, 4 * C);
    linear_d(cat_name(pn, "pw_conv1.weight"), cat_name(pn, "pw_conv1.bias"), &tct, &p1t, C, 4 * C);
    snake1d(&p1t, W_(cat_name(pn, "act.alpha")), 1, n, 4 * C, 1);
    per_frame_grn(p1, n, 4 * C, W_(cat_name(pn, "grn.gamma")), W_(cat_name(pn, "grn.beta")));
    float *p2 = sa_alloc(A, (size_t)n * C);
    T p2t = mk_tc(p2, n, C);
    linear_d(cat_name(pn, "pw_conv2.weight"), cat_name(pn, "pw_conv2.bias"), &p1t, &p2t, 4 * C, C);
    /* -> CV */
    float *side = sa_alloc(A, (size_t)C * n);
    for (int c = 0; c < C; c++) for (int t = 0; t < n; t++) side[(size_t)c * n + t] = p2[(size_t)t * C + c];
    float *base = sa_alloc(A, (size_t)C * n);
    sdelay_step(resh, CU_DELAY, C, cur, n, base);
    float *out = sa_alloc(A, (size_t)C * n);
    for (size_t i = 0; i < (size_t)C * n; i++) out[i] = base[i] + side[i];
    return out;
}

/* ==========================================================================
 * 编码端：encoder -> en_encoder -> quantizer
 * ========================================================================== */
int senc_step(SEncState *s, const float *x, int n, int64_t *tokens) {
    sa_reset(&s->A);
    int frames = n / HOP_LENGTH;

    float *cur = enc_first_block(s, x, n);
    tap("encoder.blocks.0", cur, ENC_DIMS[0], n, 0);
    int cur_n = n;

    for (int st = 0; st < ENC_STAGES; st++) {
        char nm[64], pn[96];
        snprintf(pn, sizeof(pn), "encoder.blocks.%d.0.module", 2 * st + 1);
        cur = convunit_residual(&s->A, &s->dw[st], &s->resd[st], pn, cur, ENC_DIMS[st], cur_n);
        snprintf(nm, sizeof(nm), "encoder.blocks.%d", 2 * st + 1);
        tap(nm, cur, ENC_DIMS[st], cur_n, 0);

        int Co = ENC_DIMS[st + 1], stride = ENC_STRIDES[st];
        int nout = cur_n / stride;
        float *dn = sa_alloc(&s->A, (size_t)Co * nout);
        char w1[96], b1[96], n1[96], c1[96];
        snprintf(w1, sizeof(w1), "encoder.blocks.%d.0.weight", 2 * st + 2);
        snprintf(b1, sizeof(b1), "encoder.blocks.%d.0.bias",   2 * st + 2);
        snprintf(n1, sizeof(n1), "encoder.blocks.%d.1.weight", 2 * st + 2);
        snprintf(c1, sizeof(c1), "encoder.blocks.%d.1.bias",   2 * st + 2);
        sconv_step_d(&s->down[st], w1, b1, ENC_DIMS[st], Co, stride, stride, 1, 1, cur, cur_n, dn);
        T dnt = mk_cv(dn, Co, nout);
        channel_norm_first(&dnt, W_(n1), W_(c1), 1, Co, nout);
        cur = dn; cur_n = nout;
        snprintf(nm, sizeof(nm), "encoder.blocks.%d", 2 * st + 2);
        tap(nm, cur, Co, cur_n, 0);
    }

    /* blocks[2*ENC_STAGES+1]: 末尾 Stage（3k/6k = blocks[7]，1k = blocks[9]）
     * 注：所有现有配置的 ENC_DEPTHS 均为 1，故这里只取 1 个残差单元。 */
    int fblk = 2 * ENC_STAGES + 1;
    char fpn[64], fnm[64];
    snprintf(fpn, sizeof(fpn), "encoder.blocks.%d.0.module", fblk);
    snprintf(fnm, sizeof(fnm), "encoder.blocks.%d", fblk);
    cur = convunit_residual(&s->A, &s->dw[ENC_STAGES], &s->resd[ENC_STAGES], fpn,
                            cur, ENC_DIMS[ENC_STAGES], cur_n);
    tap(fnm, cur, ENC_DIMS[ENC_STAGES], cur_n, 0);

    /* blocks[2*ENC_STAGES+2]: ENC_DIMS[last] -> FEATURE_DIM (k=3)
     * 3k/6k = blocks[8] (88->64)，1k = blocks[10] (64->64) */
    float *fin = sa_alloc(&s->A, (size_t)FEATURE_DIM * cur_n);
    char fw[64], fb[64], fnm2[64];
    int cblk = 2 * ENC_STAGES + 2;
    snprintf(fw,  sizeof(fw),  "encoder.blocks.%d.weight", cblk);
    snprintf(fb,  sizeof(fb),  "encoder.blocks.%d.bias",   cblk);
    snprintf(fnm2, sizeof(fnm2), "encoder.blocks.%d", cblk);
    sconv_step_d(&s->fin, fw, fb, ENC_DIMS[ENC_STAGES], FEATURE_DIM, 3, 1, 1, 1, cur, cur_n, fin);
    tap(fnm2, fin, FEATURE_DIM, cur_n, 0);

    /* en_encoder: CV -> TC -> LocalTrans(depth=1) */
    float *tc = sa_alloc(&s->A, (size_t)cur_n * FEATURE_DIM);
    for (int t = 0; t < cur_n; t++)
        for (int c = 0; c < FEATURE_DIM; c++) tc[(size_t)t * FEATURE_DIM + c] = fin[(size_t)c * cur_n + t];
    float *eo = sa_alloc(&s->A, (size_t)cur_n * FEATURE_DIM);
    slt_forward(&s->lt, tc, cur_n, eo);
    tap("en_encoder", eo, FEATURE_DIM, cur_n, 1);

    /* quantizer: 产出真实传输载荷 indices */
    T qt = mk_tc(eo, cur_n, FEATURE_DIM);
    T qo = mk_tc(sa_alloc(&s->A, (size_t)cur_n * FEATURE_DIM), cur_n, FEATURE_DIM);
    quantizer_forward(&qt, &qo, tokens, cur_n);
    return frames;
}

/* ==========================================================================
 * 解码端：quantizer -> en_decoder -> decoder
 * ========================================================================== */
static float *enhance_block(SDecState *s, int st, const float *cur, int dim, int n) {
    static const int pk[ENH_NBR] = ENH_POOL;
    static const int dl[ENH_NBR] = ENH_DIL;
    static const int bd[ENH_NBR] = ENH_DELAY;
    SArena *A = &s->A;
    char pfx[64]; snprintf(pfx, sizeof(pfx), "decoder.blocks.%d", 2 + 3 * st);
    SEnhState *E = &s->enh[st];

    float *h   = sa_alloc(A, (size_t)ENH_NBR * n);
    float *tpv = sa_alloc(A, (size_t)n);
    float *cvo = sa_alloc(A, (size_t)n);
    for (int br = 0; br < ENH_NBR; br++) {
        strendpool_step(&E->tp[br], pk[br], 1, cur, n, tpv);      /* xi = x[:, 0:1, :] */
        {
            char tn[96]; snprintf(tn, sizeof(tn), "decoder.blocks.%d.b%d.tp", 2 + 3 * st, br);
            tap(tn, tpv, 1, n, 0);
        }
        char wn[96]; snprintf(wn, sizeof(wn), "%s.blocks.%d.1", pfx, br);
        sconv_step_d(&E->cvh[br], cat_name(wn, "weight"), cat_name(wn, "bias"),
                     1, 1, CU_KERNEL, 1, dl[br], 1, tpv, n, cvo);
        {
            char cn2[96]; snprintf(cn2, sizeof(cn2), "decoder.blocks.%d.b%d.cv", 2 + 3 * st, br);
            tap(cn2, cvo, 1, n, 0);
        }
        int d = ENH_MAXDELAY - bd[br];
        if (d > 0) sdelay_step(&E->brd[br], d, 1, cvo, n, &h[(size_t)br * n]);
        else       cpy(&h[(size_t)br * n], cvo, (size_t)n);
    }
    /* merge_layer: FrameNorm1d(4) -> Conv1d(4->dim, k=1) */
    float *fm = sa_alloc(A, (size_t)4 * n);
    sframenorm_step(&E->fn, 4, FN_WINDOW, FN_ALPHA,
                    W_(cat_name(pfx, "merge_layer.0.weight")), W_(cat_name(pfx, "merge_layer.0.bias")),
                    FN_EPS, h, n, fm);
    {
        char mn[96]; snprintf(mn, sizeof(mn), "decoder.blocks.%d.merge0", 2 + 3 * st);
        tap(mn, fm, 4, n, 0);
    }
    float *mg = sa_alloc(A, (size_t)dim * n);
    sconv_step_d(&E->mc, cat_name(pfx, "merge_layer.1.weight"), cat_name(pfx, "merge_layer.1.bias"),
                 4, dim, 1, 1, 1, 1, fm, n, mg);
    {
        char mn[96]; snprintf(mn, sizeof(mn), "decoder.blocks.%d.merge1", 2 + 3 * st);
        tap(mn, mg, dim, n, 0);
    }
    /* y = xd + y * xd , xd = delay(x, 23) */
    float *xd = sa_alloc(A, (size_t)dim * n);
    sdelay_step(&E->xd, ENH_MAXDELAY, dim, cur, n, xd);
    float *out = sa_alloc(A, (size_t)dim * n);
    for (size_t i = 0; i < (size_t)dim * n; i++) out[i] = xd[i] + mg[i] * xd[i];
    return out;
}

static float *legacy_residual(SDecState *s, int r, const float *cur, int C, int n) {
    static const int lgd[3] = LG_DIL;
    int dil = lgd[r];
    SArena *A = &s->A;
    char pfx[96]; snprintf(pfx, sizeof(pfx), "decoder.blocks.13.block.0.%d.module", r);

    float *s0 = sa_alloc(A, (size_t)C * n); cpy(s0, cur, (size_t)C * n);
    T s0t = mk_cv(s0, C, n);
    snake1d(&s0t, W_(cat_name(pfx, "block.0.alpha")), C, n, C, 0);
    float *c1 = sa_alloc(A, (size_t)C * n);
    sconv_step_d(&s->ldw[r], cat_name(pfx, "block.1.weight"), cat_name(pfx, "block.1.bias"),
                 C, C, CU_KERNEL, 1, dil, 1, s0, n, c1);
    float *s1 = sa_alloc(A, (size_t)C * n); cpy(s1, c1, (size_t)C * n);
    T s1t = mk_cv(s1, C, n);
    snake1d(&s1t, W_(cat_name(pfx, "block.2.alpha")), C, n, C, 0);
    float *c2 = sa_alloc(A, (size_t)C * n);
    sconv_step_d(&s->lone[r], cat_name(pfx, "block.3.weight"), cat_name(pfx, "block.3.bias"),
                 C, C, 1, 1, 1, 1, s1, n, c2);

    float *base = sa_alloc(A, (size_t)C * n);
    sdelay_step(&s->lrd[r], CU_DELAY * dil, C, cur, n, base);
    float *out = sa_alloc(A, (size_t)C * n);
    for (size_t i = 0; i < (size_t)C * n; i++) out[i] = base[i] + c2[i];
    return out;
}

int sdec_step(SDecState *s, const int64_t *tokens, int nt, float *wav) {
    sa_reset(&s->A);
    /* ---- quantizer 反量化 -> features ---- */
    float *qf = sa_alloc(&s->A, (size_t)nt * FEATURE_DIM);
    T qft = mk_tc(qf, nt, FEATURE_DIM);
    quantizer_to_features(tokens, &qft, nt);

    /* ---- en_decoder: LocalTrans(depth=1)，输出 TC -> CV ---- */
    float *ed = sa_alloc(&s->A, (size_t)nt * FEATURE_DIM);
    slt_forward(&s->lt, qf, nt, ed);
    float *cur = sa_alloc(&s->A, (size_t)FEATURE_DIM * nt);
    for (int t = 0; t < nt; t++)
        for (int c = 0; c < FEATURE_DIM; c++) cur[(size_t)c * nt + t] = ed[(size_t)t * FEATURE_DIM + c];
    tap("en_decoder", cur, FEATURE_DIM, nt, 0);

    /* ---- blocks[0]: 64 -> 88 (k=3) ---- */
    float *c0 = sa_alloc(&s->A, (size_t)DEC_DIMS[0] * nt);
    sconv_step_d(&s->c0, "decoder.blocks.0.weight", "decoder.blocks.0.bias",
                 FEATURE_DIM, DEC_DIMS[0], 3, 1, 1, 1, cur, nt, c0);
    cur = c0; int n = nt;
    tap("decoder.blocks.0", cur, DEC_DIMS[0], n, 0);

    for (int st = 0; st < DEC_STAGES; st++) {
        char nm[64], pn[96];
        int dim = DEC_DIMS[st];
        /* Stage = Sequential(StreamResidualUnit × DEC_DEPTHS[st])
         * 3k/6k 每 stage 1 个；1k 的 decoder_depths=[2,2,1,1] -> 前两个 stage 各 2 个。
         * 权重名 decoder.blocks.{1+3st}.{j}.module... */
        int rbase = dec_res_base(st);
        for (int j = 0; j < DEC_DEPTHS[st]; j++) {
            snprintf(pn, sizeof(pn), "decoder.blocks.%d.%d.module", 1 + 3 * st, j);
            cur = convunit_residual(&s->A, &s->dw[rbase + j], &s->resd[rbase + j], pn, cur, dim, n);
        }
        snprintf(nm, sizeof(nm), "decoder.blocks.%d", 1 + 3 * st);
        tap(nm, cur, dim, n, 0);

        cur = enhance_block(s, st, cur, dim, n);
        snprintf(nm, sizeof(nm), "decoder.blocks.%d", 2 + 3 * st);
        tap(nm, cur, dim, n, 0);

        /* up_layer: 1x1 conv -> CausalUpsample -> ChannelNorm */
        int nxt = (st + 1 < DEC_STAGES) ? DEC_DIMS[st + 1] : DEC_DIMS[DEC_STAGES];
        int S = DEC_STRIDES[st], k = 2 * S + 1;
        float *upc = sa_alloc(&s->A, (size_t)nxt * n);
        char uw[96], ub[96];
        snprintf(uw, sizeof(uw), "decoder.blocks.%d.0.weight", 3 + 3 * st);
        snprintf(ub, sizeof(ub), "decoder.blocks.%d.0.bias",   3 + 3 * st);
        sconv_step_d(&s->upc[st], uw, ub, dim, nxt, 1, 1, 1, 1, cur, n, upc);
        float *up = sa_alloc(&s->A, (size_t)nxt * n * S);
        char sw[96], sb[96];
        snprintf(sw, sizeof(sw), "decoder.blocks.%d.1.smooth.weight", 3 + 3 * st);
        snprintf(sb, sizeof(sb), "decoder.blocks.%d.1.smooth.bias",   3 + 3 * st);
        supsample_step_d(&s->ups[st], sw, sb, nxt, S, k, upc, n, up);
        int nout = n * S;
        char nw[96], nb[96];
        snprintf(nw, sizeof(nw), "decoder.blocks.%d.2.weight", 3 + 3 * st);
        snprintf(nb, sizeof(nb), "decoder.blocks.%d.2.bias",   3 + 3 * st);
        T upt = mk_cv(up, nxt, nout);
        channel_norm_first(&upt, W_(nw), W_(nb), 1, nxt, nout);
        cur = up; n = nout;
        snprintf(nm, sizeof(nm), "decoder.blocks.%d", 3 + 3 * st);
        tap(nm, cur, nxt, n, 0);
    }

    /* ---- blocks[13]: LastBlock ---- */
    int lastC = DEC_DIMS[DEC_STAGES];
    for (int r = 0; r < 3; r++) cur = legacy_residual(s, r, cur, lastC, n);
    float *sn = sa_alloc(&s->A, (size_t)lastC * n); cpy(sn, cur, (size_t)lastC * n);
    T snt = mk_cv(sn, lastC, n);
    snake1d(&snt, W_("decoder.blocks.13.block.1.alpha"), lastC, n, lastC, 0);

    float *out = sa_alloc(&s->A, (size_t)n);
    sconv_step_d(&s->oconv, "decoder.blocks.13.block.2.weight", "decoder.blocks.13.block.2.bias",
                 lastC, 1, CU_KERNEL, 1, 1, 1, sn, n, out);
    for (int i = 0; i < n; i++) wav[i] = tanhf(out[i]);
    tap("decoder.blocks.13", wav, 1, n, 0);
    return n;
}
