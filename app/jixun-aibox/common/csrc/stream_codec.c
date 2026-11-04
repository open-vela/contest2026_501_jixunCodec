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
 *   en_encoder (LocalEncoder)   local_trans depth=1, W=400
 *   quantizer (LearnableFSQ)
 *   en_decoder (LocalDecoder)   local_trans depth=2, W=400
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
#define A_FB_HIDDEN 32 /* structured first-block pruning; A decoder unchanged */
#include "quantizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "jx_amax.h"
static unsigned long a_pair_hits, a_cu_hits, a_qreuse_hits;
static unsigned long d_cu_hits, d_qreuse_hits;
static int d_opt_on(const char *key) {
    const char *value=getenv(key);return !value || value[0]!='0';
}
#include <time.h>
#include <stdint.h>
static int a_detail;
static uint64_t a_us[32];
static const char *a_labels[]={"first_branches","first_pair","dw0","dw1","dw2","dw3",
 "l1_0","l1_1","l1_2","l1_3","l2_0","l2_1","l2_2","l2_3",
 "resadd0","resadd1","resadd2","resadd3","down0","down1","down2",
 "norm0","norm1","norm2","finalconv","attention","quant"};
static inline uint64_t a_tick(void){
 if(!a_detail)return 0;
 struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
 return (uint64_t)t.tv_sec*1000000u+t.tv_nsec/1000;
}
static inline void a_rec(int i,uint64_t begin){if(a_detail)a_us[i]+=a_tick()-begin;}

static int a_opt_on(const char *key) {
    const char *v=getenv(key);
    if(!v && !strcmp(key,"JX_A_REUSE"))return 0; /* measured faster at internal16 */
    return !v || v[0]!='0';
}
static float *a_keep(SArena *a,float *cur,size_t count) {
    if(!a_opt_on("JX_A_REUSE"))return cur;
    if(cur!=a->d)memmove(a->d,cur,count*sizeof(float));
    a->off=count;jx_amax_kill();return a->d;
}

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
/* LoRa 版：SPI3 DMA(300KB)+两份权重后 PSRAM 连续块不足 4MB，先降 3.5MB；
 * 若 sa_alloc 溢出，日志会打印精确 need/off/cap。 */
#define ARENA_FLOATS ((size_t)700 * 1024)
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

    /* 老/新模型自适应：没有 smooth 权重就走线性插值，不要 W_() 打 MISSING。 */
    {
        const float *sw = wt_get(swname, NULL, NULL);
        if (sw == NULL)
            return supsample_step(h, C, stride, k, NULL, NULL, x, n_in, y);
        return supsample_step(h, C, stride, k, sw, sbname ? W_(sbname) : NULL, x, n_in, y);
    }
}

/* ==========================================================================
 * 编码端状态
 * ========================================================================== */
struct SEncState {
    SHist tp[FB_NBR];        /* 5 条支路的 CausalTrendPool 历史 */
    SHist cvh[FB_NBR];       /* 5 条支路的因果卷积历史 (k=7, lp=6) */
    SHist brd[FB_NBR];       /* 支路对齐延迟线 Δ = 47 - d_i */
    SHist xd;                /* 直连 x 的延迟线 (47) */
    SHist c1h, c2h;          /* conv_1(20->32,k=1) / conv_2(33->16,k=1) */
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
#include "d_norm.inc"

typedef struct {
    SHist tp[ENH_NBR], cvh[ENH_NBR], brd[ENH_NBR];
    SHist fn;                /* merge_layer.0 = FrameNorm1d(4) */
    float fn_kernel[FN_WINDOW];
    DFrameNorm fn_fast;
    SHist mc;                /* merge_layer.1 = Conv1d(4->dim, k=1) */
    SHist xd;                /* EnhanceBlock.delay = 23 */
} SEnhState;

struct SDecState {
    int chunk_capacity;
    int norm_mode; /* per reset: scan / double recurrence / hardware float */
    SLocalTrans lt;          /* en_decoder (depth=2) */
    SHist c0;                /* blocks 0 (k=3, lp=2) */
    SHist dw[MAX_DEC_RES], resd[MAX_DEC_RES];
                             /* 各 stage 的 StreamResidual，按 DEC_DEPTHS 线性展开
                              * （1k: [2,2,1,1] -> 6 个；3k/6k: 全 1 -> 4 个） */
    SEnhState enh[4];        /* blocks 2/5/8/11 */
    SHist upc[4];            /* up_layer 0 = 1x1 conv */
    SHist ups[4];            /* up_layer 1 = CausalUpsample */
    SHist ldw[3], lone[3], lrd[3];   /* blocks 13: 3 个 LegacyUnit */
    SHist oconv;             /* blocks 13.block.2 (k=7, lp=6) */
    float *ls0, *ls1, *ls2;  /* blocks 13 三个 LegacyUnit 复用的 scratch */
    size_t ls_cap;
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
    shist_reset(&s->c2h, A_FB_HIDDEN+1, 0, 4);
    for (int i = 0; i < ENC_STAGES + 1; i++) {
        shist_reset(&s->dw[i], ENC_DIMS[i], 6, 24);
        shist_reset(&s->resd[i], ENC_DIMS[i], CU_DELAY, 16);
    }
    for (int i = 0; i < ENC_STAGES; i++) shist_reset(&s->down[i], ENC_DIMS[i], 0, 4);
    shist_reset(&s->fin, ENC_DIMS[ENC_STAGES], 2, 8);
    slt_reset(&s->lt);
}

SEncState *senc_new(void) {
    extern void jx_maxalign_config(int);
    extern int jx_maxalign_selftest(void);
    const char *align=getenv("JX_A_MAXALIGN");jx_maxalign_config(!align||align[0]!='0');
    const char *check=getenv("JX_A_MAXCHECK");if(check&&check[0]=='1'&&jx_maxalign_selftest())abort();
    a_pair_hits=a_cu_hits=a_qreuse_hits=0;
    const char *detail=getenv("JX_A_DETAIL");a_detail=detail&&detail[0]=='1';memset(a_us,0,sizeof(a_us));
    SEncState *s = (SEncState*)calloc(1, sizeof(SEncState));
    if (!s) { fprintf(stderr, "senc_new: OOM\n"); exit(1); }
    /* 先分配 slt 的小块（dpb 约 6KB），再分配大 arena：
     * 否则 3MB arena 会把 PSRAM 大块拿走，剩余碎片连 6KB 都拿不到。 */
    slt_init(&s->lt, "en_encoder.local_trans", 1, EN_WINDOW);
    /* Measured baseline16 <=2.26MB; reuse32 <=2.17MB. Keep >=0.19MB margin. */
    sa_init(&s->A, (size_t)600*1024);
    senc_reset(s);
    return s;
}

void senc_free(SEncState *s) {
    if (!s) return;
    if(a_detail)for(int i=0;i<27;i++)printf("[A-detail] %s us=%llu\n",a_labels[i],(unsigned long long)a_us[i]);
    printf("[A-opt] pair=%lu residual=%lu qreuse=%lu\n",a_pair_hits,a_cu_hits,a_qreuse_hits);
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
    s->norm_mode=d_opt_on("JX_D_NORMFAST")?(d_opt_on("JX_D_NORMFLOAT")?2:1):0;
    static const int pk[ENH_NBR] = ENH_POOL;
    static const int dl[ENH_NBR] = ENH_DIL;
    static const int bd[ENH_NBR] = ENH_DELAY;
    static const int lgd[3] = LG_DIL;
    /* 流式一步的 token 数（与 port/jx_lora.c 的 SXRX_TPP 一致）。
     * ESP32 上 PSRAM 很紧：shist_ensure 扩容时要同时持有新旧两块，
     * 正好是最容易 OOM 的时刻。这里按"该步长下各 stage 的最大帧数"
     * 一次性把容量给够，之后运行期不再扩容。 */
    const int nstep = s->chunk_capacity;
    int nst[DEC_STAGES + 1];
    {
        int nn = nstep;
        for (int st = 0; st < DEC_STAGES; st++) { nst[st] = nn; nn *= DEC_STRIDES[st]; }
        nst[DEC_STAGES] = nn;      /* 最后一块的帧数（3k/16token -> 1536） */
    }
    shist_reset(&s->c0, FEATURE_DIM, 2, 2 + nst[0]);
    {
        /* 按 DEC_DEPTHS 展开：第 st 个 stage 有 DEC_DEPTHS[st] 个 StreamResidual，
         * 通道数都是 DEC_DIMS[st]（up_layer 在 stage 之后才改变通道） */
        int idx = 0;
        for (int st = 0; st < DEC_STAGES; st++) {
            for (int j = 0; j < DEC_DEPTHS[st]; j++) {
                shist_reset(&s->dw[idx], DEC_DIMS[st], 6, 6 + nst[st]);
                shist_reset(&s->resd[idx], DEC_DIMS[st], CU_DELAY, d_opt_on("JX_D_DADD") ? CU_DELAY : CU_DELAY + nst[st] + 8);
                idx++;
            }
        }
    }
    for (int st = 0; st < 4; st++) {
        for (int br = 0; br < ENH_NBR; br++) {
            int H = (pk[br] <= 1) ? 1 : 2 * (pk[br] - 1);
            shist_reset(&s->enh[st].tp[br], 1, 0, H + nst[st] + 8);
            shist_reset(&s->enh[st].cvh[br], 1, 6 * dl[br], 6 * dl[br] + nst[st] + 8);
            int d = ENH_MAXDELAY - bd[br];
            shist_reset(&s->enh[st].brd[br], 1, d, (d > 0 ? d : 1) + nst[st] + 8);
        }
        /* Same coefficients and accumulation order as the reference, once per reset. */
        {
            double total=0;
            for(int j=0;j<FN_WINDOW;j++) {
                double value=(double)FN_ALPHA*pow(1.0-(double)FN_ALPHA,(double)(FN_WINDOW-1-j));
                s->enh[st].fn_kernel[j]=(float)value;total+=value;
            }
            for(int j=0;j<FN_WINDOW;j++)s->enh[st].fn_kernel[j]=(float)((double)s->enh[st].fn_kernel[j]/total);
        }
        memset(&s->enh[st].fn_fast,0,sizeof(s->enh[st].fn_fast));
        shist_reset(&s->enh[st].fn, 4, FN_WINDOW - 1, FN_WINDOW - 1 + nst[st] + 8);
        shist_reset(&s->enh[st].mc, 4, 0, 4);
        shist_reset(&s->enh[st].xd, DEC_DIMS[st], ENH_MAXDELAY,
                    d_opt_on("JX_D_MERGE") ? ENH_MAXDELAY : ENH_MAXDELAY + nst[st] + 8);
        int nxt = (st + 1 < DEC_STAGES) ? DEC_DIMS[st + 1] : DEC_DIMS[DEC_STAGES];
        shist_reset(&s->upc[st], DEC_DIMS[st], 0, 1);
        int S = DEC_STRIDES[st], k = 2 * S + 1;
        shist_reset(&s->ups[st], nxt, (k - 1 + S - 1) / S,
                    (k - 1 + S - 1) / S + nst[st] + 8);
    }
    for (int r = 0; r < 3; r++) {
        shist_reset(&s->ldw[r], DEC_DIMS[DEC_STAGES], 6 * lgd[r],
                    6 * lgd[r] + nst[DEC_STAGES]);
        shist_reset(&s->lone[r], DEC_DIMS[DEC_STAGES], 0, 1);
        shist_reset(&s->lrd[r], DEC_DIMS[DEC_STAGES], 3 * lgd[r],
                    d_opt_on("JX_D_MERGE") ? 3 * lgd[r] : 3 * lgd[r] + nst[DEC_STAGES] + 8);
    }
    shist_reset(&s->oconv, DEC_DIMS[DEC_STAGES], 6, 6 + nst[DEC_STAGES]);

    /* blocks 13 复用 scratch：3 个张量够（a: s0/s1, b: c1/c2, c: base/out） */
    {
        size_t need = (size_t)DEC_DIMS[DEC_STAGES] * (size_t)nst[DEC_STAGES];
        free(s->ls0); free(s->ls1); free(s->ls2);
        s->ls_cap = 0;
        s->ls0 = (float*)malloc(need * sizeof(float));
        s->ls1 = (float*)malloc(need * sizeof(float));
        s->ls2 = (float*)malloc(need * sizeof(float));
        if (!s->ls0 || !s->ls1 || !s->ls2)
          {
            fprintf(stderr, "sdec_reset: OOM scratch need=%zu\n", need);
            exit(1);
          }
        s->ls_cap = need;
    }
    slt_reset(&s->lt);
}

SDecState *sdec_new_chunk(int max_tokens) {
    SDecState *s = (SDecState*)calloc(1, sizeof(SDecState));
    if (!s) { fprintf(stderr, "sdec_new: OOM\n"); exit(1); }
    s->chunk_capacity=(max_tokens>0 && max_tokens<=16)?max_tokens:16;
    slt_init(&s->lt, "en_decoder.local_trans", EN_DEC_DEPTH, EN_WINDOW);
    /* Reserve only W+chunk rows up front. Runtime doubling to256 both wastes
     * memory and briefly holds old/new KV buffers during constrained receive. */
    for(int i=0;i<s->lt.depth;i++) {
        int cap=s->lt.W+s->chunk_capacity;
        s->lt.kvk[i].d=(float *)malloc((size_t)s->lt.HD*cap*sizeof(float));
        s->lt.kvv[i].d=(float *)malloc((size_t)s->lt.HD*cap*sizeof(float));
        if(!s->lt.kvk[i].d || !s->lt.kvv[i].d){fprintf(stderr,"sdec_new: OOM KV reserve\n");exit(1);}
        s->lt.kvk[i].cap=s->lt.kvv[i].cap=cap;
    }
    size_t capacity=ARENA_FLOATS;
#if !defined(SQM_RATE_1K)
    #if defined(SQM_RATE_3K)
    /* Measured full-fallback16 <=2.41MB, optimized16 1.98MB. Leave space
     * for receive/play queues; reserve no unused 2.87MB arena. */
    if(max_tokens>0 && max_tokens<=16)capacity=(size_t)600*1024;
    if(max_tokens>0 && max_tokens<=8)capacity=(size_t)320*1024;
#else
    if(max_tokens>0 && max_tokens<=8)capacity=(size_t)460*1024;
#endif
#endif
    sa_init(&s->A, capacity);
    sdec_reset(s);                  /* arena 之后一次性拿历史缓存，避免运行时扩容 */
    return s;
}

SDecState *sdec_new(void) { return sdec_new_chunk(16); }

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
    free(s->ls0); free(s->ls1); free(s->ls2); s->ls0 = s->ls1 = s->ls2 = NULL; s->ls_cap = 0;
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

    uint64_t branch_begin=a_tick();
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
    float *raw=sa_alloc(A,(size_t)n);
    sdelay_step(&s->xd,FB_MAXDELAY,1,x,n,raw);
    a_rec(0,branch_begin);
    if(a_opt_on("JX_A_PAIR")) {
        float *paired=sa_alloc(A,(size_t)ENC_DIMS[0]*n);
        T ht=mk_cv(h,20,n),rt=mk_cv(raw,1,n),yt=mk_cv(paired,ENC_DIMS[0],n);
        if(jx_first_pair_i8(W_("encoder.blocks.0.conv_1.weight"),W_("encoder.blocks.0.conv_1.bias"),
                           W_("encoder.blocks.0.conv_2.weight"),W_("encoder.blocks.0.conv_2.bias"),
                           &ht,&rt,&yt,A_FB_HIDDEN)==0){a_pair_hits++;return paired;}
        A->off-=(size_t)ENC_DIMS[0]*n;
    }
    /* conv_1: 20 -> 32 (1x1, 天然因果) */
    float *h1 = sa_alloc(A, (size_t)A_FB_HIDDEN * n);
    sconv_step_d(&s->c1h, "encoder.blocks.0.conv_1.weight", "encoder.blocks.0.conv_1.bias",
                 20, A_FB_HIDDEN, 1, 1, 1, 1, h, n, h1);
    for (size_t i = 0; i < (size_t)A_FB_HIDDEN * n; i++) h1[i] = gelu_f(h1[i]);
    /* cat([h1, causal_delay(x, 47)]) -> 33ch */
    float *hc = sa_alloc(A, (size_t)(A_FB_HIDDEN+1) * n);
    cpy(hc, h1, (size_t)A_FB_HIDDEN * n);
    cpy(&hc[(size_t)A_FB_HIDDEN*n],raw,(size_t)n);
    float *out = sa_alloc(A, (size_t)ENC_DIMS[0] * n);
    sconv_step_d(&s->c2h, "encoder.blocks.0.conv_2.weight", "encoder.blocks.0.conv_2.bias",
                 A_FB_HIDDEN+1, ENC_DIMS[0], 1, 1, 1, 1, hc, n, out);
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
#include "a_fused.inc"
#include "d_fused.inc"
int senc_step(SEncState *s, const float *x, int n, int64_t *tokens) {
    sa_reset(&s->A);
    int frames = n / HOP_LENGTH;

    uint64_t first_begin=a_tick();
    float *cur = enc_first_block(s, x, n);
    a_rec(1,first_begin);
    tap("encoder.blocks.0", cur, ENC_DIMS[0], n, 0);
    cur=a_keep(&s->A,cur,(size_t)ENC_DIMS[0]*n);
    int cur_n = n;

    for (int st = 0; st < ENC_STAGES; st++) {
        char nm[64], pn[96];
        snprintf(pn, sizeof(pn), "encoder.blocks.%d.0.module", 2 * st + 1);
        cur = a_convunit_residual(&s->A, &s->dw[st], &s->resd[st], pn, cur, ENC_DIMS[st], cur_n);
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
        uint64_t down_begin=a_tick();
        sconv_step_d(&s->down[st], w1, b1, ENC_DIMS[st], Co, stride, stride, 1, 1, cur, cur_n, dn);
        a_rec(18+st,down_begin);
        uint64_t norm_begin=a_tick();
        T dnt = mk_cv(dn, Co, nout);
        channel_norm_first(&dnt, W_(n1), W_(c1), 1, Co, nout);
        a_rec(21+st,norm_begin);
        cur = dn; cur_n = nout;
        snprintf(nm, sizeof(nm), "encoder.blocks.%d", 2 * st + 2);
        tap(nm, cur, Co, cur_n, 0);
        cur=a_keep(&s->A,cur,(size_t)Co*cur_n);
    }

    /* blocks[2*ENC_STAGES+1]: 末尾 Stage（3k/6k = blocks[7]，1k = blocks[9]）
     * 注：所有现有配置的 ENC_DEPTHS 均为 1，故这里只取 1 个残差单元。 */
    int fblk = 2 * ENC_STAGES + 1;
    char fpn[64], fnm[64];
    snprintf(fpn, sizeof(fpn), "encoder.blocks.%d.0.module", fblk);
    snprintf(fnm, sizeof(fnm), "encoder.blocks.%d", fblk);
    cur = a_convunit_residual(&s->A, &s->dw[ENC_STAGES], &s->resd[ENC_STAGES], fpn,
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
    uint64_t final_begin=a_tick();
    sconv_step_d(&s->fin, fw, fb, ENC_DIMS[ENC_STAGES], FEATURE_DIM, 3, 1, 1, 1, cur, cur_n, fin);
    a_rec(24,final_begin);
    tap(fnm2, fin, FEATURE_DIM, cur_n, 0);
    fin=a_keep(&s->A,fin,(size_t)FEATURE_DIM*cur_n);

    /* en_encoder: CV -> TC -> LocalTrans(depth=1) */
    float *tc = sa_alloc(&s->A, (size_t)cur_n * FEATURE_DIM);
    for (int t = 0; t < cur_n; t++)
        for (int c = 0; c < FEATURE_DIM; c++) tc[(size_t)t * FEATURE_DIM + c] = fin[(size_t)c * cur_n + t];
    float *eo = sa_alloc(&s->A, (size_t)cur_n * FEATURE_DIM);
    uint64_t attn_begin=a_tick();
    slt_forward(&s->lt, tc, cur_n, eo);
    a_rec(25,attn_begin);
    tap("en_encoder", eo, FEATURE_DIM, cur_n, 1);

    /* quantizer: 产出真实传输载荷 indices */
    T qt = mk_tc(eo, cur_n, FEATURE_DIM);
    T qo = mk_tc(sa_alloc(&s->A, (size_t)cur_n * FEATURE_DIM), cur_n, FEATURE_DIM);
    uint64_t quant_begin=a_tick();
    quantizer_forward(&qt, &qo, tokens, cur_n);
    a_rec(26,quant_begin);
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
    if(s->norm_mode) {
        (s->norm_mode==2?d_framenorm_float:d_framenorm_step)(&E->fn_fast,E->fn_kernel,
                        W_(cat_name(pfx,"merge_layer.0.weight")),
                        W_(cat_name(pfx,"merge_layer.0.bias")),h,n,fm);
    } else {
    sframenorm_step_cached(&E->fn, 4, FN_WINDOW, FN_ALPHA,
                    W_(cat_name(pfx, "merge_layer.0.weight")), W_(cat_name(pfx, "merge_layer.0.bias")),
                    FN_EPS, h, n, fm, d_opt_on("JX_D_NORMCACHE") ? E->fn_kernel : NULL);
    }
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
    extern int sdelay_madd_step(SHist *,int,int,const float *,int,const float *,float *);
    if(d_opt_on("JX_D_MERGE")) {
        float *out=sa_alloc(A,(size_t)dim*n);
        if(sdelay_madd_step(&E->xd,ENH_MAXDELAY,dim,cur,n,mg,out))return out;
    }
    float *xd = sa_alloc(A, (size_t)dim * n);
    sdelay_step(&E->xd, ENH_MAXDELAY, dim, cur, n, xd);
    float *out = sa_alloc(A, (size_t)dim * n);
    for (size_t i = 0; i < (size_t)dim * n; i++) out[i] = xd[i] + mg[i] * xd[i];
    return out;
}

static float *legacy_residual(SDecState *s, int r, const float *cur, int C, int n) {
    static const int lgd[3] = LG_DIL;
    int dil = lgd[r];
    char pfx[96]; snprintf(pfx, sizeof(pfx), "decoder.blocks.13.block.0.%d.module", r);
    size_t cn = (size_t)C * (size_t)n;

    if (cn > s->ls_cap) {
        fprintf(stderr, "legacy_residual: scratch 不足 need=%zu cap=%zu\n", cn, s->ls_cap);
        exit(1);
    }

    /* 三个 LegacyUnit 串行执行，上一轮的中间结果不再需要；
     * 复用同一组 scratch，避免每个 unit 在 arena 里留 6 份大张量
     * （3 个 unit 曾占 ~1.2MB，是 sdec_step arena 峰值的大头）。 */
    /* cur 可能等于上一轮返回的 ls2，必须把输入快照留在 d，
     * 中间计算只用 a/b，否则 sdelay_step 读到的输入已被覆盖。 */
    float *d = s->ls2;   /* 输入快照 / 最终输出 */
    float *a = s->ls0;   /* scratch */
    float *b = s->ls1;   /* scratch */

    if (cur != d) cpy(d, cur, cn);
    if(d_opt_on("JX_D_LEGACY")) {
        T source=mk_cv(d,C,n), first=mk_cv(a,C,n);
        snake1d_copy(&first,&source,W_(cat_name(pfx,"block.0.alpha")));
        if(g_stap)tap(cat_name(pfx,".snake0"),a,C,n,0);
        const float *alpha=W_(cat_name(pfx,"block.2.alpha"));
        jx_conv_post_used=0;jx_post_snake_prep(alpha,C);
        jx_conv_post_alpha=alpha;jx_conv_post=2;
        sconv_step_d(&s->ldw[r],cat_name(pfx,"block.1.weight"),cat_name(pfx,"block.1.bias"),
                    C,C,CU_KERNEL,1,dil,1,a,n,b);
        if(g_stap)tap(cat_name(pfx,".conv7_snake"),b,C,n,0);
        jx_conv_post=0;jx_conv_post_alpha=NULL;
        if(jx_conv_post_used!=2) {
            T middle=mk_cv(b,C,n);snake1d(&middle,alpha,C,n,C,0);
        }
        sconv_step_d(&s->lone[r],cat_name(pfx,"block.3.weight"),cat_name(pfx,"block.3.bias"),
                    C,C,1,1,1,1,b,n,a);
        if(g_stap)tap(cat_name(pfx,".conv1"),a,C,n,0);
        if(d_opt_on("JX_D_MERGE") && sdelay_add_step(&s->lrd[r],CU_DELAY*dil,C,d,n,a,a))return a;
        sdelay_step(&s->lrd[r],CU_DELAY*dil,C,d,n,b);
        for(size_t i=0;i<cn;i++)d[i]=b[i]+a[i];
        return d;
    }
    cpy(a, d, cn);
    T s0t = mk_cv(a, C, n);
    snake1d(&s0t, W_(cat_name(pfx, "block.0.alpha")), C, n, C, 0);
    sconv_step_d(&s->ldw[r], cat_name(pfx, "block.1.weight"), cat_name(pfx, "block.1.bias"),
                 C, C, CU_KERNEL, 1, dil, 1, a, n, b);

    cpy(a, b, cn);
    T s1t = mk_cv(a, C, n);
    snake1d(&s1t, W_(cat_name(pfx, "block.2.alpha")), C, n, C, 0);
    sconv_step_d(&s->lone[r], cat_name(pfx, "block.3.weight"), cat_name(pfx, "block.3.bias"),
                 C, C, 1, 1, 1, 1, a, n, b);

    /* a 已无用，用来接 base；d 保持输入快照不动 */
    sdelay_step(&s->lrd[r], CU_DELAY * dil, C, d, n, a);
    for (size_t i = 0; i < cn; i++) d[i] = a[i] + b[i];
    return d;
}

int sdec_step(SDecState *s, const int64_t *tokens, int nt, float *wav) {
    sa_reset(&s->A);
    /* ---- quantizer 反量化 -> features ---- */
    float *qf = sa_alloc(&s->A, (size_t)nt * FEATURE_DIM);
    T qft = mk_tc(qf, nt, FEATURE_DIM);
    quantizer_to_features(tokens, &qft, nt);

    /* ---- en_decoder: LocalTrans(depth=2)，输出 TC -> CV ---- */
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
            cur = d_convunit_residual(&s->A, &s->dw[rbase + j], &s->resd[rbase + j], pn, cur, dim, n);
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
        if(g_stap)tap(cat_name(uw,".done"),upc,nxt,n,0);
        float *up = sa_alloc(&s->A, (size_t)nxt * n * S);
        char sw[96], sb[96];
        snprintf(sw, sizeof(sw), "decoder.blocks.%d.1.smooth.weight", 3 + 3 * st);
        snprintf(sb, sizeof(sb), "decoder.blocks.%d.1.smooth.bias",   3 + 3 * st);
        supsample_step_d(&s->ups[st], sw, sb, nxt, S, k, upc, n, up);
        if(g_stap)tap(cat_name(sw,".done"),up,nxt,n*S,0);
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
    for (int r = 0; r < 3; r++) {
        cur = legacy_residual(s, r, cur, lastC, n);
        char label[64];snprintf(label,sizeof(label),"decoder.legacy.%d",r);
        tap(label,cur,lastC,n,0);
    }
    float *sn = cur; /* Final activation consumes the dead residual output. */
    T snt = mk_cv(sn, lastC, n);
    snake1d(&snt, W_("decoder.blocks.13.block.1.alpha"), lastC, n, lastC, 0);

    float *out = sa_alloc(&s->A, (size_t)n);
    sconv_step_d(&s->oconv, "decoder.blocks.13.block.2.weight", "decoder.blocks.13.block.2.bias",
                 lastC, 1, CU_KERNEL, 1, 1, 1, sn, n, out);
    for (int i = 0; i < n; i++) wav[i] = tanhf(out[i]);
    tap("decoder.blocks.13", wav, 1, n, 0);
    return n;
}
