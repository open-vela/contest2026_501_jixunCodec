/* ============================================================================
 * stream_trans.c — LocalTrans / LocalMHA / FeedForward 的增量实现。
 * 语义对齐 l3ac/stream_cache.py 的 _step_localtrans / _step_mha。
 *
 * KV cache 布局：行 = key 的时间下标 j，行内索引 h*dh+d。
 * cache 保存"位置编码之前"的尾部 W 行（与 stream_cache 的处理一致）。
 * ========================================================================== */
#include "stream_trans.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* --------------------------- 小工具 --------------------------- */
/* TC [rows][in] 上的 Linear：y = x @ Wᵀ（复用 nn_ops.linear 的 double 累加） */
static void lin_tc(const float *W, const float *b, const float *x,
                   int rows, int in, float *y, int out) {
    T xt; xt.d = (float*)x; xt.ndim = 3;
    xt.shape[0] = 1; xt.shape[1] = rows; xt.shape[2] = in; xt.len = rows * in;
    T yt; yt.d = y; yt.ndim = 3;
    yt.shape[0] = 1; yt.shape[1] = rows; yt.shape[2] = out; yt.len = rows * out;
    linear(W, b, &xt, &yt, in, out);
}
/* weight-only INT8 版：y = scale @ (int8_W @ xᵀ) + b（double 累加） */
static void lin_tc_q(const int8_t *Wi, const float *sc, const float *b, const float *x,
                     int rows, int in, float *y, int out) {
    T xt; xt.d = (float*)x; xt.ndim = 3;
    xt.shape[0] = 1; xt.shape[1] = rows; xt.shape[2] = in; xt.len = rows * in;
    T yt; yt.d = y; yt.ndim = 3;
    yt.shape[0] = 1; yt.shape[1] = rows; yt.shape[2] = out; yt.len = rows * out;
    linear_q(Wi, sc, b, &xt, &yt, in, out);
}

/* 三个 scratch 是同容量池：扩容会 realloc 全部三个缓冲，因此必须"先一次性
 * 预留到本轮最大需求、再取指针"。否则先取到的指针会被后一次扩容变成悬垂指针
 * （en_encoder 上 K < n*inner 时必然踩中：sim/attn 被 o2 的扩容 realloc 掉）。 */
static void scr_reserve(SLocalTrans *t, int need) {
    if (need <= t->scr_cap) return;
    int ncap = t->scr_cap > 0 ? t->scr_cap : 1024;
    while (ncap < need) ncap *= 2;
    t->scr1 = (float*)realloc(t->scr1, (size_t)ncap * sizeof(float));
    t->scr2 = (float*)realloc(t->scr2, (size_t)ncap * sizeof(float));
    t->scr3 = (float*)realloc(t->scr3, (size_t)ncap * sizeof(float));
    if (!t->scr1 || !t->scr2 || !t->scr3) { fprintf(stderr, "scr_reserve: OOM\n"); exit(1); }
    t->scr_cap = ncap;
}

/* DynamicPositionBias.mlp：Linear(1,H)->SiLU->Linear(H,H)->SiLU->Linear(H,heads)
 * weight-only INT8 版：w0/w1/w2 为 int8 权重，sc0/sc1/sc2 为对应每输出通道 scale，
 * 偏置 b0/b1/b2 仍为 float（PTQ 不量化 bias）。 */
static void dpb_fill_q(float *tab, int W, int heads, int H,
                       const int8_t *w0, const float *sc0, const float *b0,
                       const int8_t *w1, const float *sc1, const float *b1,
                       const int8_t *w2, const float *sc2, const float *b2) {
    float h1[512], h2[512];
    for (int d = 0; d <= W; d++) {
        for (int i = 0; i < H; i++) {
            float a = b0[i] + (float)sc0[i] * (float)w0[i] * (float)d;
            h1[i] = a / (1.f + expf(-a));
        }
        for (int i = 0; i < H; i++) {
            float a = b1[i];
            for (int j = 0; j < H; j++) a += (float)sc1[i] * (float)w1[(size_t)i * H + j] * h1[j];
            h2[i] = a / (1.f + expf(-a));
        }
        for (int hd = 0; hd < heads; hd++) {
            float a = b2[hd];
            for (int j = 0; j < H; j++) a += (float)sc2[hd] * (float)w2[(size_t)hd * H + j] * h2[j];
            tab[(size_t)d * heads + hd] = a;
        }
    }
}

/* 与 dpb_fill_q 对应的 float 版（无 int8 时回退）：w0/w1/w2 为 float 权重，
 * 偏置 b0/b1/b2 仍为 float。DPB 的距离表对 d=0..W 预计算并查表。 */
static void dpb_fill(float *tab, int W, int heads, int H,
                     const float *w0, const float *b0,
                     const float *w1, const float *b1,
                     const float *w2, const float *b2) {
    float h1[512], h2[512];
    for (int d = 0; d <= W; d++) {
        for (int i = 0; i < H; i++) {
            float a = b0[i] + w0[i] * (float)d;
            h1[i] = a / (1.f + expf(-a));
        }
        for (int i = 0; i < H; i++) {
            float a = b1[i];
            for (int j = 0; j < H; j++) a += w1[(size_t)i * H + j] * h1[j];
            h2[i] = a / (1.f + expf(-a));
        }
        for (int hd = 0; hd < heads; hd++) {
            float a = b2[hd];
            for (int j = 0; j < H; j++) a += w2[(size_t)hd * H + j] * h2[j];
            tab[(size_t)d * heads + hd] = a;
        }
    }
}

/* ============================ 生命周期 ============================ */
void slt_free(SLocalTrans *t) {
    for (int i = 0; i < 8; i++) { skv_free(&t->kvk[i]); skv_free(&t->kvv[i]); }
    if (t->dpb) { free(t->dpb); t->dpb = NULL; }
    if (t->scr1) { free(t->scr1); t->scr1 = NULL; }
    if (t->scr2) { free(t->scr2); t->scr2 = NULL; }
    if (t->scr3) { free(t->scr3); t->scr3 = NULL; }
    t->scr_cap = 0;
}

void slt_reset(SLocalTrans *t) {
    for (int i = 0; i < t->depth; i++) { t->kvk[i].len = 0; t->kvv[i].len = 0; }
}

void slt_init(SLocalTrans *t, const char *pfx, int depth, int W) {
    memset(t, 0, sizeof(*t));
    t->depth = depth; t->W = W;
    t->scr_cap = 0; t->scr1 = t->scr2 = t->scr3 = NULL;

    int shp[4], n_;
    const char *nm;

    nm = cat_name(pfx, "layers.0.0.norm.weight");
    wt_get(nm, shp, &n_); t->C = shp[0];

    nm = cat_name(pfx, "layers.0.0.to_qkv.weight");
    wt_get(nm, shp, &n_); t->inner = shp[0] / 3;

    nm = cat_name(pfx, "dynamic_pos_bias.mlp.4.weight");
    wt_get(nm, shp, &n_); t->heads = shp[0];

    nm = cat_name(pfx, "dynamic_pos_bias.mlp.0.weight");
    wt_get(nm, shp, &n_); int H = shp[0];

    nm = cat_name(pfx, "layers.0.1.1.weight");
    wt_get(nm, shp, &n_); t->hid = shp[0]; t->half = t->hid / 2;

    t->dh = t->inner / t->heads;
    t->HD = t->heads * t->dh;

    /* 逐层权重指针 */
    for (int i = 0; i < depth; i++) {
        char b1[200], b2[200];
        snprintf(b1, sizeof(b1), "%s.layers.%d.0", pfx, i);
        snprintf(b2, sizeof(b2), "%s.layers.%d.1", pfx, i);
        t->nw[i]  = W_(cat_name(b1, "norm.weight"));
        t->nb[i]  = W_(cat_name(b1, "norm.bias"));
        t->wqkv[i]= W_(cat_name(b1, "to_qkv.weight"));
        t->wo[i]  = W_(cat_name(b1, "to_out.weight"));
        t->fnw[i] = W_(cat_name(b2, "0.weight"));
        t->fnb[i] = W_(cat_name(b2, "0.bias"));
        t->fw1[i] = W_(cat_name(b2, "1.weight"));
        t->fw2[i] = W_(cat_name(b2, "4.weight"));
        /* weight-only INT8 变体（量化 Linear）；NULL => 回退 float W_ */
        t->wqkv_q[i] = wt_get_i8(cat_name(b1, "to_qkv.weight"), &t->wqkv_sc[i], NULL, NULL, NULL);
        t->wo_q[i]   = wt_get_i8(cat_name(b1, "to_out.weight"), &t->wo_sc[i], NULL, NULL, NULL);
        t->fw1_q[i]  = wt_get_i8(cat_name(b2, "1.weight"), &t->fw1_sc[i], NULL, NULL, NULL);
        t->fw2_q[i]  = wt_get_i8(cat_name(b2, "4.weight"), &t->fw2_sc[i], NULL, NULL, NULL);
    }

    /* 距离表 d = 0..W（int8 权重 + 每通道 scale） */
    t->dpb = (float*)malloc((size_t)(W + 1) * t->heads * sizeof(float));
    if (!t->dpb) { fprintf(stderr, "slt_init: OOM dpb\n"); exit(1); }
    const float *sc0 = NULL, *sc1 = NULL, *sc2 = NULL;
    t->dpb_w0_q = wt_get_i8(cat_name(pfx, "dynamic_pos_bias.mlp.0.weight"), &sc0, NULL, NULL, NULL);
    const float *db0 = W_(cat_name(pfx, "dynamic_pos_bias.mlp.0.bias"));
    t->dpb_w1_q = wt_get_i8(cat_name(pfx, "dynamic_pos_bias.mlp.2.weight"), &sc1, NULL, NULL, NULL);
    const float *db1 = W_(cat_name(pfx, "dynamic_pos_bias.mlp.2.bias"));
    t->dpb_w2_q = wt_get_i8(cat_name(pfx, "dynamic_pos_bias.mlp.4.weight"), &sc2, NULL, NULL, NULL);
    const float *db2 = W_(cat_name(pfx, "dynamic_pos_bias.mlp.4.bias"));
    t->dpb_w0_sc = sc0; t->dpb_w1_sc = sc1; t->dpb_w2_sc = sc2;
    if (t->dpb_w0_q)
        dpb_fill_q(t->dpb, W, t->heads, H,
                   t->dpb_w0_q, sc0, db0, t->dpb_w1_q, sc1, db1, t->dpb_w2_q, sc2, db2);
    else
        /* 无 int8（纯 float 头 / 对照实验）：用 float 权重填充 DPB 距离表 */
        dpb_fill(t->dpb, W, t->heads, H,
                 W_(cat_name(pfx, "dynamic_pos_bias.mlp.0.weight")), db0,
                 W_(cat_name(pfx, "dynamic_pos_bias.mlp.2.weight")), db1,
                 W_(cat_name(pfx, "dynamic_pos_bias.mlp.4.weight")), db2);

    for (int i = 0; i < depth; i++) { t->kvk[i].d = t->kvv[i].d = NULL; }
}

/* ============================ 单层 LocalMHA ============================ */
/* x/out: TC [n][C] */
static void mha_step(SLocalTrans *t, int L, const float *x, int n, float *out) {
    const int C = t->C, Hh = t->heads, dh = t->dh, inner = t->inner, W = t->W, HD = t->HD;
    const float scale = (float)(1.0f / sqrtf((float)dh));

    /* 1) prenorm (nn.LayerNorm, eps=1e-5) */
    float *xn = (float*)malloc((size_t)n * C * sizeof(float));
    if (!xn) { fprintf(stderr, "mha_step: OOM xn\n"); exit(1); }
    memcpy(xn, x, (size_t)n * C * sizeof(float));
    layernorm_tc(xn, n, C, t->nw[L], t->nb[L], 1e-5f);

    /* 2) qkv 投影 */
    float *qkv = (float*)malloc((size_t)n * 3 * inner * sizeof(float));
    if (!qkv) { fprintf(stderr, "mha_step: OOM qkv\n"); exit(1); }
    if (t->wqkv_q[L]) lin_tc_q(t->wqkv_q[L], t->wqkv_sc[L], NULL, xn, n, C, qkv, 3 * inner);
    else              lin_tc(t->wqkv[L], NULL, xn, n, C, qkv, 3 * inner);
    free(xn);

    /* 3) KV cache：本轮新 key/value 追加到 [old, old+n) */
    SKV *ck = &t->kvk[L], *cv = &t->kvv[L];
    int old = ck->len;
    int K = old + n;
    skv_ensure(ck, HD, K); skv_ensure(cv, HD, K);
    for (int i = 0; i < n; i++) {
        const float *row = &qkv[(size_t)i * 3 * inner];
        float *kd = &ck->d[(size_t)(old + i) * HD];
        float *vd = &cv->d[(size_t)(old + i) * HD];
        for (int h = 0; h < Hh; h++) {
            memcpy(kd + h * dh, row + inner + h * dh, (size_t)dh * sizeof(float));
            memcpy(vd + h * dh, row + 2 * inner + h * dh, (size_t)dh * sizeof(float));
        }
    }

    /* 4) 注意力（flash 风格：KV 按块载入片上，一批 query 复用，每个 KV 元素只读一次）
     *
     *    tiling 顺序 = KV-block(外层) -> head -> query(内层)。
     *    每个 KV 块先整体从 cache(ck/cv) 搬入片上缓冲 Kblk/Vblk 一次，
     *    再被本块覆盖的 n 个 query × Hh 个 head 全部复用；cache 中每个
     *    KV 元素在整个 step 内只被搬运一次（原始实现为 n×Hh×K 次）。
     *    归一化用 online-softmax：每个 (query,head) 维护运行最大值 m、
     *    运行分母 l、运行输出累加 o2，逐块增量更新；处理完所有可见 key
     *    后收敛到与两遍 softmax 等价的结果（仅浮点求和顺序不同）。
     *
     *    Bk 为 KV tile 大小，可按目标片上 SRAM 预算调小；调小不改变 KV
     *    读取总次数（每个元素仍只读一次），只改片上缓冲占用与分块开销。 */
    int Bk = 64;                                   /* KV tile（ESP32 可降到 16~32 以适配 SRAM） */
    float *Kblk = (float*)malloc((size_t)Bk * HD * sizeof(float));
    float *Vblk = (float*)malloc((size_t)Bk * HD * sizeof(float));
    float *sc   = (float*)malloc((size_t)Bk * sizeof(float));   /* 块内临时 score */
    float *m    = (float*)malloc((size_t)n * Hh * sizeof(float));
    float *l    = (float*)malloc((size_t)n * Hh * sizeof(float));
    float *o2   = (float*)malloc((size_t)n * inner * sizeof(float));
    if (!Kblk || !Vblk || !sc || !m || !l || !o2) { fprintf(stderr, "mha_step: OOM attn\n"); exit(1); }
    for (int q = 0; q < n; q++)
        for (int h = 0; h < Hh; h++) { m[q * Hh + h] = -1e30f; l[q * Hh + h] = 0.0f; }
    memset(o2, 0, (size_t)n * inner * sizeof(float));

    int j_lo = (old > W) ? (old - W) : 0;          /* 窗口内最早可见 key（更早的不可能命中） */
    for (int jb = j_lo; jb < K; jb += Bk) {
        int Bj = (jb + Bk < K) ? Bk : (K - jb);
        /* 把 [jb, jb+Bj) 的 K/V 整体搬入片上缓冲（本块每个 KV 元素只读一次） */
        for (int jj = 0; jj < Bj; jj++) {
            memcpy(&Kblk[jj * HD], &ck->d[(size_t)(jb + jj) * HD], (size_t)HD * sizeof(float));
            memcpy(&Vblk[jj * HD], &cv->d[(size_t)(jb + jj) * HD], (size_t)HD * sizeof(float));
        }
        for (int h = 0; h < Hh; h++) {
            int hoff = h * dh;
            for (int q = 0; q < n; q++) {
                int qabs = old + q;
                /* 4a. 块内各有效 key 的 score，并求块内最大值 */
                float bmx = -1e30f;
                for (int jj = 0; jj < Bj; jj++) {
                    int j = jb + jj;
                    int d = qabs - j;
                    float s;
                    if (d < 0 || d > W) {
                        s = -1e30f;
                    } else {
                        const float *kr = &Kblk[jj * HD + hoff];
                        const float *qr = &qkv[(size_t)q * 3 * inner + hoff];
                        acc_t dot = 0;
                        for (int e = 0; e < dh; e++) dot += (acc_t)qr[e] * (acc_t)kr[e];
                        s = (float)(dot * (acc_t)scale + (acc_t)t->dpb[(size_t)d * Hh + h]);
                    }
                    sc[jj] = s;
                    if (s > bmx) bmx = s;
                }
                if (bmx <= -1e30f) continue;        /* 本块对该 query 无任何可见 key */
                /* 4b. online-softmax 增量更新 */
                float om = m[q * Hh + h];
                float nm = (bmx > om) ? bmx : om;
                float corr = expf(om - nm);
                float nl = l[q * Hh + h] * corr;
                float *od = &o2[(size_t)q * inner + hoff];
                for (int e = 0; e < dh; e++) od[e] *= corr;
                for (int jj = 0; jj < Bj; jj++) {
                    float s = sc[jj];
                    if (s <= -1e30f) continue;       /* 因果/窗口掩码外的 key */
                    float w = expf(s - nm);
                    nl += w;
                    const float *vv = &Vblk[jj * HD + hoff];
                    for (int e = 0; e < dh; e++) od[e] += w * vv[e];
                }
                m[q * Hh + h] = nm;
                l[q * Hh + h] = nl;
            }
        }
    }
    /* 4c. 最终归一化：o2 /= l（online-softmax 收敛后等价于两遍 softmax） */
    for (int h = 0; h < Hh; h++) {
        int hoff = h * dh;
        for (int q = 0; q < n; q++) {
            float inv = (l[q * Hh + h] > 0) ? (1.0f / l[q * Hh + h]) : 0.0f;
            float *od = &o2[(size_t)q * inner + hoff];
            for (int e = 0; e < dh; e++) od[e] *= inv;
        }
    }
    free(Kblk); free(Vblk); free(sc); free(m); free(l);
    free(qkv);

    /* 5) to_out */
    if (t->wo_q[L]) lin_tc_q(t->wo_q[L], t->wo_sc[L], NULL, o2, n, inner, out, C);
    else            lin_tc(t->wo[L], NULL, o2, n, inner, out, C);
    free(o2);  /* A variant owns this temporary; release after projection. */

    /* 6) cache 只保留最后 W 行 */
    int keep = (K > W) ? W : K;
    if (keep < K) {
        for (int q = 0; q < 2; q++) {
            SKV *s = q ? cv : ck;
            memmove(s->d, &s->d[(size_t)(K - keep) * HD], (size_t)keep * HD * sizeof(float));
        }
    }
    ck->len = keep; cv->len = keep;
}

/* ============================ 单层 FeedForward ============================ */
/* GEGLU: x, gate = chunk(2, -1); out = x * gelu(gate) */
static void ff_step(SLocalTrans *t, int L, const float *x, int n, float *out) {
    const int C = t->C, hid = t->hid, half = t->half;
    float *xn = (float*)malloc((size_t)n * C * sizeof(float));
    float *h  = (float*)malloc((size_t)n * hid * sizeof(float));
    if (!xn || !h) { fprintf(stderr, "ff_step: OOM\n"); exit(1); }
    memcpy(xn, x, (size_t)n * C * sizeof(float));
    layernorm_tc(xn, n, C, t->fnw[L], t->fnb[L], 1e-5f);
    if (t->fw1_q[L]) lin_tc_q(t->fw1_q[L], t->fw1_sc[L], NULL, xn, n, C, h, hid);
    else             lin_tc(t->fw1[L], NULL, xn, n, C, h, hid);
    free(xn);
    for (int i = 0; i < n; i++) {
        const float *hr = &h[(size_t)i * hid];
        for (int j = 0; j < half; j++) h[(size_t)i * half + j] = hr[j] * gelu_f(hr[half + j]);
    }
    if (t->fw2_q[L]) lin_tc_q(t->fw2_q[L], t->fw2_sc[L], NULL, h, n, half, out, C);
    else             lin_tc(t->fw2[L], NULL, h, n, half, out, C);
    free(h);
}

/* ============================ LocalTrans 栈 ============================ */
int slt_forward(SLocalTrans *t, const float *x, int n, float *out) {
    float *cur = (float*)malloc((size_t)n * t->C * sizeof(float));
    float *tmp = (float*)malloc((size_t)n * t->C * sizeof(float));
    if (!cur || !tmp) { fprintf(stderr, "slt_forward: OOM\n"); exit(1); }
    memcpy(cur, x, (size_t)n * t->C * sizeof(float));

    for (int L = 0; L < t->depth; L++) {
        /* x = attn(x) + x ; x = ff(x) + x  —— 与 LocalTrans.forward 逐字对应 */
        mha_step(t, L, cur, n, tmp);
        for (int i = 0; i < n * t->C; i++) cur[i] += tmp[i];
        ff_step(t, L, cur, n, tmp);
        for (int i = 0; i < n * t->C; i++) cur[i] += tmp[i];
    }
    memcpy(out, cur, (size_t)n * t->C * sizeof(float));
    free(cur); free(tmp);
    return n;
}
