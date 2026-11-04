#ifndef STREAM_TRANS_H
#define STREAM_TRANS_H
/* ============================================================================
 * stream_trans.h — local-attention transformer 的增量（KV-cache）流式实现。
 *
 * 对应 stream_cache.py 的 _step_localtrans / _step_mha，以及
 * local_trans.py 的 LocalTrans（exact_windowsize=True）。
 *
 * 可见范围（与 local_attention 库 exact_windowsize=True 的因果掩码等价）：
 *     query i 可见 key j  <=>  0 <= (K - n + i - j) <= W
 * 其中 n = 本轮 query 数，K = cache 里的 key 总数。距离 d = K-n+i-j 同时
 * 决定 dynamic position bias：库里的 mlp(|W+i-jj|) 经 offset = W+n-K 平移后
 * 恰好退化为 mlp(d)，所以只需预计算一张 (W+1) x heads 的距离表。
 * ========================================================================== */
#include "nn_ops.h"
#include "stream_ops.h"

typedef struct {
    int depth, C, heads, dh, W, hid, half, inner, HD;
    SKV kvk[8], kvv[8];              /* 逐层 KV cache（各保留最后 W 个 key） */
    float *dpb;                      /* [(W+1)][heads] 距离 -> bias */
    /* 逐层权重指针（init 时解析一次，避免每 chunk 反复查表） */
    const float *nw[8], *nb[8], *wqkv[8], *wo[8];
    const float *fnw[8], *fnb[8], *fw1[8], *fw2[8];
    /* weight-only INT8 权重指针 + 每通道 scale（NULL => 回退 float）。
     * LocalTrans 的 to_qkv / to_out / fw1 / fw2 均为 Linear，已被 PTQ 量化。 */
    const int8_t *wqkv_q[8], *wo_q[8], *fw1_q[8], *fw2_q[8];
    const float  *wqkv_sc[8], *wo_sc[8], *fw1_sc[8], *fw2_sc[8];
    /* DynamicPositionBias.mlp 的 int8 权重 + scale（float bias 仍走 W_） */
    const int8_t *dpb_w0_q, *dpb_w1_q, *dpb_w2_q;
    const float  *dpb_w0_sc, *dpb_w1_sc, *dpb_w2_sc;
    /* 临时缓冲（按需增长） */
    float *scr1, *scr2, *scr3;
    int scr_cap;
} SLocalTrans;

/* pfx  : "en_encoder.local_trans" / "en_decoder.local_trans"
 * depth: LocalTrans 层数（en_encoder depth=1, en_decoder depth=2）
 * W    : local window（= model_config.EN_WINDOW，3kbps = 400） */
void slt_init(SLocalTrans *t, const char *pfx, int depth, int W);
void slt_free(SLocalTrans *t);
void slt_reset(SLocalTrans *t);

/* x -> out，均为 TC [n][C]。返回 n。 */
int slt_forward(SLocalTrans *t, const float *x, int n, float *out);

#endif /* STREAM_TRANS_H */
