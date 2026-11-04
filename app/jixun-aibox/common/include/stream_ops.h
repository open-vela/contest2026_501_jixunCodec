#ifndef STREAM_OPS_H
#define STREAM_OPS_H
/* ============================================================================
 * stream_ops.h — 严格因果增量算子库（1:1 对齐 l3ac/stream_cache.py）
 *
 * 本文件是"state-cache 流式引擎"的 C 实现基础层：每个算子只消费本轮新数据，
 * 内部维护固定长度的历史（hist / KV cache / 延迟线），计算量 O(chunk)、
 * 内存与音频时长无关。语义与 Python 参考实现逐行对应，见各函数注释。
 *
 * 布局约定（全部 batch = 1）：
 *   CV = channels-first [C][T]，即 row-major 的 t 连续；
 *   TC = time-first     [T][C]，即 row-major 的 c 连续（PyTorch 的 channels_last）。
 * 只有需要 Linear / LayerNorm / attention 的地方才用 TC。
 *
 * 关键等价性质（来自编码端 B=1）：
 *   对任意 t，output[t] 只依赖 input[<= t]  =>  整段前向 ≡ 逐块流式前向
 * ========================================================================== */
#include <stddef.h>   /* size_t：Linux 下 nn_ops.h 不传递包含，必须显式引入 */
#include "nn_ops.h"

/* ---------------------------------------------------------------------------
 * bump 分配器：每个 chunk 复位一次，避免逐层 malloc/free 抖动
 * ------------------------------------------------------------------------- */
typedef struct { float *d; size_t cap; size_t off; size_t max_off; } SArena;

void   sa_init(SArena *a, size_t cap);
void   sa_reset(SArena *a);
float *sa_alloc(SArena *a, size_t n);          /* n 个 float；越界则扩容 */
void   sa_free(SArena *a);

/* ---------------------------------------------------------------------------
 * 通用历史缓冲：CV 布局 [C][len]，动态容量。
 * 语义对应 stream_cache._Slot.hist（沿时间轴保留的输入尾部）。
 * ------------------------------------------------------------------------- */
typedef struct { float *d; int len; int cap; } SHist;

/* 复位：分配 cap（至少 cap_hint），并把前 init_len 列清零、len=init_len。
 * 因果卷积的"离线整段实现是 F.pad(x,(left_pad,0))"对应 init_len = left_pad，
 * 即序列起点之前的那段零；延迟线对应 init_len = n（前 n 个输出就是零）。 */
void shist_reset(SHist *h, int C, int init_len, int cap_hint);
void shist_free(SHist *h);
/* 保证 cap >= need（会按新行宽重新排布已有数据） */
void shist_ensure(SHist *h, int C, int need);

/* ---------------------------------------------------------------------------
 * attention K/V cache：[len][HD]，HD = heads * dim_head
 * 语义对应 stream_cache._Slot.aux['k' / 'v']（保存位置编码之前的尾部 W 个）
 * ------------------------------------------------------------------------- */
typedef struct { float *d; int len; int cap; } SKV;
void skv_free(SKV *s);
void skv_ensure(SKV *s, int HD, int need);     /* len 保持不变，只扩容量 */

/* ---------------------------------------------------------------------------
 * 1. 因果卷积  (_step_conv)
 *
 *    buf = [hist | x] ; n_out = (L-K-1)/stride + 1 ; y = conv(buf)[:n_out]
 *    hist = buf[n_out*stride:]
 *    K = (k-1)*dilation
 *
 * 返回本轮新增输出帧数 n_out（0 = 连一个输出都凑不齐，此时 x 被并入 hist）。
 * y 由调用者提供，容量需 >= Co * sconv_nout(...)。
 * ------------------------------------------------------------------------- */
int sconv_nout(const SHist *h, int n_in, int k, int dil, int stride);
int sconv_step(SHist *h, const float *W, const float *b,
               int Ci, int Co, int k, int stride, int dil, int groups,
               const float *x, int n_in, float *y);
/* weight-only INT8 版（定点推理）：权重为 int8 + 每通道 scale，激活仍 float。
 * 与 sconv_step 逐行对应，仅在卷积内核用 int8 累加 y=scale[o]*Σ(int8*w)x+b。 */
int sconv_step_q(SHist *h, const int8_t *Wi, const float *sc, const float *b,
                 int Ci, int Co, int k, int stride, int dil, int groups,
                 const float *x, int n_in, float *y);

/* ---------------------------------------------------------------------------
 * 2. 因果 TrendPool  (_step_trendpool)：avg_pool(max_pool(|x|))，左侧补零
 *    rate=1：输出与输入等长；hist 保留尾部 2*(k-1)。返回 n_in。
 * ------------------------------------------------------------------------- */
int strendpool_step(SHist *h, int k, int C, const float *x, int n_in, float *y);

/* ---------------------------------------------------------------------------
 * 3. 因果 FrameNorm  (_step_framenorm)：因果有限长指数加权统计
 *    mu[t]  = Σ w_j x[t-j] ; var[t] = Σ w_j x[t-j]² - mu²  (w_j ∝ α(1-α)^j)
 *    y[t] = (x[t]-mu[t]) / sqrt(var[t]+eps) * weight[c] + bias[c]
 *    rate=1；hist 保留尾部 window-1。返回 n_in。
 * ------------------------------------------------------------------------- */
int sframenorm_step(SHist *h, int C, int window, float alpha,
                    const float *w, const float *b, float eps,
                    const float *x, int n_in, float *y);

/* ---------------------------------------------------------------------------
 * 4. 因果上采样  (_step_upsample)：零阶保持 + depthwise 因果平滑卷积
 *    u = repeat_interleave(buf, S) ; y = smooth(u) ; 取尾部 n_in*S
 *    hist 保留尾部 ceil((k-1)/S)。返回 n_in * S。
 *    sw: depthwise 权重 [C][k]（导出形状 [C,1,k]），sb: [C] 或 NULL。
 * ------------------------------------------------------------------------- */
int supsample_step(SHist *h, int C, int stride, int k,
                   const float *sw, const float *sb,
                   const float *x, int n_in, float *y);
/* weight-only INT8 版（定点推理）：sw 为 int8 + 每通道 scale。 */
int supsample_step_q(SHist *h, int C, int stride, int k,
                     const int8_t *sw, const float *sc, const float *sb,
                     const float *x, int n_in, float *y);

/* ---------------------------------------------------------------------------
 * 5. 延迟补偿线  (_step_delay)：y[t] = x[t-n]，初始 n 个零。
 *    两侧延迟不等时把延迟较小的一侧右移 Δ = (max_delay - d)。
 *    返回 n_in。
 * ------------------------------------------------------------------------- */
int sdelay_step(SHist *h, int n, int C, const float *x, int n_in, float *y);

/* ---------------------------------------------------------------------------
 * 逐元素算子
 * ------------------------------------------------------------------------- */
/* PerFrameGRN（channels_last，TC [n][C]）：y = (1+gamma)*x + beta */
void per_frame_grn(float *x, int n, int C, const float *gamma, const float *beta);

/* nn.LayerNorm(C)（TC [n][C]，对最后一维归一，eps 默认 1e-5） */
void layernorm_tc(float *x, int n, int C, const float *w, const float *b, float eps);

/* A32 optional data-flow kernels. Zero means caller must use its fallback. */
int sconv_dw7_tc(SHist *,const float *,const float *,int,const float *,int,float *);
int sdelay_add_step(SHist *,int,int,const float *,int,const float *,float *);

/* Optional caller-owned immutable coefficients; NULL retains reference calculation. */
int sframenorm_step_cached(SHist *,int,int,float,const float *,const float *,float,
                          const float *,int,float *,const float *);

#endif /* STREAM_OPS_H */
