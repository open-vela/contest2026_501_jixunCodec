#ifndef NN_OPS_H
#define NN_OPS_H
/* ============================================================================
 * nn_ops.h — low-level NN primitives (C port of l3ac/layers.py)
 *
 * This header is the foundation of the codec. It mirrors the *atomic*
 * operators defined in the Python package `l3ac/layers.py` (snake, channel
 * norm, GRN, linear, conv1d, upsample, instance norm) plus the shared tensor
 * type and the weight table lookup. Every higher-level module
 * (blocks / transformer / encoder / decoder / quantizer) includes this header.
 *
 * Model-level hyper-parameters (encoder/decoder dims, strides, local-attention
 * config, quantizer levels) live here too so there is a single source of truth
 * that the Python `configs/3kbps.toml` must match.
 * ========================================================================== */
#include <stdint.h>
#include <math.h>

/* ---- 累加精度（部署执行效率关键）----
 * 默认 float：与 Python float32（量化/推理路径本就 float32）对齐，且在无
 * double-FPU 的 MCU（如 ESP32-S3）上比 double 快 5–10×（后者是软件模拟）。
 * 累加改为 float 后 C 与 Python 的浮点结合律一致，token 对齐通常反而更优。
 * -DACC_DOUBLE 回到 double：更准但慢，用于桌面黄金比对 / 复现历史 PESQ 数值。 */
#ifndef ACC_DOUBLE
  typedef float  acc_t;
#else
  typedef double acc_t;
#endif

/* ---- model config: every rate-dependent quantity is declared in
 *      model_config.h and DEFINED per rate in model_config_*.c
 *      (generated from the Python toml by the weight-export script). ---- */
#include "model_config.h"

/* ---- numeric constants ---- */
#define XNN_EPS 1e-8f   /* match xnn.EPS used by Python ChannelNorm/GRN/Snake */
#define EPS 1e-8f
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- simple row-major tensor ---- */
typedef struct {
    float *d;   /* data, row-major */
    int   ndim;
    int   shape[4];
    int   len;  /* product of shape */
} T;

T  t_alloc(int ndim, const int *shape);
void t_free(T *t);
T  t_zeros_like(const T *src);
void t_copy(T *dst, const T *src);

/* ---- weight lookup (weights_data.h, exported from the Python model) ---- */
/* Returns pointer into WT_FDATA (float section) for tensor `name`, writes shape (4) and n. NULL if missing. */
const float* wt_get(const char *name, int *shape4, int *n_out);
int wt_has(const char *name);
/* Convenience wrapper around wt_get that also prints a warning on missing weight. */
const float* W_(const char *name);
/* Quantized (weight-only INT8) lookup: returns pointer into WT_IDATA for a
 * quantized tensor, plus its per-channel scale pointer/length. Returns NULL if
 * the name is not a quantized weight. */
const int8_t* wt_get_i8(const char *name, const float **scale, int *scale_n, int *shape4, int *n_out);
const int8_t* W_i8(const char *name, const float **scale, int *scale_n);
/* Build "pfx.suffix" into a small rotating static buffer (returns a pointer to
 * static storage; not thread-safe, fine for single-threaded inference). Shared
 * by every module that names weights by prefix (blocks / transformer /
 * quantizer / encoder / decoder). Declared here (not static) so it is visible
 * across TUs when the modules are compiled separately. */
const char* cat_name(const char *pfx, const char *suffix);
#ifdef __NuttX__
/* Call under the native voice claim, before/after all model workers. */
int wt_names_begin(void);
void wt_names_end(void);
#endif

/* ---- activations / norms (l3ac/layers.py) ---- */
void snake1d(T *x, const float *alpha, int channels, int dim1, int dim2, int data_format);
void gelu_apply(float *x,int n);
void channel_norm_last(T *x, const float *w, const float *b, int B, int Tt, int C);
void channel_norm_first(T *x, const float *w, const float *b, int B, int C, int Tt);
void grn_last(T *x, const float *gamma, const float *beta, int B, int Tt, int C);
void grn_first(T *x, const float *gamma, const float *beta, int B, int C, int Tt);
/* nn.InstanceNorm1d(C, affine=True): per-sample, per-channel normalize over T (eps given). */
void instance_norm1d(T *x, const float *w, const float *b, int B, int C, int Tt, float eps);

/* ---- layers (l3ac/layers.py) ---- */
/* y[B,O] = x[B,I] @ W[O,I] + b ; applied to every trailing row of (..., in). */
void linear(const float *W, const float *b, const T *x, T *y, int in, int out);
/* weight-only INT8 变体：y[o] = scale[o] * Σ Wi[o,j]*x[j] + b[o]（double 累加）。 */
void linear_q(const int8_t *Wi, const float *sc, const float *b, const T *x, T *y, int in, int out);
/* x[B,Ci,T] -> y[B,Co,T] (groups-supported, channels_first layout). */
void conv1d(const float *W, const float *b, const T *x, T *y,
            int Ci, int Co, int k, int stride, int pad, int groups, int T_in);
void conv1d_d(const float *W, const float *b, const T *x, T *y,
              int Ci, int Co, int k, int stride, int pad, int groups, int T_in, int dil);
void conv1d_dw(const float *W, const float *b, const T *x, T *y, /* depthwise (groups=Ci=Co) */
               int C, int k, int stride, int pad, int T_in);
/* nn.Upsample(mode='linear', align_corners=False). */
void upsampling_linear(const T *x, T *y, int B, int C, int Tin, int scale);

/* exact GELU (PyTorch default, erf formulation) — mirrors torch.nn.GELU */
static inline float gelu_f(float x) {
    return 0.5f * x * (1.f + erff(x * 0.70710678f));
}

#endif /* NN_OPS_H */
