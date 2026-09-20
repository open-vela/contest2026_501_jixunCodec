#ifndef NN_OPS_H
#define NN_OPS_H

/* r142: JX_HOT -- ?????????????? BSS/??..iram1???????????
 * irom0 (flash XIP) ?????????? MSPI ??????????? PSRAM ?????
 * iram0_0_seg ? 320KB ???????? 60KB?????????
 * ?????? .iram1 ???? iram0_0_seg?? esp32s3_sections.ld??
 * JX_HOT_IRAM=0 ?????????? */
#ifndef JX_HOT
#  if defined(__XTENSA__) && defined(JX_HOT_IRAM) && JX_HOT_IRAM
#    define JX_HOT __attribute__((section(".iram1")))
#  else
#    define JX_HOT
#  endif
#endif

#include "jx_fastmath.h"
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

/* ---- model config: every rate-dependent quantity is declared in
 *      model_config.h and DEFINED per rate in model_config_*.c
 *      (generated from the Python toml by the weight-export script). ---- */
#include "model_config.h"

/* ---- numeric constants ---- */
#define XNN_EPS 1e-8f   /* match xnn.EPS used by Python ChannelNorm/GRN/Snake */
#define EPS 1e-8f

/* ---- time chunking ----
 * first_block / conv_unit 在全采样率上的中间张量是 O(通道数 x 时长) 的
 * （first_block 峰值 197 x Tt 个 float，3 秒音频约 38MB），板上放不下。
 * 这两个算子只在时间上做局部运算，因此按时间切块 + halo 计算，
 * 每块结果与整段计算逐位一致；可用环境变量 JX_CHUNK 覆盖（0 = 关闭）。 */
/* 2026-09-13 第七轮板端扫描（set JX_CHUNK）：1024 -> 12 650 ms，
 * 512 -> 13 130，256 -> 14 250，128 -> 16 550，64 -> 19 550，
 * 2048 -> 12 410 ms（更优），4096 会撑爆 6.2MB 堆而 PANIC。
 * 分块越小越差：每块的 t_alloc/t_free 与 halo 重算的固定开销，
 * 远大于把中间张量塞进 L1 带来的收益。 */
/* r69: 2048 -> 4096. 实测（板端 A/B，同一固件只改 env）
 * CHUNK=2048 -> 4400ms，4096 -> 4240ms；5120/8192 会 OOM panic（堆只剩
 * ~5.7MB / 最大连续块 4.26MB），4096 是当前内存预算下的安全上限。 */
#define JX_CHUNK_DEFAULT 2048   /* r180: 2048 修复 loop3 OOM，h1 修复+NULL 检查 */

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

/* r174：16 字节对齐的堆分配（PIE ee.vld/st.128 要求）。free/realloc 兼容。 */
void *jx_amalloc(size_t bytes);
void *jx_arealloc(void *p, size_t newbytes);
void  jx_afree(void *p);
T  t_alloc(int ndim, const int *shape);
/* nn_ops.c 里的“本次算子被跳过”计数器：任何一步内存拿不到时都把它抬起来，
 * 上层收尾时据此报“本次结果无效”，而不是静默出错。 */
extern int jx_conv_skip;
void t_free(T *t);
T  t_zeros_like(const T *src);
void t_copy(T *dst, const T *src);

/* ---- weight lookup (weights_data.h, exported from the Python model) ---- */
/* Returns pointer into WT_DATA for tensor `name`, writes shape (4) and n. NULL if missing. */
const float* wt_get(const char *name, int *shape4, int *n_out);
int wt_has(const char *name);
/* 权重常数块（.rodata -> flash XIP）的裸指针与长度，仅供板端存储带宽微基准用。 */
const float* jx_weight_blob(void);
int jx_weight_floats(void);
/* 把整块权重从 flash(.rodata) 拷一份到堆/PSRAM，幂等。返回 0 表示成功或
 * 未启用，-1 表示分配失败。由 JX_WT 环境变量控制（0=不搬，非0=搬）。 */
int jx_weight_init(void);
/* 1 = 权重当前驻留堆(PSRAM)，0 = 仍在 flash XIP。 */
int jx_weight_in_heap(void);
/* Convenience wrapper around wt_get that also prints a warning on missing weight. */
const float* W_(const char *name);
/* Build "pfx.suffix" into a small rotating static buffer (returns a pointer to
 * static storage; not thread-safe, fine for single-threaded inference). Shared
 * by every module that names weights by prefix (blocks / transformer /
 * quantizer / encoder / decoder). Declared here (not static) so it is visible
 * across TUs when the modules are compiled separately. */
const char* cat_name(const char *pfx, const char *suffix);

/* ---- activations / norms (l3ac/layers.py) ---- */
void snake1d(T *x, const float *alpha, int channels, int dim1, int dim2, int data_format);
void channel_norm_last(T *x, const float *w, const float *b, int B, int Tt, int C);
void channel_norm_first(T *x, const float *w, const float *b, int B, int C, int Tt);
void grn_last(T *x, const float *gamma, const float *beta, int B, int Tt, int C);
void grn_first(T *x, const float *gamma, const float *beta, int B, int C, int Tt);
/* nn.InstanceNorm1d(C, affine=True): per-sample, per-channel normalize over T (eps given). */
void instance_norm1d(T *x, const float *w, const float *b, int B, int C, int Tt, float eps);

/* ---- layers (l3ac/layers.py) ---- */
/* y[B,O] = x[B,I] @ W[O,I] + b ; applied to every trailing row of (..., in). */
void linear(const float *W, const float *b, const T *x, T *y, int in, int out);
/* x[B,Ci,T] -> y[B,Co,T] (groups-supported, channels_first layout). */
void conv1d(const float *W, const float *b, const T *x, T *y,
            int Ci, int Co, int k, int stride, int pad, int groups, int T_in);
void conv1d_d(const float *W, const float *b, const T *x, T *y,
              int Ci, int Co, int k, int stride, int pad, int groups, int T_in, int dil);
void conv1d_dw(const float *W, const float *b, const T *x, T *y, /* depthwise (groups=Ci=Co) */
               int C, int k, int stride, int pad, int T_in);
void jx_conv1d_dw_tc(const float *W, const float *b, const T *x, T *y,
                     int C, int k, int stride, int pad, int T_in);
/* nn.Upsample(mode='linear', align_corners=False). */
void upsampling_linear(const T *x, T *y, int B, int C, int Tin, int scale);

/* ---- int8 / PIE 内核（csrc/int8_kernels.c）--------------------------------
 * 下面三个函数在 int8 路径可用时返回 0（结果已写入 y），否则返回 -1，
 * 调用方退回原来的浮点内核。jx_pie_enable() 打开 CPENABLE 里的 PIE 位
 * （板端出厂只开了 CP0=FPU，不开的话 PIE 指令会报 EXCCAUSE=0x23）。 */
int  jx_i8_enabled(void);
void jx_pie_enable(void);
int32_t jx_dot_i8_probe(const signed char *a, const signed char *b, int n16);
/* 第三十六轮：dot 阶段「结构」对照探针（只用于 jixun mem [16] 计时） */
void jx_dotcol_i8_probe(const signed char *A, const signed char *B, int nvec,
                        int stride, int nout, int32_t *out);
void jx_dotcol_i8_norb(const signed char *A, const signed char *B, int nvec,
                       int stride, int nout, int32_t *out);
void jx_dotmj_i8_probe(const signed char *A, const signed char *B,
                       int N, int nj, int pad, int32_t *out);
void jx_qacc_dot16_probe(const signed char *wT, const signed char *x,
                         int nchunks, int32_t out[16]);
const signed char *jx_wq_blob_ptr(void);
void jx_wq_init(void);
int  jx_linear_i8(const float *W, const float *b, const float *x, float *y,
                  int rows, int in, int out);
/* 2026-09-13 第六轮：linear 融合扩展。把 conv_unit 的 snake / GRN 折进 linear
 * 的 epilogue / 量化 prologue，省掉 p1 大张量的 2 次往返（PSRAM 只有 ~55MB/s）。
 * 表达式与 snake1d / grn_last 逐项一致，结果与分立调用逐位相同（ss 的行内
 * 累加次序与 grn_last 的 i 序一致）。任一条件不满足 int8 快路径时返回 -1，
 * 调用方退回分立调用。 */
typedef struct {
  const float *snake_alpha;   /* [out]，epilogue 施加 snake；NULL 关闭 */
  float       *ss_rows;       /* [rows]，snake 输出平方的行累加（调用方清零） */
  const float *grn_gamma;     /* [in]，量化前施加 GRN 仿射；NULL 关闭 */
  const float *grn_beta;      /* [in] */
  float        grn_nx;        /* grn_last 的 n_x = g/(g+eps) */
  /* --- r49：l1 一次做完 l2 的量化 prologue ---------------------------------
   * q_gamma/q_beta 是量化前要施加的 GRN 仿射系数（长度 = out）。out_q/out_sc
   * 非空且 oblk == out（epilogue 一次拿到整行）且 out%16==0 时，ex 会在
   * epilogue 里直接算出 l2 要的 int8 输入与逐行 scale，并置 out_qdone=1。
   * 此时调用方必须保证 grn_nx 精确等于 1.0f，否则结果不等价（调用方校验）。
   * out_q 长度 rows*out（in16 == out），out_sc 长度 rows。------------------- */
  const float *q_gamma;     /* [out] 融合量化前的 GRN gamma */
  const float *q_beta;      /* [out] */
  signed char *out_q;       /* [rows][out]，l1 写好的量化输入 */
  float       *out_sc;      /* [rows] */
  int         *out_qdone;   /* ex 完成融合写入后置 1（调用前请清零） */
  /* --- l2 侧：输入已由 l1 量化好，跳过 prologue ------------------------ */
  const signed char *in_q;  /* [rows][in16]，非 NULL 启用 */
  const float       *in_sc; /* [rows] */
  /* ConvUnit 输入侧 ChannelNorm(last) 融合：在 l1 每行量化前现场归一化，
   * 不再先把整块 c1 归一化写回。长度均为 in。 */
  const float       *in_norm_w;
  const float       *in_norm_b;
} jx_linear_fuse_t;
int  jx_linear_i8_ex(const float *W, const float *b, const float *x, float *y,
                     int rows, int in, int out, const jx_linear_fuse_t *fuse);
int  jx_linear_i8_cf(const float *W, const float *b, const float *x, float *y,
                     int rows, int in, int out, const jx_linear_fuse_t *fuse);
/* 注意力 QKᵀ 的 int8 批量点积（见 int8_kernels.c） */
float jx_qrow_i8(const float *x, signed char *q, int n);
void  jx_qk_i8(const signed char *q16, const signed char *K, int nk, int32_t *out);
int   jx_attn_qk_i8_enabled(void);   /* QKᵀ int8，默认关（精度不足，见 .c 注释） */
int   jx_attn_sv_i8_enabled(void);   /* attn·V int8，默认开 */
int32_t jx_sv_i8(const signed char *W, const signed char *V, int nblk);
float jx_qrow_i16(const float *x, signed short *q, int n);
void  jx_qk_i16(const signed short *q16, const signed short *K, int nk, int32_t *out);
int   jx_attn_qk_i16_enabled(void);   /* QKᵀ int16，默认开 */
int  jx_conv1d_i8(const float *W, const float *b, const T *x, T *y,
                  int Ci, int Co, int k, int stride, int pad, int groups,
                  int T_in, int T_out, int dil);

/* ---- 卷积 epilogue 融合的后处理（2026-09-14 第三十三轮）------------------------
 * jx_conv_post: 调用方设置；1 = 让 int8 卷积的 epilogue 顺带做 GELU。
 * jx_conv_post_used: 由 jx_conv1d_i8 回填，只有它 == jx_conv_post 才说明
 * 后处理已经在 int8 快路径里做掉了（否则调用方要自己补一次浮点版）。 */
extern int jx_conv_post;
extern int jx_conv_post_used;
/* jx_conv_post == 2（snake）时，逐输出通道的 alpha 表（长度 = Co）。 */
extern const float *jx_conv_post_alpha;
/* 预计算 snake 后处理系数表（sa = alpha[i]+eps, si = 1/sa）。 */
void jx_post_snake_prep(const float *alpha, int n);
/* snake1d 的"拷贝版"：dst = snake(src)，一趟完成（原来要 t_copy + snake1d 两趟）。
 * B==1、channels-first，与 snake1d(&x, alpha, 1, C, Tt, 0) 逐位相同。 */
void snake1d_copy(T *dst, const T *src, const float *alpha);
/* conv1d_d + snake 一次做完：int8 快路径可用时 snake 折进 epilogue，
 * 否则退回 conv1d_d + snake1d，结果逐位相同。JX_SNAKEFUSE=0 关闭。 */
void conv1d_d_snake(const float *W, const float *b, const T *x, T *y,
                    int Ci, int Co, int k, int stride, int pad, int groups,
                    int T_in, int dil, const float *alpha);
/* conv1d + gelu 一次做完：int8 快路径可用时 gelu 折进 epilogue（省掉一整趟
 * 大张量读+写），否则退回 conv1d + 逐元素 gelu，结果与原来逐位相同。
 * JX_GELFUSE=0 可关闭融合做 A/B。 */
void conv1d_gelu(const float *W, const float *b, const T *x, T *y,
                 int Ci, int Co, int k, int stride, int pad, int groups, int T_in);

/* exact GELU (PyTorch default, erf formulation) — mirrors torch.nn.GELU */
static inline float gelu_f(float x) {
    return 0.5f * x * (1.f + jx_erff(x * 0.70710678f));
}


/* r56：双核并行 for。fn(arg, lo, hi) 处理 [lo,hi) 这一段；行间无依赖时可安全二分。 */
int  jx_pf_run(int n, void (*fn)(void *, int, int), void *arg);
int  jx_pf_run_min(int n, void (*fn)(void *, int, int), void *arg, int minn);
int  jx_pf_on(void);
#endif /* NN_OPS_H */
int jx_clk_probe_cmd(void);   /* r151: 实测 CPU 频率 */
