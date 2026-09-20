#ifndef JXPROF_H
#define JXPROF_H

/* ============================================================================
 * jxprof.h — 轻量算子级计时探针（板端排查性能瓶颈用）
 *
 * 只统计“调用次数少、单次耗时长”的算子：每次进入/退出各取一次
 * clock_gettime(CLOCK_MONOTONIC)，几十~几百次调用的开销可忽略。
 * 通过环境变量关闭：JX_PROF=0（默认开启，开销 < 1ms）。
 *
 * 用法：
 *     JXP_BEG(JP_CONV1D);
 *     ... 函数体 ...
 *     JXP_END(JP_CONV1D);
 * 同一个槽位可以嵌套（比如 conv1d 被 conv_unit 调用），两个都会累加，
 * 因此解读时看“最外层”的那个槽位即可。
 * ========================================================================== */
#include <time.h>

enum {
  JP_TOTAL = 0,      /* jx_encode / jx_decode 全程 */
  JP_ENCODER,        /* encoder_forward */
  JP_ENENCODER,      /* en_encoder_forward */
  JP_FIRSTBLOCK,     /* first_block */
  JP_CONVUNIT,       /* conv_unit / conv_unit_add */
  JP_ENHANCE,        /* enhance_block */
  JP_LEGACY,         /* legacy_unit */
  JP_LOCALTRANS,     /* local_transformer */
  JP_ENDECODER,      /* en_decoder_forward */
  JP_DECODER,        /* decoder_forward */
  JP_QUANT,          /* quantizer_forward / quantizer_to_features */
  JP_CONV1D,         /* 所有 conv1d / conv1d_d */
  JP_LINEAR,         /* 所有 linear */
  JP_SNAKE,          /* snake1d */
  JP_GELU,           /* gelu / 其它逐元素超越函数循环 */
  JP_TRENDPOOL,      /* trend_pool */
  JP_UPSAMPLE,       /* upsampling_linear */
  JP_NORMS,          /* channel_norm_* / grn_* / instance_norm1d */
  JP_PERMUTE,        /* 纯搬运/转置循环 */
  JP_FBR,            /* first_block 的 5 条 trendpool 分支（含 conv1d_d + memcpy） */
  JP_FBC1,           /* first_block conv_1 (20->80) */
  JP_FBC2,           /* first_block conv_2 (81->16) */
  JP_FBCAT,          /* first_block 的 81 通道拼接（纯搬运） */
  JP_CUDW,           /* conv_unit: dw_conv + 入 permute */
  JP_CUNORM,         /* conv_unit: channel_norm + linear C->4C + snake + grn */
  JP_CUOUT,          /* conv_unit: linear 4C->C + 出 permute */
  JP_I8Q,            /* int8: activation quantization prologue */
  JP_I8MM,           /* int8: main dot-product loop + epilogue */
  JP_I8S,            /* int8: scratch setup / weight lookup */
  JP_K1Q,            /* int8 per-t 路径：im2col 量化 prologue（Pass A 求 max + Pass B 量化写） */
  JP_K1MM,           /* int8 per-t 路径：按 co 的点积批（jx_dotrow_i8） */
  JP_MISC,           /* r69：分块 halo 拷贝 / 逐元素累加等搬运 glue */
  JP_N
};

extern double jx_prof_ms[JP_N];
extern unsigned jx_prof_cnt[JP_N];
extern double   jx_elems_snake;
extern double   jx_elems_sinsq;   /* fused-epilogue jx_sinsq() call count */
extern double   jx_alloc_mb;      /* t_alloc 累计分配的 MB（清零流量的量级指标） */
/* 常开 MAC 统计（conv1d/conv1d_d 与 linear），用于板端核算真实算力速率 */
extern double   jx_mac_conv, jx_mac_linear;
extern long     jx_n_conv, jx_n_linear;

static inline double jx_prof_now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

void jx_prof_reset(void);
void jx_prof_dump(void);

/* ---- 按卷积形状的耗时直方图（诊断用，JX_NO_PROFILE 下为空） ---- */
void jx_ct_add(int Ci, int Co, int k, int g, int dil, double ms, double mac, int Tin, int Tout);
void jx_ct_dump(void);
void jx_ct_reset(void);
extern int jx_conv_path;   /* 最近一次 conv 走的路径：0=fp32 1=dense 2=pre_q 3=per_t */

/* JX_NO_PROFILE（主机回归）下必须完全展开成空语句：任何插桩都会改变 GCC 的
 * FMA 收缩选择，破坏 m0 与参考的逐字节一致性。 */
#ifdef JX_NO_PROFILE
#  define JX_CT_BEG(tin,tout)         ((void)0)
#  define JX_CT_END(ci,co,k,g,dil,mac) ((void)0)
#else
#  define JX_CT_BEG(tin,tout)         double _jx_ct0 = jx_prof_now(); int _jx_ct_tin = (tin), _jx_ct_tout = (tout)
#  define JX_CT_END(ci,co,k,g,dil,mac) jx_ct_add((ci),(co),(k),(g),(dil), jx_prof_now() - _jx_ct0, (mac), _jx_ct_tin, _jx_ct_tout)
#endif

/* 同一个函数里同一槽位只能出现一次；若需多次测量同一槽位（例如 conv_unit
 * 里前后两次 permute），把 JXP_BEG/JXP_END 包在各自的一对大括号里即可。
 * 函数中途 return 时请在 return 前补一次 JXP_END。 */
#ifdef JX_NO_PROFILE
/* 发布/回归构建：探针展开成空语句，生成的机器码与未插桩时完全一致，
 * 从而保持与 PC 参考逐字节一致（插桩会改变 GCC 的 FMA 收缩选择，
 * 曾导致 1740 个 token 里有 1 个 token 的 1 个 FSQ 维跳到相邻量化中心）。 */
#define JXP_BEG(i) ((void)0)
#define JXP_END(i) ((void)0)
#else
#define JXP_BEG(i) double _jx_p_##i = (jx_prof_off ? 0.0 : jx_prof_now())
#define JXP_END(i) do { if (!jx_prof_off) { jx_prof_ms[i] += jx_prof_now() - _jx_p_##i; jx_prof_cnt[i]++; } } while (0)
#endif

extern int jx_prof_off;

#endif /* JXPROF_H */
