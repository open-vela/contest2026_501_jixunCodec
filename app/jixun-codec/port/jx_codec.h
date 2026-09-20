/****************************************************************************
 * apps/jixun-codec/port/jx_codec.h
 *
 * 极讯 AI Codec —— 板端封装层
 *
 * 数据流（与 PC 端 test/codec_main.c 的 send/recv 完全一致）：
 *   编码: 16kHz 单声道 int16 PCM
 *         -> encoder_forward -> en_encoder_forward -> quantizer_forward
 *         -> 量化索引 idx[]（每 token VQ_TOTAL_BITS 位，这就是空口载荷）
 *   解码: idx[] -> quantizer_to_features -> en_decoder_forward
 *         -> decoder_forward -> float PCM（[-1,1]）
 ****************************************************************************/

#ifndef __APPS_JIXUN_CODEC_PORT_JX_CODEC_H
#define __APPS_JIXUN_CODEC_PORT_JX_CODEC_H

#include <stdint.h>

/* 分段耗时（毫秒） */
typedef struct
{
  double pre_ms;        /* 编码：前处理 + encoder CNN + en_encoder 注意力 */
  double quant_ms;      /* 编码：FSQ 量化 */
  double deq_attn_ms;   /* 解码：反量化 + en_decoder 注意力 */
  double dec_ms;        /* 解码：decoder CNN */
  double total_ms;      /* 编码 + 解码 总计 */
} jx_timing_t;

extern jx_timing_t jx_last_timing;

/* 当前编译进的码率名："1k" / "3k" / "6k" */
const char *jx_rate_name(void);

/* 每 token 位数（1k=17，3k/6k=18） */
int jx_token_bits(void);

/* 编码：pcm 为 n_samples 个 16kHz 单声道样本（int16）。
 * 成功返回 0，并置 *out_idx（调用方 free）、*out_ntok（token 数）、
 * *out_Tt（帧数，= padded_samples / HOP_LENGTH）。 */
int jx_encode(const int16_t *pcm, int n_samples,
              int64_t **out_idx, int *out_ntok, int *out_Tt);

/* 解码：把 Tt 帧的 idx 还原成 origT 个浮点样本（[-1,1]）。
 * 成功返回 0，*out_pcm 由调用方 free。 */
int jx_decode(const int64_t *idx, int Tt, int origT, float **out_pcm);

/****************************************************************************
 * 2026-09-18：24ms 小包流式编解码（相对区间语义）
 *
 * 判据已改：**不再要求与整段批处理逐位一致**。流式 codec 本来就按相对区间
 * 处理，输出与批处理不同是正常的 —— 实测也支持这点：分块结果对左侧回看 L、
 * 右侧余量 R 都近乎免疫，说明模型只在乎相对窗内的结构，不在乎绝对位置。
 *
 * 参数（4 帧/包）：
 *   一帧         = HOP_LENGTH = 96 样本 = 6ms
 *   一个包       = 4 帧 = 384 样本 = 24ms
 *   一个 token   = 18 bit，4 token = 72 bit = 9 字节整
 *   一个注意力窗 = 400 帧 = 100 包 = 2400ms = 900 字节
 *
 * 为什么 4 帧不是 3 帧(18ms)：400÷4=100 整除，包边界能对齐注意力窗格；
 * 400÷3 除不尽，掩码相位永远差一点。且 4×18bit 正好 9 字节。
 *
 * 语义：每包在固定长度滚动缓冲里算，缓冲 = 1 个注意力窗(400帧) + 右侧
 * 卷积 halo，只取最新 4 帧输出。卷积局部、归一化走滑动窗、注意力用对齐后的
 * 局部帧号 —— 全部相对区间。
 ****************************************************************************/

#define JX_PKT_FRAMES        4
#define JX_PKT_SAMPLES       (JX_PKT_FRAMES * HOP_LENGTH)   /* 384 = 24ms */
#define JX_NET_WINDOW_FRAMES 400                            /* = EN_WINDOW */
#define JX_STREAM_WINDOW_DEFAULT 256                        /* fast-encoder default */
#define JX_PKT_BITS          (JX_PKT_FRAMES * 18)           /* 72 bit */
#define JX_PKT_BYTES         (JX_PKT_BITS / 8)              /* 9 字节 */

struct jx_stream_s;

struct jx_stream_s *jx_stream_open(int halo_samples);
void jx_stream_close(struct jx_stream_s *st);

/* 喂一个 24ms 包（JX_PKT_SAMPLES 个样本）。成功时输出本包的 token
 *（JX_PKT_FRAMES 个，18bit 有效）与对应解码 PCM（JX_PKT_SAMPLES 个 float）。
 * tok_out 长度 >= JX_PKT_FRAMES；pcm_out 长度 >= JX_PKT_SAMPLES。
 * 返回 0 成功，<0 失败。 */
int jx_stream_feed(struct jx_stream_s *st, const int16_t *pkt,
                   int64_t *tok_out, float *pcm_out);

/* 批量版本：一次喂 npkt 个连续的 24ms 包，只跑一轮编码/解码。
 * tok_out 长度 >= npkt*JX_PKT_FRAMES，pcm_out 长度 >= npkt*JX_PKT_SAMPLES。 */
int jx_stream_feed_batch(struct jx_stream_s *st, const int16_t *pkt, int npkt,
                         int64_t *tok_out, float *pcm_out);

int jx_stream_pkts(const struct jx_stream_s *st);

/* 分段累计耗时（毫秒）：卷积 / 编码注意力 / 解码总计 /
 * 反量化到特征 / 解码注意力 / 解码 CNN。 */
void jx_stream_times(const struct jx_stream_s *st,
                     double *conv_ms, double *attn_ms, double *dec_ms,
                     double *deq_ms, double *endec_ms, double *decnn_ms);

#endif /* __APPS_JIXUN_CODEC_PORT_JX_CODEC_H */
