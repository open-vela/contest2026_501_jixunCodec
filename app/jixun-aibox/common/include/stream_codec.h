#ifndef STREAM_CODEC_H
#define STREAM_CODEC_H
/* ============================================================================
 * stream_codec.h — 因果流式编解码入口（sender / receiver 两个独立状态机）。
 *
 * 与 Python 参考的对应关系（eval/test_fixedpoint.py::simulate_streaming）：
 *     sender   = senc_new()                       编码端状态机
 *     receiver = sdec_new()                       解码端状态机
 *     每段:  senc_step(chunk) -> tokens -> sdec_step(tokens) -> wav
 *
 * 因果（0 lookahead）保证"边编码边解码"与"先编码完再解码"输出完全一致，
 * 因此顶层可以按 chunk 交替推进，就是真实流式循环。
 *
 * 输入约定：x 是峰值归一后的 16kHz 单声道 float；n 必须是 HOP_LENGTH 的整数倍。
 * ========================================================================== */
#include "nn_ops.h"
#include "stream_ops.h"
#include "stream_trans.h"

/* 分层验证钩子：每完成一个顶层 block 回调一次。
 * name 形如 "encoder.blocks.3" / "en_encoder" / "decoder.blocks.6"。
 * ch_last=1 表示 d 是 TC [T][C]，否则 CV [C][T]。 */
typedef void (*STapCb)(void *ud, const char *name, const float *d, int C, int T, int ch_last);
extern STapCb g_stap;
extern void  *g_stap_ud;

typedef struct SEncState SEncState;
typedef struct SDecState SDecState;

SEncState *senc_new(void);
void       senc_free(SEncState *s);
void       senc_reset(SEncState *s);
size_t     senc_peak(const SEncState *s);   /* arena 峰值（float 数），部署调参用 */
/* 推 n 个样本过编码器；tokens 容量需 >= n / HOP_LENGTH。返回产出的 token 数。 */
int        senc_step(SEncState *s, const float *x, int n, int64_t *tokens);

SDecState *sdec_new(void);
/* Explicit bounded chunk uses the smaller arena; default retains 16-token capacity. */
SDecState *sdec_new_chunk(int max_tokens);
void       sdec_free(SDecState *s);
void       sdec_reset(SDecState *s);
size_t     sdec_peak(const SDecState *s);   /* arena 峰值（float 数），部署调参用 */
/* 推 nt 个 token 过解码器；wav 容量需 >= nt * HOP_LENGTH。返回产出的样本数。 */
int        sdec_step(SDecState *s, const int64_t *tokens, int nt, float *wav);

#endif /* STREAM_CODEC_H */
