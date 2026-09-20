#ifndef BLOCKS_H
#define BLOCKS_H
/* ============================================================================
 * blocks.h — reusable network building blocks (C port of l3ac/modules.py).
 *
 * These are the *shared* block classes used by both the encoder and the
 * decoder in the Python package:
 *   - conv_unit()      -> modules.ConvUnit
 *   - first_block()    -> modules.V3FirstBlock  (encoder stem)
 *   - enhance_block()  -> modules.EnhanceBlock  (decoder residual branch)
 *   - legacy_unit()    -> modules.LegacyUnit   (decoder last block unit)
 * Each function takes a weight-name prefix so the same code drives every
 * occurrence of the block in the model.
 * ========================================================================== */
#include "nn_ops.h"

/* ConvUnit at given weight prefix (e.g. "encoder.blocks.1.0.module").
 * Forward: dw_conv -> permute -> ChannelNorm -> pw_conv1 -> Snake -> GRN
 *           -> pw_conv2 -> permute ; residual add is done by the caller. */
void conv_unit(const T *x, T *y, const char *pfx);

/* 与 conv_unit() 等价，但把结果直接累加回 x，省掉一份全尺寸输出张量。
 * encoder 在全速率段用它做残差累加（板上内存吃紧）。 */
void conv_unit_add(T *x, const char *pfx);

/* V3FirstBlock: 5 trend-pool branches (k=7, dil=1) -> conv_1(GELU) -> cat
 * original channel -> conv_2 ; output [B,16,T]. (encoder.blocks.0) */
void first_block(const T *x, T *y);

/* EnhanceBlock(dim): 4 trend-pool branches (xi=x[:,0:1]) -> cat(4ch) ->
 * InstanceNorm(4) -> Conv1d(4->dim)=merge_layer ; then y = x + y*x.
 * prefix pfx is e.g. "decoder.blocks.2" (the EnhanceBlock submodule). */
void enhance_block(const T *x, T *y, const char *pfx, int dim);

/* LegacyUnit(dim): Snake -> Conv(k=7,p=3*dil) -> Snake -> Conv(k=1).
 * dilation = {1,3,9} from ResidualLegacyUnit ; padding = 3*dil.
 * Residual add is done by the caller. (decoder.blocks.13) */
void legacy_unit(const T *x, T *y, const char *pfx, int dil);

/* legacy_unit() 的时间分块版本：residual add 在函数内部就地完成，
 * 常驻大张量从 4 个 [B,C,T] 降到 1 个 + 2 个小块。
 * 数值与 legacy_unit() + 调用方逐元素累加逐位一致。 */
void legacy_unit_add(T *x, const char *pfx, int dil);

#endif /* BLOCKS_H */
