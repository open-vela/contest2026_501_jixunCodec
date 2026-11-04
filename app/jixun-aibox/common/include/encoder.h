#ifndef ENCODER_H
#define ENCODER_H
/* ============================================================================
 * encoder.h — encoding side of the network (C port of l3ac en_codec.py).
 *
 *   - encoder_forward()   -> modules.Encoder  (CNN stem + residual + downsample
 *                            + final projection). x[B,1,T] -> out[B,64,T'].
 *   - en_encoder_forward()-> LocalEncoder    (1 LocalMHA layer, channels_last).
 *                            x[B,C,T'] -> out[B,T',C].
 * These are the two stages that run before the quantizer on the encode path.
 * ========================================================================== */
#include "nn_ops.h"
#include "blocks.h"
#include "transformer.h"

/* modules.Encoder.forward: CNN encoder (compress_rates=[6,4,4]). */
void encoder_forward(const T *x, T *out);

/* LocalEncoder.forward (en_encoder): permute to [B,T',C] then LocalTransformer(depth1). */
void en_encoder_forward(const T *x, T *out);

#endif /* ENCODER_H */
