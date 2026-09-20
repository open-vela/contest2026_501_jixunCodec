#ifndef DECODER_H
#define DECODER_H
/* ============================================================================
 * decoder.h — decoding side of the network (C port of l3ac en_codec.py).
 *
 *   - decoder_forward()    -> modules.Decoder  (CNN: conv stem + 4x
 *                             [ConvUnit + EnhanceBlock + upsample] + legacy
 *                             last block). x[B,64,T'] -> out[B,1,T].
 *   - en_decoder_forward() -> LocalDecoder   (2 LocalMHA layers, channels_last).
 *                            x[B,T',C] -> out[B,C,T'].
 * These are the two stages that run after the quantizer on the decode path.
 * ========================================================================== */
#include "nn_ops.h"
#include "blocks.h"
#include "transformer.h"

/* modules.Decoder.forward. */
void decoder_forward(const T *x, T *out);

/* LocalDecoder.forward (en_decoder): LocalTransformer(depth2) then permute [B,C,T']. */
void en_decoder_forward(const T *x, T *out);

#endif /* DECODER_H */
