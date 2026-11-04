#ifndef NN_INFER_H
#define NN_INFER_H
/* ============================================================================
 * nn_infer.h — umbrella header for the sqcodec-light-infer C port.
 *
 * The implementation is split into modules that mirror the Python `l3ac`
 * package layout, so debugging maps 1:1 onto the source tree:
 *
 *   nn_ops.h/.c        <- l3ac/layers.py       (snake, channel_norm, grn,
 *                                                linear, conv1d, upsample,
 *                                                instance_norm, tensor, weights)
 *   blocks.h/.c        <- l3ac/modules.py      (ConvUnit, V3FirstBlock,
 *                                                EnhanceBlock, LegacyUnit)
 *   transformer.h/.c   <- l3ac local_attention (LocalMHA, FeedForward,
 *                                                LocalTransformer)
 *   encoder.h/.c       <- l3ac en_codec        (Encoder CNN + LocalEncoder)
 *   decoder.h/.c       <- l3ac en_codec        (Decoder CNN + LocalDecoder)
 *   quantizer.h/.c     <- l3ac/vq/fsq.py       (LearnableFSQ)
 *   fsq_codec.h/.c     <- direction-1 Gray bit codec (protocol)
 *
 * Ground-truth module layout for the 3kbps config:
 *   encoder     : modules.Encoder   (compress_rates=[6,4,4], dims=[16,32,64,88], feat=64)
 *   en_encoder  : LocalEncoder      (1 LocalMHA, window=400 causal, dynamic_pos_bias)
 *   quantizer   : VQEmbed(LearnableFSQ levels=[8,8,8,7,7,7]) project_in 64->6, project_out 6->64
 *   en_decoder  : LocalDecoder      (2 LocalMHA layers)
 *   decoder     : modules.Decoder   (decode_rates=[4,4,3,2], dims=[88,56,40,28,16], legacy last)
 *
 * Weights come from weights_data.h (weight_norm unwrapped, float32,
 * [out,in]/[Co,Ci,k] row-major).
 *
 * Include this header to get every declaration. To build, compile each
 * module .c (nn_ops.c, blocks.c, transformer.c, encoder.c, decoder.c,
 * quantizer.c) plus fsq_codec.c separately and link them (see test/build.sh).
 * ========================================================================== */
#include "nn_ops.h"
#include "blocks.h"
#include "transformer.h"
#include "encoder.h"
#include "decoder.h"
#include "quantizer.h"
#include "fsq_codec.h"

#endif /* NN_INFER_H */
