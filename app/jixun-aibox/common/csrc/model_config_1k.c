/* ============================================================================
 * model_config_1k.c — rate configuration for the 1kbps model.
 *
 * Values mirror configs/1kbps.toml [nn_config]. Link this TU instead of
 * model_config_3k.c / model_config_6k.c to get a 1kbps build, together with
 * the matching weights_data.h / golden.h.
 *
 * DIFFERENCES vs 3k/6k (handled generically by the C code):
 *   - FEATURE_DIM = 56 (not 64) -> transformer heads/dim_head differ, but they
 *     are READ FROM THE WEIGHT SHAPES (transformer.c), so no per-rate constant.
 *   - ENC_STRIDES = {6,5,3} -> HOP_LENGTH = 90.
 *   - DEC_DIMS / DEC_STRIDES differ; DEC_DEPTHS = {2,2,1,1,1}: stages 0 and 1
 *     repeat the ConvUnit TWICE (decoder.blocks.{1,4}.{0,1}.module). The decoder
 *     stage loop already iterates DEC_DEPTHS[s], so this just works.
 *   - VQ_LEVELS = {7,7,7,7,7,4} -> 17-bit FSQ (last dim needs 2 bits).
 * ========================================================================== */
#include "model_config.h"

const int FEATURE_DIM   = 56;
const int SR            = 16000;
const int HOP_LENGTH    = 6 * 5 * 3;          /* product(ENC_STRIDES) */

const int ENC_STAGES    = 3;
const int ENC_DIMS[]    = {16, 24, 48, 64};
const int ENC_STRIDES[] = {6, 5, 3};
const int ENC_DEPTHS[]  = {1, 1, 1, 1};

const int DEC_STAGES    = 4;
const int DEC_DIMS[]    = {64, 44, 32, 24, 16};
const int DEC_STRIDES[] = {5, 3, 3, 2};
const int DEC_DEPTHS[]  = {2, 2, 1, 1, 1};

const int VQ_DIM        = 6;
const int VQ_LEVELS[]   = {7, 7, 7, 7, 7, 4};
const int VQ_BITS[]     = {3, 3, 3, 3, 3, 2};   /* ceil(log2(L)) */
const int VQ_OFFSETS[]  = {0, 3, 6, 9, 12, 15};
const int VQ_TOTAL_BITS = 17;

const int EN_WINDOW      = 250;  /* from 1kbps.toml en_coder_window_size (unused in compressed path) */

/* heads / dim_head / FFN-hidden are read from the weight shapes at runtime. */

/* 1kbps uses the COMPRESSED en path (en_coder_compress_rate=3):
 *   EnCodec.en_encoder = CompressedLocalEncoderWithCache(...)  -- NOTE: en_codec.py
 *     HARDCODES depth=3 for the encoder (independent of the toml's en_coder_depth),
 *     so first_layer_depth = 3//2 = 1, giving
 *     = down_trans.trans(LocalTrans d=1, win=750) + down_layer(Conv1d k=3,s=3)
 *     + local_trans(LocalTrans d=3-1=2, win=250)
 *   EnCodec.en_decoder = CompressedLocalDecoderWithCache(depth=en_coder_depth=2, rate=3)
 *     = local_trans(LocalTrans d=depth-2=0 -> identity) + up_trans(Upsample x3
 *     + LocalTrans d=2, win=750)
 * EN_ENC_DEPTH / EN_DEC_DEPTH are unused here (set 0 for clarity). */
const int EN_USE_COMPRESSED = 1;
const int EN_ENC_COMPRESS_RATE = 3;
const int EN_ENC_DEPTH   = 0;   /* unused (compressed) */
const int EN_DEC_DEPTH   = 0;   /* unused (compressed) */
const int EN_ENC_DOWN_DEPTH = 1; const int EN_ENC_DOWN_WIN = 750;
const int EN_ENC_LOCAL_DEPTH = 2; const int EN_ENC_LOCAL_WIN = 250;
const int EN_DEC_UP_DEPTH = 2; const int EN_DEC_UP_WIN = 750;
const int EN_DEC_LOCAL_DEPTH = 0; const int EN_DEC_LOCAL_WIN = 250;
