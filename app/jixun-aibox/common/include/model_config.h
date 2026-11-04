#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H
#include <stdint.h>

/* ============================================================================
 * model_config.h — rate-dependent model configuration (single source of truth)
 *
 * Every quantity that differs between codec rates (3kbps / 6kbps / 1kbps) is
 * declared here as `extern const` and DEFINED in a matching model_config_*.c
 * (generated from the Python toml by the export script, or hand-written).
 *
 * The C inference code reads all of these from here, so ONE codebase serves
 * every rate: just link the matching model_config_*.c together with the
 * corresponding weights_data.h / golden.h.
 *
 * Topology that is identical across all rates (encoder = 3 stride stages +
 * 1 final 1x1; decoder = 4 stages each [ConvUnit, Enhance, Up] + legacy last;
 * block indices 0..13 fixed) is NOT configurable — only the numeric values
 * (dims, strides, depths, quantizer levels, local-attn params) are.
 * ========================================================================== */

/* ---- scalars ---- */
extern const int FEATURE_DIM;            /* quantizer / transformer channel dim */
extern const int SR;                     /* sample rate (16000) */
extern const int HOP_LENGTH;             /* = product(ENC_STRIDES) */

/* ---- encoder (modules.Encoder) ---- */
extern const int ENC_STAGES;             /* = len(ENC_DIMS) - 1 (=3) */
extern const int ENC_DIMS[];             /* [stem, d1, d2, d3]      len = ENC_STAGES+1 */
extern const int ENC_STRIDES[];          /* per-stage conv stride    len = ENC_STAGES */
extern const int ENC_DEPTHS[];           /* per-block ConvUnit count len = ENC_STAGES+1 */

/* ---- decoder (modules.Decoder) ---- */
extern const int DEC_STAGES;             /* = len(DEC_DIMS) - 1 (=4) */
extern const int DEC_DIMS[];             /* [d0..d_last]             len = DEC_STAGES+1 */
extern const int DEC_STRIDES[];          /* per-stage upsample scale len = DEC_STAGES */
extern const int DEC_DEPTHS[];           /* per-stage ConvUnit count len = DEC_STAGES+1 */

/* ---- quantizer (learnable_fsq) ---- */
extern const int VQ_DIM;                 /* number of FSQ levels (6) */
extern const int VQ_LEVELS[];            /* per-dim codebook size    len = VQ_DIM */
extern const int VQ_BITS[];              /* bits per dim = ceil(log2(L)) len = VQ_DIM */
extern const int VQ_OFFSETS[];           /* bit offset (prefix sum)  len = VQ_DIM */
extern const int VQ_TOTAL_BITS;          /* sum(VQ_BITS) */

/* ---- local attention (LocalEncoder / LocalDecoder, plus compressed variants) ---- */
/* The number of heads, head dim, and FFN hidden size are READ FROM THE WEIGHT
 * SHAPES inside transformer.c (see local_mha_layer / feed_forward), so they are
 * NOT declared here.  What IS config-driven:
 *   - EN_USE_COMPRESSED : 0 => standard LocalEncoder/LocalDecoder (3k/6k);
 *                         1 => CompressedLocal{Encoder,Decoder}WithCache (1kbps),
 *                             which downsamples T by EN_ENC_COMPRESS_RATE inside
 *                             the en-path (DownTrans / UpTransV2).
 *   - EN_WINDOW         : causal window for the standard (uncompressed) en path.
 *   - EN_ENC_DEPTH / EN_DEC_DEPTH : LocalTrans depth for the standard en path
 *                             (Python hardcodes en_encoder depth=1; en_decoder
 *                             depth = toml en_coder_depth).
 *   - EN_*_DOWN/_LOCAL/_UP : per-sub-block depth + window for the compressed path. */
extern const int EN_USE_COMPRESSED;      /* 0 standard, 1 compressed (1kbps) */
extern const int EN_ENC_COMPRESS_RATE;   /* T downsample/upsample factor (1 std, 3 1kbps) */
extern const int EN_WINDOW;              /* causal window (standard en path) */
extern const int EN_ENC_DEPTH;           /* standard en_encoder LocalTrans depth (hardcoded=1) */
extern const int EN_DEC_DEPTH;           /* standard en_decoder LocalTrans depth (= toml en_coder_depth) */
extern const int EN_ENC_DOWN_DEPTH;      /* compressed: down_trans.trans depth */
extern const int EN_ENC_DOWN_WIN;        /* compressed: down_trans window */
extern const int EN_ENC_LOCAL_DEPTH;     /* compressed: en_encoder.local_trans depth */
extern const int EN_ENC_LOCAL_WIN;       /* compressed: en_encoder.local_trans window */
extern const int EN_DEC_UP_DEPTH;        /* compressed: up_trans.trans depth */
extern const int EN_DEC_UP_WIN;          /* compressed: up_trans window */
extern const int EN_DEC_LOCAL_DEPTH;     /* compressed: en_decoder.local_trans depth (0 => identity) */
extern const int EN_DEC_LOCAL_WIN;       /* compressed: en_decoder.local_trans window */

#endif /* MODEL_CONFIG_H */
