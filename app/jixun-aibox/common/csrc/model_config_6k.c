/* ============================================================================
 * model_config_6k.c — rate configuration for the 6kbps STREAM (causal) model.
 *
 * Values mirror l3ac/configs/stream_light_6kbps.toml [network_config]
 * (the "light" 6kbps streaming setup trained on the server). Link this TU
 * together with -D SQM_RATE_6K to get a 6kbps build, and point -I at the
 * matching weights/weights_data_6k.h.
 *
 * Topology is identical to 3kbps (3 encoder stages, 4 decoder stages, fixed
 * block indices 0..13); only the striding differs:
 *   3k : compress_rates=[6,4,4]  -> HOP=96,  decode_rates=[4,4,3,2]
 *   6k : compress_rates=[6,4,2]  -> HOP=48,  decode_rates=[4,3,2,2]
 * Because the channel dims (ENC_DIMS/DEC_DIMS) and the quantizer (VQ) are
 * identical, the weight TENSOR SHAPES are byte-for-byte the same as 3kbps —
 * only the stride (a runtime conv arg) and the weight VALUES differ. So the
 * entire C inference codebase is reused unchanged; only this file + the
 * weights_6k/ header are swapped.
 *
 * Heads / dim_head / FFN-hidden are READ FROM THE WEIGHT SHAPES at runtime
 * (transformer.c), so they are NOT declared here.
 * ========================================================================== */
#include "model_config.h"

const int FEATURE_DIM   = 64;
const int SR            = 16000;
const int HOP_LENGTH    = 6 * 4 * 2;          /* product(ENC_STRIDES) = 48 */

const int ENC_STAGES    = 3;
const int ENC_DIMS[]    = {16, 32, 64, 88};
const int ENC_STRIDES[] = {6, 4, 2};          /* 48x downsample */
const int ENC_DEPTHS[]  = {1, 1, 1, 1};

const int DEC_STAGES    = 4;
const int DEC_DIMS[]    = {88, 56, 40, 28, 16};
const int DEC_STRIDES[] = {4, 3, 2, 2};       /* 48x upsample */
const int DEC_DEPTHS[]  = {1, 1, 1, 1, 1};

const int VQ_DIM        = 6;
const int VQ_LEVELS[]   = {8, 8, 8, 7, 7, 7};
const int VQ_BITS[]     = {3, 3, 3, 3, 3, 3};   /* 18 bit/frame */
const int VQ_OFFSETS[]  = {0, 3, 6, 9, 12, 15};
const int VQ_TOTAL_BITS = 18;                  /* 333.33 Hz * 18 = 6000 bps */

const int EN_WINDOW      = 400;  /* from stream_light_6kbps.toml en_coder_window_size */

/* heads / dim_head / FFN-hidden are read from the weight shapes at runtime. */

/* 6kbps uses the STANDARD en path (en_coder_compress_rate=1): LocalEncoder
 * depth is hardcoded=1 by Python; LocalDecoder depth = toml en_coder_depth (=2). */
const int EN_USE_COMPRESSED = 0;
const int EN_ENC_COMPRESS_RATE = 1;
const int EN_ENC_DEPTH   = 1;   /* Python hardcodes LocalEncoder depth=1 */
const int EN_DEC_DEPTH   = 2;   /* LocalDecoder depth = toml en_coder_depth */
const int EN_ENC_DOWN_DEPTH = 0; const int EN_ENC_DOWN_WIN = 0;
const int EN_ENC_LOCAL_DEPTH = 0; const int EN_ENC_LOCAL_WIN = 0;
const int EN_DEC_UP_DEPTH = 0; const int EN_DEC_UP_WIN = 0;
const int EN_DEC_LOCAL_DEPTH = 0; const int EN_DEC_LOCAL_WIN = 0;
