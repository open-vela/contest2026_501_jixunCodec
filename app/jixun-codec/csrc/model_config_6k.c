/* ============================================================================
 * model_config_6k.c — rate configuration for the 6kbps model.
 *
 * Values mirror configs/6kbps.toml [nn_config] (the "light" 6kbps setup the
 * user trained on the server). Link this TU instead of model_config_3k.c to
 * get a 6kbps build, together with the matching weights_data.h / golden.h.
 *
 * Topology is identical to 3kbps (3 encoder stages, 4 decoder stages); only
 * the striding and quantizer window differ.  Heads / dim_head / FFN-hidden are
 * read from the weight shapes at runtime (transformer.c), so they are not set
 * here.
 * ========================================================================== */
#include "model_config.h"

const int FEATURE_DIM   = 64;
const int SR            = 16000;
const int HOP_LENGTH    = 6 * 4 * 2;          /* product(ENC_STRIDES) */

const int ENC_STAGES    = 3;
const int ENC_DIMS[]    = {16, 32, 64, 88};
const int ENC_STRIDES[] = {6, 4, 2};
const int ENC_DEPTHS[]  = {1, 1, 1, 1};

const int DEC_STAGES    = 4;
const int DEC_DIMS[]    = {88, 56, 40, 28, 16};
const int DEC_STRIDES[] = {4, 3, 2, 2};
const int DEC_DEPTHS[]  = {1, 1, 1, 1, 1};

const int VQ_DIM        = 6;
const int VQ_LEVELS[]   = {8, 8, 8, 7, 7, 7};
const int VQ_BITS[]     = {3, 3, 3, 3, 3, 3};
const int VQ_OFFSETS[]  = {0, 3, 6, 9, 12, 15};
const int VQ_TOTAL_BITS = 18;

const int EN_WINDOW      = 400;  /* from 6kbps.toml en_coder_window_size */

/* heads / dim_head / FFN-hidden are read from the weight shapes at runtime. */

/* 6kbps uses the STANDARD en path (en_coder_compress_rate=1), so en_encoder
 * depth is hardcoded=1 by Python (NOT the toml en_coder_depth=2); LocalDecoder
 * depth = toml en_coder_depth (=2). */
const int EN_USE_COMPRESSED = 0;
const int EN_ENC_COMPRESS_RATE = 1;
const int EN_ENC_DEPTH   = 1;   /* Python hardcodes LocalEncoder depth=1 (uncompressed path) */
const int EN_DEC_DEPTH   = 2;   /* LocalDecoder depth = toml en_coder_depth */
const int EN_ENC_DOWN_DEPTH = 0; const int EN_ENC_DOWN_WIN = 0;
const int EN_ENC_LOCAL_DEPTH = 0; const int EN_ENC_LOCAL_WIN = 0;
const int EN_DEC_UP_DEPTH = 0; const int EN_DEC_UP_WIN = 0;
const int EN_DEC_LOCAL_DEPTH = 0; const int EN_DEC_LOCAL_WIN = 0;
