#include "fsq_codec.h"

/* Standalone implementation of the Direction-1 codec primitives.
 * All logic lives in the header (static inline); this TU exists so the
 * functions can be linked explicitly and unit-tested if desired.
 *
 * For a full ESP32-S3-EYE decode you additionally need, after obtaining the
 * per-level indices `out[0..5]` via learnable_index_to_level_indices():
 *   1. quant_centers[d][out[d]]  (learnable centers, from model weights)
 *   2. act_value -> inv_act (tanh-style: x*2-1)
 *   3. en_decoder -> decoder  (the lightweight NN, e.g. int8 on ESP32)
 * This file ONLY guarantees the index<->level Gray mapping (encode + decode)
 * matches Python. */
