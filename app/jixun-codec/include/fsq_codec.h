#ifndef FSQ_CODEC_H
#define FSQ_CODEC_H

#include <stdint.h>

/* Direction-1 transmission codec for learnable_fsq (ESP32-S3-EYE decode side).
 *
 * Encoder packs 6 quantization-level indices, each Gray-coded into a per-dim
 * bit field (low bit = dim 0), into a single VQ_TOTAL_BITS integer. A single
 * bit flip on the channel therefore corrupts exactly ONE quantization level
 * -> per-level error resilience. This header mirrors, bit-for-bit, LearnableFSQ
 * in sqcodec-light/src/l3ac/vq/fsq.py (level_indices_to_index / index_to_level_indices).
 *
 * The levels / bit layout are RATE-SPECIFIC and come from model_config.h
 * (VQ_LEVELS / VQ_BITS / VQ_OFFSETS / VQ_TOTAL_BITS).  e.g.
 *   3kbps/6kbps: L=[8,8,8,7,7,7]  bits=[3,3,3,3,3,3]  -> 18 total
 *   1kbps      : L=[7,7,7,7,7,4]  bits=[3,3,3,3,3,2]  -> 17 total
 *
 * Two primitives are provided (both directions):
 *   - level_indices_to_learnable_index()  : ENCODE  (levels -> packed idx)
 *   - learnable_index_to_level_indices()  : DECODE  (packed idx -> levels)
 * They are exact inverses of each other for all valid codes.
 */

#include "model_config.h"
#define FSQ_DIM VQ_DIM
extern const int VQ_LEVELS[];
extern const int VQ_BITS[];
extern const int VQ_OFFSETS[];
extern const int VQ_TOTAL_BITS;
/* alias the historical FSQ_* names onto the rate-config VQ_* symbols */
#define FSQ_LEVELS       VQ_LEVELS
#define FSQ_BITS_PER_DIM VQ_BITS
#define FSQ_BIT_OFFSETS  VQ_OFFSETS
#define FSQ_TOTAL_BITS   VQ_TOTAL_BITS

/* Binary-reflected Gray code (matches fsq.py gray_encode). */
static inline uint32_t gray_encode_u32(uint32_t x) {
    return x ^ (x >> 1);
}

/* Gray -> binary. Parallel XOR form, equivalent to fsq.py gray_decode loop. */
static inline uint32_t gray_decode_u32(uint32_t g) {
    g ^= g >> 1;
    g ^= g >> 2;
    g ^= g >> 4;
    g ^= g >> 8;
    g ^= g >> 16;
    return g;
}

/* ENCODE: per-dimension level indices -> packed 18-bit integer.
 * Mirrors LearnableFSQ.level_indices_to_index. */
static inline uint32_t level_indices_to_learnable_index(const int *lv) {
    uint32_t idx = 0;
    for (int d = 0; d < FSQ_DIM; d++) {
        uint32_t gray = gray_encode_u32((uint32_t)lv[d]);
        int off = FSQ_BIT_OFFSETS[d];
        for (int b = 0; b < FSQ_BITS_PER_DIM[d]; b++) {
            idx |= ((gray >> b) & 1u) << (off + b);
        }
    }
    return idx;
}

/* DECODE: packed 18-bit integer -> per-dimension level indices.
 * Mirrors LearnableFSQ.index_to_level_indices -> bits_to_level_indices.
 * Clamps each level to [0, L-1] (defensive; valid codes never exceed). */
static inline void learnable_index_to_level_indices(uint32_t idx, int *out) {
    for (int d = 0; d < FSQ_DIM; d++) {
        uint32_t gray = 0;
        int off = FSQ_BIT_OFFSETS[d];
        int w   = FSQ_BITS_PER_DIM[d];
        for (int b = 0; b < w; b++) {
            gray |= ((idx >> (off + b)) & 1u) << b;
        }
        int lv = (int)gray_decode_u32(gray);
        if (lv < 0) lv = 0;
        if (lv > FSQ_LEVELS[d] - 1) lv = FSQ_LEVELS[d] - 1;
        out[d] = lv;
    }
}

#endif /* FSQ_CODEC_H */
