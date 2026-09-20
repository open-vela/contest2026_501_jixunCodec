#ifndef QUANTIZER_H
#define QUANTIZER_H
/* ============================================================================
 * quantizer.h — learnable finite-scalar quantizer (C port of l3ac/vq/fsq.py
 * LearnableFSQ). The bit-level Gray pack/unpack codec lives in fsq_codec.h;
 * this file is the *neural* part: project_in -> nearest level -> project_out,
 * plus the decode path that rebuilds features from transmitted indices.
 * ========================================================================== */
#include "nn_ops.h"
#include "fsq_codec.h"

/* Encode path: x[B,T,C=64] -> project_in(64->6) -> quantize -> project_out(6->64).
 * indices[B,T] are the packed 18-bit LearnableFSQ codes (via fsq_codec). */
void quantizer_forward(const T *x, T *q, int64_t *indices, int Tt);

/* Decode path: transmitted indices -> level centers -> inv_act -> project_out.
 * (Used after receiving the 18-bit codes on the decode side.) */
void quantizer_to_features(const int64_t *indices, T *q, int Tt);

#endif /* QUANTIZER_H */
