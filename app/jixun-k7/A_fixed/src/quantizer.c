/* ============================================================================
 * quantizer.c — implementations of the learnable FSQ (encode + decode paths).
 * See quantizer.h for the Python module mapping.
 * ========================================================================== */
#include "quantizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ============================ weight access ============================ */
static const float* quant_center(int d) {
    char n[64]; snprintf(n, sizeof(n), "quantizer.vq.quant_centers.%d", d);
    int shp[4], n_; const float *p = wt_get(n, shp, &n_);
    return p;
}

/* ============================ LearnableFSQ core ============================ */
static void learnable_quantize(const float *z, int D, float *q, int *level_idx) {
    /* LearnableFSQ: act=(tanh(z)+1)/2 in [0,1]; nearest center in act space; inv_act = center*2-1. */
    for (int d = 0; d < D; d++) {
        const float *centers = quant_center(d); int L = VQ_LEVELS[d];
        float az = 0.5f * (tanhf(z[d]) + 1.0f);
        int best = 0; float bd = 1e30f;
        for (int l = 0; l < L; l++) { float dist = fabsf(az - centers[l]); if (dist < bd) { bd = dist; best = l; } }
        level_idx[d] = best; q[d] = centers[best] * 2.0f - 1.0f;
    }
}

/* ============================ encode path ============================ */
void quantizer_forward(const T *x, T *q, int64_t *indices, int Tt) {
    /* x[B,T,C=64] -> project_in(64->6) -> quantize -> project_out(6->64); indices[B,T] */
    int B = x->shape[0], C = x->shape[2], D = VQ_DIM;
    T z = t_alloc(3, (int[]){B, Tt, D});
    linear(W_("quantizer.project_in.weight"), W_("quantizer.project_in.bias"), x, &z, C, D);
    T qz = t_alloc(3, (int[]){B, Tt, D});
    int *li = (int*)malloc((size_t)D * sizeof(int));
    for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tt; t++) {
        learnable_quantize(&z.d[(bi*Tt+t)*D], D, &qz.d[(bi*Tt+t)*D], li);
        uint32_t idx = level_indices_to_learnable_index(li);
        indices[bi * Tt + t] = (int64_t)idx;
    }
    free(li);
    linear(W_("quantizer.project_out.weight"), W_("quantizer.project_out.bias"), &qz, q, D, C);
    q->shape[0] = B; q->shape[1] = Tt; q->shape[2] = C; q->len = B*Tt*C;
    t_free(&z); t_free(&qz);
}

/* ============================ decode path ============================ */
void quantizer_to_features(const int64_t *indices, T *q, int Tt) {
    int B = q->shape[0], C = q->shape[2], D = VQ_DIM;
    int *li = (int*)malloc((size_t)D * sizeof(int));
    T qz = t_alloc(3, (int[]){B, Tt, D});
    for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tt; t++) {
        learnable_index_to_level_indices((uint32_t)indices[bi*Tt+t], li);
        for (int d = 0; d < D; d++) qz.d[(bi*Tt+t)*D+d] = quant_center(d)[li[d]] * 2.0f - 1.0f;
    }
    linear(W_("quantizer.project_out.weight"), W_("quantizer.project_out.bias"), &qz, q, D, C);
    q->shape[0] = B; q->shape[1] = Tt; q->shape[2] = C; q->len = B*Tt*C;
    free(li); t_free(&qz);
}
