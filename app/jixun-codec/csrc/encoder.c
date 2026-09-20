/* ============================================================================
 * encoder.c — implementations of the encode-side network.
 * See encoder.h for the Python class mapping.
 * ========================================================================== */
#include "encoder.h"
#include <stddef.h>
extern void jx_stage_h(const char *name, const void *p, size_t nbytes);
#include "jxprof.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <malloc.h>

/* ============================ modules.Encoder.forward ============================ */
void encoder_forward(const T *x, T *out) {
    JXP_BEG(JP_ENCODER);
    int B = x->shape[0], T0 = x->shape[2];
    T cur = t_alloc(3, (int[]){B, 1, T0}); t_copy(&cur, x);
    T b = t_alloc(3, (int[]){B, ENC_DIMS[0], T0}); first_block(&cur, &b); t_free(&cur); cur = b;
    jx_stage_h("enc.fb", b.d, (size_t)b.len * sizeof(float));
    int Tprev = T0;
    for (int i = 0; i < ENC_STAGES; i++) {
        int Ci = ENC_DIMS[i], Co = ENC_DIMS[i+1], st = ENC_STRIDES[i];
        char pn[64]; snprintf(pn, sizeof(pn), "encoder.blocks.%d.0.module", 2*i + 1);
        { char nm[32]; snprintf(nm, sizeof(nm), "enc.s%d.cu", i); jx_stage_h(nm, cur.d, (size_t)cur.len * sizeof(float)); }
        conv_unit_add(&cur, pn);
        T dn = t_alloc(3, (int[]){B, Co, 0});
        char wn[64], bn[64], cn[64], cbn[64];
        snprintf(wn, sizeof(wn), "encoder.blocks.%d.0.weight", 2*i + 2);
        snprintf(bn, sizeof(bn), "encoder.blocks.%d.0.bias", 2*i + 2);
        snprintf(cn, sizeof(cn), "encoder.blocks.%d.1.weight", 2*i + 2);
        snprintf(cbn, sizeof(cbn), "encoder.blocks.%d.1.bias", 2*i + 2);
        if (getenv("JX_ENC_TRACE") != NULL)
          printf("  [CU2] i=%d cur.d=%p len=%d\n", i, (void *)cur.d, (int)cur.len);
        { char nm[32]; snprintf(nm, sizeof(nm), "enc.s%d.cu2", i); jx_stage_h(nm, cur.d, (size_t)cur.len * sizeof(float)); }
        conv1d(W_(wn), W_(bn), &cur, &dn, Ci, Co, st, st, 0, 1, Tprev);
        { char nm[32]; snprintf(nm, sizeof(nm), "enc.s%d.dn", i); jx_stage_h(nm, dn.d, (size_t)dn.len * sizeof(float)); }
        { char nm[32]; snprintf(nm, sizeof(nm), "enc.s%d.raw", i); jx_stage_h(nm, dn.d, (size_t)dn.len * sizeof(float)); }
        channel_norm_first(&dn, W_(cn), W_(cbn), B, Co, dn.shape[2]); t_free(&cur); cur = dn;
        Tprev = cur.shape[2];
    }
    /* final 1x1 conv: ENC_DIMS[last] -> FEATURE_DIM (block 2*ENC_STAGES+1/.2) */
    int Cl = ENC_DIMS[ENC_STAGES];
    char pl[64]; snprintf(pl, sizeof(pl), "encoder.blocks.%d.0.module", 2*ENC_STAGES + 1);
    jx_stage_h("enc.fin.cu", cur.d, (size_t)cur.len * sizeof(float));
    conv_unit_add(&cur, pl);
    T fin = t_alloc(3, (int[]){B, FEATURE_DIM, 0});
    char ew[64], eb[64];
    snprintf(ew, sizeof(ew), "encoder.blocks.%d.weight", 2*ENC_STAGES + 2);
    snprintf(eb, sizeof(eb), "encoder.blocks.%d.bias", 2*ENC_STAGES + 2);
    conv1d(W_(ew), W_(eb), &cur, &fin, Cl, FEATURE_DIM, 3, 1, 1, 1, Tprev);
    jx_stage_h("enc.fin.conv", fin.d, (size_t)fin.len * sizeof(float));
    t_free(&cur); *out = fin;
    JXP_END(JP_ENCODER);
}

/* ============================ LocalEncoder.forward ============================ */
void en_encoder_forward(const T *x, T *out) {
    JXP_BEG(JP_ENENCODER);
    /* x[B,C,T'] -> permute [B,T',C] -> LocalTrans stack -> out[B,T',C] */
    int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
    T pl = t_alloc(3, (int[]){B, Tt, C});
    JXP_BEG(JP_PERMUTE);
    for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tt; t++) for (int c = 0; c < C; c++)
        pl.d[(bi*Tt+t)*C+c] = x->d[(bi*C+c)*Tt+t];
    JXP_END(JP_PERMUTE);

    if (EN_USE_COMPRESSED) {
        /* CompressedLocalEncoderWithCache:
         *   down_trans.trans = LocalTrans(depth=EN_ENC_DOWN_DEPTH, win=EN_ENC_DOWN_WIN)
         *   down_trans.down_layer = Conv1d(k=rate, s=rate)  -> T' -> T'/rate
         *   local_trans = LocalTrans(depth=EN_ENC_LOCAL_DEPTH, win=EN_ENC_LOCAL_WIN) */
        T dt;                       /* local_transformer 自己分配并交接出来 */
        local_transformer(&pl, &dt, "en_encoder.down_trans.trans", EN_ENC_DOWN_DEPTH, EN_ENC_DOWN_WIN);
        t_free(&pl);
        /* down_layer: permute (B,T,C)->(B,C,T), conv1d k=s=rate, permute back */
        int Td = Tt / EN_ENC_COMPRESS_RATE;   /* Tt is divisible by rate */
        T dtc = t_alloc(3, (int[]){B, C, Tt});
        for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tt; t++) for (int c = 0; c < C; c++)
            dtc.d[(bi*C+c)*Tt+t] = dt.d[(bi*Tt+t)*C+c];
        t_free(&dt);
        T dconv = t_alloc(3, (int[]){B, C, Td});
        conv1d(W_("en_encoder.down_trans.down_layer.weight"), W_("en_encoder.down_trans.down_layer.bias"),
               &dtc, &dconv, C, C, EN_ENC_COMPRESS_RATE, EN_ENC_COMPRESS_RATE, 0, 1, Tt);
        t_free(&dtc);
        T dconvp = t_alloc(3, (int[]){B, Td, C});
        for (int bi = 0; bi < B; bi++) for (int t = 0; t < Td; t++) for (int c = 0; c < C; c++)
            dconvp.d[(bi*Td+t)*C+c] = dconv.d[(bi*C+c)*Td+t];
        t_free(&dconv);
        T lt;                       /* 同上，避免再把一份全尺寸占位块丢掉 */
        local_transformer(&dconvp, &lt, "en_encoder.local_trans", EN_ENC_LOCAL_DEPTH, EN_ENC_LOCAL_WIN);
        t_free(&dconvp);
        *out = lt;
    } else {
        T tr;                       /* 同上 */
        local_transformer(&pl, &tr, "en_encoder.local_trans", EN_ENC_DEPTH, EN_WINDOW);
        t_free(&pl);
        *out = tr;
    }
    JXP_END(JP_ENENCODER);
}
