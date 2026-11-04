/* ============================================================================
 * nn_ops.c — implementations of l3ac/layers.py primitives.
 * See nn_ops.h for the Python-aligned documentation.
 * the exported weight table (weights_data_3k.h / weights_data_6k.h) is included
 * here exactly once so the tensor/weight lookups have a single home, matching
 * the original layout.
 * ========================================================================== */
#include "nn_ops.h"
#include "jx_neon.h"
#ifndef WEIGHTS_HEADER
#define WEIGHTS_HEADER "weights_data_3k.h"   /* 3k 默认；6k 经 build.sh -DWEIGHTS_HEADER 切换 */
#endif
#include WEIGHTS_HEADER
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>

/* ============================ tensor helpers ============================ */
T t_alloc(int ndim, const int *shape) {
    T t; t.ndim = ndim; int len = 1;
    for (int i = 0; i < ndim; i++) { t.shape[i] = shape[i]; len *= shape[i]; }
    t.len = len; t.d = (float*)calloc((size_t)len, sizeof(float));
    return t;
}
void t_free(T *t) { if (t->d) free(t->d); t->d = NULL; t->len = 0; }
T t_zeros_like(const T *s) { return t_alloc(s->ndim, s->shape); }
void t_copy(T *dst, const T *src) { memcpy(dst->d, src->d, (size_t)src->len * sizeof(float)); }

/* Native evaluation only: valid float fallback for quantized tensors.
 * Quantized convolution/linear continues to consume the original int8 bytes
 * and scales. This cache is not a proposed ESP32 allocation strategy. */
static float *native_float_cache[NW_TENSORS];
/* ============================ weight lookup ============================ */
const float* wt_get(const char *name, int *shape4, int *n_out) {
    for (int i = 0; i < NW_TENSORS; i++)
        if (strcmp(WT_TAB[i].name, name) == 0) {
            if (shape4) for (int j = 0; j < 4; j++) shape4[j] = WT_TAB[i].shape[j];
            if (n_out) *n_out = WT_TAB[i].n;
            if (!WT_TAB[i].q) {
                if (WT_TAB[i].off < 0 || WT_TAB[i].off + WT_TAB[i].n > NW_FLOATS) abort();
                return WT_FDATA + WT_TAB[i].off;
            }
            if (!native_float_cache[i]) {
                int n=WT_TAB[i].n, scales=WT_TAB[i].scale_n;
                if (scales<1 || n%scales || WT_TAB[i].off<0 || WT_TAB[i].off+n>NW_INT8 || WT_TAB[i].scale_off<0 || WT_TAB[i].scale_off+scales>NW_SCALES) abort();
                native_float_cache[i]=malloc((size_t)n*sizeof(float));
                if (!native_float_cache[i]) abort();
                for(int j=0;j<n;j++) native_float_cache[i][j]=(float)WT_IDATA[WT_TAB[i].off+j]*WT_SDATA[WT_TAB[i].scale_off+j/(n/scales)];
            }
            return native_float_cache[i];
        }
    return NULL;
}
int wt_has(const char *name) { for (int i = 0; i < NW_TENSORS; i++) if (!strcmp(WT_TAB[i].name, name)) return 1; return 0; }
/* Encoder and decoder execute concurrently; each thread owns its name ring.
 * The bare-metal libgcc implements emulated _Thread_local as a single global
 * address. POSIX keys provide actual thread isolation on Linux and NuttX. */
#include <pthread.h>
struct name_ring {char text[8][256];unsigned next;};
static pthread_key_t name_key;
#ifndef __NuttX__
static pthread_once_t name_once=PTHREAD_ONCE_INIT;
#endif
static int name_key_error;
static void name_key_init(void) {name_key_error=pthread_key_create(&name_key,free);}
#ifdef __NuttX__
/* Flat-build globals outlive a command's task group, while NuttX pthread keys
 * belong to that group. Claim/release one key for each voice invocation. */
int wt_names_begin(void) {name_key_init();return name_key_error;}
void wt_names_end(void) {
    void *ring=pthread_getspecific(name_key);
    if(ring){pthread_setspecific(name_key,NULL);free(ring);}
    pthread_key_delete(name_key);
}
#endif
const char* cat_name(const char *pfx, const char *suffix) {
#ifdef __NuttX__
    if(name_key_error)abort();
#else
    if(pthread_once(&name_once,name_key_init)||name_key_error)abort();
#endif
    struct name_ring *ring=pthread_getspecific(name_key);
    if(!ring) {
        ring=calloc(1,sizeof(*ring));
        if(!ring)abort();
        if(pthread_setspecific(name_key,ring)){free(ring);abort();}
    }
    unsigned i=ring->next;ring->next=(i+1)&7;
    snprintf(ring->text[i],sizeof(ring->text[i]),"%s.%s",pfx,suffix);
    return ring->text[i];
}
const float* W_(const char *name) {
    const float *p = wt_get(name, NULL, NULL);
    if (!p) fprintf(stderr, "MISSING WEIGHT: %s\n", name);
    return p;
}

/* ---- weight-only INT8 lookup ---- */
const int8_t* wt_get_i8(const char *name, const float **scale, int *scale_n, int *shape4, int *n_out) {
    for (int i = 0; i < NW_TENSORS; i++)
        if (strcmp(WT_TAB[i].name, name) == 0) {
            if (WT_TAB[i].q != 1) return NULL;            /* 非量化张量 */
            if (shape4) for (int j = 0; j < 4; j++) shape4[j] = WT_TAB[i].shape[j];
            if (n_out) *n_out = WT_TAB[i].n;
            if (scale) *scale = &WT_SDATA[WT_TAB[i].scale_off];
            if (scale_n) *scale_n = WT_TAB[i].scale_n;
            return &WT_IDATA[WT_TAB[i].off];
        }
    return NULL;
}
const int8_t* W_i8(const char *name, const float **scale, int *scale_n) {
    const int8_t *p = wt_get_i8(name, scale, scale_n, NULL, NULL);
    if (!p) fprintf(stderr, "MISSING INT8 WEIGHT: %s\n", name);
    return p;
}

/* ============================ activations / norms ============================ */
#ifdef JX_NEON_ENABLED
struct jx_snake_ctx {float *x;const float *a,*inverse;int C,T,last;};
static void jx_snake_range(void *opaque,int first,int end) {
    const struct jx_snake_ctx *c=opaque;int at=first;
    while(at<end) {
        if(c->last) {
            int channel=at%c->C,stop=at+c->C-channel;if(stop>end)stop=end;
            for(;at<stop;at++,channel++) {
                float v=c->x[at],s=sinf(c->a[channel]*v);
                c->x[at]=v+c->inverse[channel]*(s*s);
            }
        } else {
            int channel=(at/c->T)%c->C,stop=(at/c->T+1)*c->T;if(stop>end)stop=end;
            float a=c->a[channel],inv=c->inverse[channel];
            for(;at<stop;at++){float v=c->x[at],s=sinf(a*v);c->x[at]=v+inv*(s*s);}
        }
    }
}
static void jx_gelu_range(void *opaque,int first,int end) {
    float *x=opaque;for(int i=first;i<end;i++)x[i]=gelu_f(x[i]);
}
#endif
void gelu_apply(float *x,int n) {
#ifdef JX_NEON_ENABLED
    jx_parallel_run(jx_gelu_range,x,n,16,(size_t)n*64);
#else
    for(int i=0;i<n;i++)x[i]=gelu_f(x[i]);
#endif
}
static void snake_apply(float *x, const float *alpha, int n, int ch) {
    /* channels_last layout: element (i, c) lives at x[i*ch + c] */
    for (int c = 0; c < ch; c++) {
        float a = alpha[c] + XNN_EPS; float inv = 1.0f / a;
        for (int i = 0; i < n; i++) { float v = x[i * ch + c]; x[i * ch + c] = v + inv * (sinf(a * v) * sinf(a * v)); }
    }
}
void snake1d(T *x, const float *alpha, int channels, int dim1, int dim2, int data_format) {
#ifdef JX_NEON_ENABLED
    int C=data_format?x->shape[2]:x->shape[1];
    float a[C],inverse[C];
    for(int c=0;c<C;c++){a[c]=alpha[c]+XNN_EPS;inverse[c]=1.0f/a[c];}
    struct jx_snake_ctx context={x->d,a,inverse,C,data_format?x->shape[1]:x->shape[2],data_format==1};
    jx_parallel_run(jx_snake_range,&context,x->len,16,(size_t)x->len*64);
    return;
#endif
    if (data_format == 1) {
        int B = x->shape[0], Tt = x->shape[1], C = x->shape[2];
        for (int b = 0; b < B; b++) snake_apply(&x->d[b * Tt * C], alpha, Tt, C);
    } else {
        int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
        for (int b = 0; b < B; b++) for (int c = 0; c < C; c++) {
            float a = alpha[c] + XNN_EPS; float inv = 1.0f / a;
            float *p = &x->d[(b * C + c) * Tt];
            for (int i = 0; i < Tt; i++) { float v = p[i]; p[i] = v + inv * (sinf(a * v) * sinf(a * v)); }
        }
    }
}
void channel_norm_last(T *x, const float *w, const float *b, int B, int Tt, int C) {
    for (int i = 0; i < B * Tt; i++) {
        float *p = &x->d[i * C]; float mean = 0;
        for (int c = 0; c < C; c++) mean += p[c]; mean /= C;
        float var = 0; for (int c = 0; c < C; c++) { float d = p[c] - mean; var += d * d; } var /= C;
        float inv = 1.0f / sqrtf(var + XNN_EPS);
        for (int c = 0; c < C; c++) p[c] = w[c] * (p[c] - mean) * inv + b[c];
    }
}
void channel_norm_first(T *x, const float *w, const float *b, int B, int C, int Tt) {
    /* ChannelNorm(channels_first): normalize over the channel dim (C) for each (b,t). */
    for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tt; t++) {
        float mean = 0;
        for (int c = 0; c < C; c++) mean += x->d[(bi * C + c) * Tt + t];
        mean /= C;
        float var = 0; for (int c = 0; c < C; c++) { float d = x->d[(bi * C + c) * Tt + t] - mean; var += d * d; } var /= C;
        float inv = 1.0f / sqrtf(var + XNN_EPS);
        for (int c = 0; c < C; c++) {
            float v = x->d[(bi * C + c) * Tt + t];
            x->d[(bi * C + c) * Tt + t] = w[c] * (v - mean) * inv + b[c];
        }
    }
}
void grn_last(T *x, const float *gamma, const float *beta, int B, int Tt, int C) {
    /* GRN(channels_last): g = ||x|| over (T,C) per sample; n_x = g/(g+eps); gamma/beta per-channel. */
    for (int bi = 0; bi < B; bi++) {
        float *p = &x->d[bi * Tt * C]; float ss = 0;
        for (int i = 0; i < Tt * C; i++) ss += p[i] * p[i];
        float g = sqrtf(ss); float n_x = g / (g + XNN_EPS);
        for (int i = 0; i < Tt * C; i++) { int c = i % C; p[i] = gamma[c] * (p[i] * n_x) + beta[c] + p[i]; }
    }
}
void grn_first(T *x, const float *gamma, const float *beta, int B, int C, int Tt) {
    /* GRN(channels_first): g = ||x|| over (C,T) per sample; n_x = g/(g+eps); gamma/beta per-channel. */
    for (int bi = 0; bi < B; bi++) {
        float ss = 0; int base = bi * C * Tt;
        for (int i = 0; i < C * Tt; i++) ss += x->d[base + i] * x->d[base + i];
        float g = sqrtf(ss); float n_x = g / (g + XNN_EPS);
        for (int c = 0; c < C; c++) for (int t = 0; t < Tt; t++) {
            int idx = base + c * Tt + t;
            x->d[idx] = gamma[c] * (x->d[idx] * n_x) + beta[c] + x->d[idx];
        }
    }
}
void instance_norm1d(T *x, const float *w, const float *b, int B, int C, int Tt, float eps) {
    /* nn.InstanceNorm1d(C, affine=True): per-sample, per-channel normalize over T. */
    for (int bi = 0; bi < B; bi++) for (int c = 0; c < C; c++) {
        float *p = &x->d[(bi * C + c) * Tt];
        float mean = 0; for (int t = 0; t < Tt; t++) mean += p[t]; mean /= Tt;
        float var = 0; for (int t = 0; t < Tt; t++) { float d = p[t] - mean; var += d * d; } var /= Tt;
        float inv = 1.0f / sqrtf(var + eps);
        for (int t = 0; t < Tt; t++) p[t] = w[c] * (p[t] - mean) * inv + b[c];
    }
}

/* ============================ layers ============================ */
void linear(const float *W, const float *b, const T *x, T *y, int in, int out) {
    /* Apply the same Linear (rows = in -> out) to every trailing "row" of the
     * (..., in) tensor. For (B, Tt, C) input, rows = B*Tt. This is dense
     * row-major so x->len/in gives the correct count of independent rows.
     * Accumulate in double to keep cross-language agreement tight (esp. for
     * long dot products like the 4*C pw_conv in ConvUnit / FFN). */
    int rows = x->len / in;
    for (int i = 0; i < rows; i++) {
        const float *xi = &x->d[i * in]; float *yi = &y->d[i * out];
        for (int o = 0; o < out; o++) {
            const float *wr = &W[o * in]; acc_t acc = (b ? (acc_t)b[o] : 0.0);
            for (int j = 0; j < in; j++) acc += (acc_t)xi[j] * (acc_t)wr[j];
            yi[o] = (float)acc;
        }
    }
}
#ifdef JX_NEON_ENABLED
struct jx_linear_ctx {const int8_t *Wi;const float *sc,*b,*packed;const T *x;T *y;int in,out,row_first,row_end;};
static void jx_linear_range(void *opaque,int first,int end) {
    const struct jx_linear_ctx *c=opaque;
    const int8_t *Wi=c->Wi;const float *sc=c->sc,*b=c->b,*packed=c->packed;
    const T *x=c->x;T *y=c->y;int in=c->in,out=c->out;
        for(int i=c->row_first;i<c->row_end;i++) {
            const float *xi=x->d+(size_t)i*in;float *yi=y->d+(size_t)i*out;int o=first;
            for(;o+15<end;o+=16) {
                float32x4_t s0=vdupq_n_f32(0),s1=s0,s2=s0,s3=s0;
                const float *w0=packed+(size_t)o*in,*w1=w0+4*in,*w2=w1+4*in,*w3=w2+4*in;
                for(int j=0;j<in;j++) {
                    float xj=xi[j];
                    s0=vfmaq_n_f32(s0,vld1q_f32(w0+4*j),xj);
                    s1=vfmaq_n_f32(s1,vld1q_f32(w1+4*j),xj);
                    s2=vfmaq_n_f32(s2,vld1q_f32(w2+4*j),xj);
                    s3=vfmaq_n_f32(s3,vld1q_f32(w3+4*j),xj);
                }
                float values[16];vst1q_f32(values,s0);vst1q_f32(values+4,s1);vst1q_f32(values+8,s2);vst1q_f32(values+12,s3);
                for(int lane=0;lane<16;lane++)yi[o+lane]=(float)(sc[o+lane]*values[lane]+(b?(acc_t)b[o+lane]:0.0));
            }
            for(;o+3<end;o+=4) {
                float32x4_t sum=vdupq_n_f32(0);
                const float *w=packed+(size_t)o*in;
                for(int j=0;j<in;j++)sum=vfmaq_n_f32(sum,vld1q_f32(w+4*j),xi[j]);
                float values[4];vst1q_f32(values,sum);
                for(int lane=0;lane<4;lane++)yi[o+lane]=(float)(sc[o+lane]*values[lane]+(b?(acc_t)b[o+lane]:0.0));
            }
            for(;o<end;o++) {
                acc_t sum=0;
                for(int j=0;j<in;j++)sum+=(acc_t)Wi[(size_t)o*in+j]*(acc_t)xi[j];
                yi[o]=(float)(sc[o]*sum+(b?(acc_t)b[o]:0.0));
            }
        }
}
static void jx_linear_rows(void *opaque,int first,int end) {
    struct jx_linear_ctx context=*(struct jx_linear_ctx *)opaque;
    context.row_first=first;context.row_end=end;
    jx_linear_range(&context,0,context.out);
}
#endif
void linear_q(const int8_t *Wi, const float *sc, const float *b, const T *x, T *y, int in, int out) {
#ifdef JX_NEON_ENABLED
    const float *packed=jx_neon_pack_i8(Wi,in,out);
    if(packed) {
        int rows=x->len/in;
        struct jx_linear_ctx context={Wi,sc,b,packed,x,y,in,out,0,rows};
        if(rows>=12)jx_parallel_run(jx_linear_rows,&context,rows,1,(size_t)rows*in*out);
        else jx_parallel_run(jx_linear_range,&context,out,16,(size_t)rows*in*out);
        return;
    }
#endif
    /* weight-only INT8：y[o] = scale[o] * Σ Wi[o,j]*x[j] + b[o]。
     * 与 Python ptq 的 dequant 路径 y = Σ(int8*scale)*x + b 数学等价
     * （分配律），差异仅在浮点结合律（C 用 double 反更准）。 */
    int rows = x->len / in;
    for (int i = 0; i < rows; i++) {
        const float *xi = &x->d[i * in]; float *yi = &y->d[i * out];
        for (int o = 0; o < out; o++) {
            const int8_t *wr = &Wi[(size_t)o * in]; acc_t acc = 0.0;
            for (int j = 0; j < in; j++) acc += (acc_t)wr[j] * (acc_t)xi[j];
            yi[o] = (float)(sc[o] * acc + (b ? (acc_t)b[o] : 0.0));
        }
    }
}
static void conv1d_core(const float *W, const float *b, const float *xin, float *yout,
                        int Ci, int Co, int k, int stride, int pad, int groups, int T_in, int T_out, int dil) {
    int cpg = Ci / groups;
    for (int co = 0; co < Co; co++) {
        int g = co / (Co / groups);
        for (int t = 0; t < T_out; t++) {
            acc_t acc = (b ? (acc_t)b[co] : 0.0); int base_ci = g * cpg;
            for (int ci = 0; ci < cpg; ci++) {
                const float *wr = &W[(co * cpg + ci) * k];
                for (int kk = 0; kk < k; kk++) {
                    int pos = t * stride - pad + kk * dil;
                    if (pos >= 0 && pos < T_in) acc += (acc_t)xin[(base_ci + ci) * T_in + pos] * (acc_t)wr[kk];
                }
            }
            yout[co * T_out + t] = (float)acc;
        }
    }
}
void conv1d(const float *W, const float *b, const T *x, T *y, int Ci, int Co, int k, int stride, int pad, int groups, int T_in) {
    int T_out = (T_in + 2 * pad - k) / stride + 1;
    int need = x->shape[0] * Co * T_out;
    if (y->len < need) y->d = (float*)realloc(y->d, (size_t)need * sizeof(float));
    y->shape[0] = x->shape[0]; y->shape[1] = Co; y->shape[2] = T_out; y->len = need;
    int B = x->shape[0];
    for (int bi = 0; bi < B; bi++)
        conv1d_core(W, b, &x->d[bi * Ci * T_in], &y->d[bi * Co * T_out], Ci, Co, k, stride, pad, groups, T_in, T_out, 1);
}
void conv1d_d(const float *W, const float *b, const T *x, T *y, int Ci, int Co, int k, int stride, int pad, int groups, int T_in, int dil) {
    int T_out = (T_in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    int need = x->shape[0] * Co * T_out;
    if (y->len < need) y->d = (float*)realloc(y->d, (size_t)need * sizeof(float));
    y->shape[0] = x->shape[0]; y->shape[1] = Co; y->shape[2] = T_out; y->len = need;
    int B = x->shape[0];
    for (int bi = 0; bi < B; bi++)
        conv1d_core(W, b, &x->d[bi * Ci * T_in], &y->d[bi * Co * T_out], Ci, Co, k, stride, pad, groups, T_in, T_out, dil);
}
void conv1d_dw(const float *W, const float *b, const T *x, T *y, int C, int k, int stride, int pad, int T_in) {
    conv1d(W, b, x, y, C, C, k, stride, pad, C, T_in);
}
void upsampling_linear(const T *x, T *y, int B, int C, int Tin, int scale) {
    int Tout = Tin * scale;
    int need = B * C * Tout;
    if (y->len < need) y->d = (float*)realloc(y->d, (size_t)need * sizeof(float));
    for (int bi = 0; bi < B; bi++) for (int c = 0; c < C; c++) for (int o = 0; o < Tout; o++) {
        /* nn.Upsample(scale_factor=s, mode='linear', align_corners=False) with integer s:
           real src coord = ((o + 0.5) * Tin / Tout - 0.5), matches PyTorch exactly.
           Use double for src/frac: for large Tin (e.g. 83520) float32 loses ~0.005 absolute
           precision in (src - floor(src)), corrupting the interp. Out-of-range -> clamp edge. */
        double src = (Tout > 1) ? ((double)o + 0.5) * (double)Tin / (double)Tout - 0.5 : 0.0;
        int i0 = (int)floor(src); double frac = src - (double)i0; int i1 = i0 + 1;
        float v0 = (i0 < 0) ? x->d[(bi * C + c) * Tin + 0] : (i0 >= Tin ? x->d[(bi * C + c) * Tin + (Tin - 1)] : x->d[(bi * C + c) * Tin + i0]);
        float v1 = (i1 < 0) ? x->d[(bi * C + c) * Tin + 0] : (i1 >= Tin ? x->d[(bi * C + c) * Tin + (Tin - 1)] : x->d[(bi * C + c) * Tin + i1]);
        y->d[(bi * C + c) * Tout + o] = v0 * (float)(1.0 - frac) + v1 * (float)frac;
    }
    y->shape[0] = B; y->shape[1] = C; y->shape[2] = Tout; y->len = B * C * Tout;
}
