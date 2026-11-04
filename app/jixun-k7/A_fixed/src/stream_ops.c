/* ============================================================================
 * stream_ops.c — 严格因果增量算子实现。语义逐行对齐 l3ac/stream_cache.py。
 *
 * 内存布局
 * --------
 * SHist 是 [C][cap] 行主序，行跨距 = cap；"活跃"数据是每行的前 len 列。
 * 因此 `行 c 的列 col` 的地址 = h->d + c*cap + col。所有逐行操作都按这个
 * 公式走（拼接 = 每行各自的 hist ++ x），不能当成一整块 memcpy。
 *
 * 数值约定（刻意与 nn_ops.c 的 conv1d_core / linear 保持一致）
 * ---------------------------------------------------------
 *   * 卷积内积、attention 的 QKᵀ / attn·V、FrameNorm 的统计量都用 double 累加，
 *     输出一次性 float 化；这样"整段"与"分块"两条路径的差异只来自求和区间，
 *     而不是累加精度。
 * ========================================================================== */
#include "stream_ops.h"
#include "jx_neon.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* 行 c 的起始地址 */
#define SROW(h, c) ((h)->d + (size_t)(c) * (size_t)(h)->cap)

/* ============================ arena ============================
 * 注意：arena 一旦分配就**不再移动**（绝不用 realloc 扩容），否则先前返回的
 * 指针会全部失效。容量上限由 sa_init 给定，越界直接报错退出（开发期工具，
 * 明确失败优于静默踩内存）。 */
void sa_init(SArena *a, size_t cap) {
    a->cap = cap ? cap : 1024; a->off = 0; a->max_off = 0;
    a->d = (float*)malloc(a->cap * sizeof(float));
    if (!a->d) { fprintf(stderr, "sa_init: OOM (%zu floats)\n", a->cap); exit(1); }
}
void sa_reset(SArena *a) { a->off = 0; }
float *sa_alloc(SArena *a, size_t n) {
    if (a->off + n > a->cap) {
        fprintf(stderr, "sa_alloc: arena 溢出 (need %zu, off %zu, cap %zu) —— 调大 sa_init 的 cap\n",
                n, a->off, a->cap);
        exit(1);
    }
    size_t end = a->off + n;
    if (end > a->max_off) a->max_off = end;
    float *p = a->d + a->off; a->off += n;
    return p;
}
void sa_free(SArena *a) { if (a->d) free(a->d); a->d = NULL; a->cap = a->off = 0; }

/* ============================ SHist ============================ */
void shist_free(SHist *h) { if (h->d) free(h->d); h->d = NULL; h->len = h->cap = 0; }

void shist_reset(SHist *h, int C, int init_len, int cap_hint) {
    int cap = cap_hint > 0 ? cap_hint : (init_len > 0 ? init_len : 1);
    if (cap < init_len) cap = init_len;
    if (cap < 1) cap = 1;
    shist_free(h);
    h->cap = cap; h->len = init_len;
    h->d = (float*)calloc((size_t)C * cap, sizeof(float));
    if (!h->d) { fprintf(stderr, "shist_reset: OOM C=%d cap=%d\n", C, cap); exit(1); }
}

/* 扩容：[C][cap_old] -> [C][cap_new]，保留每行前 len 列（逐行搬，互不干扰）。 */
void shist_ensure(SHist *h, int C, int need) {
    if (need <= h->cap) return;
    int ncap = h->cap > 0 ? h->cap : 1;
    while (ncap < need) ncap *= 2;
    float *nd = (float*)malloc((size_t)C * ncap * sizeof(float));
    if (!nd) { fprintf(stderr, "shist_ensure: OOM C=%d ncap=%d\n", C, ncap); exit(1); }
    for (int c = 0; c < C; c++)
        memcpy(&nd[(size_t)c * ncap], &h->d[(size_t)c * h->cap], (size_t)h->len * sizeof(float));
    free(h->d); h->d = nd; h->cap = ncap;
}

/* ============================ SKV ============================ */
void skv_free(SKV *s) { if (s->d) free(s->d); s->d = NULL; s->len = s->cap = 0; }

void skv_ensure(SKV *s, int HD, int need) {
    if (need <= s->cap) return;
    int ncap = s->cap > 0 ? s->cap : 16;
    while (ncap < need) ncap *= 2;
    float *nd = (float*)malloc((size_t)HD * ncap * sizeof(float));
    if (!nd) { fprintf(stderr, "skv_ensure: OOM HD=%d ncap=%d\n", HD, ncap); exit(1); }
    if (s->len > 0) memcpy(nd, s->d, (size_t)HD * s->len * sizeof(float));
    free(s->d); s->d = nd; s->cap = ncap;
}

/* ============================ 卷积内核 ============================ */
/* xbuf 行跨距 = xstride；W[Co][Ci/groups][k]；yout[Co][T_out]。
 * 累加顺序与 nn_ops.c:conv1d_core 一致（co -> t -> ci -> kk，acc_t 累加）。 */
static void conv1d_b1(const float *W, const float *b, const float *xbuf, int Ci, int xstride,
                      float *yout, int Co, int k, int stride, int T_out, int groups, int dil) {
    int cpg = Ci / groups;
    int gsz = Co / groups;
    for (int co = 0; co < Co; co++) {
        int g = co / gsz;
        const float *wrow = &W[(size_t)co * cpg * k];
        int t=0;
#ifdef JX_NEON_ENABLED
        if(stride==1)for(;t+15<T_out;t+=16) {
            float32x4_t s0=vdupq_n_f32(b?b[co]:0),s1=s0,s2=s0,s3=s0;
            for(int ci=0;ci<cpg;ci++) {
                const float *xc=xbuf+(size_t)(g*cpg+ci)*xstride+t;
                const float *wc=wrow+(size_t)ci*k;
                for(int kk=0;kk<k;kk++) {
                    const float *v=xc+kk*dil;float w=wc[kk];
                    s0=vfmaq_n_f32(s0,vld1q_f32(v),w);s1=vfmaq_n_f32(s1,vld1q_f32(v+4),w);
                    s2=vfmaq_n_f32(s2,vld1q_f32(v+8),w);s3=vfmaq_n_f32(s3,vld1q_f32(v+12),w);
                }
            }
            float *v=yout+(size_t)co*T_out+t;
            vst1q_f32(v,s0);vst1q_f32(v+4,s1);vst1q_f32(v+8,s2);vst1q_f32(v+12,s3);
        }
        if(stride==1)for(;t+3<T_out;t+=4) {
            float32x4_t sum=vdupq_n_f32(b?b[co]:0);
            for(int ci=0;ci<cpg;ci++) {
                const float *xc=xbuf+(size_t)(g*cpg+ci)*xstride+t;
                const float *wc=wrow+(size_t)ci*k;
                for(int kk=0;kk<k;kk++)sum=vfmaq_n_f32(sum,vld1q_f32(xc+kk*dil),wc[kk]);
            }
            vst1q_f32(yout+(size_t)co*T_out+t,sum);
        }
#endif
        for (; t < T_out; t++) {
            acc_t acc = (b ? (acc_t)b[co] : 0.0);
            int p0 = t * stride;
            for (int ci = 0; ci < cpg; ci++) {
                const float *xc = xbuf + (size_t)(g * cpg + ci) * xstride;
                const float *wc = &wrow[(size_t)ci * k];
                for (int kk = 0; kk < k; kk++) acc += (acc_t)xc[p0 + kk * dil] * (acc_t)wc[kk];
            }
            yout[(size_t)co * T_out + t] = (float)acc;
        }
    }
}

/* ============================ 1. sconv ============================ */
int sconv_nout(const SHist *h, int n_in, int k, int dil, int stride) {
    int K = (k - 1) * dil;
    int L = h->len + n_in;
    if (L <= K) return 0;
    return (L - K - 1) / stride + 1;
}

/* int8 + per-channel scale 版卷积内核（weight-only INT8） */
static void conv1d_b1_q_range(const int8_t *Wi, const float *sc, const float *b,
                        const float *xbuf, int Ci, int xstride,
                        float *yout, int Co, int k, int stride, int T_out, int groups, int dil, int first, int end) {
    int cpg = Ci / groups;
    int gsz = Co / groups;
    int co=first;
#ifdef JX_NEON_ENABLED
    const float *packed=(stride==1&&groups==1&&T_out%4==0)?jx_neon_pack_i8(Wi,Ci*k,Co):NULL;
    if(packed)for(;co+3<end;co+=4) {
        const float *w=packed+(size_t)co*Ci*k;
        for(int t=0;t<T_out;t+=4) {
            float32x4_t s0=vdupq_n_f32(0),s1=s0,s2=s0,s3=s0;
            for(int ci=0;ci<Ci;ci++) {
                const float *xc=xbuf+(size_t)ci*xstride+t;
                for(int kk=0;kk<k;kk++) {
                    float32x4_t v=vld1q_f32(xc+kk*dil);const float *p=w+((size_t)ci*k+kk)*4;
                    s0=vfmaq_n_f32(s0,v,p[0]);s1=vfmaq_n_f32(s1,v,p[1]);
                    s2=vfmaq_n_f32(s2,v,p[2]);s3=vfmaq_n_f32(s3,v,p[3]);
                }
            }
            float values[16];vst1q_f32(values,s0);vst1q_f32(values+4,s1);vst1q_f32(values+8,s2);vst1q_f32(values+12,s3);
            for(int c=0;c<4;c++)for(int lane=0;lane<4;lane++)
                yout[(size_t)(co+c)*T_out+t+lane]=(float)(sc[co+c]*values[c*4+lane]+(b?(acc_t)b[co+c]:0.0));
        }
    }
#endif
    for (; co < end; co++) {
        int g = co / gsz;
        const int8_t *wrow = &Wi[(size_t)co * cpg * k];
        float s = sc[co];
        int t=0;
#ifdef JX_NEON_ENABLED
        if(stride==1)for(;t+15<T_out;t+=16) {
            float32x4_t s0=vdupq_n_f32(0),s1=s0,s2=s0,s3=s0;
            for(int ci=0;ci<cpg;ci++) {
                const float *xc=xbuf+(size_t)(g*cpg+ci)*xstride+t;
                const int8_t *wc=wrow+(size_t)ci*k;
                for(int kk=0;kk<k;kk++) {
                    const float *v=xc+kk*dil;float w=(float)wc[kk];
                    s0=vfmaq_n_f32(s0,vld1q_f32(v),w);s1=vfmaq_n_f32(s1,vld1q_f32(v+4),w);
                    s2=vfmaq_n_f32(s2,vld1q_f32(v+8),w);s3=vfmaq_n_f32(s3,vld1q_f32(v+12),w);
                }
            }
            float values[16];vst1q_f32(values,s0);vst1q_f32(values+4,s1);vst1q_f32(values+8,s2);vst1q_f32(values+12,s3);
            for(int lane=0;lane<16;lane++)yout[(size_t)co*T_out+t+lane]=(float)(s*values[lane]+(b?(acc_t)b[co]:0.0));
        }
        if(stride==1)for(;t+3<T_out;t+=4) {
            float32x4_t sum=vdupq_n_f32(0);
            for(int ci=0;ci<cpg;ci++) {
                const float *xc=xbuf+(size_t)(g*cpg+ci)*xstride+t;
                const int8_t *wc=wrow+(size_t)ci*k;
                for(int kk=0;kk<k;kk++)sum=vfmaq_n_f32(sum,vld1q_f32(xc+kk*dil),(float)wc[kk]);
            }
            float values[4];vst1q_f32(values,sum);
            for(int lane=0;lane<4;lane++)yout[(size_t)co*T_out+t+lane]=(float)(s*values[lane]+(b?(acc_t)b[co]:0.0));
        }
#endif
        for (; t < T_out; t++) {
            acc_t acc = 0.0;
            int p0 = t * stride;
            for (int ci = 0; ci < cpg; ci++) {
                const float *xc = xbuf + (size_t)(g * cpg + ci) * xstride;
                const int8_t *wc = &wrow[(size_t)ci * k];
                for (int kk = 0; kk < k; kk++) acc += (acc_t)wc[kk] * (acc_t)xc[p0 + kk * dil];
            }
            yout[(size_t)co * T_out + t] = (float)(s * acc + (b ? (acc_t)b[co] : 0.0));
        }
    }
}
#ifdef JX_NEON_ENABLED
struct jx_conv_ctx {const int8_t *Wi;const float *sc,*b,*x;float *y;int Ci,xstride,Co,k,stride,T,groups,dil;};
static void jx_conv_range(void *opaque,int first,int end) {
 const struct jx_conv_ctx *c=opaque;
 conv1d_b1_q_range(c->Wi,c->sc,c->b,c->x,c->Ci,c->xstride,c->y,c->Co,c->k,c->stride,c->T,c->groups,c->dil,first,end);
}
#endif
static void conv1d_b1_q(const int8_t *Wi,const float *sc,const float *b,const float *xbuf,int Ci,int xstride,
                       float *yout,int Co,int k,int stride,int T_out,int groups,int dil) {
#ifdef JX_NEON_ENABLED
 struct jx_conv_ctx context={Wi,sc,b,xbuf,yout,Ci,xstride,Co,k,stride,T_out,groups,dil};
 jx_parallel_run(jx_conv_range,&context,Co,4,(size_t)(Ci/groups)*Co*k*T_out);
#else
 conv1d_b1_q_range(Wi,sc,b,xbuf,Ci,xstride,yout,Co,k,stride,T_out,groups,dil,0,Co);
#endif
}


int sconv_step(SHist *h, const float *W, const float *b,
               int Ci, int Co, int k, int stride, int dil, int groups,
               const float *x, int n_in, float *y) {
    int K = (k - 1) * dil;
    int L = h->len + n_in;
    if (L <= K) {                       /* 连一个输出都凑不齐：把 x 并入 hist */
        shist_ensure(h, Ci, L);
        for (int c = 0; c < Ci; c++)
            memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));
        h->len = L;
        return 0;
    }
    int n_out = (L - K - 1) / stride + 1;

    if (h->len == 0 && stride == 1) {   /* 无历史 + 无跨步：直接对 x 卷积 */
        conv1d_b1(W, b, x, Ci, n_in, y, Co, k, stride, n_out, groups, dil);
        return n_out;
    }

    shist_ensure(h, Ci, L);
    for (int c = 0; c < Ci; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));
    conv1d_b1(W, b, h->d, Ci, h->cap, y, Co, k, stride, n_out, groups, dil);

    /* 新 hist = buf 的尾部 newlen 列 = 列 [L-newlen, L)（即 buf[n_out*stride:]） */
    int newlen = L - n_out * stride;
    if (newlen > 0)
        for (int c = 0; c < Ci; c++)
            memmove(SROW(h, c), SROW(h, c) + (L - newlen), (size_t)newlen * sizeof(float));
    h->len = newlen;
    return n_out;
}

/* int8 + scale 版 sconv_step（覆盖所有流式量化卷积） */
int sconv_step_q(SHist *h, const int8_t *Wi, const float *sc, const float *b,
                 int Ci, int Co, int k, int stride, int dil, int groups,
                 const float *x, int n_in, float *y) {
    int K = (k - 1) * dil;
    int L = h->len + n_in;
    if (L <= K) {
        shist_ensure(h, Ci, L);
        for (int c = 0; c < Ci; c++)
            memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));
        h->len = L;
        return 0;
    }
    int n_out = (L - K - 1) / stride + 1;
    if (h->len == 0 && stride == 1) {
        conv1d_b1_q(Wi, sc, b, x, Ci, n_in, y, Co, k, stride, n_out, groups, dil);
        return n_out;
    }
    shist_ensure(h, Ci, L);
    for (int c = 0; c < Ci; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));
    conv1d_b1_q(Wi, sc, b, h->d, Ci, h->cap, y, Co, k, stride, n_out, groups, dil);
    int newlen = L - n_out * stride;
    if (newlen > 0)
        for (int c = 0; c < Ci; c++)
            memmove(SROW(h, c), SROW(h, c) + (L - newlen), (size_t)newlen * sizeof(float));
    h->len = newlen;
    return n_out;
}

/* ============================ 2. strendpool ============================ */
int strendpool_step(SHist *h, int k, int C, const float *x, int n_in, float *y) {
    if (k <= 1) { memcpy(y, x, (size_t)C * n_in * sizeof(float)); return n_in; }
    int H = 2 * (k - 1);
    int L = h->len + n_in;
    shist_ensure(h, C, L);
    for (int c = 0; c < C; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));

    /* causal_trend_pool（零左补齐）：
     *   m[t] = max_{u in [t-(k-1), t]} |x[u]|   (u<0 -> 0)
     *   y[t] = (1/k) * Σ_{v in [t-(k-1), t]} m[v]  (v<0 -> 0)
     * 先用滑窗最大值求 m（O(L*k)），再用前缀和求 y（O(L)）。 */
    float *m = (float*)malloc((size_t)L * sizeof(float));
    if (!m) { fprintf(stderr, "strendpool_step: OOM L=%d\n", L); exit(1); }
    int head = L - n_in;
    for (int c = 0; c < C; c++) {
        const float *bc = SROW(h, c);
        float *yc = &y[(size_t)c * n_in];
        for (int t = 0; t < L; t++) {
            int lo = t - (k - 1); if (lo < 0) lo = 0;
            float mx = 0;
            for (int u = lo; u <= t; u++) { float a = fabsf(bc[u]); if (a > mx) mx = a; }
            m[t] = mx;
        }
        /* y[t] = (1/k) Σ_{v=t-k+1}^{t} m[v]（v<0 -> 0）—— 滑窗和 */
        acc_t run = 0;
        for (int t = 0; t < L; t++) {
            run += (acc_t)m[t];
            if (t - k >= 0) run -= (acc_t)m[t - k];
            if (t >= head) yc[t - head] = (float)(run / (acc_t)k);
        }
    }
    free(m);

    int newlen = (L > H) ? H : L;
    for (int c = 0; c < C; c++)
        memmove(SROW(h, c), SROW(h, c) + (L - newlen), (size_t)newlen * sizeof(float));
    h->len = newlen;
    return n_in;
}

/* ============================ 3. sframenorm ============================ */
int sframenorm_step(SHist *h, int C, int window, float alpha,
                    const float *w, const float *b, float eps,
                    const float *x, int n_in, float *y) {
    int H = window - 1;
    if (H < 0) { memcpy(y, x, (size_t)C * n_in * sizeof(float)); return n_in; }
    int L = h->len + n_in;
    shist_ensure(h, C, L);
    for (int c = 0; c < C; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));

    /* kern[j] = α(1-α)^(window-1-j)，归一化（最新样本权重最大） */
    float kern_local[512];
    float *kern = kern_local;
    if (window > 512) kern = (float*)malloc((size_t)window * sizeof(float));
    {
        double sum = 0;
        for (int j = 0; j < window; j++) {
            double wj = (double)alpha * pow(1.0 - (double)alpha, (double)(window - 1 - j));
            kern[j] = (float)wj; sum += wj;
        }
        for (int j = 0; j < window; j++) kern[j] = (float)((double)kern[j] / sum);
    }

    int head = L - n_in;
    for (int c = 0; c < C; c++) {
        const float *bc = SROW(h, c);
        float *yc = &y[(size_t)c * n_in];
        for (int t = head; t < L; t++) {
            acc_t mu = 0, m2 = 0;
            for (int j = 0; j < window; j++) {
                int p = t - (window - 1) + j;
                if (p < 0) continue;
                acc_t v = bc[p];
                mu += (acc_t)kern[j] * v;
                m2 += (acc_t)kern[j] * v * v;
            }
            acc_t var = m2 - mu * mu;
            if (var < 0.0f) var = 0.0f;
            acc_t inv = (acc_t)(1.0f / sqrtf((float)(var + eps)));
            yc[t - head] = (float)((bc[t] - mu) * inv * (acc_t)w[c] + (acc_t)b[c]);
        }
    }
    if (kern != kern_local) free(kern);

    int newlen = (L > H) ? H : L;
    for (int c = 0; c < C; c++)
        memmove(SROW(h, c), SROW(h, c) + (L - newlen), (size_t)newlen * sizeof(float));
    h->len = newlen;
    return n_in;
}

/* ============================ 4. supsample ============================ */
int supsample_step(SHist *h, int C, int stride, int k,
                   const float *sw, const float *sb,
                   const float *x, int n_in, float *y) {
    int S = stride;
    int H = (k - 1 + S - 1) / S;
    int L = h->len + n_in;
    shist_ensure(h, C, L);
    for (int c = 0; c < C; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));

    /* y = smooth(repeat_interleave(buf, S)) 的尾部 n_in*S 个。
     * smooth 为 depthwise 因果卷积（left_pad = k-1）：
     *   out[c][t] = Σ_j sw[c][j] * u[c][t-(k-1)+j],  u[c][s] = buf[c][s/S]（越界为 0） */
    int tot = L * S;
    int base = tot - n_in * S;
    for (int c = 0; c < C; c++) {
        const float *bc = SROW(h, c);
        const float *wc = &sw[(size_t)c * k];
        float bias = sb ? sb[c] : 0.0f;
        float *yc = &y[(size_t)c * (size_t)n_in * S];
        for (int t = base; t < tot; t++) {
            acc_t acc = (acc_t)bias;
            for (int j = 0; j < k; j++) {
                int s = t - (k - 1) + j;
                if (s < 0) continue;
                acc += (acc_t)wc[j] * (acc_t)bc[s / S];
            }
            yc[t - base] = (float)acc;
        }
    }

    int newlen = (L > H) ? H : L;
    for (int c = 0; c < C; c++)
        memmove(SROW(h, c), SROW(h, c) + (L - newlen), (size_t)newlen * sizeof(float));
    h->len = newlen;
    return n_in * S;
}

/* int8 + scale 版 supsample_step（weight-only INT8 上采样平滑卷积）。
 * sw 布局 [C][k]（导出形状 [C,1,k]），scale[c] 为通道 c 的量化 scale。 */
int supsample_step_q(SHist *h, int C, int stride, int k,
                     const int8_t *sw, const float *sc, const float *sb,
                     const float *x, int n_in, float *y) {
    int S = stride;
    int H = (k - 1 + S - 1) / S;
    int L = h->len + n_in;
    shist_ensure(h, C, L);
    for (int c = 0; c < C; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));

    int tot = L * S;
    int base = tot - n_in * S;
    for (int c = 0; c < C; c++) {
        const float *bc = SROW(h, c);
        const int8_t *wc = &sw[(size_t)c * k];
        float s = sc[c];
        float bias = sb ? sb[c] : 0.0f;
        float *yc = &y[(size_t)c * (size_t)n_in * S];
        for (int t = base; t < tot; t++) {
            acc_t acc = 0.0;
            for (int j = 0; j < k; j++) {
                int sidx = t - (k - 1) + j;
                if (sidx < 0) continue;
                acc += (acc_t)wc[j] * (acc_t)bc[sidx / S];
            }
            yc[t - base] = (float)(s * acc + (acc_t)bias);
        }
    }

    int newlen = (L > H) ? H : L;
    for (int c = 0; c < C; c++)
        memmove(SROW(h, c), SROW(h, c) + (L - newlen), (size_t)newlen * sizeof(float));
    h->len = newlen;
    return n_in * S;
}

/* ============================ 5. sdelay ============================ */
int sdelay_step(SHist *h, int n, int C, const float *x, int n_in, float *y) {
    if (n <= 0) { memcpy(y, x, (size_t)C * n_in * sizeof(float)); return n_in; }
    /* 就地：hist 占各行的列 [0,n)，把 x 追到列 [n, n+n_in)，
     * 于是 concat(hist,x) 恰好落在每行的前 n+n_in 列。
     *   y[t] = concat[t]                  (t < n_in)
     *   new hist = concat[n_in, n+n_in) = 列 [n_in, n+n_in) */
    shist_ensure(h, C, n + n_in);
    for (int c = 0; c < C; c++)
        memcpy(SROW(h, c) + n, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));
    for (int c = 0; c < C; c++) {
        float *row = SROW(h, c);
        memcpy(&y[(size_t)c * n_in], row, (size_t)n_in * sizeof(float));
        memmove(row, row + n_in, (size_t)n * sizeof(float));
    }
    h->len = n;
    return n_in;
}

/* ============================ 逐元素 ============================ */
void per_frame_grn(float *x, int n, int C, const float *gamma, const float *beta) {
    for (int t = 0; t < n; t++) {
        float *p = &x[(size_t)t * C];
        for (int c = 0; c < C; c++) p[c] = (1.0f + gamma[c]) * p[c] + beta[c];
    }
}

void layernorm_tc(float *x, int n, int C, const float *w, const float *b, float eps) {
    for (int t = 0; t < n; t++) {
        float *p = &x[(size_t)t * C];
        acc_t mean = 0;
        for (int c = 0; c < C; c++) mean += (acc_t)p[c];
        mean /= (acc_t)C;
        acc_t var = 0;
        for (int c = 0; c < C; c++) { acc_t d = (acc_t)p[c] - mean; var += d * d; }
        var /= (acc_t)C;
        acc_t inv = (acc_t)(1.0f / sqrtf((float)(var + eps)));
        for (int c = 0; c < C; c++)
            p[c] = w[c] * ((float)((acc_t)p[c] - mean) * inv) + b[c];
    }
}
