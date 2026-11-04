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
static float *g_cvbuf = NULL;
static size_t g_cvbuf_cap = 0;

/* 2026-10-29：原实现是三重大标量循环（实测 ~8-10 cyc/MAC），是全机解码
 * 的最大瓶颈。改为把 hist 打包成连续 [Ci][L] 后交给 nn_ops 的 conv1d_d
 * （PIE 内核 + t 轴分块，实测 ~2 cyc/MAC）。累加次序不变（ci 外层、kk 内层），
 * 输出逐位一致；打包只是行拷贝，L<=cap。 */
static void conv1d_b1(const float *W, const float *b, const float *xbuf, int Ci, int xstride,
                      float *yout, int Co, int k, int stride, int T_out, int groups, int dil) {
    /* Exact receptive span: last output starts at (T_out-1)*stride. */
    int L = (T_out - 1) * stride + (k - 1) * dil + 1;
    const char *pack=getenv("JX_A_PACK");
    if(xstride==L && L>0 && T_out>0 && (!pack||pack[0]!='0')) {
        /* A contiguous input already has the exact receptive span. */
        T xt={0},yt={0};
        xt.d=(float *)xbuf;xt.ndim=3;xt.shape[0]=1;xt.shape[1]=Ci;xt.shape[2]=L;xt.len=Ci*L;
        yt.d=yout;yt.ndim=3;yt.shape[0]=1;yt.shape[1]=Co;yt.shape[2]=T_out;yt.len=Co*T_out;
        conv1d_d(W,b,&xt,&yt,Ci,Co,k,stride,0,groups,L,dil);return;
    }
    size_t need = (size_t)Ci * (size_t)L;
    if (L > 0 && T_out > 0 && need > 0) {
        if (need > g_cvbuf_cap) {
            float *np = (float *)realloc(g_cvbuf, need * sizeof(float));
            if (np != NULL) { g_cvbuf = np; g_cvbuf_cap = need; }
        }
        if (g_cvbuf != NULL && g_cvbuf_cap >= need) {
            for (int c = 0; c < Ci; c++)
                memcpy(g_cvbuf + (size_t)c * L, xbuf + (size_t)c * xstride,
                       (size_t)L * sizeof(float));
            T xt; T yt;
            memset(&xt, 0, sizeof(xt)); memset(&yt, 0, sizeof(yt));
            xt.d = g_cvbuf; xt.ndim = 3; xt.shape[0] = 1; xt.shape[1] = Ci;
            xt.shape[2] = L; xt.len = (int)need;
            yt.d = yout;    yt.ndim = 3; yt.shape[0] = 1; yt.shape[1] = Co;
            yt.shape[2] = T_out; yt.len = Co * T_out;
            conv1d_d(W, b, &xt, &yt, Ci, Co, k, stride, 0, groups, L, dil);
            return;
        }
    }
    /* fallback：原标量实现 */
    int cpg = Ci / groups;
    int gsz = Co / groups;
    for (int co = 0; co < Co; co++) {
        int g = co / gsz;
        const float *wrow = &W[(size_t)co * cpg * k];
        for (int t = 0; t < T_out; t++) {
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
static void conv1d_b1_q(const int8_t *Wi, const float *sc, const float *b,
                        const float *xbuf, int Ci, int xstride,
                        float *yout, int Co, int k, int stride, int T_out, int groups, int dil) {
    /* Mono k7 branches: hoist exact weight conversions once and interleave
     * independent time samples. Keep each seven-tap sum in its old order. */
    const char *fast7=getenv("JX_A_BRANCH7");
    if(Ci==1 && Co==4 && k==7 && stride==1 && dil==1 && groups==1 && (!fast7 || fast7[0]!='0')) {
      for(int co=0;co<4;co++) {
        const int8_t *w=Wi+co*7;const float s=sc[co];const double bias=b?(acc_t)b[co]:0.0;
        const acc_t w0=(acc_t)w[0],w1=(acc_t)w[1],w2=(acc_t)w[2],w3=(acc_t)w[3],w4=(acc_t)w[4],w5=(acc_t)w[5],w6=(acc_t)w[6];
        float *out=yout+(size_t)co*T_out;int t=0;
        for(;t+4<=T_out;t+=4) {
          acc_t a0=0,a1=0,a2=0,a3=0;
          a0+=w0*(acc_t)xbuf[t+0];
          a1+=w0*(acc_t)xbuf[t+1];
          a2+=w0*(acc_t)xbuf[t+2];
          a3+=w0*(acc_t)xbuf[t+3];
          a0+=w1*(acc_t)xbuf[t+1];
          a1+=w1*(acc_t)xbuf[t+2];
          a2+=w1*(acc_t)xbuf[t+3];
          a3+=w1*(acc_t)xbuf[t+4];
          a0+=w2*(acc_t)xbuf[t+2];
          a1+=w2*(acc_t)xbuf[t+3];
          a2+=w2*(acc_t)xbuf[t+4];
          a3+=w2*(acc_t)xbuf[t+5];
          a0+=w3*(acc_t)xbuf[t+3];
          a1+=w3*(acc_t)xbuf[t+4];
          a2+=w3*(acc_t)xbuf[t+5];
          a3+=w3*(acc_t)xbuf[t+6];
          a0+=w4*(acc_t)xbuf[t+4];
          a1+=w4*(acc_t)xbuf[t+5];
          a2+=w4*(acc_t)xbuf[t+6];
          a3+=w4*(acc_t)xbuf[t+7];
          a0+=w5*(acc_t)xbuf[t+5];
          a1+=w5*(acc_t)xbuf[t+6];
          a2+=w5*(acc_t)xbuf[t+7];
          a3+=w5*(acc_t)xbuf[t+8];
          a0+=w6*(acc_t)xbuf[t+6];
          a1+=w6*(acc_t)xbuf[t+7];
          a2+=w6*(acc_t)xbuf[t+8];
          a3+=w6*(acc_t)xbuf[t+9];
          out[t+0]=(float)(s*a0+bias);
          out[t+1]=(float)(s*a1+bias);
          out[t+2]=(float)(s*a2+bias);
          out[t+3]=(float)(s*a3+bias);
        }
        for(;t<T_out;t++) {
          acc_t acc=0;
          acc+=w0*(acc_t)xbuf[t+0];
          acc+=w1*(acc_t)xbuf[t+1];
          acc+=w2*(acc_t)xbuf[t+2];
          acc+=w3*(acc_t)xbuf[t+3];
          acc+=w4*(acc_t)xbuf[t+4];
          acc+=w5*(acc_t)xbuf[t+5];
          acc+=w6*(acc_t)xbuf[t+6];
          out[t]=(float)(s*acc+bias);
        }
      }
      return;
    }

    int cpg = Ci / groups;
    int gsz = Co / groups;
    for (int co = 0; co < Co; co++) {
        int g = co / gsz;
        const int8_t *wrow = &Wi[(size_t)co * cpg * k];
        float s = sc[co];
        for (int t = 0; t < T_out; t++) {
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


/* Causal k7 depthwise: preserve six input samples per channel and write
 * time-major output directly. No packed history or CV temporary is needed. */
int sconv_dw7_tc(SHist *h,const float *W,const float *b,int C,
                const float *x,int n,float *tc) {
 if(n<=0||h->len!=6)return 0;
 int L=n+6;shist_ensure(h,C,L);
 for(int c=0;c<C;c++)memcpy(SROW(h,c)+6,x+(size_t)c*n,(size_t)n*sizeof(float));
 jx_dw7_strided_tc(W,b,h->d,h->cap,tc,C,n);
 for(int c=0;c<C;c++)memmove(SROW(h,c),SROW(h,c)+n,6*sizeof(float));
 return n;
}
/* The residual delay contains exactly delay samples per channel. Read it
 * directly while adding; retain only the new tail instead of appending the
 * complete current chunk to the history. Input and output must not alias. */
int sdelay_add_step(SHist *h,int delay,int C,const float *x,int n,
                    const float *side,float *out) {
 if(delay<0||n<=0||h->len!=delay||out==x)return 0;
 for(int c=0;c<C;c++) {
  const float *xr=x+(size_t)c*n,*sr=side+(size_t)c*n;float *yr=out+(size_t)c*n;
  float *hr=SROW(h,c);int cut=n<delay?n:delay;
  for(int t=0;t<cut;t++)yr[t]=hr[t]+sr[t];
  for(int t=cut;t<n;t++)yr[t]=xr[t-delay]+sr[t];
  if(n>=delay)memcpy(hr,xr+n-delay,(size_t)delay*sizeof(float));
  else {memmove(hr,hr+n,(size_t)(delay-n)*sizeof(float));memcpy(hr+delay-n,xr,(size_t)n*sizeof(float));}
 }
 return n;
}
int sdelay_madd_step(SHist *h,int delay,int C,const float *x,int n,
                     const float *side,float *out) {
 if(delay<0||n<=0||h->len!=delay||out==x)return 0;
 for(int c=0;c<C;c++) {
  const float *xr=x+(size_t)c*n,*sr=side+(size_t)c*n;float *yr=out+(size_t)c*n;
  float *hr=SROW(h,c);int cut=n<delay?n:delay;
  for(int t=0;t<cut;t++){float v=hr[t];yr[t]=v+sr[t]*v;}
  for(int t=cut;t<n;t++){float v=xr[t-delay];yr[t]=v+sr[t]*v;}
  if(n>=delay)memcpy(hr,xr+n-delay,(size_t)delay*sizeof(float));
  else {memmove(hr,hr+n,(size_t)(delay-n)*sizeof(float));memcpy(hr+delay-n,xr,(size_t)n*sizeof(float));}
 }
 return n;
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
        const char *pool=getenv("JX_A_POOL");
        if(k<=64 && (!pool || pool[0]!='0')) {
            int q[64],head_q=0,tail_q=0;
            for(int t=0;t<L;t++) {
                while(head_q<tail_q && q[head_q&63]<=t-k)head_q++;
                float v=fabsf(bc[t]);if(!(v>=0.0f))v=0.0f;
                while(head_q<tail_q && fabsf(bc[q[(tail_q-1)&63]])<=v)tail_q--;
                q[tail_q&63]=t;tail_q++;
                m[t]=fabsf(bc[q[head_q&63]]);if(!(m[t]>=0.0f))m[t]=0.0f;
            }
        } else
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
int sframenorm_step_cached(SHist *h, int C, int window, float alpha,
                    const float *w, const float *b, float eps,
                    const float *x, int n_in, float *y, const float *cached) {
    int H = window - 1;
    if (H < 0) { memcpy(y, x, (size_t)C * n_in * sizeof(float)); return n_in; }
    int L = h->len + n_in;
    shist_ensure(h, C, L);
    for (int c = 0; c < C; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));

    /* kern[j] = α(1-α)^(window-1-j)，归一化（最新样本权重最大） */
    float kern_local[512];
    float *kern = cached ? (float *)cached : kern_local;
    if (!cached && window > 512) kern = (float*)malloc((size_t)window * sizeof(float));
    if (!cached) {
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
    if (!cached && kern != kern_local) free(kern);

    int newlen = (L > H) ? H : L;
    for (int c = 0; c < C; c++)
        memmove(SROW(h, c), SROW(h, c) + (L - newlen), (size_t)newlen * sizeof(float));
    h->len = newlen;
    return n_in;
}

int sframenorm_step(SHist *h,int C,int window,float alpha,
                     const float *w,const float *b,float eps,
                     const float *x,int n_in,float *y) {
    return sframenorm_step_cached(h,C,window,alpha,w,b,eps,x,n_in,y,NULL);
}

/* Phase indices are invariant across time. Keep every k-tap addition in
 * the original order, while eliminating division from the sample loop. */
typedef struct {
    const SHist *h;const float *w,*b;float *y;
    int n,S,k,offset[4][9];
} sup_fast_ctx;
extern int jx_pf_run_min(int,void (*)(void *,int,int),void *,int);
static void sup_fast_rows(void *arg,int lo,int hi) {
    const sup_fast_ctx *q=(const sup_fast_ctx *)arg;
    for(int c=lo;c<hi;c++) {
        const float *x=SROW(q->h,c),*w=q->w+(size_t)c*q->k;
        float *y=q->y+(size_t)c*q->n*q->S;
        float bias=q->b?q->b[c]:0;
        for(int phase=0;phase<q->S;phase++) {
            const int *off=q->offset[phase];
            for(int t=0;t<q->n;t++) {
                acc_t acc=(acc_t)bias;
                for(int j=0;j<q->k;j++)acc+=(acc_t)w[j]*(acc_t)x[t+off[j]];
                y[(size_t)t*q->S+phase]=(float)acc;
            }
        }
    }
}

/* ============================ 4. supsample ============================ */
int supsample_step(SHist *h, int C, int stride, int k,
                   const float *sw, const float *sb,
                   const float *x, int n_in, float *y) {
    int S = stride;
    int H = (k - 1 + S - 1) / S;

    /* 板端 3k/6k 的模型导出没有 decoder.blocks.N.1.smooth.weight，
     * 其 Up 层是 nn_ops.c:upsampling_linear（半像素对齐的线性插值）。
     * 博士新版导出才带 smooth 卷积。这里按 sw 是否为 NULL 自动分流：
     *   sw == NULL -> 线性插值（与整段 decoder_forward 的语义一致）
     *   sw != NULL -> 原 CausalUpsample 因果平滑卷积
     * 线性插值在帧内 j 上只依赖 x[g-1]、x[g]、x[g+1]：
     *   ya = floor((2j+1-S)/(2S)) + g，yb = ya+1，越界按整段实现的方式
     *   夹到序列端点（右端夹紧 = 末帧常数延拓）。跨包必须保留上一帧，
     *   否则每个包边界的 j=0 会丢掉 x[g-1]。 */
    if (sw == NULL) {
        int prev = h->len;
        int L, c, t, j;
        const int den = 2 * S;

        if (prev > 1) {
            for (c = 0; c < C; c++)
                memcpy(SROW(h, c), SROW(h, c) + (prev - 1), sizeof(float));
            h->len = 1;
            prev = 1;
        }
        L = prev + n_in;
        shist_ensure(h, C, L);
        for (c = 0; c < C; c++)
            memcpy(SROW(h, c) + prev, &x[(size_t)c * n_in],
                   (size_t)n_in * sizeof(float));

        for (c = 0; c < C; c++) {
            const float *xc = SROW(h, c);
            float *yc = &y[(size_t)c * (size_t)n_in * S];
            for (t = 0; t < n_in; t++) {
                int g = prev + t;
                for (j = 0; j < S; j++) {
                    int num = 2 * j + 1 - S;
                    int i0 = num / den;
                    int rem = num - i0 * den;
                    int a, b;
                    float fr;
                    if (rem < 0) { rem += den; i0 -= 1; }
                    a = g + i0; b = a + 1;
                    if (a < 0) a = 0; else if (a > L - 1) a = L - 1;
                    if (b < 0) b = 0; else if (b > L - 1) b = L - 1;
                    fr = (float)rem / (float)den;
                    yc[(size_t)t * S + j] = xc[a] * (1.0f - fr) + xc[b] * fr;
                }
            }
        }
        for (c = 0; c < C; c++)
            SROW(h, c)[0] = SROW(h, c)[L - 1];
        h->len = 1;
        return n_in * S;
    }
    int L = h->len + n_in;
    shist_ensure(h, C, L);
    for (int c = 0; c < C; c++)
        memcpy(SROW(h, c) + h->len, &x[(size_t)c * n_in], (size_t)n_in * sizeof(float));

    /* y = smooth(repeat_interleave(buf, S)) 的尾部 n_in*S 个。
     * smooth 为 depthwise 因果卷积（left_pad = k-1）：
     *   out[c][t] = Σ_j sw[c][j] * u[c][t-(k-1)+j],  u[c][s] = buf[c][s/S]（越界为 0） */
    int tot = L * S;
    int base = tot - n_in * S;
    const char *fast=getenv("JX_D_UPFAST");
    if(S<=4 && k<=9 && base>=k-1 && (!fast || fast[0]!='0')) {
        sup_fast_ctx job={.h=h,.w=sw,.b=sb,.y=y,.n=n_in,.S=S,.k=k};
        for(int phase=0;phase<S;phase++)for(int j=0;j<k;j++)
            job.offset[phase][j]=(base+phase-(k-1)+j)/S;
        jx_pf_run_min(C,sup_fast_rows,&job,2);
    } else {
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
