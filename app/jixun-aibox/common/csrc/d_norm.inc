/* Finite 400-sample exponential moments. Includes the outgoing sample term.
 * Double state avoids drift and cancellation; ESP32 work is O(C*T), not O(C*T*400).
 * State belongs to one decoder and is cleared by sdec_reset, including zero padding. */
typedef struct {
    float ring[4][FN_WINDOW];
    double mean[4], square[4];
    int pos;
} DFrameNorm;
static void d_framenorm_step(DFrameNorm *s, const float *kernel,
                            const float *w, const float *b,
                            const float *x, int n, float *y) {
    const double ratio=1.0-(double)FN_ALPHA;
    const double newest=(double)kernel[FN_WINDOW-1];
    const double outgoing=newest*pow(ratio,FN_WINDOW);
    const int start=s->pos;
    for(int c=0;c<4;c++) {
        int pos=start;double mu=s->mean[c],m2=s->square[c];
        float *ring=s->ring[c];
        const float *src=x+(size_t)c*n;float *dst=y+(size_t)c*n;
        for(int t=0;t<n;t++) {
            double v=(double)src[t],old=(double)ring[pos];
            ring[pos]=src[t];if(++pos==FN_WINDOW)pos=0;
            mu=ratio*mu+newest*v-outgoing*old;
            m2=ratio*m2+newest*v*v-outgoing*old*old;
            double variance=m2-mu*mu;if(variance<0)variance=0;
            float inv=1.0f/sqrtf((float)variance+FN_EPS);
            dst[t]=(src[t]-(float)mu)*inv*w[c]+b[c];
        }
        s->mean[c]=mu;s->square[c]=m2;
    }
    s->pos=(start+n)%FN_WINDOW;
}

/* Hardware FPU path. Keep the same finite-window subtraction; caller selects
 * the numerical mode once per decoder session, never midway through a stream. */
static void d_framenorm_float(DFrameNorm *s,const float *kernel,
                             const float *w,const float *b,const float *x,int n,float *y) {
    const float newest=kernel[FN_WINDOW-1];
    const float outgoing=(float)((double)newest*pow(1.0-(double)FN_ALPHA,FN_WINDOW));
    int start=s->pos;
    for(int c=0;c<4;c++) {
        int pos=start;float mu=(float)s->mean[c],m2=(float)s->square[c];
        float *ring=s->ring[c];const float *src=x+(size_t)c*n;float *dst=y+(size_t)c*n;
        for(int t=0;t<n;t++) {
            float v=src[t],old=ring[pos];ring[pos]=v;if(++pos==FN_WINDOW)pos=0;
            mu=(mu-FN_ALPHA*mu)+newest*v-outgoing*old;
            m2=(m2-FN_ALPHA*m2)+newest*v*v-outgoing*old*old;
            float variance=m2-mu*mu;if(variance<0)variance=0;
            dst[t]=(v-mu)*(1.0f/sqrtf(variance+FN_EPS))*w[c]+b[c];
        }
        s->mean[c]=mu;s->square[c]=m2;
    }
    s->pos=(start+n)%FN_WINDOW;
}
