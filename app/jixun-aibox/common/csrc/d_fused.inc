/* Adapted from the verified encoder kernel; decoder histories remain independent. */
/* A per-frame GRN is exactly (1+gamma)*x+beta. The fused path uses nx=1
 * and preserves causal depthwise and residual histories. */
static float *d_convunit_residual(SArena *A,SHist *dwh,SHist *resh,
                                const char *pn,const float *cur,int C,int n) {
    int reuse=0;size_t mark=A->off;
    if(!d_opt_on("JX_D_FUSE")) {
        float *result=convunit_residual(A,dwh,resh,pn,cur,C,n);
        if(reuse){memmove((float *)cur,result,(size_t)C*n*sizeof(float));A->off=mark;jx_amax_kill();return (float *)cur;}
        return result;
    }

    float *dw=sa_alloc(A,(size_t)C*n);
    float *tc=sa_alloc(A,(size_t)n*C);
    const char *dwtc=getenv("JX_D_DWTC");int dw_done=0;
    if(!dwtc||dwtc[0]!='0')dw_done=sconv_dw7_tc(dwh,W_(cat_name(pn,"dw_conv.weight")),W_(cat_name(pn,"dw_conv.bias")),C,cur,n,tc);
    if(!dw_done) {
        sconv_step_d(dwh,cat_name(pn,"dw_conv.weight"),cat_name(pn,"dw_conv.bias"),C,C,CU_KERNEL,1,1,C,cur,n,dw);
        for(int t=0;t<n;t++)for(int c=0;c<C;c++)tc[(size_t)t*C+c]=dw[(size_t)c*n+t];
    }


    if(g_stap)tap(cat_name(pn,".dwdone"),tc,C,n,1);
    T tct=mk_tc(tc,n,C);
    const float *nw=W_(cat_name(pn,"norm.weight")),*nb=W_(cat_name(pn,"norm.bias"));
    const float *w1=W_(cat_name(pn,"pw_conv1.weight")),*b1=W_(cat_name(pn,"pw_conv1.bias"));
    const float *w2=W_(cat_name(pn,"pw_conv2.weight")),*b2=W_(cat_name(pn,"pw_conv2.bias"));
    const float *alpha=W_(cat_name(pn,"act.alpha"));
    const float *gg=W_(cat_name(pn,"grn.gamma")),*gb=W_(cat_name(pn,"grn.beta"));
    float *p1=sa_alloc(A,(size_t)n*4*C),*ss=sa_alloc(A,(size_t)n),*sc=sa_alloc(A,(size_t)n);
    memset(ss,0,(size_t)n*sizeof(float));T p1t=mk_tc(p1,n,4*C);
    /* PIE q-vector loads require16-byte alignment. SArena guarantees float
     * alignment only. The byte result uses <=1/4 of the float allocation,
     * leaving ample room for up to15 bytes of alignment padding. */
    signed char *q1=(signed char *)(((uintptr_t)p1+15u)&~(uintptr_t)15u);
    jx_linear_fuse_t f1={0},f2={0};int qdone=0;
    f1.snake_alpha=alpha;f1.ss_rows=ss;f1.in_norm_w=nw;f1.in_norm_b=nb;
    f1.q_gamma=gg;f1.q_beta=gb;f1.out_q=q1;f1.out_sc=sc;f1.out_qdone=&qdone;
    int l1ok=jx_linear_i8_ex(w1,b1,tc,p1,n,C,4*C,&f1)==0,norm_done=0;
    if(!l1ok){channel_norm_last(&tct,nw,nb,1,n,C);norm_done=1;
        linear_d(cat_name(pn,"pw_conv1.weight"),cat_name(pn,"pw_conv1.bias"),&tct,&p1t,C,4*C);
        snake1d(&p1t,alpha,1,n,4*C,1);}


    if(g_stap)tap(cat_name(pn,".l1done"),p1,4*C,n,1);
    float *side=sa_alloc(A,(size_t)C*n);
    if(qdone){f2.in_q=q1;f2.in_sc=sc;}
    else{f2.grn_gamma=gg;f2.grn_beta=gb;f2.grn_nx=1.0f;}
    int l2ok=jx_linear_i8_cf(w2,b2,p1,side,n,4*C,C,&f2)==0;
    if(!l2ok){
        /* Rebuild p1 even if the successful l1 wrote quantized bytes into it. */
        if(!norm_done)channel_norm_last(&tct,nw,nb,1,n,C);
        linear_d(cat_name(pn,"pw_conv1.weight"),cat_name(pn,"pw_conv1.bias"),&tct,&p1t,C,4*C);
        snake1d(&p1t,alpha,1,n,4*C,1);per_frame_grn(p1,n,4*C,gg,gb);
        float *p2=sa_alloc(A,(size_t)n*C);T p2t=mk_tc(p2,n,C);
        linear_d(cat_name(pn,"pw_conv2.weight"),cat_name(pn,"pw_conv2.bias"),&p1t,&p2t,4*C,C);
        for(int c=0;c<C;c++)for(int t=0;t<n;t++)side[(size_t)c*n+t]=p2[(size_t)t*C+c];
    }


    if(g_stap)tap(cat_name(pn,".l2done"),side,C,n,0);
    if(l1ok&&l2ok){d_cu_hits++;if(qdone)d_qreuse_hits++;}
    float *out=reuse?(float *)cur:sa_alloc(A,(size_t)C*n);
    const char *dadd=getenv("JX_D_DADD");int add_done=0;
    if(!dadd||dadd[0]!='0')add_done=sdelay_add_step(resh,CU_DELAY,C,cur,n,side,out);
    if(!add_done) {
        float *base=dw;sdelay_step(resh,CU_DELAY,C,cur,n,base);
        for(size_t i=0;i<(size_t)C*n;i++)out[i]=base[i]+side[i];
    }

    if(reuse){A->off=mark;jx_amax_kill();}return out;
}
