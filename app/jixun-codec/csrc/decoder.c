/* ============================================================================
 * decoder.c — implementations of the decode-side network.
 * See decoder.h for the Python class mapping.
 * ========================================================================== */
#include "decoder.h"
#include "jxprof.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <malloc.h>

/* 2026-09-14 第二十三轮：堆探针。长录音（>=4s）时解码器的大块（up/cur）拿不到
 * 连续内存。mallinfo 的 mxordblk 就是"最大连续空闲块"——用它区分两种成因：
 * 总量不足（fordblks 也小）还是碎片化（fordblks 够但 mxordblk 不够）。 */
static void dec_mem(const char *tag, int s, int Tn)
{
    if (getenv("JX_DEC_TRACE") == NULL) { return; }
    struct mallinfo mi = mallinfo();
    printf("[MEMDEC] %-12s s=%d Tn=%-6d 已用=%u 空闲=%u 最大块=%u\n",
           tag, s, Tn, (unsigned)mi.uordblks, (unsigned)mi.fordblks,
           (unsigned)mi.mxordblk);
}

/* --- up 路径：1x1 conv -> UpsampleLinear -> ChannelNorm(first) ---
 * 三个算子在时间上都是局部的：conv 逐点；upsample 每个输出样点只依赖相邻两个
 * 输入样点；ChannelNorm 逐位置。所以按时间切块、每块多带 1 个输入样点就能
 * 与整段计算逐位一致，同时省掉整块 upc（[nxt,Tn]，4 秒音频约 2.0MB）。
 * ChannelNorm 逐位置，对整块 up 做一次即可，与分块无关。 */
static int dec_chunk_width(void)
{
    const char *e = getenv("JX_CHUNK");
    if (e) { int v = atoi(e); return v > 0 ? v : 0; }
    return JX_CHUNK_DEFAULT;
}

/* --- up 路径：1x1 conv -> UpsampleLinear -> ChannelNorm(first) ---
 * 2026-09-14 第二十三轮（长录音连续内存）：原来一上来就 realloc 整段
 * up[B,nxt,Tn*scale]，此时 cur[B,dim_s,Tn] 还活着。而 cur 正是上一级 up，
 * 是从当时最大的那块空闲内存里切出来的 —— 板端 5 秒实测：stage 1 出了
 * up1=2.135MB 之后，那块 6.547MB 的大空闲只剩 4.412MB，而 stage 2 要
 * 4.4835MB，差 71KB，可总空闲还有 5.41MB，就是一个能用的连续块都没有。
 *
 * 改成三步：先算小张量 uc = 1x1conv(cur) [B,nxt,Tn]（比 cur 小），把它读干净，
 * 然后 t_free(cur) —— mm_free 的相邻块合并会把空闲大块拼回去，紧接着那次
 * 整段 up 的分配就必然拿到连续内存。末段不提前放：它失败后还要靠 cur 走
 * decoder_tail_stream 兜底。
 *
 * 数值：1x1 conv 走逐位置量化、upsample 逐样点，本路径与分块宽度无关
 * （主机 1~5 秒 JX_UPW=512 vs 整段实测逐位一致），所以顺手去掉了分块循环。
 *
 * uc 优先用调用方在解码入口预留的暂存张量 sc（预留发生在堆最完整的时候，
 * 不会去啃后面唯一剩下的那块大内存）；没预留就现场分配。
 * JX_UPW 自本轮起不再起作用（up 路径恒定整段）。 */
static int decoder_up_path(T *cur, int B, int dim_s, int nxt, int Tn, int scale,
                           const float *wc, const float *bc,
                           const float *wn, const float *bn,
                           T *sc, int last_stage, T *up)
{
    int Tout   = Tn * scale;
    int uc_need = B * nxt * Tn;
    up->ndim = 3; up->shape[0] = B; up->shape[1] = nxt; up->shape[2] = Tout;
    up->len = B * nxt * Tout;

    /* 末段：先按老顺序要整段 up —— 拿不到就立刻返回，交给流式兜底，
     * 不必先把卷积算一遍（短录音末段本来就能整段跑通，不能因此退化）。 */
    if (last_stage)
      {
        float *nd0 = (float*)jx_arealloc(up->d, (size_t)up->len * sizeof(float));
        if (nd0 == NULL)
          {
            printf("[UP] 上采样输出 %d floats (%.2fMB) 分配失败\n",
                   up->len, (double)up->len * 4.0 / 1.0e6);
            dec_mem("up 大块失败", -1, Tn);
            up->d = NULL; up->len = 0; up->shape[2] = 0;
            return -1;
          }
        up->d = nd0;
      }

    T uc; int uc_own = 0;
    if (sc != NULL && sc->d != NULL && sc->len >= uc_need)
      {
        uc.d = sc->d; uc.ndim = 3;
        uc.shape[0] = B; uc.shape[1] = nxt; uc.shape[2] = Tn; uc.len = uc_need;
      }
    else
      {
        uc = t_alloc(3, (int[]){B, nxt, 0});
        uc_own = 1;
      }

    conv1d(wc, bc, cur, &uc, dim_s, nxt, 1, 1, 0, 1, Tn);
    if (uc.d == NULL || uc.len != uc_need)
      {
        printf("[UP] 1x1 卷积暂存 %d floats 分配失败\n", uc_need);
        dec_mem("upc 失败", -1, Tn);
        if (uc_own) { t_free(&uc); }
        up->d = NULL; up->len = 0; up->shape[2] = 0;
        return -1;
      }

    if (!last_stage)
      {
        /* 关键一步：cur 已经读干净了，放掉它让空闲大块合并回来 */
        t_free(cur);
        float *nd = (float*)jx_arealloc(up->d, (size_t)up->len * sizeof(float));
        if (nd == NULL)
          {
            printf("[UP] 上采样输出 %d floats (%.2fMB) 分配失败\n",
                   up->len, (double)up->len * 4.0 / 1.0e6);
            dec_mem("up 大块失败", -1, Tn);
            if (uc_own) { t_free(&uc); }
            up->d = NULL; up->len = 0; up->shape[2] = 0;
            return -1;
          }
        up->d = nd;
      }

    upsampling_linear(&uc, up, B, nxt, Tn, scale);
    if (uc_own) { t_free(&uc); }
    if (up->len != B * nxt * Tout)
      {
        printf("[UP] up 张量不完整\n");
        up->len = 0; return -1;
      }
    channel_norm_first(up, wn, bn, B, nxt, Tout);
    return 0;
}

/* ============================ 末段流式兜底（长录音） ============================ */
/* 正常路径：decoder_up_path 先把整段 up [B,nxt,Tout] 物化出来（4 秒音频 4.10MB），
 * 再在整段上跑 3 个 LegacyUnit + Snake + conv(k=7)。4 秒音频时 up(4.10MB) 与
 * 上一 stage 的 cur [B,28,32016](3.59MB) 必须同时活着 = 7.7MB；板端 8.5MB PSRAM
 * 堆里空闲有 4.84MB 但最大连续只有 2.54MB（cur 卡在中间把空闲劈成两半），
 * realloc 拿不到 4MB -> 整段输出作废。3 秒已经能在这个堆里跑通，4 秒就差这一块。
 *
 * 这里把末段整个摊平：按输出块算一小段 up（复用 decoder_up_path 那套 1 样点
 * halo 的分块，逐位一致），就地接着跑 Legacy/Snake/末层卷积，只把 [u0,u1)
 * 写回 out，永不物化整段 [nxt,Tout] 与整段 [1,16,T]。峰值降到只剩上一 stage
 * 的 cur + 几块百 KB 级工作缓冲（4 秒约 3.9MB）。
 *
 * halo：末段链是 up -> ChannelNorm -> [Snake->conv7(dil)]x3 -> Snake -> conv7(p=3)，
 * 各段感受野半径 = 27(最后那个 LegacyUnit, dil=9) + 9 + 3 + 3 = 42，
 * 取 H = 48。块 [u0-48, u1+48) 算完，[u0,u1) 这一段是精确的。
 *
 * 数值：这不是逐位一致改动 —— int8 dense-tile 的量化尺度是每 WT(默认 256) 个
 * 输出位置共用一个，与子块边界耦合；blocks.c 里 conv_unit / first_block 的
 * 分块早就是同样的性质。整段路径能分到内存时仍然走整段路径，本函数只在
 * 整段 up 分配失败时生效，用 SNR 回归确认（见 _REALTIME_ROUND22_20260914.md）。 */
static int decoder_tail_stream(const T *cur, int B, int dim_s, int nxt, int Tn, int scale,
                               const float *wc, const float *bc,
                               const float *wn, const float *bn, int last_dim, T *out)
{
    int U = Tn * scale;
    out->ndim = 3; out->shape[0] = B; out->shape[1] = 1; out->shape[2] = U;
    out->len = B * U;
    float *nd = (float*)jx_arealloc(out->d, (size_t)out->len * sizeof(float));
    if (nd == NULL)
      {
        jx_conv_skip++;
        printf("[TLS] 末段输出 %d floats 分配失败\n", out->len);
        out->d = NULL; out->len = 0; out->shape[2] = 0;
        return -1;
      }
    out->d = nd;
    if (B != 1)
      {
        jx_conv_skip++;
        printf("[TLS] 只支持 B=1（当前 B=%d）\n", B);
        out->len = 0; out->shape[2] = 0;
        return -1;
      }

    int W = dec_chunk_width();
    if (W <= 0) { W = JX_CHUNK_DEFAULT; }
    {   /* JX_TLSW 覆盖末段块宽（主机对照用；>= U 即单块，应当与整段路径逐位一致） */
        const char *we = getenv("JX_TLSW");
        if (we != NULL) { int v = atoi(we); if (v > 0) { W = v; } }
    }
    const int H = 48;
    int cap    = W + 2 * H;
    int cw_cap = (cap + scale - 1) / scale + 3;

    T xc  = t_alloc(3, (int[]){B, dim_s, cw_cap});
    T uc  = t_alloc(3, (int[]){B, nxt,   cw_cap});
    T yc  = t_alloc(3, (int[]){B, nxt,   cw_cap * scale});
    T blk = t_alloc(3, (int[]){B, nxt,   cap});
    T fin = t_alloc(3, (int[]){B, 1,     cap});
    int rc = 0;
    if (xc.d == NULL || uc.d == NULL || yc.d == NULL || blk.d == NULL || fin.d == NULL)
      {
        jx_conv_skip++;
        printf("[TLS] 工作缓冲分配失败\n");
        rc = -1; goto done;
      }
    {
    const float *sn_a  = W_("decoder.blocks.13.block.1.alpha");
    const float *fw    = W_("decoder.blocks.13.block.2.weight");
    const float *fb    = W_("decoder.blocks.13.block.2.bias");

    for (int u0 = 0; u0 < U; u0 += W)
      {
        int u1 = u0 + W; if (u1 > U) { u1 = U; }
        int a  = u0 - H; if (a  < 0) { a  = 0; }
        int e  = u1 + H; if (e  > U) { e  = U; }
        int w  = e - a;
        /* up 的全局索引 g 只依赖 conv 域索引 floor(g/scale) 与 +1（线性插值，
         * 见 upsampling_linear；kmod 的 itab 可能是 -1，所以左边必须多带 1 个
         * conv 样点，否则块首会被钳到 cur[pa] 而不是真实的 cur[pa-1]）。 */
        int pa = a / scale; if (pa > 0) { pa -= 1; }
        int pb = (e >= U) ? Tn : ((e - 1) / scale + 2);
        if (pb > Tn) { pb = Tn; }
        int cw = pb - pa;
        if (cw <= 0 || cw > cw_cap || w > cap)
          {
            jx_conv_skip++;
            printf("[TLS] 窗口越界 cw=%d w=%d cap=%d\n", cw, w, cap);
            rc = -1; goto done;
          }

        xc.shape[0] = B; xc.shape[1] = dim_s; xc.shape[2] = cw; xc.len = B * dim_s * cw;
        for (int c = 0; c < dim_s; c++)
            memcpy(&xc.d[(size_t)c * cw], &cur->d[(size_t)c * Tn + pa],
                   (size_t)cw * sizeof(float));
        conv1d(wc, bc, &xc, &uc, dim_s, nxt, 1, 1, 0, 1, cw);
        if (uc.len != B * nxt * cw)
          { jx_conv_skip++; printf("[TLS] 1x1 卷积失败\n"); rc = -1; goto done; }
        upsampling_linear(&uc, &yc, B, nxt, cw, scale);
        if (yc.len == 0)
          { jx_conv_skip++; printf("[TLS] 上采样失败\n"); rc = -1; goto done; }

        blk.shape[0] = B; blk.shape[1] = nxt; blk.shape[2] = w; blk.len = B * nxt * w;
        int off = pa * scale;              /* yc 的第 off 列 == 全局 up 的第 pa*scale 列 */
        for (int c = 0; c < nxt; c++)
            memcpy(&blk.d[(size_t)c * w], &yc.d[(size_t)c * (cw * scale) + (a - off)],
                   (size_t)w * sizeof(float));
        channel_norm_first(&blk, wn, bn, B, nxt, w);

        for (int r = 0; r < 3; r++)
          {
            char rp[64];
            snprintf(rp, sizeof(rp), "decoder.blocks.13.block.0.%d.module", r);
            legacy_unit_add(&blk, rp, (r == 0 ? 1 : r == 1 ? 3 : 9));
          }
        snake1d(&blk, sn_a, B, w, last_dim, 0);
        fin.shape[0] = B; fin.shape[1] = 1; fin.shape[2] = w; fin.len = B * w;
        conv1d(fw, fb, &blk, &fin, last_dim, 1, 7, 1, 3, 1, w);
        if (fin.len != B * w)
          { jx_conv_skip++; printf("[TLS] 末层卷积失败\n"); rc = -1; goto done; }
        jx_kt_ensure();
        {   /* 第三十九轮：tanh 的 8 个系数装载一次 */
            const jx_tanh8 TH = jx_tanh8_load();
            for (int t = 0; t < u1 - u0; t++)
                out->d[(size_t)u0 + t] = jx_tanhf8(fin.d[(size_t)(u0 - a) + t], TH);
        }
      }
    }
  done:
    t_free(&xc); t_free(&uc); t_free(&yc); t_free(&blk); t_free(&fin);
    if (rc != 0) { out->len = 0; out->shape[2] = 0; }
    return rc;
}

/* ============================ modules.Decoder.forward ============================ */
void decoder_forward(const T *x, T *out) {
    JXP_BEG(JP_DECODER);
    /* x[B,FEATURE_DIM,T'] -> out[B,1,T] (T = T'*HOP_LENGTH) */
    int B = x->shape[0], T0 = x->shape[2];
    /* 2026-09-14 第二十三轮（板端 5 秒崩机根因）：所有早退路径都必须留下一个
     * "明确无效"的张量。原来中途 stage 失败时 out->d 是未初始化的栈垃圾，
     * jx_decode 按 origT 去读 -> LoadProhibited（实测 PC=jx_decode+0x189，
     * VADDR=0x25，栈里 dec={d=0x25,ndim=3,shape=1,1,0,len=0}）。 */
    out->ndim = 3; out->shape[0] = B; out->shape[1] = 1; out->shape[2] = 0;
    out->len = 0; out->d = NULL;
    dec_mem("dec 入口", -1, T0);
    /* 2026-09-14 第二十三轮：在堆最完整的时候（实测入口最大空闲块 7.59MB）
     * 先把 up 路径要用的 1x1 卷积暂存张量 [B,nxt,Tn] 预留好。它比 up 小一个
     * 数量级，不占后面唯一剩下的那块大内存；放这里是为了不让它去啃 stage 2
     * 需要的那块 4.48MB 连续空间（见 decoder_up_path 的注释）。 */
    T sc = t_alloc(3, (int[]){1, 1, 0});
    {
        int mx = 0;
        for (int s = 0; s + 1 < DEC_STAGES; s++)
          {
            int Ts = T0;
            for (int k = 0; k <= s; k++) { Ts *= DEC_STRIDES[k]; }
            int f = B * DEC_DIMS[s + 1] * (Ts / DEC_STRIDES[s]);   /* B*nxt*Tn */
            if (f > mx) { mx = f; }
          }
        if (mx > 0)
          {
            sc = t_alloc(3, (int[]){1, 1, mx});
            if (getenv("JX_DEC_TRACE") != NULL)
              printf("[MEMDEC] 预留 1x1 卷积暂存 %d floats (%.2fMB) %s\n",
                     mx, (double)mx * 4.0 / 1.0e6, sc.d ? "成功" : "失败");
          }
    }
    T cur = t_alloc(3, (int[]){B, FEATURE_DIM, T0}); t_copy(&cur, x);
    /* block0: conv FEATURE_DIM -> DEC_DIMS[0] k=3 p=1 */
    T c = t_alloc(3, (int[]){B, DEC_DIMS[0], 0});
    conv1d(W_("decoder.blocks.0.weight"), W_("decoder.blocks.0.bias"), &cur, &c, FEATURE_DIM, DEC_DIMS[0], 3, 1, 1, 1, T0);
    t_free(&cur); cur = c;
    int Tn = cur.shape[2];
    /* stage loop: each stage = [ConvUnit x DEC_DEPTHS[s]] + Enhance + Up.
     * block indices are fixed across rates: stage s -> convu=1+3s,
     * enh=2+3s, up_layer=3+3s (matches Python modules.Decoder). Extra
     * ConvUnits for DEC_DEPTHS[s]>1 live at decoder.blocks.{convu}.{j}. */
    for (int s = 0; s < DEC_STAGES; s++) {
        /* 末段不再用卷积暂存（它的 uc_need = 768*T0 恒大于暂存的 448*T0），
         * 而末段自己的 conv_unit 光分块临时块就要 1.84MB，先把暂存还回去。
         * 主机 8.5MB 实测：不还，stage 3 的 conv_unit 就分不到那 1.84MB。 */
        if (s == DEC_STAGES - 1) { t_free(&sc); }
        int convu = 1 + 3 * s, enh = 2 + 3 * s;   /* up_layer = enh+1 = 3+3s */
        int dim_s = DEC_DIMS[s];
        /* per-stage ConvUnit(s); each is a residual add (depth>1 -> j loop) */
        for (int j = 0; j < DEC_DEPTHS[s]; j++) {
            char p[64]; snprintf(p, sizeof(p), "decoder.blocks.%d.%d.module", convu, j);
            /* conv_unit_add 就地累加，省掉一整块 [dim_s,Tn]（4 秒音频 3.6MB） */
            conv_unit_add(&cur, p);
        }
        /* enhance */
        /* 2026-09-14 第二十一轮：enhance 就地写回 cur，省掉一整块 [dim,T]。
         * 分支（trend-pool + merge_layer.0）都先算完，最后那句 y = x + merged*x
         * 逐元素只读自己那一个元素，就地写完全等价（见 blocks.c enhance_block）。 */
        char pe[64]; snprintf(pe, sizeof(pe), "decoder.blocks.%d", enh);
        enhance_block(&cur, &cur, pe, dim_s);
        /* cur 已经就地更新，不需要 t_free / 重新指向 */
        /* up: conv (dim_s -> nextdim, k=1,s=1) + Upsample(scale) + ChannelNorm(first) */
        int nxt = (s + 1 < DEC_STAGES) ? DEC_DIMS[s+1] : DEC_DIMS[DEC_STAGES];
        int scale = DEC_STRIDES[s];
        char cn[64]; snprintf(cn, sizeof(cn), "decoder.blocks.%d.0.weight", enh + 1);
        char cb[64]; snprintf(cb, sizeof(cb), "decoder.blocks.%d.0.bias", enh + 1);
        char nn[64]; snprintf(nn, sizeof(nn), "decoder.blocks.%d.2.weight", enh + 1);
        char nb[64]; snprintf(nb, sizeof(nb), "decoder.blocks.%d.2.bias", enh + 1);
        T up = t_alloc(3, (int[]){B, nxt, 0});
        /* JX_TLS=1 强制最后一个 stage 走流式末段（主机对照 / 板端试验用）。 */
        const char *tls_e = getenv("JX_TLS");
        int force_tls = (tls_e != NULL && atoi(tls_e) != 0) && (s == DEC_STAGES - 1);
        int rc_up = 0;
        if (!force_tls)
            rc_up = decoder_up_path(&cur, B, dim_s, nxt, Tn, scale,
                                    W_(cn), W_(cb), W_(nn), W_(nb),
                                    &sc, (s == DEC_STAGES - 1), &up);
        if (rc_up != 0 || force_tls)
          {
            /* 整段 up 拿不到连续内存（长录音）。只有最后一个 stage 才能走流式
             * 兜底：它直接产出 out，跳过后面的 decoder.blocks.13。 */
            if (s == DEC_STAGES - 1)
              {
                printf("[DEC] %s（Tn=%d scale=%d），转流式末段\n",
                       force_tls ? "JX_TLS 强制流式末段" : "末段整段 up 分配失败", Tn, scale);
                dec_mem("末段流转前", s, Tn);
                T tls = t_alloc(3, (int[]){B, 1, 0});
                /* 失败时 decoder_tail_stream 已经把 tls 标成 len=0 并抬了
                 * jx_conv_skip，调用方据此判“本次结果无效”。 */
                (void)decoder_tail_stream(&cur, B, dim_s, nxt, Tn, scale,
                                          W_(cn), W_(cb), W_(nn), W_(nb),
                                          DEC_DIMS[DEC_STAGES], &tls);
                t_free(&sc); t_free(&up); t_free(&cur);
                *out = tls;
                JXP_END(JP_DECODER);
                return;
              }
            jx_conv_skip++;
            dec_mem("stage up 失败", s, Tn);
            printf("[DEC] stage %d 的 up 分配失败，无法继续\n", s);
            t_free(&sc); t_free(&up); t_free(&cur);
            out->ndim = 3; out->shape[0] = B; out->shape[1] = 1; out->shape[2] = 0; out->len = 0;
            JXP_END(JP_DECODER);
            return;
          }
        t_free(&cur); cur = up; Tn = cur.shape[2];
        dec_mem("stage 出", s, Tn);
    }
    /* legacy last block (decoder.blocks.13): 3x Residual(LegacyUnit) + Snake + conv(DEC_DIMS[DEC_STAGES]->1) + Tanh */
    int last_dim = DEC_DIMS[DEC_STAGES];
    for (int r = 0; r < 3; r++) {
        char rp[64]; snprintf(rp, sizeof(rp), "decoder.blocks.13.block.0.%d.module", r);
        /* 2026-09-14 第二十一轮：改成时间分块 + 就地累加（blocks.c legacy_unit_add）。
         * 整段展开时 cur/lu/s0/c1 四块 [1,16,T] 同时活着，2 秒音频就是 8.2MB，
         * 超过 PSRAM 堆总量 -> conv1d_d 的 realloc 必失败、卷积被跳过。数值逐位不变。 */
        legacy_unit_add(&cur, rp, (r == 0 ? 1 : r == 1 ? 3 : 9));
    }
    /* 2026-09-13 第九轮：原来先把 cur 整块复制到 sn 再就地 snake。
     * cur 在这之后只被 free，没有任何读者，直接原地 snake 即可
     * （T=16032 时省掉 2MB 往返，数值不变）。 */
    snake1d(&cur, W_("decoder.blocks.13.block.1.alpha"), B, Tn, last_dim, 0);
    T fin = t_alloc(3, (int[]){B, 1, 0}); conv1d(W_("decoder.blocks.13.block.2.weight"), W_("decoder.blocks.13.block.2.bias"), &cur, &fin, last_dim, 1, 7, 1, 3, 1, Tn);
    JXP_BEG(JP_GELU);
    jx_kt_ensure();
    { const jx_tanh8 TH = jx_tanh8_load();
      for (int i = 0; i < fin.len; i++) fin.d[i] = jx_tanhf8(fin.d[i], TH); }
    JXP_END(JP_GELU);
    t_free(&sc);
    t_free(&cur); *out = fin;
    JXP_END(JP_DECODER);
}

/* ============================ LocalDecoder.forward ============================ */
void en_decoder_forward(const T *x, T *out) {
    JXP_BEG(JP_ENDECODER);
    /* x[B,T',C] -> LocalTrans stack -> permute [B,C,T'] */
    int B = x->shape[0], Tt = x->shape[1], C = x->shape[2];

    if (EN_USE_COMPRESSED) {
        /* CompressedLocalDecoderWithCache:
         *   local_trans = LocalTrans(depth=EN_DEC_LOCAL_DEPTH, win=EN_DEC_LOCAL_WIN)
         *                (depth 0 -> identity for 1kbps)
         *   up_trans = Upsample(scale=rate) then LocalTrans(depth=EN_DEC_UP_DEPTH, win=EN_DEC_UP_WIN) */
        T lt;                       /* local_transformer 自己分配并交接出来 */
        if (EN_DEC_LOCAL_DEPTH > 0)
            local_transformer(x, &lt, "en_decoder.local_trans", EN_DEC_LOCAL_DEPTH, EN_DEC_LOCAL_WIN);
        else
          {
            lt = t_alloc(3, (int[]){B, Tt, C});
            t_copy(&lt, x);
          }
        /* up_layer: permute (B,T,C)->(B,C,T), upsample linear scale=rate, permute back */
        T ltc = t_alloc(3, (int[]){B, C, Tt});
        for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tt; t++) for (int c = 0; c < C; c++)
            ltc.d[(bi*C+c)*Tt+t] = lt.d[(bi*Tt+t)*C+c];
        t_free(&lt);
        int Tu = Tt * EN_ENC_COMPRESS_RATE;
        T up = t_alloc(3, (int[]){B, C, Tu});
        upsampling_linear(&ltc, &up, B, C, Tt, EN_ENC_COMPRESS_RATE);
        t_free(&ltc);
        T upp = t_alloc(3, (int[]){B, Tu, C});
        /* permute (B,C,Tu)->(B,Tu,C) channels_last. NOTE: dest is (B,Tu,C) so the
         * destination offset MUST be (bi*Tu+t)*C+c, NOT (bi*C+c)*Tu+t (which would
         * keep the data in channels-first layout and scramble it -> garbage/explosion
         * in the following up_trans.trans). The encoder's down_layer permute uses the
         * same correct (bi*Td+t)*C+c form. */
        for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tu; t++) for (int c = 0; c < C; c++)
            upp.d[(bi*Tu+t)*C+c] = up.d[(bi*C+c)*Tu+t];
        t_free(&up);
        T ut;                       /* 同上 */
        local_transformer(&upp, &ut, "en_decoder.up_trans.trans", EN_DEC_UP_DEPTH, EN_DEC_UP_WIN);
        t_free(&upp);
        T perm = t_alloc(3, (int[]){B, C, Tu});
        for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tu; t++) for (int c = 0; c < C; c++)
            perm.d[(bi*C+c)*Tu+t] = ut.d[(bi*Tu+t)*C+c];
        t_free(&ut);
        *out = perm;
    } else {
        T tr;                       /* 同上 */
        local_transformer(x, &tr, "en_decoder.local_trans", EN_DEC_DEPTH, EN_WINDOW);
        T perm = t_alloc(3, (int[]){B, C, Tt});
        for (int bi = 0; bi < B; bi++) for (int t = 0; t < Tt; t++) for (int c = 0; c < C; c++)
            perm.d[(bi*C+c)*Tt+t] = tr.d[(bi*Tt+t)*C+c];
        t_free(&tr);
        *out = perm;
    }
    JXP_END(JP_ENDECODER);
}
