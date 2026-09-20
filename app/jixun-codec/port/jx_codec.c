/****************************************************************************
 * apps/jixun-codec/port/jx_codec.c
 *
 * 极讯 AI Codec 板端封装：PCM(int16) <-> 量化索引 idx
 *
 * 流程与 PC 端 test/codec_main.c 的 encode_to_idx / decode_from_idx 一一对应，
 * 只把接口换成 int16 PCM 并加分段计时。
 ****************************************************************************/

#include <nuttx/config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nn_infer.h"
#include "fsq_codec.h"
#include "jx_codec.h"

/****************************************************************************
 * 码率标识（编译期决定）
 ****************************************************************************/

#if defined(SQM_RATE_1K)
#  define JX_RATE_STR "1k"
#elif defined(SQM_RATE_6K)
#  define JX_RATE_STR "6k"
#elif defined(SQM_RATE_3K)
#  define JX_RATE_STR "3k"
#else
#  define JX_RATE_STR "3k"
#endif

const char *jx_rate_name(void) { return JX_RATE_STR; }
int jx_token_bits(void)        { return VQ_TOTAL_BITS; }

/****************************************************************************
 * 计时
 ****************************************************************************/

jx_timing_t jx_last_timing;

static double jx_now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/****************************************************************************
 * 前处理：右补零到 HOP_LENGTH * EN_ENC_COMPRESS_RATE 的整数倍
 * （与 Python codec.py 的 fill_length 语义一致）
 ****************************************************************************/

static float *jx_preprocess(const int16_t *pcm, int n, int *out_len)
{
  const int frame = HOP_LENGTH * EN_ENC_COMPRESS_RATE;
  const int pad   = (frame - (n % frame)) % frame;
  const int total = n + pad;

  float *buf = (float *)malloc(sizeof(float) * (size_t)total);
  if (buf == NULL)
    {
      return NULL;
    }

  for (int i = 0; i < n; i++)
    {
      buf[i] = (float)pcm[i] / 32768.0f;
    }

  for (int i = n; i < total; i++)
    {
      buf[i] = 0.0f;
    }

  *out_len = total;
  return buf;
}

/****************************************************************************
 * 公共 API：编码
 ****************************************************************************/

/****************************************************************************
 * 竞态定位探针（JX_SH=1）：把各阶段张量按字节做 FNV-1a 64 打印指纹。
 * 用途：双核结果不可复现时，逐级比对两次冷启动的指纹，找出**第一处**分歧。
 ****************************************************************************/
void jx_stage_h(const char *name, const void *p, size_t nbytes)
{
  static int on = -1;
  if (on < 0)
    {
      const char *e = getenv("JX_SH");
      on = (e != NULL && e[0] != (char)48);
    }
  if (!on) { return; }
  unsigned long long h = 1469598103934665603ull;
  if (p != NULL)
    {
      const unsigned char *b = (const unsigned char *)p;
      for (size_t i = 0; i < nbytes; i++)
        {
          h = (h ^ b[i]) * 1099511628211ull;
        }
    }
  else
    {
      h = 0ull;
    }
  printf("  [SH] %-26s %016llx  %u B%s\n", name, h, (unsigned)nbytes,
         p == NULL ? "  (NULL)" : "");
  /* r168（2026-09-17 双核不确定性）：数值幅度探针（JX_SHM=1）。
   * 逐字节哈希只能回答"同/不同"，回答不了"差多少"。这里再算三档
   * **容差哈希**：每个 float 先按绝对容差 1e-6 / 1e-4 / 1e-2 量化再哈希。
   * 比较两次运行：匹配到哪一档，就说明最大绝对偏差落在那档以下，
   * 从而直接判定"舍入级差异（无害）"还是"真错（有害）"。 */
  {
    static int mon = -1;
    if (mon < 0)
      {
        const char *e = getenv("JX_SHM");
        mon = (e != NULL && e[0] != (char)48);
      }
    if (mon && p != NULL && nbytes >= 4u && (nbytes & 3u) == 0u)
      {
        const float *f = (const float *)p;
        size_t n = nbytes / 4u; size_t i;
        float mx = 0.0f; double sum = 0.0;
        unsigned long long h6 = 1469598103934665603ull;
        unsigned long long h4 = h6, h2 = h6;
        for (i = 0; i < n; i++)
          {
            float v = f[i];
            float a = (v < 0.0f) ? -v : v;
            float sg = (v >= 0.0f) ? 0.5f : -0.5f;
            int q6, q4, q2;
            if (a > mx) { mx = a; }
            sum += (double)v;
            if (a > 2000.0f)
              { q6 = q4 = q2 = (v > 0.0f) ? 2000000000 : -2000000000; }
            else
              {
                q6 = (int)(v * 1000000.0f + sg);
                q4 = (int)(v * 10000.0f + sg);
                q2 = (int)(v * 100.0f + sg);
              }
            h6 = (h6 ^ (unsigned long long)(unsigned)q6) * 1099511628211ull;
            h4 = (h4 ^ (unsigned long long)(unsigned)q4) * 1099511628211ull;
            h2 = (h2 ^ (unsigned long long)(unsigned)q2) * 1099511628211ull;
          }
        { /* r170：把匹配 JX_SHDT 的阶段的**前 JX_SHDN 个 float** 以 8 位十六进制逐行打印，
           * 用来在 PC 上逐元素比较两次运行的张量内容（判"位移"还是"个别元素脏"）。 */
          static long _dn = -1; static const char *_dt = NULL;
          if (_dn < 0)
            {
              const char *_e1 = getenv("JX_SHDN"); const char *_e2 = getenv("JX_SHDT");
              _dn = (_e1 != NULL) ? atol(_e1) : 0;
              _dt = (_e2 != NULL) ? _e2 : "";
            }
          if (_dn > 0 && p != NULL && (nbytes & 3u) == 0u && _dt[0] != 0 && strstr(name, _dt) != NULL)
            {
              size_t _m = nbytes / 4u; size_t _i;
              if (_m > (size_t)_dn) { _m = (size_t)_dn; }
              for (_i = 0; _i < _m; _i++)
                {
                  unsigned _w; memcpy(&_w, &f[_i], 4);
                  printf("%08x%c", _w, ((_i & 7u) == 7u || _i + 1u == _m) ? (char)10 : (char)32);
                }
            } }
        { unsigned b0 = 0, b1 = 0, b2 = 0, b3 = 0;
          if (n > 0) { memcpy(&b0, &f[0], 4); }
          if (n > 1) { memcpy(&b1, &f[1], 4); }
          if (n > 2) { memcpy(&b2, &f[2], 4); }
          if (n > 3) { memcpy(&b3, &f[3], 4); }
          printf("  [SX] %-26s p=%p mx=%-11.6g sum=%-14.7g w1e-6=%016llx w1e-4=%016llx w1e-2=%016llx f0..3=%08x,%08x,%08x,%08x\n",
                 name, p, (double)mx, sum, h6, h4, h2, b0, b1, b2, b3); }
      }
  }
}

int jx_encode(const int16_t *pcm, int n_samples,
              int64_t **out_idx, int *out_ntok, int *out_Tt)
{
  int T0 = 0;
  float *padded = jx_preprocess(pcm, n_samples, &T0);
  if (padded == NULL)
    {
      return -1;
    }

  const int B  = 1;
  const int Tt = T0 / HOP_LENGTH;

  double t0 = jx_now_ms();

  jx_stage_h("enc.pre.padded", padded, (size_t)T0 * sizeof(float));
  T xin = t_alloc(3, (int[]){B, 1, T0});
  for (int t = 0; t < T0; t++)
    {
      xin.d[t] = padded[t];
    }

  /* encoder_forward 内部用 `*out = <新张量>` 交接结果，会整体替换 out->d，
   * 所以这里不能预分配（预分配的那块会被丢弃，白白泄漏一份全尺寸张量）。 */
  T enc;
  encoder_forward(&xin, &enc);

  jx_stage_h("enc.encoder_forward", enc.d, (size_t)enc.len * sizeof(float));
  T enenc;
  en_encoder_forward(&enc, &enenc);

  double t1 = jx_now_ms();

  /* 压缩路径（1kbps）下 en_encoder 已把时序压到 Tt/EN_ENC_COMPRESS_RATE */
  const int Tlat = Tt / EN_ENC_COMPRESS_RATE;

  jx_stage_h("enc.en_encoder", enenc.d, (size_t)enenc.len * sizeof(float));
  int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)B * (size_t)Tlat);
  if (idx == NULL)
    {
      t_free(&xin); t_free(&enc); t_free(&enenc); free(padded);
      return -1;
    }

  T enq = t_alloc(3, (int[]){B, Tlat, FEATURE_DIM});
  quantizer_forward(&enenc, &enq, idx, Tlat);

  double t2 = jx_now_ms();

  jx_stage_h("enc.idx", idx, (size_t)Tlat * sizeof(int64_t));
  t_free(&xin); t_free(&enc); t_free(&enenc); t_free(&enq);
  free(padded);

  jx_last_timing.pre_ms   = t1 - t0;
  jx_last_timing.quant_ms = t2 - t1;
  jx_last_timing.total_ms = t2 - t0;

  *out_idx  = idx;
  *out_ntok = Tlat;
  *out_Tt   = Tt;
  return 0;
}

/****************************************************************************
 * 公共 API：解码
 ****************************************************************************/

int jx_decode(const int64_t *idx, int Tt, int origT, float **out_pcm)
{
  const int B    = 1;
  const int Tlat = Tt / EN_ENC_COMPRESS_RATE;
  const int T0   = Tt * HOP_LENGTH;

  double t0 = jx_now_ms();

  T enq = t_alloc(3, (int[]){B, Tlat, FEATURE_DIM});
  quantizer_to_features(idx, &enq, Tlat);

  jx_stage_h("dec.enq", enq.d, (size_t)enq.len * sizeof(float));
  T endec;                       /* 同上：由 en_decoder_forward 分配 */
  en_decoder_forward(&enq, &endec);

  double t1 = jx_now_ms();

  T dec;                         /* 同上：由 decoder_forward 分配 */
  jx_stage_h("dec.en_decoder", endec.d, (size_t)endec.len * sizeof(float));
  memset(&dec, 0, sizeof(dec));  /* 防呆：decoder_forward 一定会写，但不留未初始化态 */
  decoder_forward(&endec, &dec);

  double t2 = jx_now_ms();

  /* 2026-09-14 第二十三轮：解码器可以"明确放弃"（长录音时堆里凑不出连续大块），
   * 这时 dec.d==NULL / dec.len==0。原来没有这道检查，直接按 origT 读 dec.d
   * -> LoadProhibited 崩机。宁可报错返回 -1，也不要死机。 */
  if (dec.d == NULL || dec.len <= 0)
    {
      printf("[JX] 解码结果无效：d=%p len=%d 需要 %d，放弃本次\n",
             (void *)dec.d, dec.len, origT);
      t_free(&enq); t_free(&endec); t_free(&dec);
      return -1;
    }

  jx_stage_h("dec.decoder", dec.d, (size_t)dec.len * sizeof(float));
  float *out = (float *)malloc(sizeof(float) * (size_t)origT);
  if (out == NULL)
    {
      t_free(&enq); t_free(&endec); t_free(&dec);
      return -1;
    }

  memset(out, 0, sizeof(float) * (size_t)origT);
  {
    int copy = (dec.len < origT) ? dec.len : origT;
    for (int i = 0; i < copy; i++)
      {
        out[i] = dec.d[i];
      }
  }

  t_free(&enq); t_free(&endec); t_free(&dec);

  jx_last_timing.deq_attn_ms = t1 - t0;
  jx_last_timing.dec_ms      = t2 - t1;
  jx_last_timing.total_ms   += (t2 - t0);

  *out_pcm = out;
  return 0;
}

/****************************************************************************
 * 24ms 小包流式（相对区间语义）—— 朴素实现
 *
 * 每个包都对「1 个注意力窗 + 右侧 halo + 本包」的滚动缓冲重跑一遍编解码，
 * 只取本包那 4 帧的 token 与 PCM。语义是对的（卷积边界拿到真实数据、
 * 注意力缓冲起点对齐窗格、归一化走滑动窗），但**代价是每包 O(2.4s) 的算力**，
 * 吞吐还达不到实时 —— 这一版的作用是把流水线、包协议和听感先跑通。
 *
 * 下一步（吞吐）：给注意力加 k/v 缓存，只对新增帧算 query，key/value 从缓存取，
 * 那时每包代价降到 O(24ms)，才谈得上实时。这是 transformer 的结构改动。
 ****************************************************************************/

struct jx_stream_s
{
  int16_t *buf;      /* 滚动缓冲：[window + halo + pkt] 样本 */
  int      cap;      /* 容量（样本） */
  int      have;     /* 已填入的真实样本数 */
  int      halo;     /* 右侧 halo（样本，已对齐到整帧） */
  int      pkts;     /* 已喂包数 */
  int      win;      /* 注意力窗帧数 */

  /* 2026-09-18 增补：把「卷积」和「注意力」拆开跑。
   *
   * 起因（r194 实测）：朴素实现每包对整条窗口重跑编码器，编码器第一级输出
   * 必须是全窗口尺寸 [1,16,T0] —— 400 帧 → 2.51MB，而板上堆的最大连续块
   * 只有 2.36MB，直接 OOM；同时耗时 3090ms/包 = 129 倍实时。
   *
   * 关键认识：**2.4 秒的上下文只有注意力需要，卷积不需要**（感受野只有
   * 几十个样本）。所以：
   *   卷积链 → 只喂「本包 + 左右各 5 帧 halo」= 1344 样本，张量 84KB
   *   注意力链 → 喂整个特征环（400 帧 × 64 通道 = 100KB 级）
   * 两堵墙一起解掉。 */
  float   *feat;     /* 特征环：FEATURE_DIM × win（channels-first，与 encoder 输出一致） */
  int64_t *tokr;     /* token 环：win 个 token，解码侧用 */

  /* 分段计时。放结构体里（堆上）而不是全局 —— dram0_0_seg 已溢出 64 字节，
   * 任何新的 .data/.bss 都会让链接失败。 */
  double   t_conv, t_attn, t_dec;
  double   t_deq, t_endec, t_decnn;

  /* 编码侧 LocalEncoder 的跨包 K/V 缓存（标准 3k/6k 路径，depth=1）。
   * 缓存按“最近 win 帧”保存，步长恰好一个包；每包只对新到的 4 帧算 QKV，
   * 旧帧的 K/V 直接复用，因此不再重跑整圈 400 帧注意力。 */
  int      ec_on;
  int      ec_cap;
  int      ec_count;
  int      ec_inner;
  int      ec_heads;
  int      ec_dh;
  int      ec_dpb_h;
  int      ec_ff_hid;
  int      ec_ff_half;
  float   *ec_k;
  float   *ec_v;
  float   *ec_bias;
  const float *ec_wqkv;
  const float *ec_wo;
  const float *ec_wmha_n;
  const float *ec_bmha_n;
  const float *ec_ff_nw;
  const float *ec_ff_nb;
  const float *ec_ff_w0;
  const float *ec_ff_w2;
  const float *ec_dpb_w0;
  const float *ec_dpb_b0;
  const float *ec_dpb_w1;
  const float *ec_dpb_b1;
  const float *ec_dpb_w2;
  const float *ec_dpb_b2;
};

/* Small-tensor linear: direct FP32 is deliberately kept here.  The generic
 * int8 linear path uses shared per-core scratch and is not safe to call
 * re-entrantly from this streaming state machine. */
static void jx_stream_linear_f(const float *w, const float *b,
                               const T *x, T *y, int in, int out)
{
  int rows = x->len / in;
  const char *i8e = getenv("JX_STREAM_I8");

  if ((i8e == NULL || i8e[0] != (char)48) &&
      x->d != NULL && y->d != NULL &&
      jx_linear_i8_ex(w, b, x->d, y->d, rows, in, out, NULL) == 0)
    {
      y->shape[0] = x->shape[0];
      y->shape[1] = x->shape[1];
      y->shape[2] = out;
      y->len = rows * out;
      return;
    }

  for (int r = 0; r < rows; r++)
    {
      const float *xr = &x->d[(size_t)r * in];
      float *yr = &y->d[(size_t)r * out];
      for (int o = 0; o < out; o++)
        {
          const float *wr = &w[(size_t)o * in];
          float acc = (b != NULL) ? b[o] : 0.0f;
          for (int i = 0; i < in; i++) { acc += wr[i] * xr[i]; }
          yr[o] = acc;
        }
    }
  y->shape[0] = x->shape[0];
  y->shape[1] = x->shape[1];
  y->shape[2] = out;
  y->len = rows * out;
}

static int jx_stream_ffn(const struct jx_stream_s *st, const T *x, T *out)
{
  const int C = FEATURE_DIM;
  const int Tn = x->shape[1];
  const int hid = st->ec_ff_hid;
  const int half = st->ec_ff_half;
  T a = t_alloc(3, (int[]){1, Tn, C});
  T h = t_alloc(3, (int[]){1, Tn, hid});
  T g = t_alloc(3, (int[]){1, Tn, half});

  if (a.d == NULL || h.d == NULL || g.d == NULL)
    {
      t_free(&a); t_free(&h); t_free(&g);
      return -1;
    }

  t_copy(&a, x);
  channel_norm_last(&a, st->ec_ff_nw, st->ec_ff_nb, 1, Tn, C);
  jx_stream_linear_f(st->ec_ff_w0, NULL, &a, &h, C, hid);

  for (int r = 0; r < Tn; r++)
    {
      const float *h0 = &h.d[(size_t)r * hid];
      const float *h1 = h0 + half;
      float *gr = &g.d[(size_t)r * half];
      jx_kt_ensure();
      const jx_erf9 KE = jx_erf9_load();
      for (int j = 0; j < half; j++)
        {
          gr[j] = h0[j] * jx_gelu_fast(h1[j], KE);
        }
    }

  jx_stream_linear_f(st->ec_ff_w2, NULL, &g, out, half, C);
  t_free(&a); t_free(&h); t_free(&g);
  return 0;
}

/* DynamicPositionBias 的 1D MLP：与 transformer.c 的同名实现保持同一算式。 */
static void jx_stream_dpb(const struct jx_stream_s *st, float rel, float *out)
{
  float h1[64];
  float h2[64];

  for (int i = 0; i < st->ec_dpb_h; i++)
    {
      float a = st->ec_dpb_b0[i] + st->ec_dpb_w0[i] * rel;
      h1[i] = jx_silu(a);
    }

  for (int i = 0; i < st->ec_dpb_h; i++)
    {
      float a = st->ec_dpb_b1[i];
      for (int j = 0; j < st->ec_dpb_h; j++)
        {
          a += st->ec_dpb_w1[i * st->ec_dpb_h + j] * h1[j];
        }
      h2[i] = jx_silu(a);
    }

  for (int h = 0; h < st->ec_heads; h++)
    {
      float a = st->ec_dpb_b2[h];
      for (int j = 0; j < st->ec_dpb_h; j++)
        {
          a += st->ec_dpb_w2[h * st->ec_dpb_h + j] * h2[j];
        }
      out[h] = a;
    }
}

static int jx_stream_enc_cache_init(struct jx_stream_s *st, int win_frames)
{
  int qkv_shape[4], qkv_n;
  int dpb_shape[4], dpb_n;
  int dpb0_shape[4], dpb0_n;
  int ff_shape[4], ff_n;
  const float *wqkv;
  int init_frames;

  if (EN_USE_COMPRESSED != 0 || EN_ENC_DEPTH != 1)
    {
      return 0;
    }
  {
    const char *e = getenv("JX_ENC_CACHE");
    if (e != NULL && atoi(e) == 0)
      {
        return 0;
      }
  }

  wqkv = wt_get("en_encoder.local_trans.layers.0.0.to_qkv.weight",
                qkv_shape, &qkv_n);
  if (wqkv == NULL || qkv_shape[0] % 3 != 0)
    {
      return 0;
    }

  wt_get("en_encoder.local_trans.dynamic_pos_bias.mlp.4.weight",
         dpb_shape, &dpb_n);
  wt_get("en_encoder.local_trans.dynamic_pos_bias.mlp.0.weight",
         dpb0_shape, &dpb0_n);
  if (dpb_n <= 0 || dpb0_n <= 0)
    {
      return 0;
    }
  wt_get("en_encoder.local_trans.layers.0.1.1.weight", ff_shape, &ff_n);
  if (ff_n <= 0 || ff_shape[0] <= 0 || (ff_shape[0] & 1) != 0)
    {
      return 0;
    }

  st->ec_cap   = win_frames;
  st->ec_inner = qkv_shape[0] / 3;
  st->ec_heads = dpb_shape[0];
  st->ec_dpb_h = dpb0_shape[0];
  st->ec_ff_hid = ff_shape[0];
  st->ec_ff_half = ff_shape[0] / 2;
  if (st->ec_heads <= 0 || st->ec_inner % st->ec_heads != 0)
    {
      return 0;
    }
  st->ec_dh = st->ec_inner / st->ec_heads;
  init_frames = st->ec_cap;

  st->ec_k = (float *)jx_amalloc(sizeof(float) * (size_t)st->ec_cap *
                                 (size_t)st->ec_inner);
  st->ec_v = (float *)jx_amalloc(sizeof(float) * (size_t)st->ec_cap *
                                 (size_t)st->ec_inner);
  st->ec_bias = (float *)malloc(sizeof(float) * (size_t)st->ec_cap *
                                (size_t)st->ec_heads);
  if (st->ec_k == NULL || st->ec_v == NULL || st->ec_bias == NULL)
    {
      jx_afree(st->ec_k); jx_afree(st->ec_v); free(st->ec_bias);
      st->ec_k = NULL; st->ec_v = NULL; st->ec_bias = NULL;
      return 0;
    }

  st->ec_wqkv   = wqkv;
  st->ec_wo     = W_("en_encoder.local_trans.layers.0.0.to_out.weight");
  st->ec_wmha_n = W_("en_encoder.local_trans.layers.0.0.norm.weight");
  st->ec_bmha_n = W_("en_encoder.local_trans.layers.0.0.norm.bias");
  st->ec_ff_nw  = W_("en_encoder.local_trans.layers.0.1.0.weight");
  st->ec_ff_nb  = W_("en_encoder.local_trans.layers.0.1.0.bias");
  st->ec_ff_w0  = W_("en_encoder.local_trans.layers.0.1.1.weight");
  st->ec_ff_w2  = W_("en_encoder.local_trans.layers.0.1.4.weight");
  st->ec_dpb_w0 = W_("en_encoder.local_trans.dynamic_pos_bias.mlp.0.weight");
  st->ec_dpb_b0 = W_("en_encoder.local_trans.dynamic_pos_bias.mlp.0.bias");
  st->ec_dpb_w1 = W_("en_encoder.local_trans.dynamic_pos_bias.mlp.2.weight");
  st->ec_dpb_b1 = W_("en_encoder.local_trans.dynamic_pos_bias.mlp.2.bias");
  st->ec_dpb_w2 = W_("en_encoder.local_trans.dynamic_pos_bias.mlp.4.weight");
  st->ec_dpb_b2 = W_("en_encoder.local_trans.dynamic_pos_bias.mlp.4.bias");
  if (st->ec_wo == NULL || st->ec_wmha_n == NULL || st->ec_bmha_n == NULL ||
      st->ec_ff_nw == NULL || st->ec_ff_nb == NULL || st->ec_ff_w0 == NULL ||
      st->ec_ff_w2 == NULL || st->ec_dpb_w0 == NULL || st->ec_dpb_b0 == NULL ||
      st->ec_dpb_w1 == NULL || st->ec_dpb_b1 == NULL || st->ec_dpb_w2 == NULL ||
      st->ec_dpb_b2 == NULL)
    {
      jx_afree(st->ec_k); jx_afree(st->ec_v); free(st->ec_bias);
      st->ec_k = NULL; st->ec_v = NULL; st->ec_bias = NULL;
      return 0;
    }

  for (int r = 0; r < st->ec_cap; r++)
    {
      jx_stream_dpb(st, (float)r, &st->ec_bias[r * st->ec_heads]);
    }

  /* 缓存开头对应现有实现的补零预热段：zero -> norm 的输出就是 bias。 */
  {
    T zx = t_alloc(3, (int[]){1, 1, FEATURE_DIM});
    T zq = t_alloc(3, (int[]){1, 1, st->ec_inner * 3});
    if (zx.d == NULL || zq.d == NULL)
      {
        t_free(&zx); t_free(&zq);
        jx_afree(st->ec_k); jx_afree(st->ec_v); free(st->ec_bias);
        st->ec_k = NULL; st->ec_v = NULL; st->ec_bias = NULL;
        return 0;
      }
    for (int c = 0; c < FEATURE_DIM; c++) { zx.d[c] = st->ec_bmha_n[c]; }
    jx_stream_linear_f(st->ec_wqkv, NULL, &zx, &zq,
                       FEATURE_DIM, st->ec_inner * 3);
    for (int i = 0; i < init_frames; i++)
      {
        memcpy(&st->ec_k[(size_t)i * st->ec_inner],
               &zq.d[st->ec_inner], sizeof(float) * (size_t)st->ec_inner);
        memcpy(&st->ec_v[(size_t)i * st->ec_inner],
               &zq.d[2 * st->ec_inner], sizeof(float) * (size_t)st->ec_inner);
      }
    t_free(&zx); t_free(&zq);
  }

  st->ec_count = st->ec_cap;
  st->ec_on = 1;
  return 1;
}

static int jx_stream_enc_cache_step_n(struct jx_stream_s *st, int Tn, T *out)
{
  const int C = FEATURE_DIM;
  const int inner = st->ec_inner;
  const int heads = st->ec_heads;
  const int dh = st->ec_dh;
  const float scale = 1.0f / sqrtf((float)dh);
  float sims[512];
  T pl, xn, qkv, attn, mh, y, ff;
  double _ep_t0 = 0.0, _ep_t1 = 0.0, _ep_t2 = 0.0, _ep_t3 = 0.0;
  double _ep_t4 = 0.0, _ep_t5 = 0.0, _ep_t6 = 0.0;
  int _ep_on = (getenv("JX_ECPROF") != NULL);

  if (!st->ec_on || st->ec_cap > 512)
    {
      return -1;
    }

  pl = t_alloc(3, (int[]){1, Tn, C});
  xn = t_alloc(3, (int[]){1, Tn, C});
  qkv = t_alloc(3, (int[]){1, Tn, inner * 3});
  attn = t_alloc(3, (int[]){1, Tn, inner});
  mh = t_alloc(3, (int[]){1, Tn, C});
  y = t_alloc(3, (int[]){1, Tn, C});
  ff = t_alloc(3, (int[]){1, Tn, C});
  if (pl.d == NULL || xn.d == NULL || qkv.d == NULL || attn.d == NULL ||
      mh.d == NULL || y.d == NULL || ff.d == NULL)
    {
      t_free(&pl); t_free(&xn); t_free(&qkv); t_free(&attn);
      t_free(&mh); t_free(&y); t_free(&ff);
      return -1;
    }

  for (int c = 0; c < C; c++)
    {
      const float *src = &st->feat[(size_t)c * st->win + st->win - Tn];
      for (int i = 0; i < Tn; i++) { pl.d[i * C + c] = src[i]; }
    }
  t_copy(&xn, &pl);
  _ep_t0 = jx_now_ms();
  channel_norm_last(&xn, st->ec_wmha_n, st->ec_bmha_n, 1, Tn, C);
  _ep_t1 = jx_now_ms();
  jx_stream_linear_f(st->ec_wqkv, NULL, &xn, &qkv, C, inner * 3);
  _ep_t2 = jx_now_ms();

  if (st->ec_count == st->ec_cap)
    {
      memmove(st->ec_k, &st->ec_k[(size_t)Tn * inner],
              sizeof(float) * (size_t)(st->ec_cap - Tn) * (size_t)inner);
      memmove(st->ec_v, &st->ec_v[(size_t)Tn * inner],
              sizeof(float) * (size_t)(st->ec_cap - Tn) * (size_t)inner);
      st->ec_count -= Tn;
    }

  for (int i = 0; i < Tn; i++)
    {
      memcpy(&st->ec_k[(size_t)(st->ec_count + i) * inner],
             &qkv.d[(size_t)i * inner * 3 + inner],
             sizeof(float) * (size_t)inner);
      memcpy(&st->ec_v[(size_t)(st->ec_count + i) * inner],
             &qkv.d[(size_t)i * inner * 3 + 2 * inner],
             sizeof(float) * (size_t)inner);
    }
  st->ec_count += Tn;

  for (int h = 0; h < heads; h++)
    {
      for (int i = 0; i < Tn; i++)
        {
          const int qpos = st->ec_count - Tn + i;
          const int nkey = qpos + 1;
          const int start = st->ec_count - nkey;
          const float *qrow = &qkv.d[(size_t)i * inner * 3 + h * dh];
          float mx = -1e30f;

          for (int j = start; j <= qpos; j++)
            {
              const float *krow = &st->ec_k[(size_t)j * inner + h * dh];
              float s = 0.0f;
              for (int d = 0; d < dh; d++) { s += qrow[d] * krow[d]; }
              s = s * scale + st->ec_bias[(size_t)(qpos - j) * heads + h];
              sims[j - start] = s;
              if (s > mx) { mx = s; }
            }

          float sum = 0.0f;
          for (int j = start; j <= qpos; j++)
            {
              float e = jx_expf_neg(sims[j - start] - mx);
              sims[j - start] = e;
              sum += e;
            }
          float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
          for (int d = 0; d < dh; d++)
            {
              float acc = 0.0f;
              for (int j = start; j <= qpos; j++)
                {
                  acc += sims[j - start] *
                         st->ec_v[(size_t)j * inner + h * dh + d];
                }
              attn.d[(size_t)i * inner + h * dh + d] = acc * inv;
            }
        }
    }

  _ep_t3 = jx_now_ms();
  jx_stream_linear_f(st->ec_wo, NULL, &attn, &mh, inner, C);
  _ep_t4 = jx_now_ms();
  for (int i = 0; i < pl.len; i++) { y.d[i] = pl.d[i] + mh.d[i]; }
  if (jx_stream_ffn(st, &y, &ff) != 0)
    {
      t_free(&pl); t_free(&xn); t_free(&qkv); t_free(&attn);
      t_free(&mh); t_free(&y); t_free(&ff);
      return -1;
    }
  _ep_t5 = jx_now_ms();
  for (int i = 0; i < y.len; i++) { y.d[i] += ff.d[i]; }
  _ep_t6 = jx_now_ms();
  if (_ep_on)
    {
      printf("[EC] Tn=%d norm %.2f qkv %.2f attn %.2f wo %.2f ffn %.2f add %.2f total %.2f ms\n",
             Tn, _ep_t1 - _ep_t0, _ep_t2 - _ep_t1, _ep_t3 - _ep_t2,
             _ep_t4 - _ep_t3, _ep_t5 - _ep_t4, _ep_t6 - _ep_t5,
             _ep_t6 - _ep_t0);
    }
  *out = y;

  t_free(&pl); t_free(&xn); t_free(&qkv); t_free(&attn);
  t_free(&mh); t_free(&ff);
  return 0;
}

static int jx_stream_enc_cache_step(struct jx_stream_s *st, T *out)
{
  return jx_stream_enc_cache_step_n(st, JX_PKT_FRAMES, out);
}

/* 解码环长度（帧）。与编码环解耦：编码侧注意力需要 2.4 秒上下文，解码侧
 * 不需要那么长，而解码恰恰是每包耗时的大头（r196 实测 5690/6260 ms = 91%，
 * 每包把整个 2.4 秒环重新解码一遍）。JX_DECWIN 可以扫出「解码环长度 ↔
 * 每包耗时/音质」的曲线。默认 4 帧，避免每条小包都重跑长解码环。 */
static int jx_decwin_frames(int win)
{
  const char *e = getenv("JX_DECWIN");

  if (e != NULL)
    {
      int v = atoi(e);

      if (v >= JX_PKT_FRAMES && v <= win) { return v; }
    }
  /* 解码侧不需要和编码侧一样长的上下文；默认只保留当前包所需的 4 帧。
   * 这是延迟优先的默认值，需要做长解码环 A/B 时用 JX_DECWIN 覆盖。 */
  return JX_PKT_FRAMES;
}

/* 卷积窗口：本包 + 左右各 5 帧 halo。1344 样本 = 14 帧（整帧对齐） */
#define JX_CONV_HALO_FRAMES 5
#define JX_CONV_WIN_SMP \
  (JX_PKT_SAMPLES + 2 * JX_CONV_HALO_FRAMES * HOP_LENGTH)

static int jx_conv_halo(const struct jx_stream_s *st)
{
  const char *e = getenv("JX_CONV_HALO");
  int h = (e != NULL) ? atoi(e) : JX_CONV_HALO_FRAMES;
  int maxh = (st->cap - JX_PKT_SAMPLES) / (2 * HOP_LENGTH);

  if (h < 0) { h = 0; }
  if (h > maxh) { h = maxh; }
  return h;
}

struct jx_stream_s *jx_stream_open(int halo_samples)
{
  struct jx_stream_s *st;
  int win_frames = JX_STREAM_WINDOW_DEFAULT;

  /* 窗口可用 JX_STREAMWIN 覆盖（帧）。
   * 为什么需要这个开关：朴素实现每包都要对整条窗口重跑编码器，而编码器
   * 第一级输出是 [1,16,T0] 全尺寸张量 —— 400 帧窗口 = 39296 样本 = 2.51MB，
   * 板上堆的最大连续块只有 2.36MB，直接 OOM（r193 实测）。
   * 先用小窗口把流水线端到端跑通、拿到 token/PCM、验证稳定性；
   * 窗口调回 400 要靠「卷积只算新增帧 + 注意力走 k/v 环形缓存」来省内存。 */
  {
    const char *e = getenv("JX_STREAMWIN");

    if (e != NULL)
      {
        int v = atoi(e);

        if (v >= JX_PKT_FRAMES && v <= JX_NET_WINDOW_FRAMES)
          {
            win_frames = v;
          }
      }
  }

  if (halo_samples < 0) { halo_samples = 0; }
  /* halo 必须对齐到整帧，否则产出区间的帧边界不是整数 */
  halo_samples = (halo_samples / HOP_LENGTH) * HOP_LENGTH;

  st = (struct jx_stream_s *)malloc(sizeof(*st));
  if (st == NULL) { return NULL; }
  /* 必须清零：计时字段（t_conv/t_attn/t_dec）与新增字段全靠这里初始化。
   * 漏掉会读到堆里的垃圾（r197 实测卷积计时打印出 -2.58e94）。 */
  memset(st, 0, sizeof(*st));

  st->halo = halo_samples;
  st->cap  = win_frames * HOP_LENGTH + halo_samples + JX_PKT_SAMPLES;
  /* 缓冲**一开始就按全长算**，前面补零代表「语音还没开始」。
   * 不能从 0 开始累积：模型的最小输入长度由 first_block 的感受野决定
   * （5 条 dilated 卷积，k=7/pad=1 在 T=4 帧时输出长度为 4+2-7+1=0，
   * 零长度张量会把后续流程搞死 —— r192 实测第一包就卡在这）。 */
  st->have = st->cap;
  st->pkts = 0;

  st->buf = (int16_t *)malloc(sizeof(int16_t) * (size_t)st->cap);
  if (st->buf == NULL) { free(st); return NULL; }
  memset(st->buf, 0, sizeof(int16_t) * (size_t)st->cap);

  st->win  = win_frames;
  st->feat = (float *)malloc(sizeof(float) * (size_t)FEATURE_DIM * (size_t)win_frames);
  st->tokr = (int64_t *)malloc(sizeof(int64_t) * (size_t)win_frames);
  if (st->feat == NULL || st->tokr == NULL)
    {
      free(st->buf); free(st->feat); free(st->tokr); free(st);
      return NULL;
    }
  memset(st->feat, 0, sizeof(float) * (size_t)FEATURE_DIM * (size_t)win_frames);
  memset(st->tokr, 0, sizeof(int64_t) * (size_t)win_frames);

  /* 标准 3k/6k 的编码注意力只有一个 LocalTrans 层，可以直接做跨包 K/V
   * 缓存。缓存分配失败或模型是压缩路径时自动回退到整圈重算。 */
  if (jx_stream_enc_cache_init(st, win_frames) != 0)
    {
      printf("[JX] 编码 K/V 缓存已启用: %d 帧, inner=%d, heads=%d\n",
             st->ec_cap, st->ec_inner, st->ec_heads);
    }

  return st;
}

void jx_stream_close(struct jx_stream_s *st)
{
  if (st == NULL) { return; }
  jx_afree(st->ec_k);
  jx_afree(st->ec_v);
  free(st->ec_bias);
  free(st->feat);
  free(st->tokr);
  free(st->buf);
  free(st);
}

int jx_stream_pkts(const struct jx_stream_s *st)
{
  return (st != NULL) ? st->pkts : -1;
}

/* 分段累计耗时（毫秒）：卷积 / 注意力 / 解码 */
void jx_stream_times(const struct jx_stream_s *st,
                     double *conv_ms, double *attn_ms, double *dec_ms,
                     double *deq_ms, double *endec_ms, double *decnn_ms)
{
  if (st == NULL) { return; }
  if (conv_ms != NULL) { *conv_ms = st->t_conv; }
  if (attn_ms != NULL) { *attn_ms = st->t_attn; }
  if (dec_ms  != NULL) { *dec_ms  = st->t_dec;  }
  if (deq_ms  != NULL) { *deq_ms  = st->t_deq;  }
  if (endec_ms != NULL) { *endec_ms = st->t_endec; }
  if (decnn_ms != NULL) { *decnn_ms = st->t_decnn; }
}

int jx_stream_feed(struct jx_stream_s *st, const int16_t *pkt,
                   int64_t *tok_out, float *pcm_out)
{
  if (st == NULL || pkt == NULL || tok_out == NULL || pcm_out == NULL)
    {
      return -1;
    }

  /* 1) 固定长度滚动：整体左移一个包，新包追加到尾部。
   *    左移量恒为 JX_PKT_SAMPLES，而 100 包 = 400 帧 = 恰好一个注意力窗，
   *    所以缓冲起点始终落在窗格的整数倍上 —— 注意力用局部帧号算掩码就
   *    等价于用绝对帧号（这就是「相对区间」）。 */
  memmove(st->buf, &st->buf[JX_PKT_SAMPLES],
          sizeof(int16_t) * (size_t)(st->cap - JX_PKT_SAMPLES));
  memcpy(&st->buf[st->cap - JX_PKT_SAMPLES], pkt,
         sizeof(int16_t) * JX_PKT_SAMPLES);

  /* 2) 卷积链：只喂「本包 + 左右各 5 帧 halo」这一小段。
   *    这是解内存墙的关键 —— 中间张量从 [1,16,39296]=2.51MB 降到
   *    [1,16,1344]=84KB，省 30 倍。 */
  {
    const int halo = jx_conv_halo(st);
    const int n    = JX_PKT_SAMPLES + 2 * halo * HOP_LENGTH;
    const int fr   = n / HOP_LENGTH;
    const int base = fr - halo - JX_PKT_FRAMES;
    const int16_t *w = &st->buf[st->cap - n];
    T xin = t_alloc(3, (int[]){1, 1, n});
    T enc;
    double t_conv0 = jx_now_ms();

    if (xin.d == NULL) { return -4; }
    for (int i = 0; i < n; i++) { xin.d[i] = (float)w[i] / 32768.0f; }

    encoder_forward(&xin, &enc);
    t_free(&xin);
    if (enc.d == NULL || enc.len < FEATURE_DIM * fr) { t_free(&enc); return -5; }

    memmove(st->feat, &st->feat[(size_t)FEATURE_DIM * JX_PKT_FRAMES],
            sizeof(float) * (size_t)FEATURE_DIM * (size_t)(st->win - JX_PKT_FRAMES));
    for (int c = 0; c < FEATURE_DIM; c++)
      {
        for (int i = 0; i < JX_PKT_FRAMES; i++)
          {
            st->feat[(size_t)c * st->win + st->win - JX_PKT_FRAMES + i] =
                enc.d[(size_t)c * fr + base + i];
          }
      }
    t_free(&enc);
    st->t_conv += jx_now_ms() - t_conv0;
  }

  /* 3) 编码侧注意力。缓存命中时只处理新增 4 帧；否则保留原来的整圈重算路径。 */
  if (st->ec_on)
    {
      T enc_out;
      T enq_new = t_alloc(3, (int[]){1, JX_PKT_FRAMES, FEATURE_DIM});
      int64_t idx_new[JX_PKT_FRAMES];
      double t_attn0 = jx_now_ms();

      if (enq_new.d == NULL) { return -6; }
      if (jx_stream_enc_cache_step(st, &enc_out) != 0)
        {
          t_free(&enq_new);
          return -7;
        }

      quantizer_forward(&enc_out, &enq_new, idx_new, JX_PKT_FRAMES);
      memmove(st->tokr, &st->tokr[JX_PKT_FRAMES],
              sizeof(int64_t) * (size_t)(st->win - JX_PKT_FRAMES));
      for (int i = 0; i < JX_PKT_FRAMES; i++)
        {
          st->tokr[st->win - JX_PKT_FRAMES + i] = idx_new[i];
          tok_out[i] = idx_new[i];
        }

      t_free(&enc_out);
      t_free(&enq_new);
      st->t_attn += jx_now_ms() - t_attn0;
    }
  else
    {
    T ring = t_alloc(3, (int[]){1, FEATURE_DIM, st->win});
    T enenc;
    T enq;
    int64_t *idx = NULL;
    double t_attn0 = jx_now_ms();

    if (ring.d == NULL) { return -6; }
    memcpy(ring.d, st->feat,
           sizeof(float) * (size_t)FEATURE_DIM * (size_t)st->win);

    en_encoder_forward(&ring, &enenc);
    t_free(&ring);
    if (enenc.d == NULL) { t_free(&enenc); return -7; }

    enq = t_alloc(3, (int[]){1, st->win, FEATURE_DIM});
    idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)st->win);
    if (enq.d == NULL || idx == NULL)
      { t_free(&enenc); t_free(&enq); free(idx); return -8; }

    quantizer_forward(&enenc, &enq, idx, st->win);

    memmove(st->tokr, &st->tokr[JX_PKT_FRAMES],
            sizeof(int64_t) * (size_t)(st->win - JX_PKT_FRAMES));
    for (int i = 0; i < JX_PKT_FRAMES; i++)
      {
        st->tokr[st->win - JX_PKT_FRAMES + i] =
            idx[st->win - JX_PKT_FRAMES + i];
        tok_out[i] = st->tokr[st->win - JX_PKT_FRAMES + i];
      }

    t_free(&enenc);
    t_free(&enq);
    free(idx);
    st->t_attn += jx_now_ms() - t_attn0;
    }

  /* 4) 解码：token 环 -> 特征 -> 解码器，取末尾（halo 之前）一个包 */
  /* single-packet skipdec: measure encoder-only realtime cost */
  { const char *skipdec = getenv("JX_SKIP_DEC");
    if (skipdec != NULL && skipdec[0] == '1')
      { for (int i = 0; i < JX_PKT_SAMPLES; i++) pcm_out[i] = 0.0f;
        st->pkts++; return 0; } }
  {
    const int dw = jx_decwin_frames(st->win);
    const int64_t *tail = &st->tokr[st->win - dw];
    T enq2 = t_alloc(3, (int[]){1, dw, FEATURE_DIM});
    T endec;
    T dec;
    int off;
    double t_dec0 = jx_now_ms();
    double t_deq0 = t_dec0;

    if (enq2.d == NULL) { return -9; }
    quantizer_to_features(tail, &enq2, dw);

    st->t_deq += jx_now_ms() - t_deq0;
    double t_endec0 = jx_now_ms();

    en_decoder_forward(&enq2, &endec);
    t_free(&enq2);
    if (endec.d == NULL) { t_free(&endec); return -10; }

    st->t_endec += jx_now_ms() - t_endec0;
    double t_decnn0 = jx_now_ms();

    memset(&dec, 0, sizeof(dec));
    decoder_forward(&endec, &dec);
    t_free(&endec);
    if (dec.d == NULL) { t_free(&dec); return -11; }

    st->t_decnn += jx_now_ms() - t_decnn0;

    off = dec.len - st->halo - JX_PKT_SAMPLES;
    if (off < 0) { off = 0; }

    for (int i = 0; i < JX_PKT_SAMPLES; i++)
      {
        int si = off + i;
        pcm_out[i] = (si < dec.len) ? dec.d[si] : 0.0f;
      }
    t_free(&dec);
    st->t_dec += jx_now_ms() - t_dec0;
  }

  st->pkts++;
  return 0;
}

/* 批量推理入口：一次吃 npkt 个 24ms 网络包，只跑一轮编解码。
 * 这样卷积 halo、量化 prologue/epilogue 和解码器固定开销会被多个包摊薄；
 * 返回的 tok_out/pcm_out 长度分别为 npkt*JX_PKT_FRAMES / npkt*JX_PKT_SAMPLES。 */

int jx_stream_feed_batch(struct jx_stream_s *st, const int16_t *pkt, int npkt,
                         int64_t *tok_out, float *pcm_out)
{
  const int nframes = npkt * JX_PKT_FRAMES;
  const int n_smp   = npkt * JX_PKT_SAMPLES;

  if (st == NULL || pkt == NULL || tok_out == NULL || pcm_out == NULL || npkt < 1)
    {
      return -1;
    }
  if (nframes > st->win)
    {
      return -2;
    }

  memmove(st->buf, &st->buf[n_smp],
          sizeof(int16_t) * (size_t)(st->cap - n_smp));
  memcpy(&st->buf[st->cap - n_smp], pkt,
         sizeof(int16_t) * (size_t)n_smp);

  {
    const int halo = jx_conv_halo(st);
    const int n    = n_smp + 2 * halo * HOP_LENGTH;
    const int fr   = n / HOP_LENGTH;
    const int base = fr - halo - nframes;
    const int16_t *w = &st->buf[st->cap - n];
    T xin = t_alloc(3, (int[]){1, 1, n});
    T enc;
    double t_conv0 = jx_now_ms();

    if (xin.d == NULL) { return -4; }
    for (int i = 0; i < n; i++) { xin.d[i] = (float)w[i] / 32768.0f; }
    encoder_forward(&xin, &enc);
    t_free(&xin);
    if (enc.d == NULL || enc.len < FEATURE_DIM * fr) { t_free(&enc); return -5; }

    memmove(st->feat, &st->feat[(size_t)FEATURE_DIM * nframes],
            sizeof(float) * (size_t)FEATURE_DIM * (size_t)(st->win - nframes));
    for (int c = 0; c < FEATURE_DIM; c++)
      {
        for (int i = 0; i < nframes; i++)
          {
            st->feat[(size_t)c * st->win + st->win - nframes + i] =
                enc.d[(size_t)c * fr + base + i];
          }
      }
    t_free(&enc);
    st->t_conv += jx_now_ms() - t_conv0;
  }

  if (st->ec_on)
    {
      T enc_out;
      T enq_new = t_alloc(3, (int[]){1, nframes, FEATURE_DIM});
      int64_t *idx_new = (int64_t *)malloc(sizeof(int64_t) * (size_t)nframes);
      double t_attn0 = jx_now_ms();

      if (enq_new.d == NULL || idx_new == NULL)
        {
          t_free(&enq_new); free(idx_new);
          return -6;
        }
      if (jx_stream_enc_cache_step_n(st, nframes, &enc_out) != 0)
        {
          t_free(&enq_new); free(idx_new);
          return -7;
        }

      quantizer_forward(&enc_out, &enq_new, idx_new, nframes);
      memmove(st->tokr, &st->tokr[nframes],
              sizeof(int64_t) * (size_t)(st->win - nframes));
      for (int i = 0; i < nframes; i++)
        {
          st->tokr[st->win - nframes + i] = idx_new[i];
          tok_out[i] = idx_new[i];
        }
      t_free(&enc_out); t_free(&enq_new); free(idx_new);
      st->t_attn += jx_now_ms() - t_attn0;
    }
  else
    {
      T ring = t_alloc(3, (int[]){1, FEATURE_DIM, st->win});
      T enenc, enq;
      int64_t *idx = NULL;
      double t_attn0 = jx_now_ms();

      if (ring.d == NULL) { return -6; }
      memcpy(ring.d, st->feat,
             sizeof(float) * (size_t)FEATURE_DIM * (size_t)st->win);
      en_encoder_forward(&ring, &enenc);
      t_free(&ring);
      if (enenc.d == NULL) { t_free(&enenc); return -7; }
      enq = t_alloc(3, (int[]){1, st->win, FEATURE_DIM});
      idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)st->win);
      if (enq.d == NULL || idx == NULL)
        { t_free(&enenc); t_free(&enq); free(idx); return -8; }
      quantizer_forward(&enenc, &enq, idx, st->win);
      memmove(st->tokr, &st->tokr[nframes],
              sizeof(int64_t) * (size_t)(st->win - nframes));
      for (int i = 0; i < nframes; i++)
        {
          st->tokr[st->win - nframes + i] = idx[st->win - nframes + i];
          tok_out[i] = st->tokr[st->win - nframes + i];
        }
      t_free(&enenc); t_free(&enq); free(idx);
      st->t_attn += jx_now_ms() - t_attn0;
    }

  {
    const char *skipdec = getenv("JX_SKIP_DEC");

    if (skipdec != NULL && skipdec[0] == (char)49)
      {
        for (int i = 0; i < n_smp; i++) { pcm_out[i] = 0.0f; }
        st->pkts += npkt;
        return 0;
      }
  }

  {
    const char *skip = getenv("JX_SKIP_DEC");

    if (skip != NULL && skip[0] == '1')
      {
        st->pkts += npkt;
        return 0;
      }
  }

  {
    const char *dwe = getenv("JX_BATCH_DECWIN");
    int dw = (dwe != NULL) ? atoi(dwe) : 64;
    if (dw < nframes) { dw = nframes; }
    if (dw > st->win) { dw = st->win; }
    const int64_t *tail = &st->tokr[st->win - dw];
    T enq2 = t_alloc(3, (int[]){1, dw, FEATURE_DIM});
    T endec, dec;
    int off;
    double t_dec0 = jx_now_ms();
    double t_deq0 = t_dec0;

    if (enq2.d == NULL) { return -9; }
    quantizer_to_features(tail, &enq2, dw);
    st->t_deq += jx_now_ms() - t_deq0;
    double t_endec0 = jx_now_ms();
    en_decoder_forward(&enq2, &endec);
    t_free(&enq2);
    if (endec.d == NULL) { t_free(&endec); return -10; }
    st->t_endec += jx_now_ms() - t_endec0;
    double t_decnn0 = jx_now_ms();
    memset(&dec, 0, sizeof(dec));
    decoder_forward(&endec, &dec);
    t_free(&endec);
    if (dec.d == NULL) { t_free(&dec); return -11; }
    st->t_decnn += jx_now_ms() - t_decnn0;

    off = dec.len - st->halo - n_smp;
    if (off < 0) { off = 0; }
    for (int i = 0; i < n_smp; i++)
      {
        int si = off + i;
        pcm_out[i] = (si < dec.len) ? dec.d[si] : 0.0f;
      }
    t_free(&dec);
    st->t_dec += jx_now_ms() - t_dec0;
  }

  st->pkts += npkt;
  return 0;
}
