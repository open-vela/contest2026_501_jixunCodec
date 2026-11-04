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
#include "stream_codec.h"

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

int jx_codec_rate_id(void)
{
  const char *r = jx_rate_name();
  if (r[0] == '1') { return 0; }
  if (r[0] == '6') { return 2; }
  return 1;
}

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
  SEncState *enc;
  SDecState *dec;
  int pkts;
  double conv_ms, attn_ms, dec_ms;
};

/* The tap executes synchronously in this encoder's worker. No shared owner. */
static void a_timing_tap(void *ud, const char *name, const float *d,
                         int C, int Tn, int last)
{
  double *v=(double *)ud;
  double now=jx_now_ms();
  if (!strcmp(name,"en_encoder")) v[2]+=now-v[0];
  else if (!strncmp(name,"encoder.blocks.",15)) v[1]+=now-v[0];
  v[0]=now;
  (void)d;(void)C;(void)Tn;(void)last;
}

struct jx_stream_s *jx_stream_open(int halo_samples)
{
  struct jx_stream_s *st=calloc(1,sizeof(*st));
  if (!st) return NULL;
  st->enc=senc_new();
  if (!st->enc) {free(st);return NULL;}
  /* TX requests skip-decoder. Never allocate an unused2.8MB decoder arena. */
  const char *skip=getenv("JX_SKIP_DEC");
  if (!skip || skip[0]!='1') st->dec=sdec_new();
  printf("[A32-opt] causal encoder ready hidden=32 window=128 decoder_depth=1 internal_tokens=16/32 external=first4/max8\n");
  (void)halo_samples;
  return st;
}

void jx_stream_close(struct jx_stream_s *st)
{
  if (!st) return;
  printf("[A] encoder_arena_peak_bytes=%u\n",(unsigned)(senc_peak(st->enc)*sizeof(float)));
  senc_free(st->enc);
  if (st->dec) sdec_free(st->dec);
  free(st);
}

int jx_stream_pkts(const struct jx_stream_s *st) {return st?st->pkts:-1;}

void jx_stream_times(const struct jx_stream_s *st, double *conv_ms,
                     double *attn_ms,double *dec_ms,double *deq_ms,
                     double *endec_ms,double *decnn_ms)
{
  if (!st) return;
  if(conv_ms)*conv_ms=st->conv_ms;
  if(attn_ms)*attn_ms=st->attn_ms;
  if(dec_ms)*dec_ms=st->dec_ms;
  if(deq_ms)*deq_ms=0;
  if(endec_ms)*endec_ms=0;
  if(decnn_ms)*decnn_ms=st->dec_ms;
}

int jx_stream_feed_batch(struct jx_stream_s *st,const int16_t *pkt,int npkt,
                         int64_t *tokens,float *pcm)
{
  if(!st||!pkt||!tokens||!pcm||npkt<1||npkt>32)return -1;
  float input[32*96];
  const char *step_env=getenv("JX_A_STEP"),*reuse_env=getenv("JX_A_REUSE");
  int step=(step_env&&atoi(step_env)==32)?32:16;
  if(!reuse_env||reuse_env[0]=='0')step=16; /* default reuse is off: require explicit opt-in for32 */
  const int n=npkt*JX_PKT_SAMPLES;
  for(int off=0;off<n;){
    int take=n-off;if(take>step*96)take=step*96;
    for(int i=0;i<take;i++)input[i]=(float)pkt[off+i]/32768.0f;
    double v[3]={jx_now_ms(),0,0};
    STapCb oldtap=g_stap;void *oldud=g_stap_ud;
    g_stap=a_timing_tap;g_stap_ud=v;
    int got=senc_step(st->enc,input,take,tokens+off/HOP_LENGTH);
    g_stap=oldtap;g_stap_ud=oldud;
    st->conv_ms+=v[1];st->attn_ms+=v[2];
    if(got!=take/HOP_LENGTH)return -2;
    if(st->dec){
      double t=jx_now_ms();
      if(sdec_step(st->dec,tokens+off/HOP_LENGTH,got,pcm+off)!=take)return -3;
      st->dec_ms+=jx_now_ms()-t;
    }else memset(pcm+off,0,(size_t)take*sizeof(float));
    off+=take;
  }
  st->pkts+=npkt;
  return 0;
}

int jx_stream_feed(struct jx_stream_s *st,const int16_t *pkt,int64_t *tok,float *pcm)
{return jx_stream_feed_batch(st,pkt,1,tok,pcm);}
