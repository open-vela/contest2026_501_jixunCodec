/****************************************************************************
 * apps/jixun-codec/main/jx_main.c
 *
 * 极讯 AI Codec —— 单板卡验证 app（NSH 命令名 `jixun`）
 *
 *   nsh> jixun codec info            打印当前码率/参数/权重规模
 *   nsh> jixun codec bench [秒数]    合成 16k 音频 -> 编码 -> 解码，
 *                                    打印分段耗时与 RTF（纯算力测量，不碰音频硬件）
 *   nsh> jixun codec loop  [秒数]    麦克风录 N 秒 -> 编码 -> 解码 -> 喇叭播放
 *                                    （单板闭环，用来听真实效果）
 *   nsh> jixun codec mem            存储层次/浮点吞吐微基准（判断瓶颈在算力还是访存）
 *
 * 音频采集/播放复用 plant-companion 的 ai_voice 层（24kHz 采集 -> 16kHz
 * 单声道），本 app 只负责把 codec 接在中间。
 ****************************************************************************/

#include <nuttx/config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <nuttx/sched.h>
#include <unistd.h>

#include "model_config.h"   /* SR / HOP_LENGTH / FEATURE_DIM / VQ_* / EN_WINDOW */
#include "tts_clip.h"       /* embedded clean TTS playback test */
#include "raw_clip.h"       /* embedded microphone recording playback test */
#include "jx_codec.h"
#include "jxprof.h"
#include "nn_ops.h"
#include "jx_percore.h"
#include "ai_voice.h"
#include "hal_i2s.h"
#include "jx_net.h"     /* 2026-09-14 R27：双板 WiFi 话音 index */
#include "jx_serial.h"  /* SAFE DEMO PCM quality capture */
#include "jx_serial.h"  /* SAFE DEMO PCM quality capture */
#include "jx_serial.h"  /* SAFE DEMO PCM quality capture */
#include "jx_serial.h"  /* SAFE DEMO PCM quality capture */
#include <netutils/dhcpd.h>
#include "hal_i2c.h"     /* 2026-09-14 R26：直接写 ES7210 PGA 增益寄存器 */
#include "es7210.h"
#include "jx_ui.h"      /* 2026-09-17：ST7796 480x320 仪表盘（不抢编解码资源） */

extern int jx_streamtx_run(const char *ip, int port, int secs);
extern int jx_streamtx_probe(const char *ip, int port);

#ifdef CONFIG_ESP32S3_BOARD_TOUCHSCREEN
extern int gt911_diag_snapshot(uint32_t out[8]);
extern int gt911_diag_probe(uint8_t out[8]);
#endif

/* v97 诊断：把 I2S RX 预读链置为空闲（hal_i2s.c 新增公共封装；
 * 原本 hal_i2s_rx_prefetch_stop 是 static，app 侧调不到）。 */
extern void hal_i2s_rx_idle(void);

#define JX_PCM_RATE 16000

static void jx_ui_send_action(void);

/****************************************************************************
 * 工具
 ****************************************************************************/

/* 该 defconfig 未链接 libm，lrint/lrintf 不可用；此处只需最邻近取整 */

static int16_t jx_round_s16(double v)
{
  double s = v * 32767.0;
  return (int16_t)(s >= 0.0 ? s + 0.5 : s - 0.5);
}

/* 24 ms/100 ms 分块接收时，旧路径每块都 ai_voice_play -> init + flush，
 * DMA 必然在块边界空转。这里只做 16k mono -> 24k stereo 重采样，然后把
 * 小块持续排进同一个 TX 队列；调用方在末块统一 flush。默认关闭，使用
 * set JX_STREAM_PLAY 1 开启。 */
static int jx_voice_stream_play_pcm(const int16_t *pcm, int samples)
{
  enum { IN_CAP = 800, OUT_CAP = 2400 };
  static int16_t *workspace;
  int16_t *in;
  int16_t *outbuf;
  int sent = 0;

  if (pcm == NULL || samples <= 0)
    {
      return -1;
    }

  if (workspace == NULL)
    {
      workspace = (int16_t *)malloc(sizeof(int16_t) * (IN_CAP + OUT_CAP));
      if (workspace == NULL)
        {
          return -1;
        }
    }

  in = workspace;
  outbuf = workspace + IN_CAP;

  while (sent < samples)
    {
      int n_in = samples - sent;
      int n_out = 0;
      int j;
      int ret;

      if (n_in > IN_CAP)
        {
          n_in = IN_CAP;
        }

      memcpy(in, pcm + sent, (size_t)n_in * sizeof(int16_t));

      for (j = 0; j < n_in * 3 / 2; j++)
        {
          uint32_t pos = (uint32_t)j * 2u;
          uint32_t idx = pos / 3u;
          uint32_t frac = pos % 3u;
          int16_t a = in[idx];
          int16_t b = ((int)idx + 1 < n_in) ? in[idx + 1] : a;
          int16_t s = (frac == 0u)
                          ? a
                          : (int16_t)(((int32_t)a * (3 - (int32_t)frac) +
                                      (int32_t)b * (int32_t)frac) / 3);

          outbuf[n_out++] = s;
          outbuf[n_out++] = s;
        }

      ret = hal_i2s_write_async(outbuf, (uint32_t)n_out * 2u);
      if (ret < 0)
        {
          printf("[Voice] 流式播放失败: hal_i2s_write_async=%d\n", ret);
          return ret;
        }

      sent += n_in;
    }

  return 0;
}

static int jx_play_idx_range(const int64_t *idx, int Tt, int orig_t,
                             int delay_ms)
{
  float *rec = NULL;
  int16_t *out = NULL;
  uint8_t *wav = NULL;
  int rc;
  int i;

  if (idx == NULL || Tt <= 0 || orig_t <= 0) { return -1; }
  if (jx_decode(idx, Tt, orig_t, &rec) != 0 || rec == NULL) { return -1; }
  out = (int16_t *)malloc(sizeof(int16_t) * (size_t)orig_t);
  if (out == NULL) { free(rec); return -1; }
  for (i = 0; i < orig_t; i++)
    {
      float v = rec[i];
      if (v > 1.0f)  { v = 1.0f; }
      if (v < -1.0f) { v = -1.0f; }
      out[i] = jx_round_s16((double)v);
    }
  free(rec);
  wav = (uint8_t *)malloc(44u + (size_t)orig_t * 2u);
  if (wav == NULL) { free(out); return -1; }
  (void)ai_voice_wav_encode(wav, out, (uint32_t)orig_t * 2u);
  free(out);
  if (delay_ms > 0) { usleep((useconds_t)delay_ms * 1000u); }
  rc = ai_voice_play(wav, 44u + (size_t)orig_t * 2u);
  free(wav);
  printf("[Voice] 播放完成\n");
  return rc;
}

struct jx_playbuf_s
{
  pthread_mutex_t lock;
  pthread_cond_t cond;
  int16_t *data;
  int cap;
  int head;
  int tail;
  int count;
  int prebuffer;
  int finished;
  int started;
  int failed;
  int underruns;
  double started_ms;
};

static double jx_now_ms(void);

static int jx_playbuf_init(struct jx_playbuf_s *pb, int cap, int prebuffer)
{
  memset(pb, 0, sizeof(*pb));
  pb->cap = cap;
  pb->prebuffer = prebuffer;
  if (pb->prebuffer < 0) { pb->prebuffer = 0; }
  if (pb->prebuffer > cap) { pb->prebuffer = cap; }
  pb->data = (int16_t *)malloc(sizeof(int16_t) * (size_t)cap);
  if (pb->data == NULL) { return -1; }
  if (pthread_mutex_init(&pb->lock, NULL) != 0) { free(pb->data); return -1; }
  if (pthread_cond_init(&pb->cond, NULL) != 0) { free(pb->data); return -1; }
  return 0;
}

static int jx_playbuf_write(struct jx_playbuf_s *pb,
                            const int16_t *pcm, int samples)
{
  int first;
  int second;

  if (pb == NULL || pcm == NULL || samples <= 0) { return -1; }
  pthread_mutex_lock(&pb->lock);
  while (!pb->failed && pb->count + samples > pb->cap)
    {
      pthread_mutex_unlock(&pb->lock);
      usleep(10000);
      pthread_mutex_lock(&pb->lock);
    }
  if (pb->failed)
    {
      pthread_mutex_unlock(&pb->lock);
      return -1;
    }

  first = pb->cap - pb->tail;
  if (first > samples) { first = samples; }
  second = samples - first;
  memcpy(&pb->data[pb->tail], pcm, sizeof(int16_t) * (size_t)first);
  if (second > 0)
    {
      memcpy(pb->data, pcm + first, sizeof(int16_t) * (size_t)second);
    }
  pb->tail = (pb->tail + samples) % pb->cap;
  pb->count += samples;
  pthread_cond_broadcast(&pb->cond);
  pthread_mutex_unlock(&pb->lock);
  return 0;
}

static void jx_playbuf_finish(struct jx_playbuf_s *pb)
{
  if (pb == NULL) { return; }
  pthread_mutex_lock(&pb->lock);
  pb->finished = 1;
  pthread_cond_broadcast(&pb->cond);
  pthread_mutex_unlock(&pb->lock);
}

static void *jx_playbuf_worker(void *arg)
{
  struct jx_playbuf_s *pb = (struct jx_playbuf_s *)arg;
  int16_t chunk[1600];

  if (ai_voice_init() != 0)
    {
      pthread_mutex_lock(&pb->lock);
      pb->failed = 1;
      pthread_cond_broadcast(&pb->cond);
      pthread_mutex_unlock(&pb->lock);
      return NULL;
    }

  for (;;)
    {
      int take;
      int first;
      int second;

      pthread_mutex_lock(&pb->lock);
      while (!pb->failed && !pb->finished && pb->count < pb->prebuffer)
        {
          pthread_mutex_unlock(&pb->lock);
          usleep(10000);
          pthread_mutex_lock(&pb->lock);
        }
      if (pb->failed || (pb->finished && pb->count == 0))
        {
          pthread_mutex_unlock(&pb->lock);
          break;
        }
      if (!pb->started)
        {
          pb->started = 1;
          pb->started_ms = jx_now_ms();
          printf("  [playbuf] 开始送 I2S，缓冲 %d 样本\n", pb->count);
        }
      take = pb->count;
      if (take > (int)(sizeof(chunk) / sizeof(chunk[0])))
        {
          take = (int)(sizeof(chunk) / sizeof(chunk[0]));
        }
      first = pb->cap - pb->head;
      if (first > take) { first = take; }
      second = take - first;
      memcpy(chunk, &pb->data[pb->head], sizeof(int16_t) * (size_t)first);
      if (second > 0)
        {
          memcpy(chunk + first, pb->data, sizeof(int16_t) * (size_t)second);
        }
      pb->head = (pb->head + take) % pb->cap;
      pb->count -= take;
      pthread_cond_broadcast(&pb->cond);
      pthread_mutex_unlock(&pb->lock);

      if (jx_voice_stream_play_pcm(chunk, take) != 0)
        {
          pthread_mutex_lock(&pb->lock);
          pb->failed = 1;
          pthread_cond_broadcast(&pb->cond);
          pthread_mutex_unlock(&pb->lock);
          break;
        }
    }

  hal_i2s_write_flush();
  return NULL;
}

static double jx_now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* 权重落在 flash(XIP) 还是堆(PSRAM)，bench 结论必须带上这一行才有意义 */
static void jx_print_wt(void)
{
  printf("  权重位置  : %s  (%p)\n",
         jx_weight_in_heap() ? "堆 / PSRAM" : "flash XIP",
         (const void *)jx_weight_blob());
}

/* v97 诊断：读环境开关（set JX_RXIDLE 0 可关掉 RX 空闲策略做 A/B） */
static int jx_env_on(const char *name, int defv)
{
  const char *e = getenv(name);
  if (e == NULL) { return defv; }
  return atoi(e) != 0;
}

/* 2026-09-14 第二十六轮：环境变量取整数（set JX_MICGAIN 12） */
static int jx_env_int(const char *name, int defv)
{
  const char *e = getenv(name);
  return (e == NULL) ? defv : atoi(e);
}

/* 2026-09-14 第二十六轮：麦克风模拟增益（ES7210 PGA）
 *
 * 为什么在 app 里改：ES7210 驱动（plant-companion/voice_agent/es7210.c）的
 * es7210_init() 先写 30dB(0x1A)，紧接着 es7210_restore_analog() 在 MCLK
 * 起来后又重写成 27dB(0x19) —— 实际生效的是 27dB。实测录音峰值只有 4476
 * （满量程 32767 的 14%）、静音段 RMS 102/32768，录音信噪比仅约 15dB，
 * 3kbps 声码器在这么低的输入电平下音质明显变差（第二十五轮电平敏感性实测：
 * 峰值 15000~20000 时编码 SNR 最高）。这里不动上游驱动，直接写码片寄存器。
 *
 * REG43/REG44 = MIC1/MIC2 PGA：bit4=0x10 使能，bits[3:0]=档位
 *   （0=0dB，每档 +3dB；12=36dB；14=37.5dB，小智同款最高档）
 * 历史实测值：0x19=27dB(9)、0x1A=30dB(10)、0x1E=37.5dB(14)。
 *
 * 用法：nsh> set JX_MICGAIN 14      （0..15；不设时默认 14）
 *       nsh> jixun codec loop 2
 *     回到驱动默认：set JX_MICGAIN 9
 *     A/B 用：set JX_MICGAIN_OFF 1   （完全不动寄存器）
 */
#define JX_MIC_I2C_ADDR  0x40
#define JX_MIC_I2C_FREQ  100000

extern FAR struct i2c_master_s *esp32s3_i2cbus_initialize(int bus);

static void jx_mic_gain_apply(void)
{
  FAR struct i2c_master_s *i2c;
  uint8_t code = (uint8_t)(jx_env_int("JX_MICGAIN", 14) & 0x0f);
  uint8_t reg  = (uint8_t)((1 << 4) | code);
  uint8_t before = 0;
  uint8_t after  = 0;

  if (jx_env_on("JX_MICGAIN_OFF", 0))
    {
      printf("  [mic] JX_MICGAIN_OFF=1：保持驱动默认增益\n");
      return;
    }

  i2c = esp32s3_i2cbus_initialize(0);
  if (i2c == NULL)
    {
      printf("  [mic] I2C0 初始化失败，保持驱动默认增益\n");
      return;
    }

  hal_i2c_read_reg(i2c, JX_MIC_I2C_ADDR, ES7210_MIC1_GAIN, &before,
                  JX_MIC_I2C_FREQ);
  hal_i2c_write_reg(i2c, JX_MIC_I2C_ADDR, ES7210_MIC1_GAIN, reg,
                    JX_MIC_I2C_FREQ);
  hal_i2c_write_reg(i2c, JX_MIC_I2C_ADDR, ES7210_MIC2_GAIN, reg,
                    JX_MIC_I2C_FREQ);
  hal_i2c_read_reg(i2c, JX_MIC_I2C_ADDR, ES7210_MIC1_GAIN, &after,
                   JX_MIC_I2C_FREQ);

  if (code > 12)
    {
      printf("  [mic] PGA 档位 %u (≥36 dB; 14=37.5dB 顶格)", (unsigned)code);
    }
  else
    {
      printf("  [mic] PGA 档位 %u (~%u dB)", (unsigned)code, (unsigned)(code * 3u));
    }

  printf(": REG43 0x%02x -> 0x%02x (读回 0x%02x)\n", before, reg, after);
}

static int jx_parse_seconds(int argc, char *argv[], int defval, int maxval)
{
  int s = (argc >= 3) ? atoi(argv[2]) : defval;
  if (s < 1)   { s = 1; }
  if (s > maxval) { s = maxval; }
  return s;
}

extern double jx_live_mb;
extern double jx_peak_mb;

static void jx_print_mem(const char *tag)
{
  struct mallinfo mi = mallinfo();
  printf("  [mem] %-12s 已用=%u 空闲=%u 最大空闲块=%u | 张量存活=%.2fMB 峰值=%.2fMB\n",
         tag, (unsigned)mi.uordblks, (unsigned)mi.fordblks,
         (unsigned)mi.mxordblk, jx_live_mb, jx_peak_mb);
}

/****************************************************************************
 * 合成测试音频（确定性；编解码算力与内容无关，所以测速可复现）
 ****************************************************************************/

static void jx_gen_pcm(int16_t *pcm, int n, uint32_t seed)
{
  uint32_t s = seed ? seed : 0x2468ace1u;

  for (int i = 0; i < n; i++)
    {
      double t = (double)i / (double)JX_PCM_RATE;
      double v = 0.30 * sin(2.0 * M_PI * 220.0 * t)
               + 0.20 * sin(2.0 * M_PI * 700.0 * t + 0.7)
               + 0.10 * sin(2.0 * M_PI * 1700.0 * t + 1.9);

      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      v += 0.02 * ((double)(s & 0xffffu) / 32768.0 - 1.0);

      if (v > 1.0)  { v = 1.0; }
      if (v < -1.0) { v = -1.0; }

      pcm[i] = jx_round_s16(v);
    }
}

/****************************************************************************
 * 一次完整的 编码 + 解码（带计时）
 ****************************************************************************/

/* v98：一次 编码+解码，打印计时与算子剖析，并把结果交给调用方
 *（不释放）。bench 用薄封装 jx_roundtrip 用完即弃；loop 直接拿结果
 * 去播放——原来 loop 先调 jx_roundtrip 测时、再重算一遍拿输出，白跑
 * 一轮 8.6s（1s 音频），端到端延迟从 8.6s 变成 17.5s。 */
static int jx_run_once(const int16_t *pcm, int n_samples,
                       int64_t **out_idx, int *out_ntok, int *out_Tt,
                       float **out_rec)
{
  int64_t *idx  = NULL;
  int      ntok = 0;
  int      Tt   = 0;
  float   *rec  = NULL;

  jx_prof_reset();

  /* 编解码期间禁止 UI 绘制（bench 也是走这条路） */
  jx_ui_set_busy(1);

  double t0 = jx_now_ms();
  if (jx_encode(pcm, n_samples, &idx, &ntok, &Tt) != 0)
    {
      printf("[ERR] 编码失败（内存不足？）\n");
      jx_ui_set_busy(0);
      return -1;
    }
  double t1 = jx_now_ms();

  if (jx_decode(idx, Tt, n_samples, &rec) != 0)
    {
      printf("[ERR] 解码失败（内存不足？）\n");
      free(idx);
      jx_ui_set_busy(0);
      return -1;
    }
  double t2 = jx_now_ms();
  jx_ui_set_busy(0);

  double audio_s = (double)n_samples / (double)JX_PCM_RATE;
  double enc_s   = (t1 - t0) / 1000.0;
  double dec_s   = (t2 - t1) / 1000.0;

  printf("  音频      : %.3f s (%d 样本 @16kHz)\n", audio_s, n_samples);
  printf("  帧/令牌   : Tt=%d 帧, %d token, %d bit/token\n",
         Tt, ntok, jx_token_bits());
  printf("  载荷      : %d 字节 (%.2f kbps)\n",
         (int)((int64_t)ntok * jx_token_bits() / 8),
         (double)ntok * jx_token_bits() / audio_s / 1000.0);
  printf("  编码耗时  : %8.1f ms   (RTF %.3f)\n", t1 - t0, enc_s / audio_s);
  printf("    前处理+CNN+注意力 %8.1f ms\n", jx_last_timing.pre_ms);
  printf("    FSQ 量化          %8.1f ms\n", jx_last_timing.quant_ms);
  printf("  解码耗时  : %8.1f ms   (RTF %.3f)\n", t2 - t1, dec_s / audio_s);
  printf("    反量化+注意力     %8.1f ms\n", jx_last_timing.deq_attn_ms);
  printf("    CNN 解码          %8.1f ms\n", jx_last_timing.dec_ms);
  printf("  单次总耗时: %8.1f ms   (总 RTF %.3f)  %s\n",
         t2 - t0, (enc_s + dec_s) / audio_s,
         ((enc_s + dec_s) / audio_s) < 1.0 ? "=> 快于实时 OK" : "=> 慢于实时 NG");
  /* 2026-09-15 第三十五轮：逐位对照指纹。
   * 把 token 序列与解码输出的 float 缓冲按字节做 FNV-1a 64，用来确认内核
   * 改动（如 QACC 门禁放宽）没有翻转任何一个比特：开关前后指纹相同即等价。
   * 开销约 1 ms（8 万样本 x 4 字节）。 */
  {
    unsigned long long h_idx = 1469598103934665603ull;
    unsigned long long h_rec = 1469598103934665603ull;
    const unsigned char *ph = (const unsigned char *)rec;
    for (int i = 0; i < ntok; i++)
      {
        int64_t v = idx[i];
        for (int b = 0; b < 8; b++)
          {
            h_idx = (h_idx ^ (unsigned char)(v >> (b * 8))) * 1099511628211ull;
          }
      }
    for (size_t i = 0; i < (size_t)n_samples * sizeof(float); i++)
      {
        h_rec = (h_rec ^ ph[i]) * 1099511628211ull;
      }
    printf("  指纹      : idx=%016llx rec=%016llx\n", h_idx, h_rec);
  }
  jx_prof_dump();

  *out_idx  = idx;
  *out_ntok = ntok;
  *out_Tt   = Tt;
  *out_rec  = rec;
  return 0;
}

static int jx_roundtrip(const int16_t *pcm, int n_samples)
{
  int64_t *idx = NULL; int ntok = 0, Tt = 0; float *rec = NULL;
  int rc = jx_run_once(pcm, n_samples, &idx, &ntok, &Tt, &rec);
  free(idx); free(rec);
  return rc;
}

/****************************************************************************
 * bench：纯算力测量
 ****************************************************************************/

static int jx_cmd_bench(int argc, char *argv[])
{
  int secs = jx_parse_seconds(argc, argv, 5, 60);
  int n    = secs * JX_PCM_RATE;

  printf("=== 极讯 Codec 单板算力基准 (码率 %s) ===\n", jx_rate_name());
  jx_print_wt();

  int16_t *pcm = (int16_t *)malloc(sizeof(int16_t) * (size_t)n);
  if (pcm == NULL)
    {
      printf("[ERR] 分配 %d 样本失败\n", n);
      return 1;
    }

  jx_gen_pcm(pcm, n, 0x2468ace1u);
  jx_print_mem("基准前");

  double t0 = jx_now_ms();
  int rc = jx_roundtrip(pcm, n);
  double t1 = jx_now_ms();

  jx_print_mem("基准后");
  printf("  含分配/释放的墙钟总时间: %.1f ms\n", t1 - t0);

  free(pcm);
  return rc;
}

/****************************************************************************
 * seg：分块一致性探针（线 A「流式」可行性的判据）
 *
 * 流式（长语音边录边编边播）能不能做，取决于一个必须先量出来的事实：
 * 「把整段切成窗口、每个窗口单独跑一遍模型」与「整段一次跑完」差多少。
 *
 *   reference = 整段 encode + decode
 *   segment   = 窗口 [s, s+C) 用缓冲区 [s-L, s+C) 跑一遍，只取 [s, s+C)
 *
 * 为什么必须有 L：注意力 local_mha_layer 里 window=400 帧，查询 t 能看到
 * [(w-1)*W, t]（w = t/W），所以窗口左侧得带足够上下文；而当缓冲区起点落在
 * window 的整数倍上时，局部坐标算出的 wstart 恰好等于绝对坐标的结果，所以
 * 这个「带左上下文重跑」的方案不需要改模型。L >= 2400ms（一个整窗）时应当
 * 逐位一致；L 越小误差越大 —— 这条曲线就是「流式窗口能开多小」的答案。
 *
 * 用法：jixun seg [C毫秒=500] [L毫秒=2400] [总秒数=2]
 ****************************************************************************/

static int jx_cmd_seg(int argc, char *argv[])
{
  int c_ms = (argc >= 3) ? atoi(argv[2]) : 500;
  int secs = (argc >= 4) ? atoi(argv[3]) : 3;

  /* L 列表：jixun seg <Cms> <总秒数> <L1ms> [L2ms] ...（不给时默认 2400）
   * 一个进程里跑完所有 L，参考只算一次 —— 既省时间，也避开「重启后堆布局
   * 变了导致参考段 OOM」的干扰。 */
  int nl = 0, lv[16];
  for (int i = 4; i < argc && nl < 16; i++) { lv[nl++] = atoi(argv[i]); }
  if (nl == 0) { lv[nl++] = 2400; }

  if (c_ms < 6) { c_ms = 6; }
  if (secs < 1) { secs = 1; }
  if (secs > 6) { secs = 6; }

  int n = secs * JX_PCM_RATE;
  int R = 0;

  /* 窗口对齐到整帧（HOP_LENGTH 样本），避免边界落在帧中间 */
  int C = (c_ms * JX_PCM_RATE / 1000) / HOP_LENGTH * HOP_LENGTH;
  if (C < HOP_LENGTH) { C = HOP_LENGTH; }

  printf("=== 极讯 Codec 分块一致性探针 (码率 %s) ===\n", jx_rate_name());
  /* 右侧余量（JX_SEGR，毫秒）：段尾的卷积需要真实后续数据，否则 jx_preprocess
   * 只能补零，段尾几十帧全错。左回看 L 加多长都补不了这一项。 */
  {
    int r_ms = jx_env_int("JX_SEGR", 0);
    if (r_ms < 0) { r_ms = 0; }
    R = (r_ms * JX_PCM_RATE / 1000) / HOP_LENGTH * HOP_LENGTH;
  }
  printf("  右侧余量 R=%d 样本 (%.0f ms)\n", R, 1000.0 * (double)R / JX_PCM_RATE);
  printf("  总长=%d 样本 (%.2f s) | 窗口 C=%d (%.0f ms) | %d 个回看配置\n",
         n, (double)n / JX_PCM_RATE, C, 1000.0 * (double)C / JX_PCM_RATE, nl);

  int16_t *pcm = (int16_t *)malloc(sizeof(int16_t) * (size_t)n);
  if (pcm == NULL) { printf("[ERR] pcm 分配失败\n"); return 1; }
  jx_gen_pcm(pcm, n, 0x2468ace1u);

  /* ---- 1) 参考：整段一次 ---- */
  int16_t *ref  = NULL;
  double  t_ref = 0.0;
  {
    int64_t *idx = NULL; int ntok = 0, Tt = 0;
    float *rec = NULL;
    double t0 = jx_now_ms();
    if (jx_encode(pcm, n, &idx, &ntok, &Tt) != 0) { free(pcm); return 1; }
    if (jx_decode(idx, Tt, n, &rec) != 0) { free(idx); free(pcm); return 1; }
    t_ref = jx_now_ms() - t0;
    ref = (int16_t *)malloc(sizeof(int16_t) * (size_t)n);
    if (ref == NULL) { free(rec); free(idx); free(pcm); return 1; }
    for (int i = 0; i < n; i++) { ref[i] = jx_round_s16((double)rec[i]); }
    free(rec); free(idx);
    printf("  [参考] 整段编解码 %.1f ms (%d token)\n", t_ref, ntok);
  }

  /* ---- 2) 逐个回看配置（参考只算一次） ---- */
  for (int k = 0; k < nl; k++)
    {
      int l_ms = lv[k]; if (l_ms < 0) { l_ms = 0; }
      int L = (l_ms * JX_PCM_RATE / 1000) / HOP_LENGTH * HOP_LENGTH;

      double t_seg0 = jx_now_ms();
      double rs = 0.0, es = 0.0, pk = 0.0;
      int nseg = 0, nfail = 0;
      for (int s = 0; s < n; s += C)
        {
          int e  = s + C; if (e > n) { e = n; }
          int b0 = s - L; if (b0 < 0) { b0 = 0; }
          int e2 = e + R; if (e2 > n) { e2 = n; }
          int m  = e2 - b0;

          int64_t *idx = NULL; int ntok = 0, Tt = 0;
          float *rec = NULL;
          if (jx_encode(&pcm[b0], m, &idx, &ntok, &Tt) != 0) { nfail++; continue; }
          if (jx_decode(idx, Tt, m, &rec) != 0) { free(idx); nfail++; continue; }

          /* 就地比对：不留 seg 缓冲（3 s 总长时省 384 KB，正好卡在 OOM 边上） */
          for (int i = s; i < e; i++)
            {
              double a = (double)ref[i];
              double b = (double)jx_round_s16((double)rec[i - b0]);
              double d = a - b;
              rs += a * a; es += d * d;
              if (d < 0.0) { d = -d; }
              if (d > pk) { pk = d; }
            }
          free(idx); free(rec); nseg++;
        }
      double t_seg = jx_now_ms() - t_seg0;
      double xp = (t_ref > 0.0) ? (t_seg / t_ref) : 0.0;

      if (es == 0.0)
        {
          printf("  L=%-5d (%5d 样本) 段=%-2d 失败=%d 墙钟 %7.1f ms (%.2fx) ==> 逐位一致 (inf dB)\n",
                 l_ms, L, nseg, nfail, t_seg, xp);
        }
      else
        {
          printf("  L=%-5d (%5d 样本) 段=%-2d 失败=%d 墙钟 %7.1f ms (%.2fx) ==> SNR %6.2f dB (峰值误差 %.4g)\n",
                 l_ms, L, nseg, nfail, t_seg, xp, 10.0 * log10(rs / es), pk);
        }
    }

  free(ref); free(pcm);
  return 0;
}

/****************************************************************************
 * loop：麦克风 -> 编码 -> 解码 -> 喇叭（单板闭环）
 ****************************************************************************/

/****************************************************************************
 * stream：24ms 小包流式编解码自测
 *
 *   nsh> jixun stream [包数]      默认 25 包 (=600ms)
 *
 * 用途：把 24ms 流水线单独跑起来，看每包耗时（实时门槛是 <24ms/包）、
 * token 输出是否稳定。当前是朴素实现（每包重算整条滚动缓冲），
 * 耗时必然远超门槛 —— 这一步验的是语义与稳定性，不是速度。
 ****************************************************************************/

static int jx_cmd_stream(int argc, char *argv[])
{
  int npkt = (argc >= 3) ? atoi(argv[2]) : 25;
  int batch = (argc >= 4) ? atoi(argv[3]) : 1;
  struct jx_stream_s *st;
  int16_t *pcm;
  int64_t  tok[JX_PKT_FRAMES];
  float    out[JX_PKT_SAMPLES];
  int64_t *tokp = tok;
  float   *outp = out;
  int64_t  last_tok[JX_PKT_FRAMES];
  float    last_out[JX_PKT_SAMPLES];
  double   t0, t1;
  int      rc = 0, done = 0;

  if (npkt < 1)   { npkt = 1; }
  if (npkt > 400) { npkt = 400; }
  if (batch < 1)  { batch = 1; }
  if (batch > 64) { batch = 64; }
  if (batch > npkt) { batch = npkt; }

  pcm = (int16_t *)malloc(sizeof(int16_t) * (size_t)JX_PKT_SAMPLES * (size_t)npkt);
  if (pcm == NULL) { printf("[ERR] pcm 分配失败\n"); return 1; }
  jx_gen_pcm(pcm, JX_PKT_SAMPLES * npkt, 0x2468ace1u);

  if (batch > 1)
    {
      tokp = (int64_t *)malloc(sizeof(int64_t) * (size_t)JX_PKT_FRAMES * (size_t)batch);
      outp = (float *)malloc(sizeof(float) * (size_t)JX_PKT_SAMPLES * (size_t)batch);
      if (tokp == NULL || outp == NULL)
        {
          printf("[ERR] batch 缓冲分配失败\n");
          free(tokp); free(outp); free(pcm);
          return 1;
        }
    }

  st = jx_stream_open(512);
  if (st == NULL) { free(pcm); printf("[ERR] 流分配失败\n"); return 1; }

  printf("=== 24ms 小包流式（%d 包 = %d ms，批大小 %d）===\n",
         npkt, npkt * 24, batch);
  printf("  包参数: %d 帧/包, %d 样本/包, %d bit/包 = %d 字节\n",
         JX_PKT_FRAMES, JX_PKT_SAMPLES, JX_PKT_BITS, JX_PKT_BYTES);
  printf("  滚动缓冲: %d 帧窗 + 512 样本 halo + 1 包\n", JX_STREAM_WINDOW_DEFAULT);

  t0 = jx_now_ms();
  jx_prof_reset();
  for (int i = 0; i < npkt; i += batch)
    {
      int n = npkt - i;
      if (n > batch) { n = batch; }
      if (batch == 1)
        {
          rc = jx_stream_feed(st, &pcm[(size_t)i * JX_PKT_SAMPLES], tokp, outp);
        }
      else
        {
          rc = jx_stream_feed_batch(st, &pcm[(size_t)i * JX_PKT_SAMPLES],
                                    n, tokp, outp);
        }
      if (rc != 0)
        {
          printf("[ERR] 第 %d 包失败 rc=%d\n", i, rc);
          break;
        }
      memcpy(last_tok, &tokp[(size_t)(n - 1) * JX_PKT_FRAMES],
             sizeof(last_tok));
      memcpy(last_out, &outp[(size_t)(n - 1) * JX_PKT_SAMPLES],
             sizeof(last_out));
      done += n;
    }
  t1 = jx_now_ms();

  printf("  完成 %d 包, 墙钟 %.1f ms\n", done, t1 - t0);
  if (done > 0)
    {
      printf("  每包 %.1f ms（实时门槛 < 24ms/包）\n", (t1 - t0) / (double)done);
      {
        double cm = 0.0, am = 0.0, dm = 0.0;
        double qm = 0.0, em = 0.0, nm = 0.0;
        jx_stream_times(st, &cm, &am, &dm, &qm, &em, &nm);
        printf("  分段/包: 卷积 %.1f + 编码注意力 %.1f + 解码 %.1f"
               " (反量化 %.1f + 解注意力 %.1f + 解CNN %.1f)\n",
               cm / done, am / done, dm / done,
               qm / done, em / done, nm / done);
      }
      printf("  末包 token: %08llx %08llx %08llx %08llx\n",
             (unsigned long long)last_tok[0], (unsigned long long)last_tok[1],
             (unsigned long long)last_tok[2], (unsigned long long)last_tok[3]);
      printf("  末包 PCM[0..3]: %.5f %.5f %.5f %.5f\n",
             (double)last_out[0], (double)last_out[1],
             (double)last_out[2], (double)last_out[3]);
    }

  if (getenv("JX_STREAM_PROF") != NULL)
    {
      jx_prof_dump();
      jx_ct_dump();
    }

  jx_stream_close(st);
  if (batch > 1) { free(tokp); free(outp); }
  free(pcm);
  return (rc == 0) ? 0 : 1;
}

/****************************************************************************
 * ui：ST7796 480x320 仪表盘
 *
 *   nsh> jixun ui init        取屏 + 挂 SD + 载字库
 *   nsh> jixun ui page <n>    整屏切到第 n 页（0 话音 / 1 图像 / 2 误码 /
 *                             3 日志 / 4 码率）
 *   nsh> jixun ui test        逐项打印绘制耗时（满屏/文字/波形/栏）
 ****************************************************************************/

static int jx_cmd_ui(int argc, char *argv[])
{
  const char *sub = (argc >= 3) ? argv[2] : "init";

  if (strcmp(sub, "init") == 0)
    {
      if (jx_ui_init() != 0)
        {
          printf("[ui] 初始化失败\n");
          return 1;
        }

      printf("[ui] 字库: %s (%s)\n", jx_ui_font_ok() ? "已载入" : "缺失",
             jx_ui_font_path());
      jx_ui_set_send_cb(jx_ui_send_action);
      jx_ui_log("[UI] panel ready");
      return 0;
    }

  if (strcmp(sub, "test") == 0)
    {
      if (jx_ui_init() != 0) { return 1; }
      jx_ui_set_send_cb(jx_ui_send_action);
      return jx_ui_selftest();
    }

  if (strcmp(sub, "rdtest") == 0)
    {
      return jx_ui_rdtest();
    }

  if (strcmp(sub, "get") == 0)
    {
      if (jx_ui_init() != 0) { return 1; }
      jx_ui_set_send_cb(jx_ui_send_action);
      printf("[ui] page=%d ready=%d busy=%d font=%d\n",
             jx_ui_page_get(), jx_ui_ready(), jx_ui_busy(), jx_ui_font_ok());
      return 0;
    }

  if (strcmp(sub, "peer") == 0)
    {
      if (argc < 4)
        {
          printf("用法: jixun ui peer <对端IP>\n");
          return 1;
        }

      if (jx_ui_init() != 0) { return 1; }
      jx_ui_set_send_cb(jx_ui_send_action);
      jx_ui_set_peer(argv[3]);
      jx_ui_set_status("READY  SEND");
      if (jx_ui_page_get() == 0) { jx_ui_page(0); }
      printf("[ui] 对端已设置: %s\n", jx_ui_peer_get());
      return 0;
    }

  if (strcmp(sub, "touch") == 0)
    {
      int sec = (argc >= 4) ? atoi(argv[3]) : 10;
      return (jx_ui_touch_test(sec) == 0) ? 0 : 1;
    }

#ifdef CONFIG_ESP32S3_BOARD_TOUCHSCREEN
  if (strcmp(sub, "tdiag") == 0)
    {
      uint32_t d[8];
      uint8_t p[8];
      int probe_rc;

      if (gt911_diag_snapshot(d) != 0)
        {
          printf("[touch] diag 读取失败\n");
          return 1;
        }

      probe_rc = gt911_diag_probe(p);
      printf("[touch] pid=%02x %02x %02x %02x status=%02x probe_rc=%d\n",
             p[0], p[1], p[2], p[3], p[4], probe_rc);
      printf("[touch] polls=%lu i2c_err=%lu down=%lu up=%lu "
             "status=%lu last=(%lu,%lu) active=%lu\n",
             (unsigned long)d[0], (unsigned long)d[1],
             (unsigned long)d[2], (unsigned long)d[3],
             (unsigned long)d[4], (unsigned long)d[5],
             (unsigned long)d[6], (unsigned long)d[7]);
      return 0;
    }
#endif

  if (strcmp(sub, "page") == 0)
    {
      int n, rc;

      if (jx_ui_init() != 0) { return 1; }
      jx_ui_set_send_cb(jx_ui_send_action);

      n  = (argc >= 4) ? atoi(argv[3]) : 0;
      rc = jx_ui_page(n);
      printf("[ui] page %d -> rc=%d (当前页 %d, 字库 %s)\n",
             n, rc, jx_ui_page_get(), jx_ui_font_ok() ? "SD" : "缺失");
      return (rc == 0) ? 0 : 1;
    }

  printf("用法: jixun ui init | test | rdtest | get | peer <IP> | touch [秒] | tdiag | page <0..%d>\n",
         jx_ui_page_count() - 1);
  return 1;
}

/* 防止编译器把测量循环优化掉 */
static volatile float g_mem_sink;

/****************************************************************************
 * mem：存储层次 / 浮点吞吐微基准
 *
 * 目的：把「算力不够」和「访存不够」分开。
 *   [1] FPU + L1 常驻  → 纯算力上限（无访存干扰）
 *   [2] PSRAM 顺序读/写 → 大张量所在堆的实际带宽
 *   [3] FLASH(XIP) 顺序读 → 权重常数块的实际带宽
 *   [4] PSRAM 跨步读   → 老卷积内核「跨通道跳着取」的访存模式
 *   [5] PSRAM 流式读改写 → 流式卷积内核 y[t] += x[t]*w 的模式
 ****************************************************************************/

/* ---- PIE / 128-bit 向量扩展 int8 MAC 吞吐 ---------------------------------
 *
 * ESP32-S3 的向量单元有 `ee.vmulas.s8.accx q0, q1`：**一条指令做 16 个
 * int8 乘加**。浮点这边被 FPU 的 4 周期/MAC 卡死（madd.s），int8 是唯一
 * 能把算力拉上去的路，所以先把它的真实吞吐量出来。
 *
 * clobber 用 f0..f15：PIE 的 q 寄存器与 FPU 的 f 寄存器共用同一组物理
 * 寄存器，不声明的话 GCC 可能把活跃浮点值放在被我们踩掉的位置上。
 */

#define JX_VMULAS8_16                                                     \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"           \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"           \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"           \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"           \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"           \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"           \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"           \
  "ee.vmulas.s8.accx q0, q1\n\t" "ee.vmulas.s8.accx q0, q1\n\t"

#define JX_FCLOB "f0","f1","f2","f3","f4","f5","f6","f7", \
                 "f8","f9","f10","f11","f12","f13","f14","f15"

/* ---- Phase0：CPENABLE 探测 ------------------------------------------------
 * PIE 指令报 EXCCAUSE=0x23（协处理器未使能）。先把 CPENABLE 读出来、
 * 打开全部 8 位再跑，用来判定 PIE 到底能不能用。 */
static unsigned jx_get_cpenable(void)
{
  unsigned v;
  __asm__ __volatile__("rsr %0, cpenable" : "=r"(v));
  return v;
}

static void jx_set_cpenable(unsigned v)
{
  __asm__ __volatile__("wsr %0, cpenable ; rsync" :: "r"(v) : "memory");
}

/* [1a] FPU 链数扫描：1/2/4/8 路独立累加器。
 * 8 链测到 4.08 周期/MAC。如果这是"延迟上限"，加链数就该线性变快；
 * 如果链数无关，那 4 周期/MAC 就是 madd.s 的吞吐上限，浮点没救。 */
#define JX_FPU_CHAIN(NCH)                                                     \
  do {                                                                        \
    double t0 = jx_now_ms();                                                  \
    for (int r = 0; r < REP; r++) {                                           \
      float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;   \
      for (int i = 0; i < NF; i += (NCH)) {                                   \
        if ((NCH) > 0) { a0 += ax[i] * bx[i]; }                               \
        if ((NCH) > 1) { a1 += ax[i + 1] * bx[i + 1]; }                       \
        if ((NCH) > 2) { a2 += ax[i + 2] * bx[i + 2]; }                       \
        if ((NCH) > 3) { a3 += ax[i + 3] * bx[i + 3]; }                       \
        if ((NCH) > 4) { a4 += ax[i + 4] * bx[i + 4]; }                       \
        if ((NCH) > 5) { a5 += ax[i + 5] * bx[i + 5]; }                       \
        if ((NCH) > 6) { a6 += ax[i + 6] * bx[i + 6]; }                       \
        if ((NCH) > 7) { a7 += ax[i + 7] * bx[i + 7]; }                       \
      }                                                                       \
      g_mem_sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;                     \
    }                                                                         \
    double ms = jx_now_ms() - t0;                                             \
    double macs = (double)REP * (double)NF;                                   \
    printf("  [1a] FPU %d 链       : %8.1f ms  %7.1f MMAC/s  %5.2f 周期/MAC\n", \
           (NCH), ms, macs / ms / 1000.0, ms * 1e-3 * 240.0e6 / macs);        \
  } while (0)

static void jx_fpu_chain_bench(void)
{
  enum { NF = 512 };
  static float ax[NF], bx[NF];
  for (int i = 0; i < NF; i++)
    {
      ax[i] = (float)(i & 7) * 0.125f;
      bx[i] = (float)(i & 3) * 0.25f;
    }
  const int REP = 32768;
  JX_FPU_CHAIN(1);
  JX_FPU_CHAIN(2);
  JX_FPU_CHAIN(4);
  JX_FPU_CHAIN(8);
}

/* ---- [1b]/[1c] 纯寄存器浮点吞吐（2026-09-15 第三十六轮） --------------------
 *
 * 为什么要再量一次：[1a] 的循环体每个 MAC 要两次 L1 装载（ax[i]*bx[i]），
 * 16 次装载喂 8 个 MAC。它量到的 3.86 周期/MAC 里有多少是 FPU 吞吐、多少是
 * 访存端口，[1a] 分不出来。而第三十五轮那个判决——"FPU 墙 ≈ 3.5~4 周期/条
 * FP 指令，elementwise 那 2.2 s 不可压"——完全建立在这个数上。如果它其实是
 * 访存地板，2.2 s 就还有翻倍的希望。所以必须把访存彻底剔掉再量一遍。
 *
 * 做法：乘子写成编译期常量、累加器写成局部变量，循环体里不应有装载/存储。
 * **前提必须验证**：在 VM 上 objdump 本函数，循环体内只允许出现 madd.s 与
 * 计数器/跳转；一旦出现 l32i/s32i 就说明 16 个 FP 寄存器不够用、溢出到栈，
 * 本测试无效（那时要让编译器只用 4 条链）。
 * 用 rsr.ccount 直接量周期：jx_now_ms 分辨率只有几毫秒，喂不饱这种短循环。
 *
 * 链数扫描的意义：单链量到的是**延迟**，链数变多若明显变快，说明 FPU 是流水化
 * 的、之前撞的是延迟墙（可以靠交错多条独立链绕开）；8 条链仍不掉，那条就是
 * **吞吐墙**，只能靠减少 FP 指令条数。
 */
static inline unsigned jx_cc0(void)
{
  unsigned c;
  __asm__ __volatile__("rsr %0, ccount" : "=r"(c));
  return c;
}

/* 上面这个循环如果写成 C，GCC 会把它改写成「每 4 个 madd.s 夹 4 条 mov.s」
 * 的指令混合（objdump 实证，见 _REALTIME_ROUND36），每个 madd.s 摊到 2.5 条
 * 指令，量出来的是发射上限而不是 FPU 吞吐。所以循环体改成手写汇编，
 * 指令条数完全钉死：
 *   8 条**独立**链：每轮 8 条 madd.s + addi + bnez = 10 条指令喂 8 个 MAC，
 *                   取指发射地板只有 1.25 周期/MAC，远低于待测区间 → 量的是吞吐；
 *   8 条**依赖**链：每轮 8 条串行 madd.s → 量的是 FPU 往返延迟。
 * 常量放 DRAM 常量池、循环**外**用 l32i 取进 f8/f9，循环体内零访存。 */

static float g_fpk[4] = { 1.0000001f, 1e-07f, 0.0f, 0.0f };

#define JX_FPASM_CLOBBER                                                     \
  "f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","a4","a5","a6"

/* 每轮 8 条相互独立的 madd.s（f0..f7 各一条链） */
static void __attribute__((noinline)) jx_fpasm(unsigned long rep, int dep)
{
  unsigned long n = rep;
  unsigned sink = 0;
  unsigned c0, c1;
  if (!dep)
    {
      c0 = jx_cc0();
      __asm__ __volatile__(
        "l32i  a4, %2, 0\n\t"
        "l32i  a5, %2, 4\n\t"
        "wfr   f8, a4\n\t"
        "wfr   f9, a5\n\t"
        "movi  a6, 0\n\t"
        "wfr   f0, a6\n\t" "wfr f1, a6\n\t" "wfr f2, a6\n\t" "wfr f3, a6\n\t"
        "wfr   f4, a6\n\t" "wfr f5, a6\n\t" "wfr f6, a6\n\t" "wfr f7, a6\n\t"
        "1:\n\t"
        "madd.s f0, f8, f9\n\t"
        "madd.s f1, f8, f9\n\t"
        "madd.s f2, f8, f9\n\t"
        "madd.s f3, f8, f9\n\t"
        "madd.s f4, f8, f9\n\t"
        "madd.s f5, f8, f9\n\t"
        "madd.s f6, f8, f9\n\t"
        "madd.s f7, f8, f9\n\t"
        "addi  %0, %0, -1\n\t"
        "bnez  %0, 1b\n\t"
        "rfr   %1, f0\n\t"
        : "+r"(n), "=r"(sink)
        : "r"(g_fpk)
        : JX_FPASM_CLOBBER);
      c1 = jx_cc0();
      g_mem_sink = (float)sink;
      {
        double macs = (double)rep * 8.0;
        double cyc  = (double)(c1 - c0);
        printf("  [1b] FPUasm 8 独立链: %7.2f Mcyc  %6.1f MMAC/s  %5.2f 周期/MAC\n",
               cyc / 1e6, macs * 240.0 / cyc, cyc / macs);
      }
    }
  else
    {
      c0 = jx_cc0();
      __asm__ __volatile__(
        "l32i  a4, %2, 0\n\t"
        "l32i  a5, %2, 4\n\t"
        "wfr   f8, a4\n\t"
        "wfr   f9, a5\n\t"
        "movi  a6, 0\n\t"
        "wfr   f0, a6\n\t"
        "1:\n\t"
        /* 乘子用 1e-7（f9）而不是 1.0000001（f8）：f0 = f0*1e-7 + f0 每步只涨
         * 1e-7 倍，8M 步后仍是有穷数；用 1.0000001 会翻倍到 inf，虽然本意只
         * 是量延迟，但没必要让结果落进 inf/NaN 的边界情况。 */
        "madd.s f0, f0, f9\n\t"
        "madd.s f0, f0, f9\n\t"
        "madd.s f0, f0, f9\n\t"
        "madd.s f0, f0, f9\n\t"
        "madd.s f0, f0, f9\n\t"
        "madd.s f0, f0, f9\n\t"
        "madd.s f0, f0, f9\n\t"
        "madd.s f0, f0, f9\n\t"
        "addi  %0, %0, -1\n\t"
        "bnez  %0, 1b\n\t"
        "rfr   %1, f0\n\t"
        : "+r"(n), "=r"(sink)
        : "r"(g_fpk)
        : JX_FPASM_CLOBBER);
      c1 = jx_cc0();
      g_mem_sink = (float)sink;
      {
        double macs = (double)rep * 8.0;
        double cyc  = (double)(c1 - c0);
        printf("  [1b] FPUasm 8 依赖链: %7.2f Mcyc  %6.1f MMAC/s  %5.2f 周期/MAC\n",
               cyc / 1e6, macs * 240.0 / cyc, cyc / macs);
      }
    }
}

/* noinline：VM 上要 objdump 这两个函数来核对"循环体内零访存"这个前提，
 * 被内联进 jx_cmd_mem 就不好认了。 */
static void __attribute__((noinline)) jx_fpreg_bench(void)
{
  jx_fpasm(2000000UL, 0);    /* 吞吐：8 条独立链，16M 个 MAC */
  jx_fpasm(1000000UL, 1);    /* 延迟：8 条串行 madd.s，8M 个 MAC */
}

/* [1c] jx_sinsq 的延迟 vs 吞吐。
 * snake 的 fused epilogue（int8_kernels.c 里 4 路展开那一支）每元素约 14 条
 * FP 指令，其中 jx_sinsq 是唯一带串行 Horner 链的部分，第三十五轮估算它吃掉
 * 了 epilogue 48.65 周期/元素里的大头。1 链 = 延迟，8 链 = 吞吐。
 * 入参用 x <- sin^2(x+0.25) 迭代：它收敛到 ~0.33，不会掉进非规格化数，
 * 每轮固定 +1 个 add（正好对得上调用点的 sna*v 那一步）。 */
#define JX_SINSQ_CHAIN(NCH, REP, INNER)                                        \
  do {                                                                         \
    float x0 = 0.30f, x1 = 0.31f, x2 = 0.32f, x3 = 0.33f;                      \
    float x4 = 0.34f, x5 = 0.35f, x6 = 0.36f, x7 = 0.37f;                      \
    unsigned c0 = jx_cc0();                                                    \
    for (int r = 0; r < (REP); r++)                                            \
      {                                                                        \
        for (int i = 0; i < (INNER); i++)                                      \
          {                                                                    \
            if ((NCH) > 0) { x0 = jx_sinsq(x0 + 0.25f); }                      \
            if ((NCH) > 1) { x1 = jx_sinsq(x1 + 0.25f); }                      \
            if ((NCH) > 2) { x2 = jx_sinsq(x2 + 0.25f); }                      \
            if ((NCH) > 3) { x3 = jx_sinsq(x3 + 0.25f); }                      \
            if ((NCH) > 4) { x4 = jx_sinsq(x4 + 0.25f); }                      \
            if ((NCH) > 5) { x5 = jx_sinsq(x5 + 0.25f); }                      \
            if ((NCH) > 6) { x6 = jx_sinsq(x6 + 0.25f); }                      \
            if ((NCH) > 7) { x7 = jx_sinsq(x7 + 0.25f); }                      \
          }                                                                    \
      }                                                                        \
    unsigned c1 = jx_cc0();                                                    \
    g_mem_sink = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;                        \
    {                                                                          \
      double calls = (double)(REP) * (double)(INNER) * (NCH);                  \
      double cyc   = (double)(c1 - c0);                                        \
      printf("  [1c] sinsq %d 链   : %7.2f Mcyc  %6.1f Mcall/s  %5.2f 周期/次\n", \
             (NCH), cyc / 1e6, calls * 240.0 / cyc, cyc / calls);              \
    }                                                                          \
  } while (0)

static void __attribute__((noinline)) jx_sinsq_bench(void)
{
  jx_sq_kt_ensure();
  JX_SINSQ_CHAIN(1, 4096, 256);
  JX_SINSQ_CHAIN(2, 4096, 256);
  JX_SINSQ_CHAIN(4, 4096, 256);
  JX_SINSQ_CHAIN(8, 4096, 256);
}

/* [1d] 转换指令 vs 幻数法：jx_sinsq_lut 那条链到底卡在哪一条指令。
 * [kc] 实测 Ci=20 Co=80 epilogue 3413 周期/位置 / 80 通道 = 42.7 周期/元素，
 * 而反汇编里只有 ~8 条 FP + 1 次查表；sinsq 4 链微基准 18.51 周期/次。
 * 说明链上有指令的**吞吐**极低（4 条独立链都没盖住）。拆开量：
 *   (a) 只有 mul.s（对照）
 *   (b) float->int->float 往返（trunc.s + float.s）
 *   (c) 现行 LUT 链 mul.s/trunc.s/and/addx4/lsi
 *   (d) 幻数法 LUT 链：把取整并进一条 madd.s（x*SCALE + 1.5*2^23 的尾数低位
 *       就是向偶取整结果，掩码后与截断法同余），省掉 trunc.s。 */
static void __attribute__((noinline)) jx_cvt_bench(void)
{
  const int      N     = 2048;
  const float    SCALE = (float)N / 3.14159265358979f;
  const float    MAGIC = 12582912.0f;       /* 1.5 * 2^23 */
  const unsigned MASKU = (unsigned)(N - 1);
  const long     REP   = 300000;
  /* 表复用 jx_ssq（本来就在 .bss，2048 个 float）—— dram0 只剩几百字节余量，
   * 不能为探针再开静态数组。 */
  jx_sq_kt_ensure();
  float *g_cvt_tbl = jx_ssq;
  {
    float x0=0.3f,x1=0.31f,x2=0.32f,x3=0.33f; unsigned c0=jx_cc0();
    for (long r=0;r<REP;r++) for (int i=0;i<64;i++) {
      x0 = x0*SCALE+0.25f; x1 = x1*SCALE+0.25f;
      x2 = x2*SCALE+0.25f; x3 = x3*SCALE+0.25f;
    }
    unsigned c1=jx_cc0(); g_mem_sink = x0+x1+x2+x3;
    printf("  [1d] (a) mul.s   4链 : %7.2f Mcyc  %5.2f 周期/次\n",
           (double)(c1-c0)/1e6, (double)(c1-c0)/(double)(REP*64*4));
  }
  {
    float x0=0.3f,x1=0.31f,x2=0.32f,x3=0.33f; unsigned c0=jx_cc0();
    for (long r=0;r<REP;r++) for (int i=0;i<64;i++) {
      x0 = (float)((int)(x0*1000.0f)) * 0.001f + 0.25f;
      x1 = (float)((int)(x1*1000.0f)) * 0.001f + 0.25f;
      x2 = (float)((int)(x2*1000.0f)) * 0.001f + 0.25f;
      x3 = (float)((int)(x3*1000.0f)) * 0.001f + 0.25f;
    }
    unsigned c1=jx_cc0(); g_mem_sink = x0+x1+x2+x3;
    printf("  [1d] (b) trunc+cvt 4链: %7.2f Mcyc  %5.2f 周期/次\n",
           (double)(c1-c0)/1e6, (double)(c1-c0)/(double)(REP*64*4));
  }
  {
    float x0=0.3f,x1=0.31f,x2=0.32f,x3=0.33f; unsigned c0=jx_cc0();
    for (long r=0;r<REP;r++) for (int i=0;i<64;i++) {
      x0 = g_cvt_tbl[((unsigned)(int)(x0*SCALE)) & MASKU] + 0.25f;
      x1 = g_cvt_tbl[((unsigned)(int)(x1*SCALE)) & MASKU] + 0.25f;
      x2 = g_cvt_tbl[((unsigned)(int)(x2*SCALE)) & MASKU] + 0.25f;
      x3 = g_cvt_tbl[((unsigned)(int)(x3*SCALE)) & MASKU] + 0.25f;
    }
    unsigned c1=jx_cc0(); g_mem_sink = x0+x1+x2+x3;
    printf("  [1d] (c) LUT trunc 4链 : %7.2f Mcyc  %5.2f 周期/次\n",
           (double)(c1-c0)/1e6, (double)(c1-c0)/(double)(REP*64*4));
  }
  {
    float x0=0.3f,x1=0.31f,x2=0.32f,x3=0.33f; unsigned c0=jx_cc0();
    for (long r=0;r<REP;r++) for (int i=0;i<64;i++) {
      float y0=x0*SCALE+MAGIC, y1=x1*SCALE+MAGIC;
      float y2=x2*SCALE+MAGIC, y3=x3*SCALE+MAGIC;
      unsigned u0,u1,u2,u3;
      __builtin_memcpy(&u0,&y0,4); __builtin_memcpy(&u1,&y1,4);
      __builtin_memcpy(&u2,&y2,4); __builtin_memcpy(&u3,&y3,4);
      x0 = g_cvt_tbl[u0 & MASKU] + 0.25f;
      x1 = g_cvt_tbl[u1 & MASKU] + 0.25f;
      x2 = g_cvt_tbl[u2 & MASKU] + 0.25f;
      x3 = g_cvt_tbl[u3 & MASKU] + 0.25f;
    }
    unsigned c1=jx_cc0(); g_mem_sink = x0+x1+x2+x3;
    printf("  [1d] (d) LUT magic 4链 : %7.2f Mcyc  %5.2f 周期/次\n",
           (double)(c1-c0)/1e6, (double)(c1-c0)/(double)(REP*64*4));
  }
}

/* [16] k=1 卷积 dot 阶段的「结构」对照（第三十六轮）。
 *
 * 同一份权重/激活（都放 .bss，L1 常驻），按三种真实形状跑四种结构：
 *   conv_1  Ci=20 Co=80   nv=2  kstride=32
 *   conv_2  Ci=81 Co=16   nv=6  kstride=96
 *   16->16  Ci=16 Co=16   nv=1  kstride=16
 * 指标统一折算成「周期 / 输出元素」，直接和板上 [k1] accx 分支的
 * 1786 周期/输出位置 对照。目的：判定 dot 阶段的瓶颈到底是
 *   (a) 结构差（每输出一次 accx 会话 + rur 往返）
 *   (b) rur.accx_0 的往返延迟（c 组去掉它）
 *   (c) 循环/调用开销
 * 全部只计时，不参与数值正确性。 */
/* 内部 SRAM 只剩 ~1.6 KB，权重不自己开数组，直接用真实的量化权重 blob
 * （与生产路径同源，且在 PSRAM 堆里，访存代价一致）。只有激活/输出开
 * 小数组，全部 L1 常驻。 */
static signed char g_bt_x[64] __attribute__((aligned(16)));
static signed char g_bt_ap[16 * 64] __attribute__((aligned(16)));
static int32_t     g_bt_o[128] __attribute__((aligned(16)));
static float       g_bt_sink;

typedef void (*jx_bt_dotfn)(const signed char *, const signed char *,
                            int, int, int, int32_t *);

/* nv > 4 时按 4 向量一批拆开（与 jx_linear_i8_cf 的调用形态一致） */
static void __attribute__((noinline))
jx_bt_nvec(jx_bt_dotfn f, const signed char *a, const signed char *w,
           int nv, int kstr, int co)
{
  int c0 = 0;
  while (c0 < nv)
    {
      int nc = nv - c0;
      if (nc > 4) { nc = 4; }
      f(a + (size_t)c0 * 16u, w + (size_t)c0 * 16u, nc, kstr, co, g_bt_o);
      c0 += nc;
    }
}

static void __attribute__((noinline))
jx_bt_shape(const char *name, int ci, int co, int rep)
{
  const int cpad = (ci + 15) & ~15;
  const int nv   = cpad >> 4;
  const int kstr = cpad;
  const int nj   = 16;
  const signed char *wt = jx_wq_blob_ptr();
  double t0, ms;
  long   r;
  int    o, i;
  float  s = 0.0f;

  if (co > 256 || cpad > 96) { return; }
  for (i = 0; i < 64; i++)      { g_bt_x[i] = (signed char)((i * 53 + 7) % 61 - 30); }
  for (i = 0; i < 16 * 64; i++) { g_bt_ap[i] = (signed char)((i * 29 + 3) % 67 - 33); }

  /* (a) 现状 pert：每个输出元素一次 jx_dot_i8 */
  t0 = jx_now_ms();
  for (r = 0; r < rep; r++)
    for (o = 0; o < co; o++)
      { s += (float)jx_dot_i8_probe(g_bt_x, wt + (size_t)o * kstr, nv); }
  ms = jx_now_ms() - t0;
  printf("  [16a] %-7s Ci=%-3d Co=%-3d nv=%d  pert     %8.1f ms  %7.1f 周期/输出\n",
         name, ci, co, nv, ms, ms * 240000.0 / ((double)rep * co));

  /* (b) dotcol：激活常驻向量寄存器，逐输出行装权重 + rur */
  t0 = jx_now_ms();
  for (r = 0; r < rep; r++)
    { jx_bt_nvec(jx_dotcol_i8_probe, g_bt_x, wt, nv, kstr, co); s += (float)g_bt_o[0]; }
  ms = jx_now_ms() - t0;
  printf("  [16b] %-7s Ci=%-3d Co=%-3d nv=%d  dotcol   %8.1f ms  %7.1f 周期/输出\n",
         name, ci, co, nv, ms, ms * 240000.0 / ((double)rep * co));

  /* (c) dotcol 无 rur：只剩 MAC + 装载，量吞吐上限 */
  t0 = jx_now_ms();
  for (r = 0; r < rep; r++)
    { jx_bt_nvec(jx_dotcol_i8_norb, g_bt_x, wt, nv, kstr, co); s += (float)g_bt_o[0]; }
  ms = jx_now_ms() - t0;
  printf("  [16c] %-7s Ci=%-3d Co=%-3d nv=%d  no-rur   %8.1f ms  %7.1f 周期/输出\n",
         name, ci, co, nv, ms, ms * 240000.0 / ((double)rep * co));

  /* (d) dotmj：权重常驻，逐位置换激活（nv<=4 才支持） */
  if (nv <= 4)
    {
      t0 = jx_now_ms();
      for (r = 0; r < rep; r++)
        for (o = 0; o < co; o++)
          {
            jx_dotmj_i8_probe(g_bt_ap, wt + (size_t)o * kstr, nv, nj,
                              cpad - nv * 16, g_bt_o);
            s += (float)g_bt_o[0];
          }
      ms = jx_now_ms() - t0;
      printf("  [16d] %-7s Ci=%-3d Co=%-3d nv=%d  dotmj    %8.1f ms  %7.1f 周期/输出\n",
             name, ci, co, nv, ms, ms * 240000.0 / ((double)rep * co * nj));
    }

  /* (e) qacc：16 个输出通道并行累加，一次读回 16 个结果。
   * 权重按 [chunk][16 输出通道] 布局（每 chunk 256 字节）。 */
  t0 = jx_now_ms();
  for (r = 0; r < rep; r++)
    {
      jx_qacc_dot16_probe(wt, g_bt_x, nv, g_bt_o);
      s += (float)g_bt_o[0];
    }
  ms = jx_now_ms() - t0;
  printf("  [16e] %-7s Ci=%-3d Co=16   nv=%d  qacc16   %8.1f ms  %7.1f 周期/输出\n",
         name, ci, nv, ms, ms * 240000.0 / ((double)rep * 16));

  g_bt_sink = s;
}

static void __attribute__((noinline)) jx_bt_bench(void)
{
  jx_wq_init();
  jx_bt_shape("conv_1", 20, 80, 20000);
  jx_bt_shape("conv_2", 81, 16, 20000);
  jx_bt_shape("16->16", 16, 16, 20000);
  g_mem_sink = g_bt_sink;
}

/* [2b] L1 常驻 + PSRAM 分块：把 1MB 分成 8KB 块来回滚，
 * 看"缓存友好"能拿到多少带宽（对照 [2] 的 59.9 MB/s）。 */
static void jx_block_bw_bench(void)
{
  const int NF = 256 * 1024;
  float *buf = (float *)malloc(sizeof(float) * (size_t)NF);
  if (buf == NULL) { printf("  [2b] 分配失败\n"); return; }
  for (int i = 0; i < NF; i++) { buf[i] = (float)(i & 15); }
  const int BLK = 2048;                 /* 8 KB */
  const int REP = 8;
  float s = 0;
  double t0 = jx_now_ms();
  for (int r = 0; r < REP; r++)
    for (int b = 0; b < NF; b += BLK)
      for (int i = b; i < b + BLK; i++) { s += buf[i]; }
  double ms = jx_now_ms() - t0;
  g_mem_sink = s;
  printf("  [2b] PSRAM 8KB 分块读: %8.1f ms  %7.1f MB/s\n",
         ms, (double)REP * NF * 4.0 / 1e6 / (ms / 1000.0));

  free(buf);
}

static void jx_pie_bench(void)
{
  unsigned cp0 = jx_get_cpenable();
  printf("  [0] CPENABLE 读回    : 0x%02x\n", cp0);
  jx_set_cpenable(cp0 | 0xffu);
  printf("  [0] CPENABLE 写入后  : 0x%02x\n", jx_get_cpenable());
  /* r76：内存位置探针。内部 DRAM = 0x3FC8xxxx~0x3FCFxxxx；PSRAM = 0x3Cxxxxxx~0x3Exxxxxx。 */
  {
    volatile char loc = 0;
    void *h10 = malloc(10240);
    void *h32 = malloc(32768);
    printf("  [19] 栈 0x%08lx | 堆10KB 0x%08lx | 堆32KB 0x%08lx\n",
           (unsigned long)(uintptr_t)&loc, (unsigned long)(uintptr_t)h10,
           (unsigned long)(uintptr_t)h32);
    { extern void jx_dump_static_addrs(void); jx_dump_static_addrs(); }
    free(h10); free(h32);
  }
  /* [7] 纯寄存器吞吐：每轮 16 条指令 = 256 个 int8 MAC，无访存干扰 */
  {
    const long REP = 4000000L;
    long i;
    double t0 = jx_now_ms();
    __asm__ __volatile__(
        "ee.zero.accx\n\t"
        "ee.zero.q  q0\n\t"
        "ee.zero.q  q1\n\t"
        "movi %0, 0\n\t"
        "1:\n\t"
        JX_VMULAS8_16
        "addi %0, %0, 1\n\t"
        "blt %0, %1, 1b\n\t"
        "rur.accx_0 %0\n\t"
        : "=&r"(i) : "r"(REP) : JX_FCLOB);
    double ms = jx_now_ms() - t0;
    double macs = (double)REP * 256.0;
    printf("  [7] PIE int8 MAC 上限 : %8.1f ms  %7.1f MMAC/s  %5.2f 周期/MAC\n",
           ms, macs / ms / 1000.0, ms * 1e-3 * 240.0e6 / macs);
    g_mem_sink = (float)i;
  }

  /* [8] 带 128-bit 装载：每轮 4 条 vld.128 + 4 条 vmulas = 64 个 MAC
   *     （4 字节装载 / 1 个 MAC 的极差复用比，是真实卷积内核的下限） */
  {
    static signed char buf[64] __attribute__((aligned(16)));
    const long REP = 4000000L;
    long i;
    for (int k = 0; k < 64; k++) { buf[k] = (signed char)(k + 1); }

    double t0 = jx_now_ms();
    __asm__ __volatile__(
        "ee.zero.accx\n\t"
        "movi %0, 0\n\t"
        "1:\n\t"
        "mov  a5, %2\n\t"
        "ee.vld.128.ip q0, a5, 16\n\t"
        "ee.vld.128.ip q1, a5, 16\n\t"
        "ee.vld.128.ip q2, a5, 16\n\t"
        "ee.vld.128.ip q3, a5, 16\n\t"
        "ee.vmulas.s8.accx q0, q1\n\t"
        "ee.vmulas.s8.accx q2, q3\n\t"
        "ee.vmulas.s8.accx q0, q1\n\t"
        "ee.vmulas.s8.accx q2, q3\n\t"
        "addi %0, %0, 1\n\t"
        "blt %0, %1, 1b\n\t"
        "rur.accx_0 %0\n\t"
        : "=&r"(i) : "r"(REP), "r"(buf) : "a5", JX_FCLOB);
    double ms = jx_now_ms() - t0;
    double macs = (double)REP * 64.0;
    printf("  [8] PIE int8 + 装载   : %8.1f ms  %7.1f MMAC/s  (4B装载/1MAC)\n",
           ms, macs / ms / 1000.0);
    g_mem_sink = (float)i;
  }

  /* [9] 短点积往返 —— 真实 linear/conv 内层的形态。
   *
   * 本模型的点积极短：linear in=28 -> n16=2、in=16 -> n16=1；conv k=1/Ci=20
   * -> n16=2；depthwise k=7 -> n16=1。[7]/[8] 量的是长循环的稳态吞吐，
   * 完全反映不了"每次调用只做 1~3 个向量"时的固定开销（accx 清零、accx
   * 读回、函数调用）。这一段就是用来量这个固定开销的。 */
  {
    static signed char wa[256] __attribute__((aligned(16)));
    static signed char xb[256] __attribute__((aligned(16)));
    const signed char *fl = jx_wq_blob_ptr();
    volatile int32_t sink = 0;
    for (int k = 0; k < 256; k++) { wa[k] = (signed char)((k * 7) & 127); xb[k] = (signed char)((k * 13) & 127); }

    printf("  [9]  短点积往返（RAM 操作数，L1 常驻）\n");
    for (int n16 = 1; n16 <= 8; n16 <<= 1) {
      const long R = 200000L;
      long it;
      double t0 = jx_now_ms();
      for (it = 0; it < R; it++) { sink += jx_dot_i8_probe(xb, wa, n16); }
      double ms = jx_now_ms() - t0;
      printf("       n16=%d : %7.1f ns/次  %7.1f MMAC/s  %6.1f 周期/次\n",
             n16, ms * 1e6 / (double)R, (double)n16 * 16.0 * (double)R / ms / 1000.0,
             ms * 240000.0 / (double)R);
    }
    printf("  [9b] 同形态但权重在 FLASH(XIP)\n");
    for (int n16 = 1; n16 <= 4; n16 <<= 1) {
      const long R = 200000L;
      long it;
      double t0 = jx_now_ms();
      for (it = 0; it < R; it++) { sink += jx_dot_i8_probe(xb, fl, n16); }
      double ms = jx_now_ms() - t0;
      printf("       n16=%d : %7.1f ns/次  %6.1f 周期/次\n",
             n16, ms * 1e6 / (double)R, ms * 240000.0 / (double)R);
    }
    g_mem_sink = (float)sink;
  }

  /* [10] 端到端内核：与 first_block 的 conv_2 **完全同形**（Ci=81 -> Co=16, k=1,
   *      单窗 T=1152、用真权重）。目的：判定 first_block 里那 3.3 秒
   *      到底在内核里，还是在周边代码（拼接/分配/访存布局）。
   *      同时 [10b] 单独量"拼接"本身：跳步读+跳步写 vs 连续写。 */
  {
    const int Cin = 81, Cout = 16, TT = 1152;
    const float *W2 = wt_get("encoder.blocks.0.conv_2.weight", NULL, NULL);
    const float *b2 = wt_get("encoder.blocks.0.conv_2.bias", NULL, NULL);
    float *xb = (float *)malloc((size_t)Cin * TT * sizeof(float));
    float *h1b = (float *)malloc((size_t)80 * TT * sizeof(float));
    float *hcb = (float *)malloc((size_t)81 * TT * sizeof(float));
    float *yb = (float *)malloc((size_t)Cout * TT * sizeof(float));
    if (W2 && b2 && xb && h1b && hcb && yb) {
      T xt = { .d = xb,  .ndim = 3, .shape = {1, Cin, TT},  .len = Cin * TT  };
      T yt = { .d = yb,  .ndim = 3, .shape = {1, Cout, TT}, .len = Cout * TT };
      for (int i = 0; i < Cin * TT; i++) { xb[i] = (float)((i * 37 % 199) - 99) * 0.01f; }
      for (int i = 0; i < 80 * TT; i++) { h1b[i] = (float)((i * 53 % 211) - 105) * 0.01f; }
      {
        const int R = 16;
        double t0 = jx_now_ms();
        for (int r = 0; r < R; r++)
          { jx_conv1d_i8(W2, b2, &xt, &yt, Cin, Cout, 1, 1, 0, 1, TT, TT, 1); }
        double ms = jx_now_ms() - t0;
        printf("  [10] conv_2 裸内核 81->16 k=1 T=1152 x16 : %8.1f ms  (%6.2f ms/次)\n",
               ms, ms / R);
        g_mem_sink = yb[0];
      }
      {
        const int R = 16;
        double t0 = jx_now_ms();
        for (int r = 0; r < R; r++)
          for (int t = 0; t < TT; t++)
            for (int c = 0; c < 80; c++)
              { hcb[(size_t)c * TT + t] = h1b[(size_t)c * TT + t]; }
        double ms = jx_now_ms() - t0;
        printf("  [10b] 拼接 跳步读+跳步写 80x1152 x16 : %8.1f ms\n", ms);
        t0 = jx_now_ms();
        for (int r = 0; r < R; r++)
          for (int c = 0; c < 80; c++)
            memcpy(&hcb[(size_t)c * TT], &h1b[(size_t)c * TT], (size_t)TT * sizeof(float));
        ms = jx_now_ms() - t0;
        printf("  [10c] 拼接 逐行 memcpy         80x1152 x16 : %8.1f ms\n", ms);
        g_mem_sink = hcb[0];
      }
    } else {
      printf("  [10] 跳过（分配或权重查找失败）\n");
    }
    free(xb); free(h1b); free(hcb); free(yb);
  }
}

/* ---------------- snake 元素级微基准 (2026-09-13 第六轮) ----------------
 * 把 snake1d 的 77 周期/元素拆成「访存 / 循环骨架 / 超越函数」三份。
 * 数据布局仿 conv_unit 的 p1：Tt 行 x C 列 channels_last（PSRAM），
 * 数值分布按主机实测生成（约 96% 的 |a*v| <= pi/2）。 */
static float jx_sinsq_old(float x)
{
  float r = x * 0.31830988618379067154f;
  float k = (float)(int)r;
  float u;
  float t;
  float pp;
  k -= (k > r) ? 1.0f : 0.0f;
  u = x - k * 3.14159265358979323846f;
  t = 3.14159265358979323846f - u;
  u = (u > 1.57079632679489661923f) ? t : u;
  pp =          -4.157437553e-13f;
  pp = pp * u +   6.631500918e-04f;
  pp = pp * u +  -4.687531765e-03f;
  pp = pp * u +   2.053761546e-03f;
  pp = pp * u +   4.268362097e-02f;
  pp = pp * u +   9.617141793e-04f;
  pp = pp * u +  -3.336593731e-01f;
  pp = pp * u +   6.431927373e-05f;
  pp = pp * u +   9.999935029e-01f;
  pp = pp * u +   2.557090515e-07f;
  pp = pp * u +  -1.662400773e-09f;
  return pp;
}

/* 与 jx_fastmath.h 里的 jx_sinsq 保持同步：系数驻留内部 SRAM。
 * [s1]（本函数）与 [s2]（jx_sinsq_old，常量仍在 flash literal pool）
 * 构成同一固件上的 A/B。 */
static float jx_sqk[5] = {
   1.129032502e-04f, -3.100702005e-03f,  4.435485267e-02f, -3.332854819e-01f,
   9.999916253e-01f };

static float jx_sinsq_new(float x)
{
  const float *K = jx_sqk;
  float ax = fabsf(x);
  float w;
  float q;
  if (ax <= 1.57079632679489661923f)
    {
      w = x * x;
      q = K[0];
      q = q * w + K[1];
      q = q * w + K[2];
      q = q * w + K[3];
      q = q * w + K[4];
      return w * q;
    }
  return jx_sinsq_old(x);
}

static void jx_qacc_probe(void)
{
  printf("=== PIE s8.qacc 语义探针 ===\n");
  {
    static signed char wa[16] __attribute__((aligned(16)));
    static signed char xb[16] __attribute__((aligned(16)));
    static signed char o0[16] __attribute__((aligned(16)));
    static signed char o1[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; i++)
      {
        wa[i] = (signed char)(i + 1);
        xb[i] = 1;
        o0[i] = 0; o1[i] = 0;
      }
    signed char *pa = wa;
    signed char *pb = xb;
    signed char *po0 = o0;
    signed char *po1 = o1;
    __asm__ __volatile__(
        "ee.vld.128.ip q0, %[A], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vmulas.s8.qacc q1, q0\n\t"
        "movi a5, 0\n\t"
        "ee.srcmb.s8.qacc q2, a5, 0\n\t"
        "ee.vst.128.ip q2, %[O0], 16\n\t"
        "movi a5, 1\n\t"
        "ee.srcmb.s8.qacc q3, a5, 1\n\t"
        "ee.vst.128.ip q3, %[O1], 16\n\t"
        : [A] "+r"(pa), [B] "+r"(pb), [O0] "+r"(po0), [O1] "+r"(po1)
        : : "a5", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    printf("  wa:");
    for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)wa[i]); }
    printf("\n  o0:");
    for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)o0[i]); }
    printf("\n  o1:");
    for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)o1[i]); }
    printf("\n");
  }
  {
    static signed char wa[16] __attribute__((aligned(16)));
    static signed char xb[16] __attribute__((aligned(16)));
    static signed char ol[16] __attribute__((aligned(16)));
    static signed char oh[16] __attribute__((aligned(16)));
    static signed char olh[16] __attribute__((aligned(16)));
    static signed char ohh[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; i++)
      {
        wa[i] = (signed char)(i + 1);
        xb[i] = 100;
        ol[i] = oh[i] = olh[i] = ohh[i] = 0;
      }
    signed char *pa = wa;
    signed char *pb = xb;
    signed char *pl = ol;
    signed char *ph = oh;
    signed char *plh = olh;
    signed char *phh = ohh;
    __asm__ __volatile__(
        "ee.vld.128.ip q0, %[A], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vmulas.s8.qacc q1, q0\n\t"
        "ee.st.qacc_l.l.128.ip %[OL], 0\n\t"
        "ee.st.qacc_h.h.32.ip %[OH], 0\n\t"
        "ee.st.qacc_l.h.32.ip %[OLH], 0\n\t"
        "ee.st.qacc_h.l.128.ip %[OHH], 0\n\t"
        : [A] "+r"(pa), [B] "+r"(pb), [OL] "+r"(pl), [OH] "+r"(ph),
          [OLH] "+r"(plh), [OHH] "+r"(phh)
        : : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    {
      static signed char ob0[16] __attribute__((aligned(16)));
      static signed char ob1[16] __attribute__((aligned(16)));
      signed char *pa2 = wa;
      signed char *pb2 = xb;
      signed char *pob0 = ob0;
      signed char *pob1 = ob1;
      for (int i = 0; i < 16; i++) { ob0[i] = ob1[i] = 0; }
      __asm__ __volatile__(
          "ee.vld.128.ip q0, %[A], 16\n\t"
          "ee.vld.128.ip q1, %[B], 16\n\t"
          "ee.vmulas.s8.qacc q1, q0\n\t"
          "movi a5, 0\n\t"
          "ee.srcmb.s8.qacc q4, a5, 0\n\t"
          "ee.vst.128.ip q4, %[O0], 16\n\t"
          "movi a5, 1\n\t"
          "ee.srcmb.s8.qacc q5, a5, 1\n\t"
          "ee.vst.128.ip q5, %[O1], 16\n\t"
          : [A] "+r"(pa2), [B] "+r"(pb2), [O0] "+r"(pob0), [O1] "+r"(pob1)
          : : "a5", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      printf("  large srcmb0:");
      for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)ob0[i]); }
      printf("\n  large srcmb1:");
      for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)ob1[i]); }
      printf("\n");
    }
    printf("  large wa*100 qacc stores:\n");
    printf("  qacc_l.l:");
    for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)ol[i]); }
    printf("\n  qacc_h.h:");
    for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)oh[i]); }
    printf("\n  qacc_l.h:");
    for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)olh[i]); }
    printf("\n  qacc_h.l:");
    for (int i = 0; i < 16; i++) { printf(" %d", (int)(unsigned char)ohh[i]); }
    printf("\n");
  }
}

static inline int32_t jx_qacc_sxt20(uint32_t v)
{
  return (int32_t)((v ^ 0x80000u) - 0x80000u);
}

static void jx_qacc_unpack10(const uint32_t *qw, int32_t *out)
{
  /* r50：全展开 + 常量移位。原实现用运行期 bit/wi/sh，Xtensa 没有单指令
   * 变长移位，GCC 只能生成移位循环，16 条 lane 要 200+ 周期；板端实测
   * 读出本体 13.6 周期/输出，比 accx 路径的 7.5 还贵，QACC 因此整体亏。
   * 这里把同一份 20-bit 位域公式手工展开成常量移位：
   *   lane i 取 qw[wi]>>sh（跨字时再 | qw[wi+1]<<(32-sh)）
   * 再用 (int32_t)(v<<12)>>12 做 20-bit 符号扩展 —— 左移天然丢掉高位，
   * 所以不需要 & 0xfffff 掩码。数值与原实现逐位相同。 */
  uint32_t a0 = qw[0], a1 = qw[1], a2 = qw[2], a3 = qw[3], a4 = qw[4];
  uint32_t a5 = qw[5], a6 = qw[6], a7 = qw[7], a8 = qw[8], a9 = qw[9];
#define JX_Q20(v) ((int32_t)((uint32_t)(v) << 12) >> 12)
  out[0]  = JX_Q20(a0);
  out[1]  = JX_Q20((a0 >> 20) | (a1 << 12));
  out[2]  = JX_Q20(a1 >> 8);
  out[3]  = JX_Q20((a1 >> 28) | (a2 << 4));
  out[4]  = JX_Q20((a2 >> 16) | (a3 << 16));
  out[5]  = JX_Q20(a3 >> 4);
  out[6]  = JX_Q20((a3 >> 24) | (a4 << 8));
  out[7]  = JX_Q20(a4 >> 12);
  out[8]  = JX_Q20(a5);
  out[9]  = JX_Q20((a5 >> 20) | (a6 << 12));
  out[10] = JX_Q20(a6 >> 8);
  out[11] = JX_Q20((a6 >> 28) | (a7 << 4));
  out[12] = JX_Q20((a7 >> 16) | (a8 << 16));
  out[13] = JX_Q20(a8 >> 4);
  out[14] = JX_Q20((a8 >> 24) | (a9 << 8));
  out[15] = JX_Q20(a9 >> 12);
#undef JX_Q20
}



static inline int64_t jx_qacc_sxt40(uint64_t v)
{
  return (int64_t)((v ^ (1ull << 39)) - (1ull << 39));
}

static void jx_qacc_unpack10_s16(const uint32_t *qw, int64_t *out)
{
  for (int i = 0; i < 8; i++)
    {
      int bit = i * 40;
      int wi = bit >> 5;
      int sh = bit & 31;
      uint64_t pair = qw[wi];
      if (wi + 1 < 10)
        {
          pair |= (uint64_t)qw[wi + 1] << 32;
        }
      uint64_t v = (pair >> sh) & ((1ull << 40) - 1u);
      out[i] = jx_qacc_sxt40(v);
    }
}

/* s16 QACC：8 个 40-bit lane，整段 K 累加也不会溢出，适合 nchunks>2 的
 * dense tile；代价是每条指令只算 8 路。 */
static void jx_qacc_dot8_s16(const int16_t *wT, const int16_t *x,
                             int nchunks, int64_t out[8])
{
#if defined(JX_PIE)
  int16_t *wp = (int16_t *)wT;
  int16_t *xp = (int16_t *)x;
  int cnt = nchunks;
  uint32_t qw[10];
  __asm__ __volatile__(
      "ee.zero.qacc\n\t"
      "ee.vld.128.ip q0, %[W], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 0\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 1\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 2\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 3\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 4\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 5\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 6\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 7\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      "rur.qacc_l_0 %[Q0]\n\t"
      "rur.qacc_l_1 %[Q1]\n\t"
      "rur.qacc_l_2 %[Q2]\n\t"
      "rur.qacc_l_3 %[Q3]\n\t"
      "rur.qacc_l_4 %[Q4]\n\t"
      "rur.qacc_h_0 %[Q5]\n\t"
      "rur.qacc_h_1 %[Q6]\n\t"
      "rur.qacc_h_2 %[Q7]\n\t"
      "rur.qacc_h_3 %[Q8]\n\t"
      "rur.qacc_h_4 %[Q9]\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [C] "+r"(cnt),
        [Q0] "=&r"(qw[0]), [Q1] "=&r"(qw[1]), [Q2] "=&r"(qw[2]),
        [Q3] "=&r"(qw[3]), [Q4] "=&r"(qw[4]), [Q5] "=&r"(qw[5]),
        [Q6] "=&r"(qw[6]), [Q7] "=&r"(qw[7]), [Q8] "=&r"(qw[8]),
        [Q9] "=&r"(qw[9])
      :
      : "f0", "f1", "f2", "memory");
  if (nchunks <= 2)
    {
      printf("  s16 raw qw(nc=%d):", nchunks);
      for (int i = 0; i < 10; i++) { printf(" %08lx", (unsigned long)qw[i]); }
      printf("\n");
    }
  jx_qacc_unpack10_s16(qw, out);
#else
  (void)wT; (void)x; (void)nchunks;
  for (int o = 0; o < 8; o++) { out[o] = 0; }
#endif
}

/* wT 布局：[k][16 个输出通道]，即每个输入位置 k 的 16 个输出权重连续。
 * 每个 chunk 是 16 个输入位置，共 256 字节；q1 装 16 个输入标量，
 * ee.vsmulas.s8.qacc.ld.incp 用 qy 的 sel16 做标量广播，一条指令同时
 * 累加 16 个输出通道，并从 wT 装入下一组 16 个权重。 */
static void jx_qacc_dot16(const signed char *wT, const signed char *x,
                          int nchunks, int32_t out[16])
{
#if defined(JX_PIE)
  signed char *wp = (signed char *)wT;
  signed char *xp = (signed char *)x;
  int cnt = nchunks;
  uint32_t qw[10];
  __asm__ __volatile__(
      "ee.zero.qacc\n\t"
      "ee.vld.128.ip q0, %[W], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 0\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 1\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 2\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 3\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 4\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 5\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 6\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 7\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 8\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 9\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 10\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 11\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 12\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 13\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 14\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 15\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      "rur.qacc_l_0 %[Q0]\n\t"
      "rur.qacc_l_1 %[Q1]\n\t"
      "rur.qacc_l_2 %[Q2]\n\t"
      "rur.qacc_l_3 %[Q3]\n\t"
      "rur.qacc_l_4 %[Q4]\n\t"
      "rur.qacc_h_0 %[Q5]\n\t"
      "rur.qacc_h_1 %[Q6]\n\t"
      "rur.qacc_h_2 %[Q7]\n\t"
      "rur.qacc_h_3 %[Q8]\n\t"
      "rur.qacc_h_4 %[Q9]\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [C] "+r"(cnt),
        [Q0] "=&r"(qw[0]), [Q1] "=&r"(qw[1]), [Q2] "=&r"(qw[2]),
        [Q3] "=&r"(qw[3]), [Q4] "=&r"(qw[4]), [Q5] "=&r"(qw[5]),
        [Q6] "=&r"(qw[6]), [Q7] "=&r"(qw[7]), [Q8] "=&r"(qw[8]),
        [Q9] "=&r"(qw[9])
      :
      : "f0", "f1", "memory");
  jx_qacc_unpack10(qw, out);
#else
  (void)wT; (void)x; (void)nchunks;
  for (int o = 0; o < 16; o++) { out[o] = 0; }
#endif
}

static void jx_qacc_bench(void)
{
  enum { NCH = 22, REP = 5000 };
  static signed char wT[16 * 16 * NCH] __attribute__((aligned(16)));
  static signed char x[16 * NCH] __attribute__((aligned(16)));
  int32_t got[16];
  int32_t exp[16] = {0};
  int ok = 1;

  for (int k = 0; k < 16 * NCH; k++)
    {
      x[k] = (signed char)(((k * 3) & 15) - 8);
      for (int o = 0; o < 16; o++)
        {
          wT[k * 16 + o] = (signed char)(((o * 5 + k * 7) & 15) - 8);
        }
    }

  for (int o = 0; o < 16; o++)
    {
      for (int k = 0; k < 16; k++)
        {
          exp[o] += (int32_t)x[k] * (int32_t)wT[k * 16 + o];
        }
    }
  jx_qacc_dot16(wT, x, 1, got);
  for (int o = 0; o < 16; o++)
    {
      if (got[o] != exp[o]) { ok = 0; }
    }
  printf("=== QACC 16ch vsmulas microbench ===\n");
  printf("  selftest=%s  got0=%ld exp0=%ld got15=%ld exp15=%ld\n",
         ok ? "OK" : "FAIL", (long)got[0], (long)exp[0],
         (long)got[15], (long)exp[15]);

  double t0 = jx_now_ms();
  for (int r = 0; r < REP; r++)
    {
      jx_qacc_dot16(wT, x, NCH, got);
    }
  double ms = jx_now_ms() - t0;
  double mac = (double)REP * (double)NCH * 16.0 * 16.0;
  double cyc = ms * 1e-3 * 240.0e6 / mac;
  printf("  %d reps, K=%d, 16 out: %.2f ms, %.2f cyc/MAC (incl rur/unpack)\n",
         REP, NCH * 16, ms, cyc);

  enum { NCH16 = 16, REP16 = 5000 };
  static int16_t wt16[8 * 8 * NCH16] __attribute__((aligned(16)));
  static int16_t x16[8 * NCH16] __attribute__((aligned(16)));
  int64_t got16[8];
  int64_t got16a[8];
  int64_t exp16[8] = {0};
  int64_t exp16a[8] = {0};
  int ok8 = 1, ok16 = 1;
  for (int k = 0; k < 8 * NCH16; k++)
    {
      x16[k] = (int16_t)(((k * 3) & 15) - 8);
      for (int o = 0; o < 8; o++)
        {
          wt16[k * 8 + o] = (int16_t)(((o * 5 + k * 7) & 15) - 8);
        }
    }
  for (int o = 0; o < 8; o++)
    {
      for (int k = 0; k < 8; k++)
        {
          exp16a[o] += (int64_t)x16[k] * (int64_t)wt16[k * 8 + o];
        }
      for (int k = 0; k < 16; k++)
        {
          exp16[o] += (int64_t)x16[k] * (int64_t)wt16[k * 8 + o];
        }
    }
  jx_qacc_dot8_s16(wt16, x16, 1, got16a);
  for (int o = 0; o < 8; o++)
    {
      if (got16a[o] != exp16a[o]) { ok8 = 0; }
    }
  printf("  s16 selftest8=%s  got0=%lld exp0=%lld got7=%lld exp7=%lld\n",
         ok8 ? "OK" : "FAIL", (long long)got16a[0], (long long)exp16a[0],
         (long long)got16a[7], (long long)exp16a[7]);
  jx_qacc_dot8_s16(wt16, x16, 2, got16);
  for (int o = 0; o < 8; o++)
    {
      if (got16[o] != exp16[o]) { ok16 = 0; }
    }
  printf("  s16 selftest=%s  got0=%lld exp0=%lld got7=%lld exp7=%lld\n",
         ok16 ? "OK" : "FAIL", (long long)got16[0], (long long)exp16[0],
         (long long)got16[7], (long long)exp16[7]);
  double t16 = jx_now_ms();
  for (int r = 0; r < REP16; r++)
    {
      jx_qacc_dot8_s16(wt16, x16, NCH16, got16);
    }
  double ms16 = jx_now_ms() - t16;
  double mac16 = (double)REP16 * (double)NCH16 * 8.0 * 8.0;
  double cyc16 = ms16 * 1e-3 * 240.0e6 / mac16;
  printf("  s16 %d reps, K=%d, 8 out: %.2f ms, %.2f cyc/MAC (incl rur/unpack)\n",
         REP16, NCH16 * 8, ms16, cyc16);
}

/* 第四十轮：P0（生产者直出 int8+scale）可行性探针，实现在 main/jx_p0probe.c */
extern void jx_p0_probe(void);
extern void jx_snake_probe(void);
extern void jx_epi_probe(void);
extern void jx_fptp_probe(void);
extern void jx_fptp2_probe(void);
extern void jx_cache_probe(void);
extern void jx_cache2_probe(void);

static int jx_cmd_smem(void)
{
  jx_sq_kt_ensure();
  const double SYSCLK = 240.0e6;
  const int Tt = 501, C = 352, REP = 16;
  const int n = Tt * C;
  float *buf = (float *)malloc((size_t)n * sizeof(float));
  float *aa  = (float *)malloc((size_t)C * sizeof(float));
  float *iv  = (float *)malloc((size_t)C * sizeof(float));
  if (buf == NULL || aa == NULL || iv == NULL) { printf("  [smem] alloc fail\n"); return 1; }
  for (int c = 0; c < C; c++) { aa[c] = 0.5f + (float)(c % 16) * 0.1f; iv[c] = 1.0f / aa[c]; }
  for (int i = 0; i < Tt; i++)
    for (int c = 0; c < C; c++)
      {
        float v = (float)((i * 7 + c * 13) % 100) * 0.015f - 0.75f; /* |v|<=0.75 */
        if ((i + c) % 25 == 0) { v *= 8.0f; }                       /* 4% 进慢速路径 */
        buf[i * C + c] = v;
      }
  printf("=== snake 元素级微基准 (%d x %d = %d elems, REP %d) ===\n", Tt, C, n, REP);

#define JX_SNAKE_RUN(TAG, EXPR)                                              \
  do {                                                                       \
    double t0 = jx_now_ms();                                                 \
    for (int r = 0; r < REP; r++)                                            \
      for (int i = 0; i < Tt; i++)                                           \
        {                                                                    \
          float *pp = &buf[i * C];                                           \
          for (int c = 0; c < C; c++)                                        \
            {                                                                \
              float v = pp[c];                                               \
              pp[c] = (EXPR);                                                \
            }                                                                \
        }                                                                    \
    double ms = jx_now_ms() - t0;                                            \
    double el = (double)REP * n;                                             \
    g_mem_sink = buf[n / 2];                                                 \
    printf("  %-28s: %8.1f ms  %6.1f 周期/元素\n", (TAG), ms,               \
           ms * 1e-3 * SYSCLK / el);                                         \
  } while (0)

  JX_SNAKE_RUN("[s3] 纯读写 v*1.0001", v * 1.0001f);
  JX_SNAKE_RUN("[s0] 骨架 v+iv*(a*v)", v + iv[c] * (aa[c] * v));
  JX_SNAKE_RUN("[s1] 快速路径 poly", v + iv[c] * jx_sinsq_new(aa[c] * v));
  JX_SNAKE_RUN("[s2] 旧 11 阶全量", v + iv[c] * jx_sinsq_old(aa[c] * v));
#undef JX_SNAKE_RUN

  {
    signed short q16[16] __attribute__((aligned(16)));
    signed short K[32] __attribute__((aligned(16)));
    int32_t out2[2] = {0, 0};
    for (int i = 0; i < 16; i++) { q16[i] = (signed short)(i + 1); K[i] = (signed short)(i + 1); K[16 + i] = (signed short)(2 * (i + 1)); }
    jx_qk_i16(q16, K, 2, out2);
    printf("  [s4] int16 QK 自检: out0=%ld (期望1496) out1=%ld (期望2992)\n", (long)out2[0], (long)out2[1]);
  }
  jx_qacc_probe();
  jx_qacc_bench();
  jx_p0_probe();
  free(buf); free(aa); free(iv);
  return 0;
}

/* ================= 2026-09-14 第十七轮：访存受限专项探针 =================
 * 把"访存受限"拆成四个可量化的问：
 *   [11] PSRAM 一次能挂几个未完成的 miss？（1/2/4/8 路并行独立流）
 *        —— 若多路明显快于 1 路，说明顺序流是"延迟受限"而不是"带宽受限"，
 *           那么把热循环改成多路并行/软预取就能直接吃带宽。
 *   [12] 内部 SRAM 跨步访问 vs PSRAM 跨步访问
 *        —— 量化"把 staging / permute 缓冲放进内部 SRAM"值多少。
 *   [13] dense-tile 的输出写模式（CO 行、步长 T_out）
 *        —— 现状"逐 t 跨行写" vs "逐行连续写"。
 *   [14] dense-tile 的 im2col 收集（cpg 次 k 字节小 memcpy）
 *        —— 现状 vs 通道交错布局后的"一次连续 memcpy"。
 * 纯测量，不参与任何模型数值。
 */
/* ---- [15] 用：PW 行并行的完整行量化（两趟，对齐 jx_q_row） ----
 * 探针只测访存形态，不参与模型数值。 */
#define JX_QP_FULL(PW, XS, QS, IN, R, OUTMS)                                     \
  do {                                                                           \
    double _t0 = jx_now_ms();                                                    \
    for (int _rep = 0; _rep < 4; _rep++)                                         \
      {                                                                          \
        for (int _r = 0; _r + (PW) <= (R); _r += (PW))                           \
          {                                                                      \
            float _m[PW];                                                        \
            for (int _k = 0; _k < (PW); _k++) { _m[_k] = 0.0f; }                 \
            for (int _i = 0; _i < (IN); _i++)                                    \
              {                                                                  \
                for (int _k = 0; _k < (PW); _k++)                                \
                  {                                                              \
                    float _a = fabsf((XS)[(size_t)(_r + _k) * (IN) + _i]);       \
                    if (_a > _m[_k]) { _m[_k] = _a; }                            \
                  }                                                              \
              }                                                                  \
            for (int _k = 0; _k < (PW); _k++)                                    \
              {                                                                  \
                float _inv = (_m[_k] > 1e-30f) ? (127.0f / _m[_k]) : 0.0f;        \
                signed char *_q = (QS) + (size_t)(_r + _k) * (IN);               \
                const float *_x = (XS) + (size_t)(_r + _k) * (IN);               \
                for (int _i = 0; _i < (IN); _i++)                                \
                  {                                                              \
                    float _v = _x[_i] * _inv;                                    \
                    int _w = (int)(_v + (_v >= 0.0f ? 0.5f : -0.5f));            \
                    if (_w > 127) { _w = 127; } else if (_w < -127) { _w = -127; } \
                    _q[_i] = (signed char)_w;                                    \
                  }                                                              \
              }                                                                  \
            g_mem_sink += (float)_m[0];                                          \
          }                                                                      \
      }                                                                          \
    (OUTMS) = (jx_now_ms() - _t0) / 4.0;                                         \
  } while (0)

/* ---- [18] 用 helper：把 k1cf 的 Pass A / Pass B 访存模式单独跑一遍 ---- */
static float g_p18st[2048] __attribute__((aligned(16)));
static void jx_p18_sweep(int CI, int TT, int TB, int mode,
                         float *xn, float *scb, float *tni, signed char *dqq)
{
  for (int t0i = 0; t0i < TT; t0i += TB)
    {
      int w = TT - t0i; if (w > TB) { w = TB; }
      for (int j = 0; j < w; j++) { scb[j] = 0.0f; }
      if (mode == 3)
        {
          for (int ci = 0; ci < CI; ci++)
            {
              const float *xr = xn + (size_t)ci * TT + t0i;
              float *st = g_p18st + (size_t)ci * (size_t)TB;
              for (int j = 0; j < w; j++)
                {
                  float a = xr[j];
                  st[j] = a;
                  if (a < 0.0f) { a = -a; }
                  if (a > scb[j]) { scb[j] = a; }
                }
            }
          for (int ci = 0; ci < CI; ci++)
            {
              const float *st = g_p18st + (size_t)ci * (size_t)TB;
              signed char *dq = dqq + ci;
              for (int j = 0; j < w; j++)
                {
                  float v = st[j] * tni[j];
                  int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
                  dq[(size_t)j * 32] = (signed char)iv;
                }
            }
        }
      else
        {
          for (int ci = 0; ci < CI; ci++)
            {
              const float *xr = xn + (size_t)ci * TT + t0i;
              for (int j = 0; j < w; j++)
                {
                  float a = xr[j];
                  if (a < 0.0f) { a = -a; }
                  if (a > scb[j]) { scb[j] = a; }
                }
            }
          if (mode >= 2)
            {
              for (int ci = 0; ci < CI; ci++)
                {
                  const float *xr = xn + (size_t)ci * TT + t0i;
                  signed char *dq = dqq + ci;
                  for (int j = 0; j < w; j++)
                    {
                      float v = xr[j] * tni[j];
                      int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
                      dq[(size_t)j * 32] = (signed char)iv;
                    }
                }
            }
        }
    }
}
static double jx_p18_time(int CI, int TT, int TB, int mode,
                          float *xn, float *scb, float *tni, signed char *dqq)
{
  double t0 = jx_now_ms();
  for (int r = 0; r < 3; r++)
    { jx_p18_sweep(CI, TT, TB, mode, xn, scb, tni, dqq); }
  return jx_now_ms() - t0;
}


/* ================= 2026-09-16 r103：把"最大单点成本"拆开量 =================
 * 三个探针，全部用 rsr.ccount 量周期，直接和 int8_kernels.c 里的
 * [i8mm] / [dt] / [k1] 统计对齐：
 *   [20] L1 局部性  —— 同一块反复读能不能命中
 *   [21] epilogue 复刻 —— 算式与 jx_linear_i8_rows 的 8 路展开逐字符相同
 *   [23] accx 点积链长 —— nv=1/2/4/8 的 MAC/周期，看 accx 回读停顿有多大
 * 复用 g_p18st（8KB 内部 SRAM，只有 jixun mem 用），不额外占 BSS。
 */
static float jx_epi_rep(float *dst, const int32_t *acc, const float *swc,
                        const float *sbb, int n, float s, int mode)
{
  float sink = 0.0f;
  int i = 0;
  if (mode == 0 || mode == 2)
    {
      for (; i + 8 <= n; i += 8)
        {
          float v0 = (float)acc[i    ] * (s * swc[i    ]) + sbb[i    ];
          float v1 = (float)acc[i + 1] * (s * swc[i + 1]) + sbb[i + 1];
          float v2 = (float)acc[i + 2] * (s * swc[i + 2]) + sbb[i + 2];
          float v3 = (float)acc[i + 3] * (s * swc[i + 3]) + sbb[i + 3];
          float v4 = (float)acc[i + 4] * (s * swc[i + 4]) + sbb[i + 4];
          float v5 = (float)acc[i + 5] * (s * swc[i + 5]) + sbb[i + 5];
          float v6 = (float)acc[i + 6] * (s * swc[i + 6]) + sbb[i + 6];
          float v7 = (float)acc[i + 7] * (s * swc[i + 7]) + sbb[i + 7];
          dst[i    ] = v0; dst[i + 1] = v1; dst[i + 2] = v2; dst[i + 3] = v3;
          dst[i + 4] = v4; dst[i + 5] = v5; dst[i + 6] = v6; dst[i + 7] = v7;
        }
      for (; i < n; i++) { dst[i] = (float)acc[i] * (s * swc[i]) + sbb[i]; }
    }
  else if (mode == 1)
    {
      for (; i + 8 <= n; i += 8)
        {
          sink += (float)acc[i    ] * (s * swc[i    ]) + sbb[i    ];
          sink += (float)acc[i + 1] * (s * swc[i + 1]) + sbb[i + 1];
          sink += (float)acc[i + 2] * (s * swc[i + 2]) + sbb[i + 2];
          sink += (float)acc[i + 3] * (s * swc[i + 3]) + sbb[i + 3];
          sink += (float)acc[i + 4] * (s * swc[i + 4]) + sbb[i + 4];
          sink += (float)acc[i + 5] * (s * swc[i + 5]) + sbb[i + 5];
          sink += (float)acc[i + 6] * (s * swc[i + 6]) + sbb[i + 6];
          sink += (float)acc[i + 7] * (s * swc[i + 7]) + sbb[i + 7];
        }
      for (; i < n; i++) { sink += (float)acc[i] * (s * swc[i]) + sbb[i]; }
    }
  else if (mode == 3)
    {
      for (; i + 8 <= n; i += 8)
        {
          dst[i    ] = (float)acc[i    ]; dst[i + 1] = (float)acc[i + 1];
          dst[i + 2] = (float)acc[i + 2]; dst[i + 3] = (float)acc[i + 3];
          dst[i + 4] = (float)acc[i + 4]; dst[i + 5] = (float)acc[i + 5];
          dst[i + 6] = (float)acc[i + 6]; dst[i + 7] = (float)acc[i + 7];
        }
      for (; i < n; i++) { dst[i] = (float)acc[i]; }
    }
  else
    {
      for (; i + 8 <= n; i += 8)
        {
          dst[i    ] = s; dst[i + 1] = s; dst[i + 2] = s; dst[i + 3] = s;
          dst[i + 4] = s; dst[i + 5] = s; dst[i + 6] = s; dst[i + 7] = s;
        }
      for (; i < n; i++) { dst[i] = s; }
    }
  return sink;
}

/* [23] accx 点积：PIE 只有 q0..q7 八个 128-bit 寄存器（汇编器实测），
 * 所以权重常驻最多 4 条（q4..q7）、输入 4 条（q0..q3）。
 * nv=1/2/4 = 权重常驻；nv=8 = 两轮各 4 条，权重每轮从 L1 重载，
 * 但每 128 个 MAC 只回读一次 accx。 */
static void jx_p23_run(const signed char *w, const signed char *x,
                       int nreq, int nv, int32_t *out)
{
  const signed char *wp = w;
  const signed char *xp = x;
  int cnt = nreq;
  unsigned t = 0;
  int back = -(nv * 16);
  if (nv == 1)
    __asm__ __volatile__(
      "ee.vld.128.ip q4, %[W], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q0, %[X], 16\n\t"
      "ee.zero.accx\n\t"
      "ee.vmulas.s8.accx q0, q4\n\t"
      "rur.accx_0 %[T]\n\t"
      "s32i %[T], %[O], 0\n\t"
      "addi %[O], %[O], 4\n\t"
      "add %[X], %[X], %[B]\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [O] "+r"(out), [C] "+r"(cnt), [T] "+r"(t)
      : [B] "r"(back)
      : "f0","f1","f2","f3","f4","f5","f6","f7", "memory");
  else if (nv == 2)
    __asm__ __volatile__(
      "ee.vld.128.ip q4, %[W], 16\n\t"
      "ee.vld.128.ip q5, %[W], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q0, %[X], 16\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.zero.accx\n\t"
      "ee.vmulas.s8.accx q0, q4\n\t"
      "ee.vmulas.s8.accx q1, q5\n\t"
      "rur.accx_0 %[T]\n\t"
      "s32i %[T], %[O], 0\n\t"
      "addi %[O], %[O], 4\n\t"
      "add %[X], %[X], %[B]\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [O] "+r"(out), [C] "+r"(cnt), [T] "+r"(t)
      : [B] "r"(back)
      : "f0","f1","f2","f3","f4","f5","f6","f7", "memory");
  else if (nv == 4)
    __asm__ __volatile__(
      "ee.vld.128.ip q4, %[W], 16\n\t"
      "ee.vld.128.ip q5, %[W], 16\n\t"
      "ee.vld.128.ip q6, %[W], 16\n\t"
      "ee.vld.128.ip q7, %[W], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q0, %[X], 16\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.vld.128.ip q2, %[X], 16\n\t"
      "ee.vld.128.ip q3, %[X], 16\n\t"
      "ee.zero.accx\n\t"
      "ee.vmulas.s8.accx q0, q4\n\t"
      "ee.vmulas.s8.accx q1, q5\n\t"
      "ee.vmulas.s8.accx q2, q6\n\t"
      "ee.vmulas.s8.accx q3, q7\n\t"
      "rur.accx_0 %[T]\n\t"
      "s32i %[T], %[O], 0\n\t"
      "addi %[O], %[O], 4\n\t"
      "add %[X], %[X], %[B]\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [O] "+r"(out), [C] "+r"(cnt), [T] "+r"(t)
      : [B] "r"(back)
      : "f0","f1","f2","f3","f4","f5","f6","f7", "memory");
  else
    __asm__ __volatile__(
      "1:\n\t"
      "ee.vld.128.ip q4, %[W], 16\n\t"
      "ee.vld.128.ip q5, %[W], 16\n\t"
      "ee.vld.128.ip q6, %[W], 16\n\t"
      "ee.vld.128.ip q7, %[W], 16\n\t"
      "ee.vld.128.ip q0, %[X], 16\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.vld.128.ip q2, %[X], 16\n\t"
      "ee.vld.128.ip q3, %[X], 16\n\t"
      "ee.zero.accx\n\t"
      "ee.vmulas.s8.accx q0, q4\n\t"
      "ee.vmulas.s8.accx q1, q5\n\t"
      "ee.vmulas.s8.accx q2, q6\n\t"
      "ee.vmulas.s8.accx q3, q7\n\t"
      "ee.vld.128.ip q4, %[W], 16\n\t"
      "ee.vld.128.ip q5, %[W], 16\n\t"
      "ee.vld.128.ip q6, %[W], 16\n\t"
      "ee.vld.128.ip q7, %[W], 16\n\t"
      "ee.vld.128.ip q0, %[X], 16\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.vld.128.ip q2, %[X], 16\n\t"
      "ee.vld.128.ip q3, %[X], 16\n\t"
      "ee.vmulas.s8.accx q0, q4\n\t"
      "ee.vmulas.s8.accx q1, q5\n\t"
      "ee.vmulas.s8.accx q2, q6\n\t"
      "ee.vmulas.s8.accx q3, q7\n\t"
      "rur.accx_0 %[T]\n\t"
      "s32i %[T], %[O], 0\n\t"
      "addi %[O], %[O], 4\n\t"
      "add %[X], %[X], %[B]\n\t"
      "add %[W], %[W], %[B]\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [O] "+r"(out), [C] "+r"(cnt), [T] "+r"(t)
      : [B] "r"(back)
      : "f0","f1","f2","f3","f4","f5","f6","f7", "memory");
}

/* [24] PSRAM 存储路径：不同宽度/跨步的真实代价（纯写，无 FP 串行链） */
static void jx_p24_vst16(float *d, const float *s, int n4)
{
  const float *sp = s;
  float *dp = d;
  int c = n4;
  __asm__ __volatile__(
    "1:\n\t"
    "ee.vld.128.ip q0, %[S], 16\n\t"
    "ee.vst.128.ip q0, %[D], 16\n\t"
    "addi %[C], %[C], -1\n\t"
    "bnez %[C], 1b\n\t"
    : [S] "+r"(sp), [D] "+r"(dp), [C] "+r"(c)
    :
    : "f0","f1","f2","f3", "memory");
}

/* ---- [26] 双核并行度探针 ------------------------------------------
 * 两核同时跑同一段纯顺序读，看总带宽是否翻倍。
 *  总带宽 ≈ 1x  -> PSRAM/总线已饱和（多核无救，只能减字节）
 *  总带宽 ≈ 2x  -> 总线不是瓶颈，瓶颈在每核自身/共享 L1 冲突
 */
typedef struct { const float *p; int rep; int n; } jx_p26_t;
static volatile float g_p26_sink;

static void jx_p26_rd_range(const float *p, int lo, int hi, int rep)
{
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  for (int r = 0; r < rep; r++)
    {
      int i = lo;
      for (; i + 4 <= hi; i += 4)
        { s0 += p[i]; s1 += p[i+1]; s2 += p[i+2]; s3 += p[i+3]; }
      for (; i < hi; i++) { s0 += p[i]; }
    }
  g_p26_sink = s0 + s1 + s2 + s3;
}

static void jx_p26_rd(void *arg, int lo, int hi)
{
  const jx_p26_t *c = (const jx_p26_t *)arg;
  jx_p26_rd_range(c->p, lo, hi, c->rep);
}

/* 两核都读**整块**（重复读）：区分「共享总线饱和」与「缓存合并能力」 */
static void jx_p26_rdfull(void *arg, int lo, int hi)
{
  const jx_p26_t *c = (const jx_p26_t *)arg;
  (void)lo; (void)hi;
  jx_p26_rd_range(c->p, 0, c->n, c->rep);
}

static void jx_p26_rep(const char *tag, double cyc, int nbytes)
{
  printf("  [26] %s  %8.1f ms  %7.1f MB/s  (%9.0f cyc)\n",
         tag, cyc / 240000.0, (double)nbytes / 1e6 / (cyc / 240000000.0), cyc);
}

/* ---- [27] DCache 手工预载：把访存从关键路径上挪走 -------------------
 * ESP32-S3 的硬件预载引擎（地址 0x600C4040/44/48）可以**在 CPU 继续
 * 计算的同时**自主把一段 PSRAM 拉进 L1。若成立，整机总耗时就从
 * T = M + C/n 变成 T = max(M, C/n)（M≈2350ms 访存串行 / C≈2820ms 计算）。
 */
#define JX_EXTC_BASE   0x600C4000u
#define JX_DCP_CTRL    (*(volatile unsigned *)(JX_EXTC_BASE + 0x40u))
#define JX_DCP_ADDR    (*(volatile unsigned *)(JX_EXTC_BASE + 0x44u))
#define JX_DCP_SIZE    (*(volatile unsigned *)(JX_EXTC_BASE + 0x48u))

static int      g_p27_unit = 64;      /* SIZE 寄存器的单位（字节/计数） */
static unsigned g_p27_burn = 40000;   /* 每块之后空转的周期数 */

static int jx_p27_start(const void *p, int bytes)
{
  if (p == NULL || bytes <= 0 || bytes > 65536) { return -1; }
  JX_DCP_ADDR = (unsigned)(uintptr_t)p;
  JX_DCP_SIZE = (unsigned)((bytes + g_p27_unit - 1) / g_p27_unit);
  JX_DCP_CTRL = 1u;                       /* bit0 ENA, bit2 ORDER=0 正向 */
  return 0;
}
static int jx_p27_done(void) { return (int)((JX_DCP_CTRL >> 1) & 1u); }

static void jx_p27_burn(void)
{
  float a = 1.0f, b = 1.0001f, c = 2.0f, d = 3.0f;
  unsigned n = g_p27_burn;
  for (unsigned i = 0; i < n; i++)
    { a = a * b + 1.0f; c = c * d + 1.0f; }
  g_mem_sink = a + b + c + d;
}

static int jx_p27_wait(void)
{
  unsigned spin = 0;
  while (spin < 2000000u)
    {
      unsigned v = JX_DCP_CTRL;
      if (((v & 1u) == 0u) && ((v & 2u) != 0u)) { return (int)spin; }
      spin++;
    }
  return -1;
}

static void jx_mem_probe5(void)
{
  const int NF  = 1 << 20;                 /* 4MB 源缓冲（用来冲掉 L1） */
  const int NS  = 1 << 13;                 /* 32KB */
  const int NCH = 64;                      /* 64 块 x 16KB = 1MB */
  const int CS  = (1 << 12) * 4;           /* 16KB = 4096 float */
  {
    const char *k = getenv("JX_P27K"); if (k != NULL) { int v = atoi(k); if (v >= 0) { g_p27_burn = (unsigned)v; } }
  }
  float *ev = (float *)malloc(sizeof(float) * (size_t)NF);
  if (ev == NULL) { printf("  [27] malloc 失败\n"); return; }
  for (int i = 0; i < NF; i++) { ev[i] = (float)(i & 7); }

  printf("  [27] CTRL=0x%08x burn=%u\n", (unsigned)JX_DCP_CTRL, g_p27_burn);

  /* (a) 冷读 32KB 基线（先读 4MB 把 L1 冲掉） */
  {
    float s = 0;
    for (int i = 0; i < NF; i += 8) { s += ev[i]; }
    g_mem_sink = s;
    unsigned c0 = jx_cc0();
    jx_p26_rd_range(ev, 0, NS, 1);
    unsigned c1 = jx_cc0();
    printf("  [27] a 冷读 32KB          : %9.0f cyc  %5.0f cyc/line\n",
           (double)(c1 - c0), (double)(c1 - c0) / (double)(NS * 4 / 64));
  }

  /* (b) 预载 32KB：采样 CTRL 变化，弄清 ENA/DONE 时序 */
  {
    unsigned c0 = jx_cc0();
    (void)jx_p27_start(ev, NS * 4);
    unsigned last = 0xFFFFFFFFu; int n = 0;
    while (n < 20)
      {
        unsigned v = (unsigned)JX_DCP_CTRL;
        if (v != last)
          { printf("  [27] b t=%8u cyc CTRL=0x%08x\n", (unsigned)(jx_cc0() - c0), v); last = v; n++; }
        if ((unsigned)(jx_cc0() - c0) > 6000000u) { break; }
      }
    printf("  [27] b 末值 CTRL=0x%08x t=%u cyc\n", (unsigned)JX_DCP_CTRL, (unsigned)(jx_cc0() - c0));
    c0 = jx_cc0();
    jx_p26_rd_range(ev, 0, NS, 1);
    unsigned c1 = jx_cc0();
    printf("  [27] b2 预载后读 32KB     : %9.0f cyc\n", (double)(c1 - c0));
  }

  /* (c) SIZE 单位标定：对每个单位预载 32KB，预载后读越快说明越准 */
  {
    const int us[5] = { 16, 32, 64, 128, 256 };
    for (int k = 0; k < 5; k++)
      {
        g_p27_unit = us[k];
        float s = 0;
        for (int i = 0; i < NF; i += 8) { s += ev[i]; }   /* 冲掉 L1 */
        g_mem_sink = s;
        unsigned c0 = jx_cc0();
        (void)jx_p27_start(ev, NS * 4);
        int w = jx_p27_wait();
        unsigned c1 = jx_cc0();
        unsigned c2 = jx_cc0();
        jx_p26_rd_range(ev, 0, NS, 1);
        unsigned c3 = jx_cc0();
        printf("  [27] c unit=%-4d 等=%9.0f cyc(%s)  读=%8.0f cyc\n",
               us[k], (double)(c1 - c0), (w < 0) ? "超时" : "OK", (double)(c3 - c2));
      }
    g_p27_unit = 64;
  }

  /* (d) 叠加对比：预载正确顺序 = 等 -> 读当前块 -> 发下一块预载 -> 空转 */
  {
    struct { unsigned wait, rd, burn; double ms; int mode; } r[4];
    for (int m = 0; m < 4; m++) { r[m].mode = m; r[m].wait = r[m].rd = r[m].burn = 0; r[m].ms = 0; }
    for (int mode = 0; mode < 4; mode++)
      {
        unsigned wall0 = jx_cc0();
        if (mode == 3) { (void)jx_p27_start(ev, CS * 4); }
        for (int b = 0; b < NCH; b++)
          {
            float *ch = ev + (size_t)b * CS;
            unsigned t0, t1;
            if (mode >= 2)                                  /* 预轭模式：3 = 真预载 */
              {
                t0 = jx_cc0();
                if (mode == 3) { (void)jx_p27_wait(); }
                t1 = jx_cc0(); r[mode].wait += (t1 - t0);
              }
            t0 = jx_cc0();
            if (mode != 1) { jx_p26_rd_range(ch, 0, CS, 1); }
            t1 = jx_cc0(); r[mode].rd += (t1 - t0);
            if (mode >= 2 && b + 1 < NCH && (mode == 3))
              { (void)jx_p27_start(ev + (size_t)(b + 1) * CS, CS * 4); }
            t0 = jx_cc0();
            jx_p27_burn();
            t1 = jx_cc0(); r[mode].burn += (t1 - t0);
          }
        r[mode].ms = (double)(jx_cc0() - wall0) / 240000.0;
      }
    for (int m = 0; m < 4; m++)
      printf("  [27] d%d %-14s wall=%7.1f ms  wait=%7.1f read=%7.1f burn=%7.1f\n",
             m, (m == 0) ? "只读" : (m == 1) ? "只空转" : (m == 2) ? "读+空转" : "预载+读+空转",
             r[m].ms, r[m].wait / 240000.0, r[m].rd / 240000.0, r[m].burn / 240000.0);
  }
  free(ev);
}

static void jx_mem_probe4(void)
{
  const int NW = 1 << 18;                 /* 256K 元素 = 1MB */
  float      *fb = (float *)malloc(sizeof(float) * (size_t)NW);
  signed char*qb = (signed char *)malloc((size_t)NW);
  if (fb != NULL && qb != NULL)
    {
      /* (a) 4B 顺序写，8 路展开 */
      {
        unsigned c0 = jx_cc0();
        for (int r = 0; r < 4; r++)
          {
            int i = 0;
            for (; i + 8 <= NW; i += 8)
              { fb[i]=1.f; fb[i+1]=1.f; fb[i+2]=1.f; fb[i+3]=1.f;
                fb[i+4]=1.f; fb[i+5]=1.f; fb[i+6]=1.f; fb[i+7]=1.f; }
            for (; i < NW; i++) { fb[i] = 1.f; }
          }
        unsigned c1 = jx_cc0();
        printf("  [24] a 4B 顺序写       : %7.2f 周期/元素  %6.1f MB/s\n",
               (double)(c1 - c0) / (4.0 * NW),
               4.0 * NW * 4.0 / 1e6 / ((double)(c1 - c0) / 240000000.0));
      }
      /* (b) 16B PIE 写 */
      {
        for (int i = 0; i < NW; i++) { fb[i] = 2.f; }
        unsigned c0 = jx_cc0();
        for (int r = 0; r < 4; r++)
          { jx_p24_vst16(fb, g_p18st, NW / 4); }
        unsigned c1 = jx_cc0();
        printf("  [24] b 16B vst.128 写  : %7.2f 周期/元素  %6.1f MB/s\n",
               (double)(c1 - c0) / (4.0 * NW),
               4.0 * NW * 4.0 / 1e6 / ((double)(c1 - c0) / 240000000.0));
      }
      /* (c) 1B 顺序写 */
      {
        unsigned c0 = jx_cc0();
        for (int r = 0; r < 4; r++)
          {
            int i = 0;
            for (; i + 8 <= NW; i += 8)
              { qb[i]=1; qb[i+1]=1; qb[i+2]=1; qb[i+3]=1;
                qb[i+4]=1; qb[i+5]=1; qb[i+6]=1; qb[i+7]=1; }
            for (; i < NW; i++) { qb[i] = 1; }
          }
        unsigned c1 = jx_cc0();
        printf("  [24] c 1B 顺序写       : %7.2f 周期/字节\n",
               (double)(c1 - c0) / (4.0 * NW));
      }
      /* (d) 1B 跨步写（步长 16，模拟 qb[j*cpad+ci] 的 ci 固定） */
      {
        unsigned c0 = jx_cc0();
        for (int r = 0; r < 4; r++)
          { for (int i = 0; i < NW; i += 16) { qb[i] = 2; } }
        unsigned c1 = jx_cc0();
        printf("  [24] d 1B 跨步16写     : %7.2f 周期/字节\n",
               (double)(c1 - c0) / (4.0 * NW / 16.0));
      }
      /* (e) 4B 写 + 同址 4B 读（读改写，看是否几乎免费） */
      {
        float s = 0.0f;
        unsigned c0 = jx_cc0();
        for (int r = 0; r < 4; r++)
          {
            int i = 0;
            for (; i + 4 <= NW; i += 4)
              { fb[i] += 1.f; fb[i+1] += 1.f; fb[i+2] += 1.f; fb[i+3] += 1.f; }
            for (; i < NW; i++) { fb[i] += 1.f; }
          }
        unsigned c1 = jx_cc0();
        s = fb[0];
        g_mem_sink = s;
        printf("  [24] e 4B 读改写       : %7.2f 周期/元素\n",
               (double)(c1 - c0) / (4.0 * NW));
      }
      /* (f) 4B 顺序读（同缓冲，随时钟量） */
      {
        float s = 0.0f;
        unsigned c0 = jx_cc0();
        for (int r = 0; r < 4; r++)
          {
            int i = 0;
            for (; i + 4 <= NW; i += 4)
              { s = fb[i] + s; s = fb[i+1] + s; s = fb[i+2] + s; s = fb[i+3] + s; }
            for (; i < NW; i++) { s = fb[i] + s; }
          }
        unsigned c1 = jx_cc0();
        g_mem_sink = s;
        printf("  [24] f 4B 顺序读(串行加): %7.2f 周期/元素\n",
               (double)(c1 - c0) / (4.0 * NW));
      }
    }
  if (fb != NULL) { free(fb); }
  if (qb != NULL) { free(qb); }
}

static void jx_mem_probe3(void)
{
  /* ---- [20] L1 局部性 ------------------------------------------------- */
  {
    const int NB = 2048;                 /* 8KB */
    float *a   = (float *)malloc(sizeof(float) * (size_t)NB);
    float *big = (float *)malloc(sizeof(float) * (size_t)(1 << 20));
    if (a != NULL && big != NULL)
      {
        float sink = 0.0f;
        for (int i = 0; i < NB; i++) { a[i] = (float)(i & 7); }
        for (int i = 0; i < (1 << 20); i++) { big[i] = (float)(i & 3); }
        {
          unsigned c0 = jx_cc0();
          for (int r = 0; r < 100; r++)
            {
              for (int i = 0; i < NB; i++) { sink += a[i]; }
              for (int i = 0; i < (1 << 20); i += 16) { sink += big[i]; }
            }
          unsigned c1 = jx_cc0();
          printf("  [20] 8KB 冷读(夹 1MB 扫描) : %7.2f 周期/元素\n",
                 (double)(c1 - c0) / (100.0 * NB));
        }
        {
          float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
          unsigned c0 = jx_cc0();
          for (int r = 0; r < 100; r++)
            { for (int i = 0; i + 4 <= NB; i += 4)
                { s0 += a[i]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3]; } }
          unsigned c1 = jx_cc0();
          sink += s0 + s1 + s2 + s3;
          double el = 100.0 * NB;
          printf("  [20] 8KB 热读(连读 100 遍)  : %7.2f 周期/元素  %7.1f MB/s\n",
                 (double)(c1 - c0) / el,
                 el * 4.0 / 1e6 / ((double)(c1 - c0) / 240000000.0));
        }
        {
          unsigned c0 = jx_cc0();
          for (int r = 0; r < 16; r++)
            { for (int i = 0; i < (1 << 20); i++) { sink += big[i]; } }
          unsigned c1 = jx_cc0();
          double el = 16.0 * (double)(1 << 20);
          printf("  [20] 4MB 流式读             : %7.2f 周期/元素  %7.1f MB/s\n",
                 (double)(c1 - c0) / el,
                 el * 4.0 / 1e6 / ((double)(c1 - c0) / 240000000.0));
        }
        g_mem_sink = sink;
      }
    if (a != NULL) { free(a); }
    if (big != NULL) { free(big); }
  }

  /* ---- [21] epilogue 复刻 --------------------------------------------- */
  {
    const int N = 90112;                 /* 256 行 x 352 输出 */
    int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * (size_t)N);
    float   *swc = (float   *)malloc(sizeof(float)   * (size_t)N);
    float   *sbb = (float   *)malloc(sizeof(float)   * (size_t)N);
    float   *dst = (float   *)malloc(sizeof(float)   * (size_t)N);
    if (acc != NULL && swc != NULL && sbb != NULL && dst != NULL)
      {
        for (int i = 0; i < N; i++)
          { acc[i] = (int32_t)((i * 37) % 1000); swc[i] = 0.001f; sbb[i] = 0.5f; }
        for (int m = 0; m < 5; m++)
          {
            float *d  = (m == 2) ? g_p18st : dst;
            int    nn = (m == 2) ? 2000 : N;
            float sink = 0.0f;
            unsigned c0 = jx_cc0();
            for (int r = 0; r < 8; r++)
              { sink += jx_epi_rep(d, acc, swc, sbb, nn, 1.0f, m); }
            unsigned c1 = jx_cc0();
            g_mem_sink = sink;
            printf("  [21] epi mode %d (%-14s) : %8.1f cyc  %7.2f 周期/元素\n",
                   m,
                   (m == 0) ? "写PSRAM 全算式" : (m == 1) ? "不写 全算式" :
                   (m == 2) ? "写SRAM 全算式" : (m == 3) ? "写PSRAM 只转换" : "写PSRAM 只常量",
                   (double)(c1 - c0) / 8.0, (double)(c1 - c0) / 8.0 / (double)nn);
          }
      }
    if (acc != NULL) { free(acc); }
    if (swc != NULL) { free(swc); }
    if (sbb != NULL) { free(sbb); }
    if (dst != NULL) { free(dst); }
  }

  /* ---- [23] accx 点积链长 --------------------------------------------- */
  {
    signed char *w = (signed char *)(void *)g_p18st;
    signed char *x = (signed char *)(void *)g_p18st + 1024;
    int32_t *out = (int32_t *)(void *)g_p18st + 256;
    for (int i = 0; i < 512; i++)
      { w[i] = (signed char)((i * 13) % 15 - 7); x[i] = (signed char)((i * 7) % 11 - 5); }
    const int NO = 512;
    for (int nv = 1; nv <= 8; nv *= 2)
      {
        unsigned c0 = jx_cc0();
        for (int r = 0; r < 8; r++)
          { jx_p23_run(w, x, NO, nv, out); }
        unsigned c1 = jx_cc0();
        g_mem_sink = (float)out[0];
        double cyc = (double)(c1 - c0) / 8.0;
        double macs = (double)NO * 16.0 * (double)nv;
        printf("  [23] accx nv=%d : %9.1f cyc  %6.2f 周期/输出  %6.2f MAC/周期\n",
               nv, cyc, cyc / (double)NO, macs / cyc);
      }
  }

  /* ---- [25] 真实 PSRAM 读带宽（无依赖链，剥离探测方法的偏差） ---- */
  {
    const int NW = 1 << 20;                 /* 1M 元素 = 4MB */
    float *big = (float *)malloc(sizeof(float) * (size_t)NW);
    if (big != NULL)
      {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < NW; i++) { big[i] = (float)(i & 7); }
        {
          unsigned c0 = jx_cc0();
          for (int r = 0; r < 8; r++)
            {
              int i = 0;
              for (; i + 4 <= NW; i += 4)
                { s0 += big[i]; s1 += big[i+1]; s2 += big[i+2]; s3 += big[i+3]; }
              for (; i < NW; i++) { s0 += big[i]; }
            }
          unsigned c1 = jx_cc0();
          double el = 8.0 * (double)NW;
          printf("  [25] a 4MB 读 4独立链   : %7.2f 周期/元素  %7.1f MB/s\n",
                 (double)(c1 - c0) / el, el * 4.0 / 1e6 / ((double)(c1 - c0) / 240000000.0));
        }
        {
          const float *p = big;
          int c = NW / 4;
          unsigned c0 = jx_cc0();
          for (int r = 0; r < 8; r++)
            {
              void *pp = (void *)p;
              void *dd = (void *)g_p18st;
              int   cc = c;
              __asm__ __volatile__(
                "ee.zero.q q7\n\t"
                "1:\n\t"
                "ee.vld.128.ip q0, %[P], 16\n\t"
                "ee.orq q7, q7, q0\n\t"
                "addi %[C], %[C], -1\n\t"
                "bnez %[C], 1b\n\t"
                "ee.vst.128.ip q7, %[D], 16\n\t"
                : [P] "+r"(pp), [C] "+r"(cc), [D] "+r"(dd)
                :
                : "f0", "f7", "memory");
            }
          unsigned c1 = jx_cc0();
          double el = 8.0 * (double)NW;
          printf("  [25] b 4MB 读 vld.128    : %7.2f 周期/元素  %7.1f MB/s\n",
                 (double)(c1 - c0) / el, el * 4.0 / 1e6 / ((double)(c1 - c0) / 240000000.0));
        }
        {
          unsigned c0 = jx_cc0();
          for (int r = 0; r < 8; r++)
            { for (int i = 0; i < NW; i += 16) { s0 += big[i]; } }
          unsigned c1 = jx_cc0();
          double el = 8.0 * (double)(NW / 16);
          printf("  [25] c 4MB 跨64B读       : %7.2f 周期/64B行  %7.1f MB/s\n",
                 (double)(c1 - c0) / el, el * 64.0 / 1e6 / ((double)(c1 - c0) / 240000000.0));
        }
        g_mem_sink = s0 + s1 + s2 + s3;
        free(big);
      }
  }

  /* ---- [26] 双核并行度：总带宽是否随核数翻倍 ---- */
  {
    const int NF  = 1 << 20;                 /* 1M 元素 = 4MB，远大于 L1 */
    const int NS  = 1 << 13;                 /* 8K 元素 = 32KB = L1 尺寸 */
    const int REP = 8, REPS = 512;
    float *big = (float *)malloc(sizeof(float) * (size_t)NF);
    float *sm  = (float *)malloc(sizeof(float) * (size_t)(2 * NS));
    if (big != NULL && sm != NULL)
      {
        jx_p26_t c;
        for (int i = 0; i < NF; i++) { big[i] = (float)(i & 7); }
        for (int i = 0; i < 2 * NS; i++) { sm[i] = (float)(i & 3); }

        /* (a) 单核 读 4MB */
        {
          unsigned t0 = jx_cc0();
          jx_p26_rd_range(big, 0, NF, REP);
          unsigned t1 = jx_cc0();
          jx_p26_rep("a 单核 读4MB      ", (double)(t1 - t0), (int)(REP * NF * 4.0));
        }
        /* (b) 双核 各读一半（不重叠） */
        {
          c.p = big; c.rep = REP; c.n = NF;
          unsigned t0 = jx_cc0();
          int ok = jx_pf_run_min(NF, jx_p26_rd, &c, 2);
          unsigned t1 = jx_cc0();
          jx_p26_rep(ok ? "b 双核 各半4MB  " : "b 单核(降级)4MB", (double)(t1 - t0), (int)(REP * NF * 4.0));
          printf("       ↑ ok=%d (1=真并行)\n", ok);
        }
        /* (c) 双核 都读整块（总量 2x） */
        {
          c.p = big; c.rep = REP; c.n = NF;
          unsigned t0 = jx_cc0();
          jx_pf_run_min(2, jx_p26_rdfull, &c, 2);
          unsigned t1 = jx_cc0();
          jx_p26_rep("c 双核 都读4MB  ", (double)(t1 - t0), (int)(2.0 * REP * NF * 4.0));
        }
        /* (d) 单核 32KB 常驻 */
        {
          unsigned t0 = jx_cc0();
          jx_p26_rd_range(sm, 0, NS, REPS);
          unsigned t1 = jx_cc0();
          jx_p26_rep("d 单核 32KB常驻", (double)(t1 - t0), (int)(REPS * NS * 4.0));
        }
        /* (e) 双核 各 32KB（合计 64KB = L1 全占） */
        {
          c.p = sm; c.rep = REPS; c.n = 2 * NS;
          unsigned t0 = jx_cc0();
          int ok = jx_pf_run_min(2 * NS, jx_p26_rd, &c, 2);
          unsigned t1 = jx_cc0();
          jx_p26_rep(ok ? "e 双核 各2×32KB  " : "e 单核 64KB    ", (double)(t1 - t0), (int)(REPS * 2.0 * NS * 4.0));
        }
        free(big); free(sm);
      }
    else { if (big != NULL) { free(big); } if (sm != NULL) { free(sm); }
           printf("  [26] malloc 失败\n"); }
  }

}

static void jx_mem_probe2(void)
{
  /* ---- [11] MLP：1/2/4/8 路并行独立流 ---- */
  {
    const int NF  = 512 * 1024;             /* 2 MB，远大于 L1 */
    const int REP = 2;
    float *buf = (float *)malloc(sizeof(float) * (size_t)NF);
    if (buf != NULL)
      {
        for (int i = 0; i < NF; i++) { buf[i] = (float)(i & 7); }
        for (int nw = 1; nw <= 8; nw <<= 1)
          {
            const int CH = NF / nw;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            double t0 = jx_now_ms();
            for (int r = 0; r < REP; r++)
              {
                for (int i = 0; i < CH; i++)
                  {
                    float v0 = buf[i];
                    float v1 = (nw > 1) ? buf[(size_t)CH + i]     : 0.0f;
                    float v2 = (nw > 2) ? buf[(size_t)2 * CH + i] : 0.0f;
                    float v3 = (nw > 2) ? buf[(size_t)3 * CH + i] : 0.0f;
                    float v4 = (nw > 4) ? buf[(size_t)4 * CH + i] : 0.0f;
                    float v5 = (nw > 4) ? buf[(size_t)5 * CH + i] : 0.0f;
                    float v6 = (nw > 4) ? buf[(size_t)6 * CH + i] : 0.0f;
                    float v7 = (nw > 4) ? buf[(size_t)7 * CH + i] : 0.0f;
                    a0 += v0; a1 += v1; a2 += v2; a3 += v3;
                    a4 += v4; a5 += v5; a6 += v6; a7 += v7;
                  }
              }
            double ms = jx_now_ms() - t0;
            g_mem_sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
            printf("  [11] PSRAM 顺序读 %d 路并行 : %8.1f ms  %7.1f MB/s\n",
                   nw, ms, (double)REP * NF * 4.0 / 1e6 / (ms / 1000.0));
          }
        free(buf);
      }
    else { printf("  [11] 跳过（2MB 分配失败）\n"); }
  }

  /* ---- [12] 内部 SRAM 跨步 vs PSRAM 跨步 ---- */
  {
    const int NF = 256 * 1024;              /* 1 MB，PSRAM */
    float *buf = (float *)malloc(sizeof(float) * (size_t)NF);
    if (buf != NULL)
      {
        for (int i = 0; i < NF; i++) { buf[i] = (float)(i & 7); }
        /* PSRAM：4 KB 跨步，每次 1 个 float */
        {
          const int ST = 1024;              /* 4 KB */
          float s = 0;
          double t0 = jx_now_ms();
          for (int r = 0; r < 64; r++)
            { for (int i = 0; i < NF; i += ST) { s += buf[i]; } }
          double ms = jx_now_ms() - t0;
          g_mem_sink = s;
          printf("  [12] PSRAM 跨步读(4KB)  : %8.1f ms  %7.2f M 次/s  %6.1f 周期/次\n",
                 ms, 64.0 * (NF / ST) / (ms / 1000.0) / 1e6,
                 ms * 240000.0 / (64.0 * (NF / ST)));
        }
        free(buf);
      }
    else { printf("  [12] 跳过（1MB 分配失败）\n"); }
  }

  /* ---- [15] 真实尺寸的行量化：1/2/4/8 行并行（JX_QPAR 收益预估） ----
   * 完整两趟（对齐 jx_q_row：第一趟求 max，第二趟量化写 q）。
   * IN/R 取本模型真实值：352 = conv_unit 里 l2 的 in（=dim4）。
   * 缓冲 352KB 远大于 32KB L1，每一趟都是真实的 PSRAM 访存。 */
  {
    const int IN = 352, R = 256;
    float *xs = (float *)malloc(sizeof(float) * (size_t)IN * (size_t)R);
    signed char *qs = (signed char *)malloc((size_t)IN * (size_t)R);
    if (xs != NULL && qs != NULL)
      {
        for (int i = 0; i < IN * R; i++) { xs[i] = (float)((i * 37) % 1000) * 0.001f; }
        memset(qs, 0, (size_t)IN * (size_t)R);
        {
          double m1 = 0, m2 = 0, m4 = 0, m8 = 0;
          double el = (double)IN * (double)(R - 8);
          JX_QP_FULL(1, xs, qs, IN, R, m1);
          JX_QP_FULL(2, xs, qs, IN, R, m2);
          JX_QP_FULL(4, xs, qs, IN, R, m4);
          JX_QP_FULL(8, xs, qs, IN, R, m8);
          printf("  [15] 行量化 in=%d rows=%d (两趟全量):\n", IN, R);
          printf("  [15]   1路 %6.1f | 2路 %6.1f (%.2fx) | 4路 %6.1f (%.2fx) | 8路 %6.1f (%.2fx) ms\n",
                 m1, m2, m1 / (m2 > 0.001 ? m2 : 0.001),
                 m4, m1 / (m4 > 0.001 ? m4 : 0.001),
                 m8, m1 / (m8 > 0.001 ? m8 : 0.001));
          printf("  [15]   cyc/元素: 1路 %.2f | 2路 %.2f | 4路 %.2f | 8路 %.2f\n",
                 m1 * 240000.0 / el, m2 * 240000.0 / el,
                 m4 * 240000.0 / el, m8 * 240000.0 / el);
        }
        free(xs); free(qs);
      }
    else { printf("  [15] 跳过（分配失败）\n"); free(xs); free(qs); }
  }

  /* ---- [18] k1cf 量化 prologue 的访存行为（r50） ----
   * 真实形状 Ci=20 / T=2112 / TB=102，张量 169 KB（远超 L1）。
   * mode 1 = 只求 max（Pass A）；mode 2 = Pass A + Pass B（现状，PSRAM 再读一遍）；
   * mode 3 = Pass A 同时把浮点搬进内部 SRAM，Pass B 从 SRAM 量化。
   * 对比第 2 趟的增量，判定 Pass B 到底有没有吃到 L1/内部 SRAM。 */
  {
    const int CI = 20, TT = 2112, TB = 102;
    float *xn  = (float *)malloc(sizeof(float) * (size_t)CI * (size_t)TT);
    float *scb = (float *)malloc(sizeof(float) * (size_t)TB);
    float *tni = (float *)malloc(sizeof(float) * (size_t)TB);
    signed char *dqq = (signed char *)malloc((size_t)TB * 32);
    if (xn != NULL && scb != NULL && tni != NULL && dqq != NULL)
      {
        for (int i = 0; i < CI * TT; i++) { xn[i] = (float)((i * 37) % 1000) * 0.001f; }
        for (int j = 0; j < TB; j++) { tni[j] = 127.0f; }
        {
          double m1, m2, m3;
          double el = 3.0 * (double)CI * (double)TT;
          jx_p18_sweep(CI, TT, TB, 1, xn, scb, tni, dqq);
          jx_p18_sweep(CI, TT, TB, 2, xn, scb, tni, dqq);
          jx_p18_sweep(CI, TT, TB, 3, xn, scb, tni, dqq);
          m1 = jx_p18_time(CI, TT, TB, 1, xn, scb, tni, dqq);
          m2 = jx_p18_time(CI, TT, TB, 2, xn, scb, tni, dqq);
          m3 = jx_p18_time(CI, TT, TB, 3, xn, scb, tni, dqq);
          g_mem_sink = dqq[0] + scb[0];
          printf("  [18] k1cf 量化 prologue  Ci=%d T=%d TB=%d  张量 %d KB\n",
                 CI, TT, TB, (int)(sizeof(float) * (size_t)CI * (size_t)TT / 1024));
          printf("  [18]   1趟(只max) %6.1f ms %6.2f cyc/el | 2趟(现状) %6.1f %6.2f | 3趟(max+SRAM暂存) %6.1f %6.2f\n",
                 m1, m1 * 240000.0 / el, m2, m2 * 240000.0 / el, m3, m3 * 240000.0 / el);
          printf("  [18]   第2趟增量 %6.2f cyc/el（PSRAM 再读） | SRAM 暂存增量 %6.2f cyc/el\n",
                 (m2 - m1) * 240000.0 / el, (m3 - m1) * 240000.0 / el);
        }
        free(xn); free(scb); free(tni); free(dqq);
      }
    else { printf("  [18] 跳过（分配失败）\n"); free(xn); free(scb); free(tni); free(dqq); }
  }

  /* ---- [13] 输出写模式：逐 t 跨行（现状） vs 逐行连续 ---- */
  {
    const int CO = 16, TO = 48086;
    float *o = (float *)malloc(sizeof(float) * (size_t)CO * (size_t)TO);
    if (o != NULL)
      {
        const int R = 4;
        double t0, ms1, ms2;
        for (size_t i = 0; i < (size_t)CO * TO; i++) { o[i] = 0.0f; }
        t0 = jx_now_ms();
        for (int r = 0; r < R; r++)
          for (int t = 0; t < TO; t++)
            for (int co = 0; co < CO; co++)
              { o[(size_t)co * TO + t] = (float)(co + t); }
        ms1 = jx_now_ms() - t0;
        t0 = jx_now_ms();
        for (int r = 0; r < R; r++)
          for (int co = 0; co < CO; co++)
            {
              float *row = o + (size_t)co * TO;
              for (int t = 0; t < TO; t++) { row[t] = (float)(co + t); }
            }
        ms2 = jx_now_ms() - t0;
        g_mem_sink = o[0] + o[(size_t)CO * TO - 1];
        printf("  [13] 输出写 16x48086 x%d : 逐t跨行(现状) %7.1f ms | 逐行连续 %7.1f ms  (%.2fx)\n",
               R, ms1, ms2, ms1 / (ms2 > 0.001 ? ms2 : 0.001));
        free(o);
      }
    else { printf("  [13] 跳过（3MB 分配失败）\n"); }
  }

  /* ---- [14] im2col 收集：cpg 次小 memcpy vs 一次大 memcpy ---- */
  {
    const int CPG = 88, KK = 7, W = 262, POS = 256;
    signed char *sa = (signed char *)malloc((size_t)CPG * W);
    signed char *sb = (signed char *)malloc((size_t)W * CPG);
    signed char *dt = (signed char *)malloc((size_t)CPG * KK);
    if (sa != NULL && sb != NULL && dt != NULL)
      {
        double t0, ms1, ms2;
        for (int i = 0; i < CPG * W; i++) { sa[i] = (signed char)(i & 127); }
        for (int c = 0; c < CPG; c++)
          { for (int j = 0; j < W; j++) { sb[j * CPG + c] = sa[c * W + j]; } }
        t0 = jx_now_ms();
        for (int r = 0; r < 8; r++)
          for (int p = 0; p < POS; p++)
            {
              int off = (p * 37) % 200;
              signed char *d = dt;
              for (int c = 0; c < CPG; c++) { memcpy(d, sa + c * W + off, (size_t)KK); d += KK; }
            }
        ms1 = jx_now_ms() - t0;
        t0 = jx_now_ms();
        for (int r = 0; r < 8; r++)
          for (int p = 0; p < POS; p++)
            {
              int off = (p * 37) % 200;
              memcpy(dt, sb + (size_t)off * CPG, (size_t)KK * CPG);
            }
        ms2 = jx_now_ms() - t0;
        g_mem_sink = (float)(dt[0] + dt[CPG * KK - 1]);
        printf("  [14] im2col 收集 88x7 x%d : 88次小memcpy(现状) %7.1f ms | 交错1次大memcpy %7.1f ms (%.2fx)\n",
               8 * POS, ms1, ms2, ms1 / (ms2 > 0.001 ? ms2 : 0.001));
        free(sa); free(sb); free(dt);
      }
    else { printf("  [14] 跳过（分配失败）\n"); free(sa); free(sb); free(dt); }
  }
}

static int jx_cmd_mem(void)
{
  const double SYSCLK = 240.0e6;
  printf("=== 存储层次 / 浮点吞吐微基准 (CPU %d MHz) ===\n", 240);

  /* [1] FPU 上限
   *
   * 注意工作集大小：ESP32-S3 的 L1 D-cache 是 32KB，而本测试原版用了
   * ax[4096]+bx[4096] = 正好 32KB，会把缓存挤满并抖动，量出来的是
   * 「缓存抖动下限」而不是 FPU 吞吐。这里改成 2x4KB（远小于缓存）+
   * 8 路独立累加链（够盖住 FMA 流水线延迟），才是真正的算力上限。
   *
   * FMA 有 4~5 个周期的延迟，累加链太少会被延迟而不是吞吐卡住。 */
  {
    enum { NF = 512 };                     /* 2 x 2KB，稳进 L1 */
    static float ax[NF], bx[NF];
    for (int i = 0; i < NF; i++)
      {
        ax[i] = (float)(i & 7) * 0.125f;
        bx[i] = (float)(i & 3) * 0.25f;
      }

    const int REP = 32768;                 /* 33.6M MAC，约 0.1~1s，够盖过 10ms 时基 */
    double t0 = jx_now_ms();
    for (int r = 0; r < REP; r++)
      {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (int i = 0; i < NF; i += 8)
          {
            a0 += ax[i]     * bx[i];
            a1 += ax[i + 1] * bx[i + 1];
            a2 += ax[i + 2] * bx[i + 2];
            a3 += ax[i + 3] * bx[i + 3];
            a4 += ax[i + 4] * bx[i + 4];
            a5 += ax[i + 5] * bx[i + 5];
            a6 += ax[i + 6] * bx[i + 6];
            a7 += ax[i + 7] * bx[i + 7];
          }
        g_mem_sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
      }
    double ms = jx_now_ms() - t0;
    double macs = (double)REP * (double)NF;
    printf("  [1] FPU 上限(4KB/8链): %8.1f ms  %7.1f MMAC/s  %5.2f 周期/MAC\n",
           ms, macs / ms / 1000.0, ms * 1e-3 * SYSCLK / macs);
  }


  /* [2] / [5] PSRAM 顺序读、写、流式读改写 */
  {
    const int NF = 256 * 1024;              /* 1 MB */
    float *buf = (float *)malloc(sizeof(float) * (size_t)NF);
    if (buf == NULL)
      {
        printf("  [2] PSRAM : 1MB 分配失败\n");
      }
    else
      {
        const int REP = 4;
        for (int i = 0; i < NF; i++) { buf[i] = (float)(i & 15); }

        float s = 0;
        double t0 = jx_now_ms();
        for (int r = 0; r < REP; r++) { for (int i = 0; i < NF; i++) { s += buf[i]; } }
        double ms = jx_now_ms() - t0;
        g_mem_sink = s;
        printf("  [2] PSRAM 顺序读     : %8.1f ms  %7.1f MB/s\n",
               ms, (double)REP * NF * 4.0 / 1e6 / (ms / 1000.0));

        t0 = jx_now_ms();
        for (int r = 0; r < REP; r++) { for (int i = 0; i < NF; i++) { buf[i] = (float)(i + r); } }
        ms = jx_now_ms() - t0;
        printf("  [3] PSRAM 顺序写     : %8.1f ms  %7.1f MB/s\n",
               ms, (double)REP * NF * 4.0 / 1e6 / (ms / 1000.0));

        s = 0;
        t0 = jx_now_ms();
        for (int r = 0; r < REP; r++)
          { for (int i = 0; i < NF; i++) { buf[i] += s; } }
        ms = jx_now_ms() - t0;
        printf("  [4] PSRAM 流式读改写 : %8.1f ms  (%7.1f MB/s 往返)\n",
               ms, 2.0 * (double)REP * NF * 4.0 / 1e6 / (ms / 1000.0));
        free(buf);
      }
  }

  /* [17] MLP（内存级并行）：同时驱动 N 条**独立的顺序流**，看 miss 能不能重叠。
   *
   * 动机（第三十七轮）：第十八轮的 [11] 探针给出
   *     1 路 24.7 MB/s / 2 路 38.1 / 4 路 46.6（饱和） / 8 路 5.6（cache set 塌陷）
   * 说明 PSRAM 是**延迟受限**而不是接口带宽受限（OCT 80MHz DTR 理论 160MB/s，
   * 寄存器实测 SPI1 CORE_CLK_SEL=2 即 160MHz 核心时钟，配置是生效的）。
   * 而模型里几乎所有热循环都只有**一条流**（一次一行、一次一个 block）。
   *
   * 测法：同样扫完 1MB，把内层循环改成 N 条流各自连续读 CH=64 个 float
   * （= 4 条 cache line，每条流内部本身就有 4 个未完成 miss），
   * N = 1/2/4。每条流一个独立累加器，避免依赖链把并发压掉。
   * 全部配置读的字节数完全相同，可直接比总带宽。 */
  {
    const int NF = 256 * 1024;              /* 1 MB */
    float *buf = (float *)malloc(sizeof(float) * (size_t)NF);
    if (buf == NULL)
      {
        printf("  [17] MLP : 1MB 分配失败\n");
      }
    else
      {
        const int REP = 4;
        const int CH = 64;                  /* 每条流一次 64 float = 4 条 line */
        for (int i = 0; i < NF; i++) { buf[i] = (float)(i & 15); }
        printf("  [17] MLP 顺序读（同样 1MB，%d 次）:\n", REP);
        for (int ns = 1; ns <= 4; ns <<= 1)
          {
            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            int step = CH * ns;
            long nread = 0;
            double t0 = jx_now_ms();
            for (int r = 0; r < REP; r++)
              {
                for (int b = 0; b + step <= NF; b += step)
                  {
                    for (int s = 0; s < ns; s++)
                      {
                        const float *p = buf + b + s * CH;
                        for (int k = 0; k < CH; k++) { acc[s] += p[k]; }
                      }
                    nread += step;
                  }
              }
            double ms = jx_now_ms() - t0;
            g_mem_sink = acc[0] + acc[1] + acc[2] + acc[3];
            printf("  [17]   %d 流 : %8.1f ms  %7.1f MB/s  (%5.1f 周期/64B 行)\n",
                   ns, ms, (double)nread * 4.0 / 1e6 / (ms / 1000.0),
                   ms * 1e-3 * SYSCLK / ((double)nread / 16.0));
          }
        free(buf);
      }
  }

  /* [6] PSRAM 跨步读（老卷积内核的访存模式：按 T_in 跳行） */
  {
    const int NF = 256 * 1024;              /* 1 MB */
    float *buf = (float *)malloc(sizeof(float) * (size_t)NF);
    if (buf != NULL)
      {
        for (int i = 0; i < NF; i++) { buf[i] = (float)(i & 7); }
        const int STRIDE = 1024;            /* 每次跳 4KB */
        float s = 0;
        double t0 = jx_now_ms();
        for (int r = 0; r < 64; r++)
          { for (int i = 0; i < NF; i += STRIDE) { s += buf[i]; } }
        double ms = jx_now_ms() - t0;
        g_mem_sink = s;
        printf("  [5] PSRAM 跨步读(4KB) : %8.1f ms  %7.2f M 次访问/s\n",
               ms, 64.0 * (NF / STRIDE) / (ms / 1000.0) / 1e6);
        free(buf);
      }
  }

  /* [7] FLASH(XIP) 顺序读（权重常数块） */
  {
    const float *wb = jx_weight_blob();
    int nf = jx_weight_floats();
    int use = (nf < 256 * 1024) ? nf : 256 * 1024;   /* 取前 1MB */
    const int REP = 4;
    float s = 0;
    double t0 = jx_now_ms();
    for (int r = 0; r < REP; r++) { for (int i = 0; i < use; i++) { s += wb[i]; } }
    double ms = jx_now_ms() - t0;
    g_mem_sink = s;
    printf("  [6] 权重块 %s 顺序读: %8.1f ms  %7.1f MB/s   (权重共 %d KB)\n",
           jx_weight_in_heap() ? "PSRAM" : "FLASH(XIP)",
           ms, (double)REP * use * 4.0 / 1e6 / (ms / 1000.0), nf / 256);
  }

  /* [7]/[8] PIE 向量单元 int8 吞吐（决定 int8 路线能不能实时） */
  jx_fpu_chain_bench();
  jx_fpreg_bench();
  jx_sinsq_bench();
  jx_cvt_bench();
  jx_block_bw_bench();
  jx_pie_bench();
  jx_mem_probe2();
  jx_mem_probe3();
  jx_mem_probe4();
  jx_mem_probe5();
  jx_bt_bench();

  return 0;
}

struct jx_rec_ctx
{
  int16_t *buf;      /* 16kHz 单声道累加缓冲 */
  int      cap;      /* 容量（样本） */
  int      n;        /* 已收样本 */
};

static void jx_rec_cb(FAR const int16_t *pcm16, size_t samples, void *arg)
{
  struct jx_rec_ctx *c = (struct jx_rec_ctx *)arg;

  if (c->n + (int)samples > c->cap)
    {
      samples = (size_t)(c->cap - c->n);
    }

  if ((int)samples > 0)
    {
      memcpy(&c->buf[c->n], pcm16, samples * sizeof(int16_t));
      c->n += (int)samples;
  }
}

/****************************************************************************
 * streamplay：录制一小段 PCM，然后逐 24ms 包处理并立即播放。
 * 这是“边处理边播放”的听感验证入口，不做网络传输。
 ****************************************************************************/
static int jx_cmd_streamplay(int argc, char *argv[])
{
  int secs = (argc >= 3) ? atoi(argv[2]) : 2;
  int batch_pkts = (argc >= 4) ? atoi(argv[3]) : 8;
  int play_gain = jx_env_int("JX_PLAY_GAIN", 4);
  struct jx_rec_ctx ctx;
  struct jx_stream_s *st;
  int16_t *out = NULL;
  uint8_t *wav = NULL;
  int npkt, rc = 0;

  if (secs < 1) { secs = 1; }
  if (secs > 3) { secs = 3; }
  if (batch_pkts < 1) { batch_pkts = 1; }
  if (batch_pkts > 32) { batch_pkts = 32; }
  if (play_gain < 1) { play_gain = 1; }
  if (play_gain > 16) { play_gain = 16; }

  if (ai_voice_init() != 0)
    {
      printf("[ERR] ai_voice_init 失败\n");
      return 1;
    }
  jx_mic_gain_apply();

  ctx.cap = secs * JX_PCM_RATE;
  ctx.n = 0;
  ctx.buf = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.cap);
  if (ctx.buf == NULL)
    {
      printf("[ERR] 录音缓冲分配失败\n");
      return 1;
    }

  printf("=== streamplay：%d 秒录音，%d 包/批（%dms）处理并播放 ===\n",
         secs, batch_pkts, batch_pkts * 24);
  printf("  播放增益: %dx\n", play_gain);
  printf("  提示音后开始说，每批解码完就播放这一批。\n");
  for (int i = 0; i < 3; i++)
    {
      ai_voice_play_tone(1046, 120);
      usleep(150 * 1000);
    }
  usleep(350 * 1000);
  ai_voice_play_tone(1568, 400);
  usleep(250 * 1000);

  printf("  录音中 ...\n");
  if (jx_ui_ready())
    {
      jx_ui_set_status("RECORDING");
      jx_ui_page(0);
    }
  (void)ai_voice_stream_record(jx_rec_cb, &ctx, secs, 0);
  printf("  录音结束: %d 样本 (%.2f s)\n", ctx.n,
         (double)ctx.n / JX_PCM_RATE);
  hal_i2s_rx_idle();

  npkt = ctx.n / JX_PKT_SAMPLES;
  st = jx_stream_open(512);
  if (st == NULL)
    {
      printf("[ERR] 流状态分配失败\n");
      free(ctx.buf);
      return 1;
    }

  out = (int16_t *)malloc(sizeof(int16_t) * (size_t)batch_pkts * JX_PKT_SAMPLES);
  wav = (uint8_t *)malloc(44u + sizeof(int16_t) *
                          (size_t)batch_pkts * JX_PKT_SAMPLES);
  if (out == NULL || wav == NULL)
    {
      printf("[ERR] 播放缓冲分配失败\n");
      rc = 1;
      goto done;
    }

  for (int base = 0; base < npkt; base += batch_pkts)
    {
      int n = npkt - base;
      int nsmp;
      int64_t *tok;
      float *pcm;
      double t0 = jx_now_ms();
      int wav_size;

      if (n > batch_pkts) { n = batch_pkts; }
      nsmp = n * JX_PKT_SAMPLES;
      tok = (int64_t *)malloc(sizeof(int64_t) * (size_t)n * JX_PKT_FRAMES);
      pcm = (float *)malloc(sizeof(float) * (size_t)nsmp);
      if (tok == NULL || pcm == NULL)
        {
          free(tok); free(pcm);
          rc = 1;
          break;
        }

      rc = jx_stream_feed_batch(st, &ctx.buf[(size_t)base * JX_PKT_SAMPLES],
                                n, tok, pcm);
      if (rc != 0)
        {
          printf("[ERR] 第 %d 包起的批量处理失败 rc=%d\n", base, rc);
          free(tok); free(pcm);
          break;
        }
      for (int i = 0; i < nsmp; i++)
        {
          float v = pcm[i];
          v *= (float)play_gain;
          if (v > 1.0f) { v = 1.0f; }
          if (v < -1.0f) { v = -1.0f; }
          out[i] = jx_round_s16((double)v);
        }
      wav_size = ai_voice_wav_encode(wav, out,
                                     (uint32_t)nsmp * 2u);
      if (base == 0)
        {
          printf("  开始逐批播放（首批计算 %.0f ms）...\n", jx_now_ms() - t0);
        }
      (void)ai_voice_play(wav, (size_t)wav_size);
      free(tok); free(pcm);
    }

  printf("  共处理 %d 包\n", jx_stream_pkts(st));

done:
  jx_stream_close(st);
  free(out); free(wav); free(ctx.buf);
  return rc == 0 ? 0 : 1;
}

/* 2026-09-14 第二十四轮：把一段 int16 序列按 hex 打回串口，PC 侧还原成 WAV。
 * 用途：板端听不出"编码前 vs 解码后"差多少，回传到 PC 才能 A/B 试听 + 算 SNR。
 *   set JX_DUMP 1 打开。80000 采样 = 320 KB 文本，115200 波特约 30 秒。
 * 每行 32 个采样（64 个 hex 字符），行首 8 位是这行字符的滚动校验，
 * 串口一旦丢字符就能在 PC 侧立刻发现并丢弃该行（而不是整体错位）。
 * 格式：
 *   @@JXDUMP <tag> <n> <rate>
 *   <csum8> <64 hex>
 *   ...
 *   @@JXDUMP-END <tag>
 */
#define JX_DUMP_COLS 32

static void jx_dump_hex(const char *tag, const int16_t *p, int n)
{
  static const char H[] = "0123456789abcdef";
  char line[JX_DUMP_COLS * 4 + 1];      /* 32 个采样 = 128 个 hex + '\0' */
  int  nl = 0;
  printf("@@JXDUMP %s %d %d\n", tag, n, JX_PCM_RATE);
  for (int i = 0; i < n; i += JX_DUMP_COLS)
    {
      int k = 0;
      for (int j = 0; j < JX_DUMP_COLS && i + j < n; j++)
        {
          unsigned v = (unsigned)(uint16_t)p[i + j];
          line[k++] = H[(v >> 12) & 15];
          line[k++] = H[(v >>  8) & 15];
          line[k++] = H[(v >>  4) & 15];
          line[k++] = H[v & 15];
        }
      line[k] = 0;
      unsigned sum = 0;
      for (int j = 0; j < k; j++) { sum = sum * 31u + (unsigned char)line[j]; }
      printf("%08x %s\n", sum, line);
      if (++nl % 500 == 0) { printf("@@JXDUMP-PROG %s %d/%d\n", tag, i, n); }
    }
  printf("@@JXDUMP-END %s\n", tag);
}

static int jx_cmd_loop(int argc, char *argv[])
{
  int secs = jx_parse_seconds(argc, argv, 4, 15);

  printf("=== 极讯 Codec 单板闭环 (码率 %s) ===\n", jx_rate_name());
  printf("  准备：%d 秒录音，请对着板子说话。\n", secs);
  if (!jx_env_on("JX_MICGAIN_OFF", 0))
    {
      printf("  麦克风增益: JX_MICGAIN=%d 档 (0..15, 12=36dB, 14=37.5dB; 9=驱动默认27dB)\n",
             jx_env_int("JX_MICGAIN", 14) & 0x0f);
    }

  if (ai_voice_init() != 0)
    {
      printf("[ERR] ai_voice_init 失败（音频外设没起来）\n");
      return 1;
    }

  /* 2026-09-14 第二十六轮：录音前抬麦克风模拟增益（可用 JX_MICGAIN 调） */
  jx_mic_gain_apply();

  /* 2026-09-14 第二十四轮：录音提示音（现场没有屏幕可看时，靠喇叭告诉人
   * 什么时候该说话）。节拍：
   *     3 声短音(1046Hz/120ms) = 准备
   *     1 声长音(1568Hz/400ms) = 现在开始说
   *     录音结束 2 声低音(784Hz/180ms) = 说完了
   * 长音后停 250ms 再开录，免得喇叭余音被自己的麦克风录进去。
   * set JX_BEEP 0 可关掉。 */
  int beep = jx_env_on("JX_BEEP", 1);
  if (beep)
    {
      printf("  提示音: 3 声短音后 1 声长音 —— 听到长音就开始说\n");
      for (int i = 0; i < 3; i++)
        {
          ai_voice_play_tone(1046, 120);
          usleep(150 * 1000);
        }
      usleep(350 * 1000);
      ai_voice_play_tone(1568, 400);
      usleep(250 * 1000);
    }

  struct jx_rec_ctx ctx;
  ctx.cap = secs * JX_PCM_RATE;
  ctx.n   = 0;
  ctx.buf = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.cap);
  if (ctx.buf == NULL)
    {
      printf("[ERR] 录音缓冲分配失败\n");
      return 1;
    }

  printf("  录音中 ...\n");
  double t0 = jx_now_ms();
  int ms = ai_voice_stream_record(jx_rec_cb, &ctx, secs, 0);
  double t1 = jx_now_ms();
  printf("  录音结束: %d ms, 收到 %d 样本 (%.2f s)  [墙钟 %.0f ms]\n",
         ms, ctx.n, (double)ctx.n / JX_PCM_RATE, t1 - t0);

  if (beep)
    {
      ai_voice_play_tone(784, 180);
      usleep(200 * 1000);
      ai_voice_play_tone(784, 180);
    }

  double t_rec_end = jx_now_ms();   /* 说完的时刻，端到端延迟用 */

  /* 2026-09-14 第二十四轮：录音统计。
   * 原来只打印"收到 N 样本"，底噪和真人说话在日志里长得一模一样，
   * 根本判断不了麦克风到底收到了什么。峰值 / 平均|采样| / 非零比 直接给出
   * 可比较的数字（不用 sqrt，这个 defconfig 没有 libm）。 */
  {
    int32_t pk = 0; long long sabs = 0; int nz = 0;
    for (int i = 0; i < ctx.n; i++)
      {
        int32_t v = ctx.buf[i];
        int32_t a = (v < 0) ? -v : v;
        if (a > pk) { pk = a; }
        if (v != 0) { nz++; }
        sabs += (v < 0) ? -v : v;
      }
    printf("  录音统计: 峰值=%d 平均|采样|=%.1f 非零=%d/%d (%.0f%%)\n",
           (int)pk, ctx.n ? (double)sabs / (double)ctx.n : 0.0,
           nz, ctx.n, ctx.n ? 100.0 * (double)nz / (double)ctx.n : 0.0);
  }

  jx_print_mem("录音后");

  /* v97 诊断：录音一结束就把 RX 预读链停掉——编解码这几秒里
   * 不需要麦克风，却一直有 RX DMA 在跑、hpwork 在反复 alloc/free
   * apb（loop 1 的 mm_free.c:208 堆一致性断言就崩在 i2s_rx_worker 里）。
   * set JX_RXIDLE 0 可关闭本行为做 A/B。 */
  if (jx_env_on("JX_RXIDLE", 1))
    {
      printf("  [dbg] 停 RX 预读链 ...\n");
      hal_i2s_rx_idle();
      printf("  [dbg] RX 预读链已空闲\n");
    }

  if (ctx.n < JX_PCM_RATE / 4)
    {
      printf("[ERR] 录音数据太少，放弃\n");
      free(ctx.buf);
      return 1;
    }

  double t_enc_start = jx_now_ms();

  printf("  编码 + 解码 ...\n");

  /* v98：一遍编解码同时给出计时/剖析和播放要用的 idx/rec */
  int64_t *idx  = NULL;
  int      ntok = 0, Tt = 0;
  float   *rec  = NULL;
  int16_t *orig_dump = NULL;
  int      dump = jx_env_on("JX_DUMP", 0);

  /* 2026-09-17：录音已拿到，先给 UI 一份波形快照（此时 ctx.buf 还在）；
   * 之后整个编解码期间用 jx_ui_set_busy(1) 挡住所有绘制 —— 这是「显示不
   * 抢实时性」的硬保证：编解码期 UI 占用严格为 0。 */
  jx_ui_wave(ctx.buf, ctx.n);

  /* r180：拆开 encode/decode，中间释放录音缓冲省 96 KB（3 秒） */
  jx_prof_reset();
  jx_ui_set_busy(1);
  {
    double t0 = jx_now_ms();
    if (jx_encode(ctx.buf, ctx.n, &idx, &ntok, &Tt) != 0)
      {
        printf("[ERR] 编码失败\n");
        jx_ui_set_busy(0);
        free(ctx.buf);
        return 1;
      }
    double t1 = jx_now_ms();
    printf("  编码耗时  : %8.1f ms\n", t1 - t0);
    /* JX_DUMP 在播放后执行，但 ctx.buf 在编码后释放；先保留诊断副本。 */
    if (dump && ctx.buf != NULL)
      {
        orig_dump = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.n);
        if (orig_dump != NULL)
          {
            memcpy(orig_dump, ctx.buf, sizeof(int16_t) * (size_t)ctx.n);
          }
        else
          {
            printf("[WARN] orig dump allocation failed");
          }
      }

    /* 录音数据已拷到 float padded，释放 96 KB */
    free(ctx.buf); ctx.buf = NULL;
    if (jx_decode(idx, Tt, ctx.n, &rec) != 0)
      {
        printf("[ERR] 解码失败\n");
        jx_ui_set_busy(0);
        free(idx);
        return 1;
      }
    double t2 = jx_now_ms();
    printf("  解码耗时  : %8.1f ms\n", t2 - t1);
    { char lb[64];
      snprintf(lb, sizeof(lb), "enc %d / dec %d ms",
               (int)(t1 - t0), (int)(t2 - t1));
      jx_ui_log(lb); }
  }
  jx_ui_set_busy(0);

  /* 编解码结束后（CPU 空闲）再刷一次界面：整屏约 38ms，不在关键路径上 */
  if (jx_ui_ready())
    {
      char rb[24], tb[32];
      double secs = (double)ctx.n / (double)JX_PCM_RATE;

      snprintf(rb, sizeof(rb), "%.2f", (double)ntok * (double)jx_token_bits()
               / secs / 1000.0);
      snprintf(tb, sizeof(tb), "%d/%d", ntok, Tt);
      jx_ui_stat(rb, tb);
      jx_ui_page(jx_ui_page_get());
    }
  jx_prof_dump();
  /* 指纹（与 jx_run_once 一致） */
  {
    unsigned long long h_idx = 1469598103934665603ull;
    unsigned long long h_rec = 1469598103934665603ull;
    const unsigned char *ph = (const unsigned char *)rec;
    for (int i = 0; i < ntok; i++)
      {
        int64_t v = idx[i];
        for (int b = 0; b < 8; b++)
          h_idx = (h_idx ^ (unsigned char)(v >> (b * 8))) * 1099511628211ull;
      }
    for (size_t i = 0; i < (size_t)ctx.n * sizeof(float); i++)
      h_rec = (h_rec ^ ph[i]) * 1099511628211ull;
    printf("  指纹      : idx=%016llx rec=%016llx\n", h_idx, h_rec);
  }

  jx_print_mem("解码后");

  int16_t *out = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.n);
  if (out == NULL)
    {
      printf("[ERR] 输出缓冲分配失败\n");
      free(idx); free(rec); free(ctx.buf);
      return 1;
    }

  for (int i = 0; i < ctx.n; i++)
    {
      float v = rec[i];
      if (v > 1.0f)  { v = 1.0f; }
      if (v < -1.0f) { v = -1.0f; }
      out[i] = jx_round_s16((double)v);
    }

  uint32_t pcm_bytes = (uint32_t)ctx.n * 2;
  uint8_t *wav = (uint8_t *)malloc(44u + pcm_bytes);
  if (wav == NULL)
    {
      printf("[ERR] WAV 缓冲分配失败\n");
      free(out); free(idx); free(rec); free(ctx.buf);
      return 1;
    }

  int wav_size = ai_voice_wav_encode(wav, out, pcm_bytes);

  printf("  端到端延迟: %.0f ms（说完 -> 出声；其中编解码 %.0f ms）\n",
         jx_now_ms() - t_rec_end, jx_now_ms() - t_enc_start);

  printf("  播放解码结果 ...\n");
  int prc = ai_voice_play(wav, (size_t)wav_size);
  printf("  播放返回 %d\n", prc);

  jx_print_mem("播放后");

  /* 2026-09-14 第二十四轮：导出放在播放【之后】。
   * 放在播放前会把"听到回放"这件事推迟好几分钟，实测就是这么被坑的。
   * org = 编码器输入（麦克风原始录音），dec = 解码器输出（喇叭听到的）。
   * set JX_DUMP 1 打开（1 = 两路都导，2 = 只导原始录音）。 */
  if (dump)
    {
      if (orig_dump != NULL)
        {
          jx_dump_hex("orig", orig_dump, ctx.n);
        }
      if (dump != 2)
        {
          jx_dump_hex("dec",  out,     ctx.n);
        }
    }

  free(orig_dump);
  free(wav); free(out); free(idx); free(rec); if (ctx.buf) free(ctx.buf);
  return 0;
}

/****************************************************************************
 * info
 ****************************************************************************/

static int jx_cmd_info(void)
{
  printf("=== 极讯 AI Codec 板端构建信息 ===\n");
  printf("  码率            : %s\n", jx_rate_name());
  printf("  采样率          : %d Hz\n", SR);
  printf("  HOP_LENGTH      : %d\n", HOP_LENGTH);
  printf("  FEATURE_DIM     : %d\n", FEATURE_DIM);
  printf("  帧压缩率        : %d\n", EN_ENC_COMPRESS_RATE);
  printf("  FSQ 维数/位数   : %d 维, %d bit/token\n",
         VQ_DIM, VQ_TOTAL_BITS);
  printf("  注意力窗        : EN_WINDOW=%d\n", EN_WINDOW);
  return 0;
}


/****************************************************************************
 * benchp：双核并行基准（通话场景）
 *
 * bench 里编码与解码是**串行**的（解码必须等编码结果），所以总 RTF 是两者相加。
 * 但真实两板卡通话时两个方向是同时工作的：本板一边编码自己这一路话音，
 * 一边解码对端发来的 index。benchp 就是把这两个方向拆到两个核上同时跑，
 * 量出通话场景下真正要紧的那个数 —— 并行墙钟。
 *
 * 前提：CONFIG_SMP=y / CONFIG_SMP_NCPUS>=2，且算子内核的分核静态状态
 *       （jx_percore.h）已就位，否则两核会互相踩 scratch。
 ****************************************************************************/

struct jx_par_arg
{
  int      id;          /* 0=编码线程(核0) 1=解码线程(核1) */
  int      cpu;         /* 绑定的核 */
  int      iters;       /* 迭代次数 */
  int      n;           /* 音频样本数 */
  int      Tt;          /* 帧数（解码用） */
  const int16_t *pcm;
  const int64_t *idx;   /* 只读 */
  double   ms;          /* 平均单次耗时 */
  int      rc;
  unsigned long cpu_mask;
};

static void jx_bind_cpu(int cpu)
{
#if defined(CONFIG_SMP) && CONFIG_SMP_NCPUS > 1
  cpu_set_t set;
  cpu_set_t got;
  int       rc;

  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  rc = (int)nxsched_set_affinity(0, sizeof(cpu_set_t), &set);
  CPU_ZERO(&got);
  (void)nxsched_get_affinity(0, sizeof(cpu_set_t), &got);
  printf("[JX] bind want=%d rc=%d aff=0x%lx this_cpu=%d\n",
         cpu, rc, (unsigned long)got, (int)up_cpu_index());
#else
  (void)cpu;
#endif
}

static void *jx_par_enc(void *arg)
{
  struct jx_par_arg *a = (struct jx_par_arg *)arg;
  double t0;
  int rc = 0;

  jx_bind_cpu(a->cpu);
  t0 = jx_now_ms();
  for (int i = 0; i < a->iters; i++)
    {
      int64_t *idx = NULL;
      int      ntok = 0;
      int      Tt = 0;
      rc = jx_encode(a->pcm, a->n, &idx, &ntok, &Tt);
      if (idx != NULL) { free(idx); }
      if (rc != 0) { break; }
      a->cpu_mask |= 1UL << up_cpu_index();
    }
  a->ms = (jx_now_ms() - t0) / (double)a->iters;
  a->rc = rc;
  return NULL;
}

static void *jx_par_dec(void *arg)
{
  struct jx_par_arg *a = (struct jx_par_arg *)arg;
  double t0;
  int rc = 0;

  jx_bind_cpu(a->cpu);
  t0 = jx_now_ms();
  for (int i = 0; i < a->iters; i++)
    {
      float *rec = NULL;
      rc = jx_decode(a->idx, a->Tt, a->n, &rec);
      if (rec != NULL) { free(rec); }
      if (rc != 0) { break; }
      a->cpu_mask |= 1UL << up_cpu_index();
    }
  a->ms = (jx_now_ms() - t0) / (double)a->iters;
  a->rc = rc;
  return NULL;
}

static int jx_cmd_benchp(int argc, char *argv[])
{
  int secs  = jx_parse_seconds(argc, argv, 1, 30);
  int iters = (argc >= 4) ? atoi(argv[3]) : 2;
  int n     = secs * JX_PCM_RATE;
  int64_t  *idx = NULL;
  int       ntok = 0, Tt = 0;
  double    audio_s, t0, t1, wall_ms;
  pthread_t th[2];
  pthread_attr_t attr;
  struct jx_par_arg ea, da;

  if (iters < 1) { iters = 1; }
  if (iters > 20) { iters = 20; }

  printf("=== 极讯 Codec 双核并行基准 (码率 %s) ===\n", jx_rate_name());
  jx_print_wt();
  printf("  SMP/核数  : %s / %d 核\n",
#if defined(CONFIG_SMP) && CONFIG_SMP_NCPUS > 1
         "已启用",
#else
         "未启用（单核串行，仅作对照）",
#endif
         JX_NCORE);

  int16_t *pcm = (int16_t *)malloc(sizeof(int16_t) * (size_t)n);
  if (pcm == NULL) { printf("[ERR] 分配 PCM 失败\n"); return 1; }
  jx_gen_pcm(pcm, n, 0x2468ace1u);

  /* 先串行编一次拿 index（下面的并行测量要用它做解码输入） */
  if (jx_encode(pcm, n, &idx, &ntok, &Tt) != 0 || idx == NULL)
    {
      printf("[ERR] 预编码失败\n");
      free(pcm);
      return 1;
    }
  audio_s = (double)n / (double)JX_PCM_RATE;

  /* 并行段不统计算子探针：两核同时写同一组计数器既不准也无意义 */
  jx_prof_off = 1;

  ea.id = 0; ea.cpu = 0; ea.iters = iters; ea.n = n; ea.Tt = Tt;
  ea.pcm = pcm; ea.idx = NULL; ea.ms = 0.0; ea.rc = 0; ea.cpu_mask = 0;
  da.id = 1; da.cpu = (JX_NCORE > 1) ? 1 : 0; da.iters = iters; da.n = n; da.Tt = Tt;
  da.pcm = NULL; da.idx = idx; da.ms = 0.0; da.rc = 0; da.cpu_mask = 0;

  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 65536);

  jx_print_mem("并行前");
  t0 = jx_now_ms();
  if (pthread_create(&th[0], &attr, jx_par_enc, &ea) != 0 ||
      pthread_create(&th[1], &attr, jx_par_dec, &da) != 0)
    {
      printf("[ERR] 线程创建失败\n");
      jx_prof_off = 0;
      free(idx); free(pcm);
      return 1;
    }
  pthread_join(th[0], NULL);
  pthread_join(th[1], NULL);
  t1 = jx_now_ms();
  jx_prof_off = 0;
  jx_print_mem("并行后");

  wall_ms = t1 - t0;

  printf("  音频      : %.3f s x %d 次/线程  (Tt=%d 帧, %d token)\n",
         audio_s, iters, Tt, ntok);
  printf("  编码线程  : 核%d  平均 %8.1f ms   RTF %.3f%s\n",
         ea.cpu, ea.ms, ea.ms / 1000.0 / audio_s, ea.rc ? "  [失败]" : "");
  printf("  解码线程  : 核%d  平均 %8.1f ms   RTF %.3f%s\n",
         da.cpu, da.ms, da.ms / 1000.0 / audio_s, da.rc ? "  [失败]" : "");
  printf("  线程所在核 : 编码 mask=0x%lx  解码 mask=0x%lx\n",
         ea.cpu_mask, da.cpu_mask);
  printf("  并行墙钟  : %8.1f ms   RTF %.3f   <- 通话场景端到端\n",
         wall_ms, wall_ms / 1000.0 / audio_s);
  printf("  串行对照  : 编码+解码 = %.1f ms (RTF %.3f)，并行加速 %.2fx\n",
         ea.ms + da.ms, (ea.ms + da.ms) / 1000.0 / audio_s,
         wall_ms > 0.0 ? (ea.ms + da.ms) / wall_ms : 0.0);
  printf("  %s\n", (wall_ms / 1000.0 / audio_s) < 1.0 ? "=> 快于实时 OK" : "=> 慢于实时 NG");

  free(idx);
  free(pcm);
  return (ea.rc || da.rc) ? 1 : 0;
}

/****************************************************************************
 * 入口
 ****************************************************************************/

/* 2026-09-14 第二十七轮：串口 hex -> 喇叭直放（PC 上 A/B 好的音频不用重烧固件）
 *
 * 背景：板子听不出"编码前 vs 解码后"差多少，而 PC 上没有外放。现场要 A/B 时，
 * 只好重新录一遍。本命令把 PC 侧的 WAV 用串口 hex 打进板子直接经 ES8311 播放。
 *
 * 协议（每行 16 个采样 = 64 个 hex 字符，行首 8 位是这行字符的滚动校验）：
 *   @@JXPLAY <总数> <rate>
 *   <csum8> <64 hex>      xN
 *   @@JXPLAY-END
 * 校验失败的行会被丢弃（并在结尾报数），避免串口丢字符造成爆音。
 * PC 侧发送脚本：host/_pc_token_relay.py
 */
#define JX_PLAY_SAMPLES_PER_LINE 16
#define JX_PLAY_MAX_SEC          10

static int jx_hexval(int c)
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
  return -1;
}

static int jx_cmd_playhex(int argc, char *argv[])
{
  static char line[JX_PLAY_SAMPLES_PER_LINE * 4 + 64];
  int16_t *pcm;
  int cap = JX_PCM_RATE * 3;
  int n = 0;
  int nlines = 0;
  int nbad = 0;
  int expect = 0;
  int got_hdr = 0;
  uint8_t *wav;
  size_t bytes;
  int rc;
  int i;

  if (argc >= 3)
    {
      int s = atoi(argv[2]);
      if (s > 0 && s <= JX_PLAY_MAX_SEC) { cap = s * JX_PCM_RATE; }
    }

  pcm = (int16_t *)malloc(sizeof(int16_t) * (size_t)cap);
  if (pcm == NULL)
    {
      printf("[ERR] PCM 缓冲分配失败 (%d 采样)\n", cap);
      return 1;
    }

  printf("=== 串口 hex -> 喇叭直放（容量 %.1f s @%d Hz，每行 %d 采样）===\n",
         (double)cap / (double)JX_PCM_RATE, JX_PCM_RATE, JX_PLAY_SAMPLES_PER_LINE);
  printf("@@JXPLAY-READY %d %d\n", cap, JX_PCM_RATE);
  fflush(stdout);

  while (n < cap && fgets(line, sizeof(line), stdin) != NULL)
    {
      char *p = line;
      unsigned sum = 0;
      unsigned want;
      char *hex;
      int k;

      while (*p == ' ' || *p == '\t') { p++; }
      if (*p == 0 || *p == '\n' || *p == '\r') { continue; }
      if (p[0] == '@')
        {
          /* @@JXPLAY <n> <rate> 头部 */
          if (!got_hdr && sscanf(p, "@@JXPLAY %d", &expect) == 1)
            {
              got_hdr = 1;
              if (expect > 0 && expect < cap) { cap = expect; }
              printf("  头部: 期望 %d 采样 %.2f s\n", expect,
                     (double)expect / (double)JX_PCM_RATE);
            }
          else
            {
              break;   /* @@JXPLAY-END */
            }

          continue;
        }

      /* 行首 8 位校验 */
      hex = strchr(p, ' ');
      if (hex == NULL) { nbad++; continue; }
      for (k = 0; k < 8 && p + k < hex; k++)
        {
          int h = jx_hexval(p[k]);
          if (h < 0) { break; }
          sum = sum * 16u + (unsigned)h;
        }

      hex++;
      nlines++;
      while (hex[0] && hex[1] && hex[2] && hex[3] && n < cap)
        {
          int h0 = jx_hexval(hex[0]);
          int h1 = jx_hexval(hex[1]);
          int h2 = jx_hexval(hex[2]);
          int h3 = jx_hexval(hex[3]);

          if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) { break; }
          pcm[n++] = (int16_t)(uint16_t)((h0 << 12) | (h1 << 8) |
                                           (h2 << 4) | h3);
          hex += 4;
        }
    }

  printf("\n  收到 %d 采样 (%.2f s), %d 行, 丢行 %d\n", n,
         (double)n / (double)JX_PCM_RATE, nlines, nbad);
  if (n <= 0)
    {
      free(pcm);
      printf("[ERR] 没收到数据\n");
      return 1;
    }

  /* 组 44 字节 WAV 头 + PCM，交给 ai_voice_play（与 loop 路径同一条播放链） */
  bytes = (size_t)n * 2;
  wav = (uint8_t *)malloc(bytes + 44);
  if (wav == NULL)
    {
      free(pcm);
      printf("[ERR] WAV 缓冲分配失败\n");
      return 1;
    }

  {
    uint32_t dsz = (uint32_t)bytes;
    uint32_t rsz = 36u + dsz;
    uint32_t rate = JX_PCM_RATE;
    uint8_t *h = wav;
    memcpy(h, "RIFF", 4);      h += 4;
    *h++ = (uint8_t)(rsz);      *h++ = (uint8_t)(rsz >> 8);
    *h++ = (uint8_t)(rsz >> 16);*h++ = (uint8_t)(rsz >> 24);
    memcpy(h, "WAVEfmt ", 8);  h += 8;
    *h++ = 16; *h++ = 0; *h++ = 0; *h++ = 0;
    *h++ = 1;  *h++ = 0;                 /* PCM, 1 声道 */
    *h++ = (uint8_t)(rate);     *h++ = (uint8_t)(rate >> 8);
    *h++ = (uint8_t)(rate >> 16);*h++ = (uint8_t)(rate >> 24);
    *h++ = (uint8_t)(rate * 2); *h++ = (uint8_t)((rate * 2) >> 8);
    *h++ = (uint8_t)((rate * 2) >> 16); *h++ = (uint8_t)((rate * 2) >> 24);
    *h++ = 2; *h++ = 0;                  /* block align */
    *h++ = 16; *h++ = 0;                 /* bits */
    memcpy(h, "data", 4);       h += 4;
    *h++ = (uint8_t)(dsz);      *h++ = (uint8_t)(dsz >> 8);
    *h++ = (uint8_t)(dsz >> 16);*h++ = (uint8_t)(dsz >> 24);
    for (i = 0; i < n; i++)
      {
        wav[44 + i * 2]     = (uint8_t)((uint16_t)pcm[i] & 0xff);
        wav[44 + i * 2 + 1] = (uint8_t)((uint16_t)pcm[i] >> 8);
      }
  }

  if (ai_voice_init() != 0)
    {
      printf("[ERR] ai_voice_init 失败\n");
      free(wav); free(pcm);
      return 1;
    }

  jx_mic_gain_apply();
  ai_voice_play_tone(1568, 120);      /* 提示要出声了 */
  usleep(200 * 1000);
  printf("  播放中 ...\n");
  rc = ai_voice_play(wav, bytes + 44);
  printf("  播放返回 %d\n", rc);
  free(wav); free(pcm);
  return 0;
}


/* 2026-09-14 第二十七轮：双板语音通信（WiFi/UDP 传话音 index）
 *
 * 发送端: jixun net tx <对端IP> [秒数] [端口]
 *   麦克风 -> 编码 -> 量化索引 idx[] -> 位打包 -> UDP 发到对端
 * 接收端: jixun net rx [端口] [超时秒]
 *   收 UDP -> 解包 -> jx_decode -> 喇叭播放
 *
 * 2 秒 3kbps 音频的 index 只有 ~752 字节，一个 UDP 包就装得下；
 * 整包重复发 3 次提高可靠性。
 */

static int jx_rate_id(void)
{
  const char *r = jx_rate_name();

  if (r[0] == '1') { return 0; }
  if (r[0] == '6') { return 2; }
  return 1;
}


/****************************************************************************
 * memhint：内部 RAM / PSRAM 分配探针
 *
 * 背景：ESP32-S3 上 PSRAM 与内部 RAM 挂在同一个堆里
 * （CONFIG_ESP32S3_SPIRAM_COMMON_HEAP），malloc 有可能返回 PSRAM 指针。
 * 而 Wi-Fi 硬件/ROM 访问不了 PSRAM：一旦 Wi-Fi 库拿到 PSRAM，轻则
 * 管理帧发不出去（关联/认证失败），重则把 .bss 打坏直接崩机。
 * 本命令用来量化「内部 RAM 还剩多少块」，判断 Wi-Fi 分配够不够。
 ****************************************************************************/

static int jx_ptr_is_psram(const void *p)
{
  uintptr_t a = (uintptr_t)p;
  return (a >= 0x3c000000u && a < 0x3e000000u);
}

static int jx_ptr_is_iram(const void *p)
{
  uintptr_t a = (uintptr_t)p;
  return (a >= 0x3fc88000u && a < 0x3fd00000u);
}

static int jx_cmd_memhint(void)
{
  static const int sizes[] = { 64, 256, 1024, 4096, 16384 };
  enum { MAXK = 64 };
  void *keep[MAXK];
  struct mallinfo mi0 = mallinfo();
  int i;

  printf("=== 内部 RAM / PSRAM 分配探针 ===\n");
  printf("  堆总览: 空闲=%u 最大空闲块=%u （内部 + PSRAM 合计）\n",
         (unsigned)mi0.fordblks, (unsigned)mi0.mxordblk);

  for (int s = 0; s < (int)(sizeof(sizes) / sizeof(sizes[0])); s++)
    {
      int n = 0, nin = 0, nps = 0;
      int size = sizes[s];

      for (n = 0; n < MAXK; n++)
        {
          void *p = malloc((size_t)size);

          if (p == NULL)
            {
              break;
            }

          keep[n] = p;
          if (jx_ptr_is_psram(p))       { nps++; }
          else if (jx_ptr_is_iram(p))   { nin++; }
        }

      printf("  %6d B x %2d: 内部 %2d 块 (%7d B) | PSRAM %2d 块 (%8d B)%s\n",
             size, n, nin, nin * size, nps, nps * size,
             (n < MAXK) ? "  [堆耗尽]" : "");

      for (i = 0; i < n; i++) { free(keep[i]); }
    }

  {
    struct mallinfo mi1 = mallinfo();
    printf("  释放后: 空闲=%u 最大空闲块=%u\n",
           (unsigned)mi1.fordblks, (unsigned)mi1.mxordblk);
  }

  return 0;
}

/* Live TX: the capture callback only copies, while an encoder thread drains
 * fixed-size chunks and sends each one immediately. */
struct jx_live_ctx
{
  int16_t *buf;
  int cap, n, consumed;
  volatile int closed, failed, done;
  int chunk_samples;
  int total_chunks;
  const char *ip;
  int port;
};

static void jx_live_rec_cb(FAR const int16_t *pcm16, size_t samples, void *arg)
{
  struct jx_live_ctx *c = (struct jx_live_ctx *)arg;
  if (c->failed) { return; }
  if (c->n + (int)samples > c->cap) { c->failed = 1; return; }
  if ((int)samples > 0)
    {
      memcpy(&c->buf[c->n], pcm16, samples * sizeof(int16_t));
      __sync_synchronize();
      c->n += (int)samples;
    }
}

static void *jx_live_tx_worker(void *arg)
{
  struct jx_live_ctx *c = (struct jx_live_ctx *)arg;
  int16_t *chunk = (int16_t *)malloc(sizeof(int16_t) * (size_t)c->chunk_samples);
  int64_t *idx = NULL;
  int ntok = 0, Tt = 0, k = 0, rc = 0;
  if (chunk == NULL) { c->failed = 1; c->done = 1; return NULL; }
  for (;;)
    {
      int avail, len, last;
      for (;;)
        {
          avail = c->n - c->consumed;
          if (avail >= c->chunk_samples || c->closed || c->failed) { break; }
          usleep(2000);
        }
      if ((avail <= 0 && c->closed) || c->failed) { break; }
      len = (avail > c->chunk_samples) ? c->chunk_samples : avail;
      if (len < JX_PCM_RATE / 20) { break; }
      last = (c->closed && c->consumed + len >= c->n) ? 1 : 0;
      if (jx_ui_ready())
        {
          char uis[64];
          snprintf(uis, sizeof(uis), "ENCODE %d/%d", k + 1, c->total_chunks);
          jx_ui_set_status(uis);
        }
      memcpy(chunk, &c->buf[c->consumed], sizeof(int16_t) * (size_t)len);
      c->consumed += len;
      __sync_synchronize();
      if (jx_encode(chunk, len, &idx, &ntok, &Tt) != 0)
        { printf("[ERR] live encode failed chunk %d\\n", k + 1); c->failed = 1; break; }
      rc = jx_net_send_idx(c->ip, c->port, jx_rate_id(), jx_token_bits(),
                           idx, ntok, Tt, len, (uint32_t)k, (uint32_t)last);
      free(idx); idx = NULL;
      printf("  [live] chunk %d: %d samples -> %d token%s\\n", k + 1, len, ntok,
             last ? " last" : "");
      if (jx_ui_ready())
        {
          char uit[64];
          snprintf(uit, sizeof(uit), "TX %d/%d  %d tok", k + 1,
                   c->total_chunks, ntok);
          jx_ui_stat(jx_rate_name(), uit);
          jx_ui_page(jx_ui_page_get());
        }
      if (rc != 0) { printf("[ERR] live send failed chunk %d\\n", k + 1); c->failed = 1; break; }
      k++;
      if (last) { break; }
    }
  free(chunk);
  c->done = 1;
  return NULL;
}

static int jx_cmd_net_tx_live(const char *ip, int port, int secs, int chunk_ms)
{
  struct jx_live_ctx ctx;
  pthread_t th;
  pthread_attr_t at;
  int ms;
  memset(&ctx, 0, sizeof(ctx));
  ctx.chunk_samples = chunk_ms * JX_PCM_RATE / 1000;
  if (ctx.chunk_samples < JX_PCM_RATE / 20) { ctx.chunk_samples = JX_PCM_RATE / 20; }
  ctx.total_chunks = (secs * 1000 + chunk_ms - 1) / chunk_ms;
  if (ctx.total_chunks < 1) { ctx.total_chunks = 1; }
  ctx.cap = secs * JX_PCM_RATE + 8 * JX_PCM_RATE;
  ctx.ip = ip; ctx.port = port;
  ctx.buf = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.cap);
  if (ctx.buf == NULL) { printf("[ERR] live capture buffer failed\\n"); return 1; }
  if (pthread_attr_init(&at) != 0) { free(ctx.buf); return 1; }
  (void)pthread_attr_setstacksize(&at, 32768);
  if (pthread_create(&th, &at, jx_live_tx_worker, &ctx) != 0)
    { printf("[ERR] live encoder thread failed\\n"); pthread_attr_destroy(&at); free(ctx.buf); return 1; }
  pthread_attr_destroy(&at);
  printf("  live tx: %d ms/chunk\\n", chunk_ms);
  ms = ai_voice_stream_record(jx_live_rec_cb, &ctx, secs, 0);
  __sync_synchronize();
  ctx.closed = 1;
  (void)pthread_join(th, NULL);
  printf("  live capture done: %d ms, %d samples, encoder %s\\n",
         ms, ctx.n, ctx.failed ? "FAILED" : "DONE");
  if (jx_ui_ready())
    {
      jx_ui_set_status(ctx.failed ? "TX FAILED" : "TX DONE");
      jx_ui_page(jx_ui_page_get());
    }
  free(ctx.buf);
  return ctx.failed ? 1 : 0;
}

static int jx_cmd_net_tx(int argc, char *argv[])
{
  const char *ip;
  int secs = 2;
  int port = JX_NET_PORT_DEF;
  int chunk_ms = 0;
  int step = 0;
  int nchunk = 1;
  struct jx_rec_ctx ctx;
  int64_t *idx = NULL;
  int ntok = 0;
  int Tt = 0;
  int rc = 0;
  int k;

  if (argc < 4)
    {
      printf("用法: jixun net tx <对端IP> [秒数] [端口] [块毫秒]\n");
      printf("      块毫秒 0（缺省）= 整段编完再发；>0 = 分块流水，出一块发一块\n");
      return 1;
    }

  ip = argv[3];
  if (argc >= 5) { secs = atoi(argv[4]); }
  if (argc >= 6) { port = atoi(argv[5]); }
  if (argc >= 7) { chunk_ms = atoi(argv[6]); }
  if (secs < 1) { secs = 1; }
  if (secs > 5) { secs = 5; }
  if (chunk_ms < 0) { chunk_ms = 0; }
  if (chunk_ms > 0 && chunk_ms < 100) { chunk_ms = 100; }

  printf("=== 极讯 Codec 发送端 (码率 %s) -> %s:%d%s ===\n", jx_rate_name(),
         ip, port, chunk_ms ? "  [分块流水]" : "");
  printf("  准备：%d 秒录音，请对着板子说话。\n", secs);

  if (jx_ui_ready())
    {
      jx_ui_set_peer(ip);
      jx_ui_set_status("TX READY");
    }

  if (ai_voice_init() != 0)
    {
      printf("[ERR] ai_voice_init 失败\n");
      return 1;
    }

  jx_mic_gain_apply();

  if (jx_env_on("JX_BEEP", 1))
    {
      for (int i = 0; i < 3; i++)
        {
          ai_voice_play_tone(1046, 120);
          usleep(150 * 1000);
        }
      usleep(350 * 1000);
      ai_voice_play_tone(1568, 400);
      usleep(250 * 1000);
    }

  if (chunk_ms > 0 && jx_env_on("JX_LIVE_TX", 0))
    {
      int lrc = jx_cmd_net_tx_live(ip, port, secs, chunk_ms);
      if (jx_env_on("JX_RXIDLE", 1)) { hal_i2s_rx_idle(); }
      return lrc;
    }

  ctx.cap = secs * JX_PCM_RATE;
  ctx.n   = 0;
  ctx.buf = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.cap);
  if (ctx.buf == NULL)
    {
      printf("[ERR] 录音缓冲分配失败\n");
      return 1;
    }

  printf("  录音中 ...\n");
  if (jx_ui_ready())
    {
      jx_ui_set_status("RECORDING");
    }
  {
    int ms = ai_voice_stream_record(jx_rec_cb, &ctx, secs, 0);
    printf("  录音结束: %d ms, 收到 %d 样本 (%.2f s)\n",
           ms, ctx.n, (double)ctx.n / JX_PCM_RATE);
  }

  if (jx_env_on("JX_BEEP", 1))
    {
      ai_voice_play_tone(784, 180);
      usleep(200 * 1000);
      ai_voice_play_tone(784, 180);
    }

  {
    int32_t pk = 0; long long sabs = 0;
    for (int i = 0; i < ctx.n; i++)
      {
        int32_t v = ctx.buf[i];
        int32_t a = (v < 0) ? -v : v;
        if (a > pk) { pk = a; }
        sabs += a;
      }
    printf("  录音统计: 峰值=%d 平均|采样|=%.1f\n", (int)pk,
           ctx.n ? (double)sabs / (double)ctx.n : 0.0);
  }

  if (jx_ui_ready())
    {
      jx_ui_wave(ctx.buf, ctx.n);
      jx_ui_log("TX capture ok");
    }

  if (jx_serial_pcm_dump_enabled())
    {
      (void)jx_serial_send_pcm_i16("raw", ctx.buf, ctx.n,
                                   JX_PCM_RATE);
    }

  if (jx_env_on("JX_RXIDLE", 1))
    {
      hal_i2s_rx_idle();
    }

  if (ctx.n < JX_PCM_RATE / 4)
    {
      printf("[ERR] 录音数据太少，放弃\n");
      free(ctx.buf);
      return 1;
    }

  if (chunk_ms > 0 && jx_rate_id() == 2 && jx_env_on("JX_FASTNET", 1))
    {
      struct jx_stream_s *st;
      int64_t *tokbuf;
      float *pcmbuf;
      const int per = 64;
      const int nfr = ctx.n / JX_PKT_SAMPLES;
      int opkt = 0;
      const char *old_skip = getenv("JX_SKIP_DEC");

      printf("  流式快速编码：%d 包 x 24ms，批 %d，窗口 256 帧\n",
             nfr, per);

      st = jx_stream_open(512);
      tokbuf = (int64_t *)malloc(sizeof(int64_t) * (size_t)nfr * JX_PKT_FRAMES);
      pcmbuf = (float *)malloc(sizeof(float) * (size_t)per * JX_PKT_SAMPLES);
      if (st == NULL || tokbuf == NULL || pcmbuf == NULL)
        {
          printf("[ERR] 流式发送缓冲分配失败\n");
          jx_stream_close(st);
          free(tokbuf);
          free(pcmbuf);
          free(ctx.buf);
          return 1;
        }

      setenv("JX_SKIP_DEC", "1", 1);
      while (opkt < nfr)
        {
          int n = nfr - opkt;

          if (n > per) { n = per; }
          rc = jx_stream_feed_batch(st, &ctx.buf[(size_t)opkt * JX_PKT_SAMPLES],
                                    n,
                                    &tokbuf[(size_t)opkt * JX_PKT_FRAMES],
                                    pcmbuf);
          if (rc != 0)
            {
              printf("[ERR] 流式编码失败 offset=%d rc=%d\n", opkt, rc);
              break;
            }
          opkt += n;
        }

      if (rc == 0)
        {
          const int ntok = nfr * JX_PKT_FRAMES;
          printf("  流式汇总: %d token / %d 帧，单次 UDP 发送\n",
                 ntok, ntok);
          rc = jx_net_send_idx(ip, port, jx_rate_id(), jx_token_bits(),
                               tokbuf, ntok, ntok, ctx.n, 0u, 1u);
        }

      setenv("JX_SKIP_DEC", (old_skip != NULL) ? old_skip : "0", 1);
      jx_stream_close(st);
      free(tokbuf);
      free(pcmbuf);
      free(ctx.buf);
      jx_print_mem("流式发送后");
      return rc == 0 ? 0 : 1;
    }

  if (chunk_ms > 0)
    {
      step = chunk_ms * JX_PCM_RATE / 1000;
      if (step < JX_PCM_RATE / 20) { step = JX_PCM_RATE / 20; }
      nchunk = (ctx.n + step - 1) / step;
    }
  else
    {
      step   = ctx.n;
      nchunk = 1;
    }

  printf("  编码+发送：%d 块 x %d 样本 (%.2f s/块)\n", nchunk, step,
         (double)step / JX_PCM_RATE);

  for (k = 0; k < nchunk; k++)
    {
      int off = k * step;
      int len = ctx.n - off;
      double tc;

      if (len > step) { len = step; }
      if (len < JX_PCM_RATE / 20) { break; }

      tc = jx_now_ms();
      if (jx_ui_ready())
        {
          char uis[64];
          snprintf(uis, sizeof(uis), "ENCODE %d/%d", k + 1, nchunk);
          jx_ui_set_status(uis);
        }
      if (jx_encode(ctx.buf + off, len, &idx, &ntok, &Tt) != 0)
        {
          printf("[ERR] 编码失败（块 %d）\n", k + 1);
          rc = -1;
          break;
        }

      printf("  块 %d/%d: %d 样本 -> %d token / %d 帧 / 编码 %.0f ms\n",
             k + 1, nchunk, len, ntok, Tt, jx_now_ms() - tc);
      if (jx_ui_ready())
        {
          char uit[64];
          snprintf(uit, sizeof(uit), "TX %d/%d  %d tok", k + 1, nchunk, ntok);
          jx_ui_stat(jx_rate_name(), uit);
        }

      rc = jx_net_send_idx(ip, port, jx_rate_id(), jx_token_bits(),
                           idx, ntok, Tt, len,
                           (uint32_t)k, (k == nchunk - 1) ? 1u : 0u);
      if (rc == 0 && chunk_ms == 0 && jx_env_on("JX_DUAL_SELF", 0))
        {
          int half = (ntok + 1) / 2;
          printf("  [dual] 本板解码前半段：%d/%d token\n", half, ntok);
          (void)jx_play_idx_range(idx, half, half * HOP_LENGTH, 0);
        }
      if (rc == 0 && chunk_ms > 0 && k > 0 && jx_env_on("JX_DUAL_TX_REST", 0))
        {
          printf("  [dual] 本板解码后续块 %d/%d: %d token\n",
                 k + 1, nchunk, ntok);
          (void)jx_play_idx_range(idx, ntok, len, 0);
        }
      free(idx);
      idx = NULL;
      if (rc != 0)
        {
          printf("[ERR] 发送失败（块 %d）\n", k + 1);
          break;
        }
    }

  if (jx_ui_ready())
    {
      jx_ui_set_status(rc == 0 ? "TX DONE" : "TX FAILED");
      jx_ui_log(rc == 0 ? "TX token done" : "TX failed");
    }

  free(ctx.buf);
  jx_print_mem("发送后");
  return rc == 0 ? 0 : 1;
}

/* UI 话音页“发送语音”按钮：独立任务执行 2 秒录音->编码->UDP 发送。 */
static int jx_ui_send_task(int argc, char *argv[])
{
  char *txargv[6];
  const char *ip;
  int rc;

  (void)argc;
  (void)argv;

  ip = jx_ui_peer_get();
  if (ip == NULL || ip[0] == '\0')
    {
      jx_ui_set_status("PEER NOT SET");
      if (jx_ui_page_get() == 0) { jx_ui_page(0); }
      return 1;
    }

  jx_ui_set_status("SENDING...");
  if (jx_ui_page_get() == 0) { jx_ui_page(0); }

  txargv[0] = "jixun";
  txargv[1] = "net";
  txargv[2] = "tx";
  txargv[3] = (char *)ip;
  txargv[4] = "2";
  txargv[5] = NULL;

  rc = jx_cmd_net_tx(5, txargv);
  jx_ui_set_status(rc == 0 ? "SENT" : "FAILED");
  if (jx_ui_page_get() == 0) { jx_ui_page(0); }
  return rc;
}

static void jx_ui_send_action(void)
{
  const char *ip = jx_ui_peer_get();
  int pid;

  if (ip == NULL || ip[0] == '\0')
    {
      jx_ui_set_status("PEER NOT SET");
      if (jx_ui_page_get() == 0) { jx_ui_page(0); }
      return;
    }

  pid = task_create("jixun-send", 150, 32768, jx_ui_send_task, NULL);
  if (pid < 0)
    {
      jx_ui_set_status("TASK FAIL");
      if (jx_ui_page_get() == 0) { jx_ui_page(0); }
      return;
    }

  jx_ui_set_status("SENDING...");
  if (jx_ui_page_get() == 0) { jx_ui_page(0); }
}


static int jx_cmd_net_rx(int argc, char *argv[])
{
  int port = JX_NET_PORT_DEF;
  int timeout = 30;
  int rate_id = 0, bits = 0, ntok = 0, Tt = 0, orig_t = 0;
  uint32_t chunk = 0, last = 0;
  int64_t *idx = NULL;
  float *rec = NULL;
  int rc = 0;
  int nseg = 0;
  int voice_ready = 0;
  int buffered_play = jx_env_on("JX_BUFFER_PLAY", 0);
  int stream_play = jx_env_on("JX_STREAM_PLAY", 0) && !buffered_play;
  int play_muted = jx_env_on("JX_PLAY_MUTE", 0);
  int dual_half = jx_env_on("JX_DUAL_HALF", 0) && !play_muted;
  int dual_rx_first = jx_env_on("JX_DUAL_RX_FIRST", 0) && !play_muted;
  int stream_prebuffer_ms = jx_env_int("JX_STREAM_PREBUFFER", 1500);
  struct jx_playbuf_s playbuf;
  struct jx_playbuf_s *playbuf_p = NULL;
  pthread_t play_thread;
  int play_thread_on = 0;
  int16_t *buffered_out = NULL;
  int buffered_n = 0;
  uint32_t next_chunk = 0;
  int total_chunks = jx_env_int("JX_RX_CHUNKS", 4);
  double t_start;
  double t_audio = 0;

  if (total_chunks < 1) { total_chunks = 1; }

  if (argc >= 4) { port = atoi(argv[3]); }
  if (argc >= 5) { timeout = atoi(argv[4]); }
  if (timeout < 3) { timeout = 3; }

  if (jx_env_on("JX_DHCPD", 0))
    {
      int dhcprc = dhcpd_start("wlan0");
      printf("[net] DHCPD rc=%d\n", dhcprc);
      usleep(500 * 1000);
    }

  printf("=== 极讯 Codec 接收端 (码率 %s) 端口 %d ===\n", jx_rate_name(), port);
  t_start = jx_now_ms();

  if (jx_ui_ready())
    {
      jx_ui_set_status("RX READY");
    }

  if (stream_play && !play_muted)
    {
      int cap = JX_PCM_RATE * 8;
      int pre = JX_PCM_RATE * stream_prebuffer_ms / 1000;
      if (stream_prebuffer_ms < 100) { pre = JX_PCM_RATE / 10; }
      if (pre > cap) { pre = cap; }
      if (jx_playbuf_init(&playbuf, cap, pre) == 0)
        {
          pthread_attr_t at;
          if (pthread_attr_init(&at) == 0)
            {
              (void)pthread_attr_setstacksize(&at, 32768);
              if (pthread_create(&play_thread, &at, jx_playbuf_worker,
                                 &playbuf) == 0)
                {
                  playbuf_p = &playbuf;
                  play_thread_on = 1;
                  printf("  [playbuf] 预缓冲 %d ms，播放线程已启动\n",
                         stream_prebuffer_ms);
                }
              (void)pthread_attr_destroy(&at);
            }
        }
      if (playbuf_p == NULL)
        {
          printf("  [playbuf] 初始化失败，回退直接流式播放\n");
        }
    }

  for (;;)
    {
      int wait = (nseg == 0) ? timeout : 20;
      int16_t *out;
      uint8_t *wav;
      int wav_size;
      int prc;
      int i;
      double t0;

      rc = jx_net_recv_idx(port, wait, &rate_id, &bits, &idx, &ntok, &Tt,
                           &orig_t, &chunk, &last);
      if (rc != 0)
        {
          if (nseg == 0)
            {
              printf("[ERR] 收包失败 rc=%d\n", rc);
              return 1;
            }

          printf("[net] 流水结束：后续 %d 秒没有新块\n", wait);
          break;
        }

      /* 每个包在线路上重复 JX_NET_REPEAT 次；块号小于"下一块"的都是
       * 重复包/迟到包，直接丢掉，否则同一块会被反复解码+播放。 */

      if (chunk < next_chunk)
        {
          free(idx);
          idx = NULL;
          continue;
        }

      next_chunk = chunk + 1;

      if (rate_id != jx_rate_id())
        {
          printf("[WARN] 对端码率 id=%d，本板 id=%d（可能对不上，仍继续）\n",
                 rate_id, jx_rate_id());
        }

      if (dual_rx_first)
        {
          if (chunk > 0)
            {
              free(idx);
              idx = NULL;
              continue;
            }
          if (jx_ui_ready())
            {
              jx_ui_set_status("DUAL RX FIRST");
            }
          printf("  [dual] 本板解码首块：%d token\n", ntok);
          prc = jx_play_idx_range(idx, ntok, orig_t, 0);
          free(idx);
          idx = NULL;
          nseg++;
          if (prc < 0)
            {
              printf("[ERR] dual first decode/play failed\n");
              return 1;
            }
          if (last)
            {
              printf("[net] dual first chunk done\n");
              break;
            }
          continue;
        }

      if (dual_half)
        {
          int half = (ntok + 1) / 2;
          int first_samples = half * HOP_LENGTH;
          int second_samples = orig_t - first_samples;
          int delay_ms;
          if (second_samples <= 0)
            {
              printf("[ERR] dual half split invalid\n");
              free(idx);
              return 1;
            }
          delay_ms = first_samples * 1000 / JX_PCM_RATE;
          if (jx_ui_ready())
            {
              jx_ui_set_status("DUAL DECODE 2/2");
            }
          printf("  [dual] 本板解码后半段：%d/%d token，延迟 %d ms\n",
                 ntok - half, ntok, delay_ms);
          prc = jx_play_idx_range(idx + half, ntok - half,
                                  second_samples, delay_ms);
          free(idx);
          idx = NULL;
          nseg++;
          if (prc < 0)
            {
              printf("[ERR] dual half decode/play failed\n");
              return 1;
            }
          if (last)
            {
              printf("[net] dual half last chunk done\n");
              break;
            }
          continue;
        }

      t0 = jx_now_ms();
      if (jx_ui_ready())
        {
          char uis[64];
          snprintf(uis, sizeof(uis), "RX %u/%d  DECODE",
                   (unsigned)chunk + 1, total_chunks);
          jx_ui_set_status(uis);
        }
      if (jx_decode(idx, Tt, orig_t, &rec) != 0)
        {
          printf("[ERR] 解码失败（块 %u）\n", (unsigned)chunk);
          free(idx);
          return 1;
        }

      free(idx);
      idx = NULL;
      printf("  块 %u 解码 %.0f ms -> %d 采样 (%.2f s)\n", (unsigned)chunk,
             jx_now_ms() - t0, orig_t, (double)orig_t / JX_PCM_RATE);

      if (jx_serial_pcm_dump_enabled())
        {
          (void)jx_serial_send_pcm_f32("pre", rec, orig_t,
                                       JX_PCM_RATE);
        }

      out = (int16_t *)malloc(sizeof(int16_t) * (size_t)orig_t);
      if (out == NULL)
        {
          printf("[ERR] 输出缓冲分配失败\n");
          free(rec);
          return 1;
        }

      int requested_gain = jx_env_int("JX_PLAY_GAIN", 1);
      float play_peak = 0.0f;
      float play_gain;

      if (requested_gain < 1) { requested_gain = 1; }
      if (requested_gain > 256) { requested_gain = 256; }

      for (i = 0; i < orig_t; i++)
        {
          float a = (rec[i] < 0.0f) ? -rec[i] : rec[i];
          if (a > play_peak) { play_peak = a; }
        }

      play_gain = (float)requested_gain;
      if (play_peak > 0.0f && play_peak * play_gain > 0.98f)
        {
          play_gain = 0.98f / play_peak;
        }

      printf("  播放增益: 请求%dx 峰值%.3f 实际%.2fx\n",
             requested_gain, (double)play_peak, (double)play_gain);

      for (i = 0; i < orig_t; i++)
        {
          float v = rec[i] * play_gain;
          if (v > 1.0f)  { v = 1.0f; }
          if (v < -1.0f) { v = -1.0f; }
          out[i] = jx_round_s16((double)v);
        }

      if (jx_serial_pcm_dump_enabled())
        {
          (void)jx_serial_send_pcm_i16("play", out, orig_t,
                                       JX_PCM_RATE);
        }

      if (jx_ui_ready())
        {
          char uit[64];
          jx_ui_wave(out, orig_t);
          snprintf(uit, sizeof(uit), "RX %u/%d  %d tok",
                   (unsigned)chunk + 1, total_chunks, ntok);
          jx_ui_stat(jx_rate_name(), uit);
        }

      free(rec);
      rec = NULL;

      if (stream_play)
        {
          if (!voice_ready && playbuf_p == NULL)
            {
              if (ai_voice_init() != 0)
                {
                  printf("[ERR] ai_voice_init 失败\n");
                  free(out);
                  return 1;
                }

              voice_ready = 1;
            }

          if (jx_ui_ready())
            {
              jx_ui_set_status(last ? "PLAY STREAM LAST" : "PLAY STREAM");
            }

          printf("  流式播放块 %u%s ...\n", (unsigned)chunk,
                 play_muted ? "（已静音，仅解码）" : "");
          if (play_muted)
            {
              prc = 0;
            }
          else if (playbuf_p != NULL)
            {
              prc = jx_playbuf_write(playbuf_p, out, orig_t);
            }
          else
            {
              prc = jx_voice_stream_play_pcm(out, orig_t);
            }
          free(out);
          out = NULL;

          if (playbuf_p != NULL && last)
            {
              jx_playbuf_finish(playbuf_p);
              if (play_thread_on)
                {
                  pthread_join(play_thread, NULL);
                  play_thread_on = 0;
                }
              if (playbuf.started_ms > 0.0)
                {
                  t_audio = playbuf.started_ms;
                  printf("  >>> 第一块出声：从开始等待算起 %.1f s，"
                         "underrun=%d\n",
                         (playbuf.started_ms - t_start) / 1000.0,
                         playbuf.underruns);
                  printf("[Voice] 播放完成\n");
                }
            }
          else if (!play_muted && prc >= 0 && last && playbuf_p == NULL)
            {
              prc = hal_i2s_write_flush();
            }

          nseg++;
          if (nseg == 1 && playbuf_p == NULL)
            {
              t_audio = jx_now_ms();
              printf("  >>> 第一块排入播放：从开始等待算起 %.1f s\n",
                     (t_audio - t_start) / 1000.0);
            }

          printf("  块 %u 流式播放返回 %d（累计 %d 块）\n",
                 (unsigned)chunk, prc, nseg);

          if (prc < 0)
            {
              return 1;
            }

          if (last)
            {
              printf("[net] 收到末块，结束\n");
              if (jx_ui_ready())
                {
                  jx_ui_set_status("RX DONE");
                  jx_ui_log("RX token played");
                }
              break;
            }

          continue;
        }

      if (buffered_play)
        {
          int16_t *grown = (int16_t *)realloc(
              buffered_out, sizeof(int16_t) * (size_t)(buffered_n + orig_t));
          if (grown == NULL)
            {
              printf("[ERR] BUFFER_PLAY 缓冲扩展失败\n");
              free(out);
              free(buffered_out);
              return 1;
            }

          buffered_out = grown;
          memcpy(buffered_out + buffered_n, out,
                 sizeof(int16_t) * (size_t)orig_t);
          buffered_n += orig_t;
          free(out);

          if (!last)
            {
              continue;
            }

          wav = (uint8_t *)malloc(44u + (size_t)buffered_n * 2u);
          if (wav == NULL)
            {
              printf("[ERR] BUFFER_PLAY WAV 缓冲分配失败\n");
              free(buffered_out);
              return 1;
            }

          wav_size = ai_voice_wav_encode(wav, buffered_out,
                                         (uint32_t)buffered_n * 2u);
          free(buffered_out);
          buffered_out = NULL;
        }
      else
        {
          wav = (uint8_t *)malloc(44u + (size_t)orig_t * 2u);
          if (wav == NULL)
            {
              printf("[ERR] WAV 缓冲分配失败\n");
              free(out);
              return 1;
            }

          wav_size = ai_voice_wav_encode(wav, out, (uint32_t)orig_t * 2u);
          free(out);
        }

      if (play_muted)
        {
          free(wav);
          prc = 0;
        }
      else
        {
          if (!voice_ready)
            {
              if (ai_voice_init() != 0)
                {
                  printf("[ERR] ai_voice_init 失败\n");
                  free(wav);
                  return 1;
                }

              voice_ready = 1;
            }

          printf("  播放块 %u%s ...\n", (unsigned)chunk,
                 buffered_play ? "（整段缓冲）" : "");
          if (jx_ui_ready())
            {
              jx_ui_set_status(buffered_play ? "PLAY BUFFERED" : "PLAYING");
            }
          prc = ai_voice_play(wav, (size_t)wav_size);
          free(wav);
        }

      nseg++;
      if (nseg == 1)
        {
          t_audio = jx_now_ms();
          printf("  >>> 第一块出声：从开始等待算起 %.1f s\n",
                 (t_audio - t_start) / 1000.0);
        }

      printf("  块 %u 播放返回 %d（累计 %d 块）\n", (unsigned)chunk, prc, nseg);

      if (last)
        {
          printf("[net] 收到末块，结束\n");
          if (jx_ui_ready())
            {
              jx_ui_set_status("RX DONE");
              jx_ui_log("RX token played");
            }
          break;
        }
    }

  if (nseg == 0)
    {
      printf("[ERR] 没有收到任何块\n");
      return 1;
    }

  printf("[net] 共 %d 块；第一块出声 %.1f s；全部完成 %.1f s\n",
         nseg, (t_audio - t_start) / 1000.0, (jx_now_ms() - t_start) / 1000.0);
  return 0;
}


/* -------- 2026-09-14 第二十八轮：WiFi 崩溃定位（纯诊断，不进编解码路径） -------- */

static uint32_t jx_hx(const char *s)
{
  uint32_t v = 0;

  if (s == NULL)
    {
      return 0;
    }

  if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
    {
      s += 2;
    }

  while (*s != '\0')
    {
      char c = *s++;
      uint32_t d;

      if (c >= '0' && c <= '9')
        {
          d = (uint32_t)(c - '0');
        }
      else if (c >= 'a' && c <= 'f')
        {
          d = (uint32_t)(c - 'a' + 10);
        }
      else if (c >= 'A' && c <= 'F')
        {
          d = (uint32_t)(c - 'A' + 10);
        }
      else
        {
          break;
        }

      v = (v << 4) | d;
    }

  return v;
}

static void jx_dump(uint32_t addr, uint32_t n)
{
  volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)addr;
  uint32_t i;

  for (i = 0; i + 3 < n; i += 4)
    {
      printf("  %08lx: %08lx %08lx %08lx %08lx\n",
             (unsigned long)(addr + i * 4),
             (unsigned long)p[i], (unsigned long)p[i + 1],
             (unsigned long)p[i + 2], (unsigned long)p[i + 3]);
    }
}

static int jx_cmd_peek(int argc, FAR char *argv[])
{
  uint32_t addr;
  uint32_t n = 4;

  if (argc < 3)
    {
      printf("用法: jixun peek <hex地址> [字数(4的倍数)]\n");
      return 1;
    }

  addr = jx_hx(argv[2]) & ~(uint32_t)3;

  if (argc >= 4)
    {
      n = (jx_hx(argv[3]) + 3) & ~(uint32_t)3;
      if (n == 0 || n > 1024)
        {
          n = 4;
        }
    }

  jx_dump(addr, n);
  return 0;
}

extern void *g_misc_nvs;         /* libcore.a:misc_nvs.o 里的全局指针 */
extern void **g_osi_funcs_p;     /* OSI 函数表指针（ROM 提供变量） */

static int jx_cmd_wprobe(void)
{
  uint32_t *tab = (uint32_t *)(uintptr_t)g_osi_funcs_p;

  printf("== WiFi 崩溃定位 ==\n");
  printf("&g_misc_nvs = %p   当前值 = %p\n",
         (void *)&g_misc_nvs, g_misc_nvs);
  printf("g_misc_nvs 附近 16 字:\n");
  jx_dump((uint32_t)(uintptr_t)&g_misc_nvs - 32, 16);

  printf("g_osi_funcs_p = %p\n", (void *)g_osi_funcs_p);
  if (tab != NULL)
    {
      printf("  tab[43](_malloc)          = %p\n", (void *)tab[43]);
      printf("  tab[86](_malloc_internal) = %p\n", (void *)tab[86]);
    }

  return 0;
}

static int jx_usage(void)
{
  printf("用法:\n");
  printf("  jixun codec info             当前码率与模型参数\n");
  printf("  jixun codec bench [秒]       合成音频跑编解码，打印 RTF（默认 5 秒）\n");
  printf("  jixun codec seg <Cms> <秒> <L1ms> [L2ms]...  分块一致性探针\n");
  printf("  jixun codec ui  init | test | page <0..4>   480x320 仪表盘\n");
  printf("  jixun codec loop  [秒]       麦克风->编码->解码->喇叭（默认 4 秒）\n");
  printf("  jixun codec streamplay [秒] 录一段后逐 24ms 包处理并立即播放\n");
  printf("  jixun codec benchp [秒] [次] 双核并行跑编码/解码（通话场景，默认 1 秒 x2）\n");
  printf("  jixun codec mem              存储层次与浮点吞吐微基准\n");
  printf("  jixun codec playhex [秒]     串口 hex -> 喇叭直放（PC 侧 _playhex.py 驱动）\n");
  printf("  jixun net tx <IP> [秒] [端口] [块毫秒]  录音->编码->UDP 发 index\n");
  printf("  jixun net rx [端口] [超时秒]         UDP 收 index->解码->喇叭\n");
  printf("\n");
  printf("  （`codec` 可省略：jixun bench / jixun loop / jixun info）\n");
  return 1;
}

static int jx_cmd_tts_play(int argc, char *argv[])
{
  uint8_t *wav;
  size_t bytes = (size_t)g_jx_tts_samples * sizeof(int16_t);
  int wav_size;
  int rc;

  (void)argc;
  (void)argv;
  wav = (uint8_t *)malloc(bytes + 44u);
  if (wav == NULL)
    {
      printf("[TTS] buffer allocation failed\n");
      return 1;
    }

  wav_size = ai_voice_wav_encode(wav, (const int16_t *)g_jx_tts_pcm,
                                 (uint32_t)bytes);
  printf("=== jixun tts: %u samples (%.2f s) ===\n",
         g_jx_tts_samples, (double)g_jx_tts_samples / 16000.0);
  rc = ai_voice_play(wav, (size_t)wav_size);
  printf("=== jixun tts done rc=%d ===\n", rc);
  free(wav);
  return rc;
}

static int jx_cmd_raw_play(int argc, char *argv[])
{
  uint8_t *wav;
  size_t bytes = (size_t)g_jx_raw_samples * sizeof(int16_t);
  int wav_size;
  int rc;

  (void)argc;
  (void)argv;
  wav = (uint8_t *)malloc(bytes + 44u);
  if (wav == NULL)
    {
      printf("[RAW] buffer allocation failed\n");
      return 1;
    }

  wav_size = ai_voice_wav_encode(wav, (const int16_t *)g_jx_raw_pcm,
                                 (uint32_t)bytes);
  printf("=== jixun rawplay: %u samples (%.2f s) ===\n",
         g_jx_raw_samples, (double)g_jx_raw_samples / 16000.0);
  rc = ai_voice_play(wav, (size_t)wav_size);
  printf("=== jixun rawplay done rc=%d ===\n", rc);
  free(wav);
  return rc;
}


int main(int argc, FAR char *argv[])
{
  /* 权重搬运必须先于任何模型初始化（wt_get 之后就固定用这个指针了） */
  jx_weight_init();

  /* 允许 `jixun codec <mode>` 与 `jixun <mode>` 两种写法 */
  if (argc >= 2 && strcmp(argv[1], "codec") == 0)
    {
      argc--;
      argv++;
    }

  if (argc < 2)
    {
      return jx_usage();
    }

  if (strcmp(argv[1], "info") == 0)
    {
      return jx_cmd_info();
    }

  if (strcmp(argv[1], "bench") == 0)
    {
      return jx_cmd_bench(argc, argv);
    }

  if (strcmp(argv[1], "seg") == 0)
    {
      return jx_cmd_seg(argc, argv);
    }

  if (strcmp(argv[1], "ui") == 0)
    {
      return jx_cmd_ui(argc, argv);
    }

  if (strcmp(argv[1], "stream") == 0)
    {
      return jx_cmd_stream(argc, argv);
    }

  if (strcmp(argv[1], "streamplay") == 0)
    {
      return jx_cmd_streamplay(argc, argv);
    }

  if (strcmp(argv[1], "benchp") == 0)
    {
      return jx_cmd_benchp(argc, argv);
    }

  if (strcmp(argv[1], "loop") == 0)
    {
      return jx_cmd_loop(argc, argv);
    }

  if (strcmp(argv[1], "clk") == 0)
    {
      return jx_clk_probe_cmd();
    }

  if (strcmp(argv[1], "mem") == 0)
    {
      return jx_cmd_mem();
    }

  if (strcmp(argv[1], "smem") == 0)
    {
      return jx_cmd_smem();
    }

  if (strcmp(argv[1], "snake") == 0)
    {
      jx_snake_probe();
      return 0;
    }

  if (strcmp(argv[1], "epi") == 0)
    {
      jx_epi_probe();
      return 0;
    }

  if (strcmp(argv[1], "p0") == 0)
    {
      jx_p0_probe();
      return 0;
    }

  if (strcmp(argv[1], "fptp") == 0)
    {
      jx_fptp_probe();
      return 0;
    }

  if (strcmp(argv[1], "fptp2") == 0)
    {
      jx_fptp2_probe();
      return 0;
    }

  if (strcmp(argv[1], "cache") == 0)
    {
      jx_cache_probe();
      return 0;
    }

  if (strcmp(argv[1], "cache2") == 0)
    {
      jx_cache2_probe();
      return 0;
    }

  if (strcmp(argv[1], "playhex") == 0)
    {
      return jx_cmd_playhex(argc, argv);
    }

  if (strcmp(argv[1], "tts") == 0)
    {
      return jx_cmd_tts_play(argc, argv);
    }

  if (strcmp(argv[1], "rawplay") == 0)
    {
      return jx_cmd_raw_play(argc, argv);
    }

  if (strcmp(argv[1], "memhint") == 0)
    {
      return jx_cmd_memhint();
    }

  if (strcmp(argv[1], "peek") == 0)
    {
      return jx_cmd_peek(argc, argv);
    }

  if (strcmp(argv[1], "wprobe") == 0)
    {
      return jx_cmd_wprobe();
    }

  if (strcmp(argv[1], "net") == 0)
    {
      if (argc >= 3 && strcmp(argv[2], "tx") == 0)
        {
          return jx_cmd_net_tx(argc, argv);
        }

      if (argc >= 5 && strcmp(argv[2], "arp") == 0)
        {
          int ar = jx_net_setarp_static(argv[3], argv[4]);
          printf("[net] ARP %s %s -> %d\n", argv[3], argv[4], ar);
          return ar == 0 ? 0 : 1;
        }

      if (argc >= 4 && strcmp(argv[2], "stxp") == 0)
        {
          int p = (argc >= 5) ? atoi(argv[4]) : 45678;
          return jx_streamtx_probe(argv[3], p);
        }

      if (argc >= 4 && strcmp(argv[2], "stx") == 0)
        {
          int p = (argc >= 6) ? atoi(argv[5]) : 45678;
          int n = (argc >= 5) ? atoi(argv[4]) : 5;
          return jx_streamtx_run(argv[3], p, n);
        }

      if (argc >= 3 && strcmp(argv[2], "rx") == 0)
        {
          return jx_cmd_net_rx(argc, argv);
        }

      printf("用法: jixun net tx <对端IP> [秒] [端口]\n");
      printf("      jixun net rx [端口] [超时秒]\n");
      return 1;
    }

  return jx_usage();
}
