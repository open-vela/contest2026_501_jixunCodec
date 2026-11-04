/* Streaming record -> encode -> UDP send.
 *
 * The audio callback only appends PCM to a ring buffer.  A worker thread
 * encodes complete batches while recording is still running.
 */

#include <nuttx/config.h>
#include <nuttx/clock.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

#include "esp32s3_gpio.h"

#include "model_config.h"
#include "ai_voice.h"
#include "jx_codec.h"
#include "jx_net.h"
#include "jx_serial.h"
#include "jx_ui.h"
#include "nn_ops.h"

extern void hal_i2s_rx_idle(void);
extern void jx_lora_release(void);
extern int jx_lora_send_idx(int bits, const int64_t *idx, int ntok,
                            int tt, int orig_t, uint32_t chunk, uint32_t last);


extern const int16_t jx_testpcm[];
extern const int jx_testpcm_len;

/* 2026-10-29 PTT: hold BOOT (GPIO0) to talk, release to end the take.
 * The record call blocks in ai_voice_stream_record(), so a small thread
 * watches the key and calls ai_voice_stream_finish() on release. */

static volatile int g_ptt_watch_stop;
static volatile int g_ptt_released;

static void *jx_ptt_watch(void *arg)
{
  int up_ms = 0;

  (void)arg;

  while (!g_ptt_watch_stop)
    {
      if (esp32s3_gpioread(0) != 0)
        {
          up_ms += 20;
          if (up_ms >= 40)
            {
              g_ptt_released = 1;
              printf("[ptt] released -> finish take\n");
              ai_voice_stream_finish();
              break;
            }
        }
      else
        {
          up_ms = 0;
        }

      usleep(20000);
    }

  return NULL;
}

static int jx_ptt_enabled(void)
{
  const char *v = getenv("JX_PTT");

  return (v != NULL && v[0] != '\0' && strcmp(v, "0") != 0);
}

static double jx_tx_now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

#define JX_STREAMTX_BATCH 32
#define JX_STREAMTX_RING_SECONDS 2
#define JX_STREAMTX_QUEUE 8

/* Future context the streaming model needs to reconstruct the tail.
 * Without it the last word of a short utterance cannot be rebuilt.
 * JX_PAD_MS overrides the default; 0 disables the padding. */
#if defined(SQM_RATE_1K)
#  define JX_STREAMTX_PAD_MS 890
#elif defined(SQM_RATE_6K)
#  define JX_STREAMTX_PAD_MS 138
#else
#  define JX_STREAMTX_PAD_MS 257
#endif


/* 2026-10-21: 内置测试音，用于无麦克风条件下的全链路自动验证。
 * 3 秒 = 6 段元音样信号，每段 400ms 发声 + 100ms 静音，基频
 * 120/150/180/210/240/270 Hz。接收端对 WAV 做分段 FFT 即可判定
 * 每一段是否完整到达，从而定位丢包位置。 */
static void jx_streamtx_gen_testtone(int16_t *buf, int n, int rate)
{
  static const float seg_f0[6] = {200.0f, 230.0f, 260.0f,
                                  290.0f, 320.0f, 350.0f};
  static const float seg_f1[6] = {700.0f, 400.0f, 300.0f,
                                  500.0f, 350.0f, 600.0f};
  static const float seg_f2[6] = {1200.0f, 2200.0f, 2500.0f,
                                  900.0f, 800.0f, 1500.0f};
  int seg_len = (rate * 500) / 1000;
  int tone_len = (rate * 400) / 1000;
  int fade = rate / 100;
  int i;

  for (i = 0; i < n; i++)
    {
      int seg = i / seg_len;
      int pos = i % seg_len;
      float v = 0.0f;

      if (seg >= 6 || pos >= tone_len)
        {
          buf[i] = 0;
          continue;
        }

      {
        float t = (float)pos / (float)rate;
        float f0 = seg_f0[seg];
        float env = 1.0f;
        int h;

        if (pos < fade) { env = (float)pos / (float)fade; }
        if (pos > tone_len - fade)
          { env = (float)(tone_len - pos) / (float)fade; }

        for (h = 1; h <= 14; h++)
          {
            float f = f0 * (float)h;
            float amp;

            if (f > (float)rate * 0.45f) { break; }

            amp = 0.35f / (float)h;
            if (f > seg_f1[seg] - 120.0f && f < seg_f1[seg] + 120.0f)
              { amp *= 3.5f; }
            if (f > seg_f2[seg] - 150.0f && f < seg_f2[seg] + 150.0f)
              { amp *= 2.5f; }
            v += amp * sinf(2.0f * 3.14159265f * f * t);
          }
        v *= 5200.0f * env;
      }

      if (v > 32000.0f) { v = 32000.0f; }
      if (v < -32000.0f) { v = -32000.0f; }
      buf[i] = (int16_t)v;
    }
}

struct jx_streamtx_item
{
  int64_t  tok[JX_STREAMTX_BATCH * JX_PKT_FRAMES];
  int      ntok;
  int      tt;
  int      orig_t;
  uint32_t chunk;
  uint32_t last;
  uint32_t audio_samples;
};

struct jx_streamtx_ctx
{
  pthread_mutex_t lock;
  sem_t           wake;
  pthread_mutex_t send_lock;
  sem_t           send_wake;
  sem_t           send_space;
  struct jx_streamtx_item queue[JX_STREAMTX_QUEUE];
  int             qhead;
  int             qtail;
  int             qcount;
  int             worker_done;
  int16_t        *ring;
  int             cap;
  int             head;
  int             tail;
  double          t_cap0;
  int             count;
  int             voiced_n;
  uint32_t        input_samples;
  int             done;
  int             failed;
  int             capture_started;
  volatile int    capture_stop;
  int             after_rec;
  int             trace_tx;
  int             minpkt;
  int steady_batch;
  int             maxpkt;
  float           gate_th;
  int             alpf_on;
  /* A new take starts with the same gate/filter state as a cold boot. */
  float           capture_gain;
  float           capture_txgain;
  int             capture_hold;
  float           filter_x1, filter_x2, filter_y1, filter_y2;
  int             port;
  int             use_lora;
  char            ip[32];
  int16_t        *batch;
  int64_t        *tok;
  float          *pcm;
  int16_t        *dumpbuf;
  int             dumpn;
  int             dumpcap;
};

static void jx_streamtx_record_cb(FAR const int16_t *pcm16, size_t samples,
                                  void *arg)
{
  struct jx_streamtx_ctx *ctx = (struct jx_streamtx_ctx *)arg;
  size_t i;

  if (ctx->capture_stop)
    {
      static int stop_logged;
      if (!stop_logged && ctx->trace_tx)
        {
          stop_logged = 1;
          printf("[streamtx] cb stop seen count=%d\n", ctx->count);
        }
      return;
    }

  if (!ctx->capture_started)
    {
      ctx->capture_started = 1;
      printf("[streamtx] capture started\n");
    }

  /* Optional raw PCM copy for JX_PCM_DUMP (independent of token path). */
  if (ctx->dumpbuf != NULL && ctx->dumpn < ctx->dumpcap)
    {
      int room = ctx->dumpcap - ctx->dumpn;
      int n = (int)samples;
      if (n > room) { n = room; }
      memcpy(ctx->dumpbuf + ctx->dumpn, pcm16,
             sizeof(int16_t) * (size_t)n);
      ctx->dumpn += n;
    }

  pthread_mutex_lock(&ctx->lock);
  ctx->input_samples += (uint32_t)samples;
  {
    /* 2026-10-21: 编码前静音门。ES8311 底噪 + LoRa/电源干扰在静音段会被
     * 3 kbps 编码放大成持续的“滋滋”。按 16 ms 块估 RMS，低于 JX_GATE
     * （默认 0.004 = -48 dBFS）就把增益拉到 0，带 ~6 ms 平滑避免爆音。
     * JX_GATE=0 可关闭。 */
    /* 2026-11-01: ES8311 ADC is stable at 30 dB but ~6 dB quieter than
     * the old 36 dB setting, and the old setting collapses after 1.1 s.
     * Apply digital make-up gain so the codec sees a usable level. */
    /* 2026-10-30: 不要在回调里 getenv。NuttX 的 getenv 走全局锁，
     * 录音回调每 24 ms 一次，和主线程/编码线程的 getenv 撞上就会把
     * ctx->lock 锁死（第二轮按 BOOT 直接卡住）。启动时缓存到 ctx。 */
    float th = ctx->gate_th;
    if (th > 0.0f)
      {
        for (i = 0; i < samples; i += 256)
          {
            size_t n = samples - i;
            float e = 0.0f;
            size_t k;
            if (n > 256) { n = 256; }
            for (k = 0; k < n; k++)
              {
                float v = (float)pcm16[i + k] * (1.0f / 32768.0f);
                e += v * v;
              }
            {
              float rms = sqrtf(e / (float)n);
              /* 2026-10-21B: 音节之间会短暂低于阈值，若立刻关门就会把每个
               * 字头削掉（听感“字字字”）。检测到语音后保持 640 ms。 */
              if (rms >= th) { ctx->capture_hold = 40; ctx->voiced_n = ctx->count + (int)(i + n); }
              else if (ctx->capture_hold > 0) { ctx->capture_hold--; }
              {
                float target = (ctx->capture_hold > 0) ? 1.0f : 0.0f;
                float rate = (target > ctx->capture_gain) ? 0.02f : 0.0008f;
                for (k = 0; k < n; k++)
                  {
                    ctx->capture_gain += (target - ctx->capture_gain) * rate;
                  if (ctx->count >= ctx->cap)
                    {
                      ctx->tail = (ctx->tail + 1) % ctx->cap;
                      ctx->count--;
                    }
                    {
                      float q = (float)pcm16[i + k] * ctx->capture_gain * ctx->capture_txgain;
                      if (q > 32767.0f) { q = 32767.0f; }
                      if (q < -32768.0f) { q = -32768.0f; }
                      ctx->ring[ctx->head] = (int16_t)q;
                    }
                    ctx->head = (ctx->head + 1) % ctx->cap;
                    ctx->count++;
                  }
              }
            }
          }
      }
    else
      {
        /* 2026-10-21E: getenv 必须在循环外读一次！放在每个样本里会拖慢
         * I2S 回调，导致录音中间断档（实测 0.4~1.4 s 全零）。 */
        for (i = 0; i < samples; i++)
          {
            float v = (float)pcm16[i] * (1.0f / 32768.0f);
            if (ctx->alpf_on)
              {
                float y = 0.2704f * v + 0.5408f * ctx->filter_x1 + 0.2704f * ctx->filter_x2
                          + 0.0913f * ctx->filter_y1 - 0.1730f * ctx->filter_y2;
                ctx->filter_x2 = ctx->filter_x1; ctx->filter_x1 = v; ctx->filter_y2 = ctx->filter_y1; ctx->filter_y1 = y;
                v = y;
              }
            { float q = v * 32768.0f;
              if (q > 32767.0f) { q = 32767.0f; }
              if (q < -32768.0f) { q = -32768.0f; }
              if (ctx->count >= ctx->cap)
                {
                  ctx->tail = (ctx->tail + 1) % ctx->cap;
                  ctx->count--;
                }
              ctx->ring[ctx->head] = (int16_t)q; }
            ctx->head = (ctx->head + 1) % ctx->cap;
            ctx->count++;
          }
      }
  }
  pthread_mutex_unlock(&ctx->lock);
  sem_post(&ctx->wake);
}

/* Append silence to the capture ring so the encoder sees the future
 * context required by the streaming model.  Called after recording has
 * stopped and before ctx.done is set. */
static void jx_streamtx_pad_silence(struct jx_streamtx_ctx *ctx, int samples)
{
  while (samples > 0)
    {
      int room;
      int n;
      int i;

      pthread_mutex_lock(&ctx->lock);
      room = ctx->cap - ctx->count;
      n = samples < room ? samples : room;
      for (i = 0; i < n; i++)
        {
          ctx->ring[ctx->head] = 0;
          ctx->head = (ctx->head + 1) % ctx->cap;
          ctx->count++;
        }
      pthread_mutex_unlock(&ctx->lock);

      if (n > 0)
        {
          samples -= n;
          sem_post(&ctx->wake);
        }
      else
        {
          usleep(1000);
        }
    }
}


/* 2026-10-30: bounded waits.  The COM7/COM14 send path occasionally
 * stopped after "chunk 1: 32 packets sent": one of the three sems never
 * came back and the whole take died.  Every wait now times out and says
 * which one, so the stall point is visible instead of silent. */

static int stx_wait_to(sem_t *sem, int ms, const char *tag, int chunk)
{
  struct timespec ts;
  int ret;

  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += ms / 1000;
  ts.tv_nsec += (long)(ms % 1000) * 1000000L;
  if (ts.tv_nsec >= 1000000000L)
    {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000L;
    }

  ret = sem_timedwait(sem, &ts);
  if (ret < 0 && strcmp(tag, "send_wake") != 0)
    {
      printf("[stx] %s timeout %d ms chunk=%d\n", tag, ms, chunk);
    }

  return ret;
}

static void *jx_streamtx_sender(void *arg)
{
  struct jx_streamtx_ctx *ctx = (struct jx_streamtx_ctx *)arg;
  struct sched_param sp;
  cpu_set_t cs;
  int affinity_rc;
  int scheduler_rc;

  /* The encoder stays on CPU1 at FIFO priority 20. Run radio work on
   * CPU0 so ready packets do not wait for the encoder queue to fill.
   * A modest priority keeps system and SPI/DMA service tasks responsive. */
  CPU_ZERO(&cs);
  CPU_SET(0, &cs);
  affinity_rc = sched_setaffinity(0, sizeof(cs), &cs);
  sp.sched_priority = 30;
  scheduler_rc = sched_setscheduler(0, SCHED_FIFO, &sp);
  printf("[streamtx] sender cpu=0 priority=30 affinity_rc=%d scheduler_rc=%d\n",
         affinity_rc, scheduler_rc);

  for (;;)
    {
      int done = 0;

      stx_wait_to(&ctx->send_wake, 500, "send_wake", -1);
      for (;;)
        {
          struct jx_streamtx_item item;
          int rc;

          pthread_mutex_lock(&ctx->send_lock);
          if (ctx->qcount == 0)
            {
              done = ctx->worker_done;
              pthread_mutex_unlock(&ctx->send_lock);
              if (done)
                {
                  return NULL;
                }
              break;
            }

          item = ctx->queue[ctx->qhead];
          ctx->qhead = (ctx->qhead + 1) % JX_STREAMTX_QUEUE;
          ctx->qcount--;
          pthread_mutex_unlock(&ctx->send_lock);
          sem_post(&ctx->send_space);

          if (ctx->use_lora)
            {
              double t_snd0 = jx_tx_now_ms();
              rc = jx_lora_send_idx(jx_token_bits(), item.tok, item.ntok,
                                    item.tt, (int)item.audio_samples, item.chunk, item.last);
              if (ctx->trace_tx)
                {
                  printf("[streamtx] send chunk=%u ntok=%d ms=%.1f at=%.1f rc=%d\n",
                         (unsigned)item.chunk, item.ntok,
                         jx_tx_now_ms() - t_snd0,
                         (ctx->t_cap0 > 0.0) ? (t_snd0 - ctx->t_cap0) : -1.0,
                         rc);
                }
            }
          else
            {
              rc = jx_net_send_idx(ctx->ip, ctx->port, jx_codec_rate_id(),
                                   jx_token_bits(), item.tok, item.ntok,
                                   item.tt, item.orig_t, item.chunk, item.last);
            }
          if (rc != 0)
            {
              printf("[streamtx] send failed rc=%d chunk=%u\n",
                     rc, (unsigned)item.chunk);
              ctx->failed = 1;
            }
          else
            {
              printf("[streamtx] chunk %u: %d packets sent last=%u\n",
                     (unsigned)item.chunk, item.ntok / JX_PKT_FRAMES,
                     (unsigned)item.last);
            }
        }
    }
}

static void *jx_streamtx_worker(void *arg)
{
  struct jx_streamtx_ctx *ctx = (struct jx_streamtx_ctx *)arg;
  struct jx_stream_s *st = NULL;
  int chunk = 0;
  int rc = 0;
  int starve_ticks = 0;
  int noenc = (getenv("JX_STX_NOENC") != NULL);
  double prev_conv = 0.0;
  double prev_attn = 0.0;
  double prev_dec = 0.0;
  struct sched_param sp;

  /* 2026-10-30: 编码线程原来 FIFO 50 绑 CPU0。边录边发时它和 I2S
   * 采集/RX 预读链抢同一个核，现场第 2 轮 blk=17 之后直接停 25 秒
   * （blk=21 wall=26880ms），再往后整机无响应。改绑 CPU1、优先级
   * 降到 20：CPU0 完整留给 I2S/录音/UI，编码只填空隙。 */
  sp.sched_priority = 20;
  (void)sched_setscheduler(0, SCHED_FIFO, &sp);

  {
    cpu_set_t cs;

    CPU_ZERO(&cs);
    CPU_SET(1, &cs);
    (void)sched_setaffinity(0, sizeof(cs), &cs);
  }

  /* 2026-10-20：编码器和量化权重在录音开始前就预热。
   * 原来要等第一包 PCM 到达（录音开始后 ~100 ms）才 jx_stream_open，
   * 首次权重加载/内存分配把首帧转发推到 1 s 之后。双板 TX 只录音不播放，
   * 提前初始化不会和 I2S 播放抢链路。 */
  setenv("JX_SKIP_DEC", "1", 1);
  /* 2026-10-30: 不再强制开双核。jx_pf_worker 跨 jixun 命令复用，
   * sem 配对错位后第 2/3 轮必挂死；而且 setenv 本身走全局锁。
   * 需要并行时在启动前 set JX_PF 1。 */
  setenv("JX_PFCORE", "0", 1);
  setenv("JX_PFK", "1", 1);
  if (!noenc)
    {
      jx_i8_options_begin();
      jx_wq_init();
      {
        const char *dv=getenv("JX_A_DUAL");
        int rc_pf=jx_pf_session_begin(dv==NULL || dv[0]!='0');
        printf("[streamtx] parallel session rc=%d\n",rc_pf);
      }
      st = jx_stream_open(512);
      if (st == NULL)
        {
          ctx->failed = 1;
          jx_pf_session_end();
          jx_i8_options_end();
          return NULL;
        }
      printf("[streamtx] encoder prewarm done\n");
    }

  for (;;)
    {
      int available;
      int done;

      pthread_mutex_lock(&ctx->lock);
      available = ctx->count;
      done = ctx->done;
      pthread_mutex_unlock(&ctx->lock);
      if (available > 0 || done)
        {
          break;
        }
      usleep(2000);
    }

  for (;;)
    {
      int stop = 0;

      stx_wait_to(&ctx->wake, 500, "wake", chunk);

      /* Single-board loopback: encoding is harmless while recording, but
       * *sending* a block lets the receiver start playback.  Playback calls
       * hal_i2s_write_async(), which stops the shared I2S RX prefetch chain;
       * capture on the same board then blocks until its 3 s timeout and the
       * loop appears dead.  When JX_TX_AFTER_REC=1 we therefore wait until
       * recording has finished before draining the ring.  Two-board setups
       * leave this off and keep the original live streaming. */

      if (ctx->after_rec)
        {
          int rec_done;

          pthread_mutex_lock(&ctx->lock);
          rec_done = ctx->done;
          pthread_mutex_unlock(&ctx->lock);
          if (!rec_done)
            {
              continue;
            }
        }

      for (;;)
        {
          int available;
          int done;
          int npkt;
          int need;
          int take;
          int last;
          uint32_t audio_samples;
          int i;

          pthread_mutex_lock(&ctx->lock);
          available = ctx->count;
          done = ctx->done;
          npkt = available / JX_PKT_SAMPLES;
          if (ctx->trace_tx && chunk < 8)
            {
              printf("[streamtx] avail=%d npkt=%d pkt_samples=%d cap=%d\n",
                     available, npkt, JX_PKT_SAMPLES, ctx->cap);
            }
          /* 2026-10-20：把多个 24 ms 包攒成一批再编码，摊薄每批固定开销
           * （权重重读、halo、线程调度）。JX_TX_MINPKT 只影响“录音还没结束
           * 时最多等多少包”，不改包格式、不改算法；设 1 即原来的行为。 */
          if (!done && npkt < ((ctx->steady_batch && chunk>0)?ctx->maxpkt:ctx->minpkt) && available > 0)
            {
              pthread_mutex_unlock(&ctx->lock);
              if (++starve_ticks > 2000)
                {
                  printf("[stx] starve guard chunk=%d av=%d\n",
                         chunk, available);
                  pthread_mutex_lock(&ctx->lock);
                  ctx->done = 1;
                  pthread_mutex_unlock(&ctx->lock);
                  starve_ticks = 0;
                  continue;
                }
              usleep(3000);
              continue;
            }
          starve_ticks = 0;
          if (npkt > ctx->maxpkt) { npkt = ctx->maxpkt; }
          if (done && npkt > 0 && (npkt & 3)) { npkt = (npkt + 3) & ~3; }
          /* 2026-10-21: 首包只编 1 帧，尽快把第一包送出去；后续仍按
           * avail 成批编码。这样首声从 ~0.6 s 降到 ~0.2 s，算法不变。 */
          if (chunk == 0 && npkt > 4)
            {
              npkt = 4;
            }
          if (npkt == 0 && done && available > 0)
            {
              npkt = 4;
            }
          if (npkt == 0)
            {
              stop = (done && available == 0);
              pthread_mutex_unlock(&ctx->lock);
              break;
            }

          need = npkt * JX_PKT_SAMPLES;
          take = need < available ? need : available;
          if (chunk < 4 || (chunk % 8) == 0)
            {
              printf("[stx] c=%d av=%d npkt=%d need=%d\n",
                     chunk, available, npkt, need);
            }
          for (i = 0; i < take; i++)
            {
              ctx->batch[i] = ctx->ring[ctx->tail];
              ctx->tail = (ctx->tail + 1) % ctx->cap;
              ctx->count--;
            }
          for (; i < need; i++)
            {
              ctx->batch[i] = 0;
            }
          last = done && (available <= need);
          audio_samples = last ? ctx->input_samples : 0;
          pthread_mutex_unlock(&ctx->lock);

          if (noenc)
            {
              int ti;

              for (ti = 0; ti < npkt * JX_PKT_FRAMES; ti++)
                {
                  ctx->tok[ti] = ti & 0x3f;
                }
              rc = 0;
            }
          else
            {
              {
                double t_enc0 = jx_tx_now_ms();
                rc = jx_stream_feed_batch(st, ctx->batch, npkt, ctx->tok,
                                          ctx->pcm);
                if (ctx->trace_tx)
                  {
                    printf("[streamtx] enc chunk=%d npkt=%d ms=%.1f\n",
                           chunk, npkt, jx_tx_now_ms() - t_enc0);
                  }
              }
            }
          if (rc != 0)
            {
              printf("[streamtx] encode failed rc=%d chunk=%d\n", rc, chunk);
              ctx->failed = 1;
              stop = 1;
              break;
            }

          if (!noenc)
            {
              double conv;
              double attn;
              double dec;
              double deq;
              double endec;
              double decnn;

              jx_stream_times(st, &conv, &attn, &dec, &deq, &endec, &decnn);
              if (ctx->trace_tx)
                {
                  printf("[streamtx] timing chunk=%d conv=%.1f attn=%.1f "
                         "dec=%.1f deq=%.1f endec=%.1f decnn=%.1f\n",
                         chunk, conv - prev_conv, attn - prev_attn,
                         dec - prev_dec, deq, endec, decnn);
                }
              prev_conv = conv;
              prev_attn = attn;
              prev_dec = dec;
            }

          stx_wait_to(&ctx->send_space, 2000, "send_space", chunk);
          pthread_mutex_lock(&ctx->send_lock);
          {
            struct jx_streamtx_item *item = &ctx->queue[ctx->qtail];

            memcpy(item->tok, ctx->tok,
                   sizeof(int64_t) * (size_t)npkt * JX_PKT_FRAMES);
            item->ntok = npkt * JX_PKT_FRAMES;
            item->tt = npkt * JX_PKT_FRAMES;
            /* 2026-10-30: 这行原来是无条件 printf。连发时每块一行，
             * 串口 TX 队列被灌满后 worker 卡在 stdout 锁里，持着
             * send_lock 不退出，主线程第二轮起直接挂死（ps 显示
             * Waiting Mutex:<worker>）。改成只在 JX_TRACE_TX 下打印。 */
            if (ctx->trace_tx)
              {
                static long g_tot_in, g_tot_chunks, g_tot_tok;
                g_tot_in += need; g_tot_tok += item->ntok; g_tot_chunks++;
                printf("[streamtx] STAT chunk=%d in_samples=%d ntok=%d "
                       "cum_in=%ld cum_tok=%ld chunks=%ld\n",
                       chunk, need, item->ntok,
                       g_tot_in, g_tot_tok, g_tot_chunks);
              }
            item->orig_t = need;
            item->chunk = (uint32_t)chunk;
            item->last = (uint32_t)last;
            item->audio_samples = audio_samples;
            ctx->qtail = (ctx->qtail + 1) % JX_STREAMTX_QUEUE;
            ctx->qcount++;
          }
          pthread_mutex_unlock(&ctx->send_lock);
          sem_post(&ctx->send_wake);
          chunk++;
        }

      if (stop)
        {
          break;
        }
    }

  if (st != NULL) { jx_stream_close(st); }
  jx_pf_session_end();
  jx_i8_options_end();
  return NULL;
}


static void *jx_streamtx_probe_worker(void *arg)
{
  struct jx_streamtx_ctx *ctx = (struct jx_streamtx_ctx *)arg;
  int64_t idx[4] = {0, 1, 2, 3};
  int rc = 0;
  int i;

  printf("[streamtx] probe worker start\n");
  for (i = 0; i < 8; i++)
    {
      idx[0] = i;
      rc = jx_net_send_idx(ctx->ip, ctx->port, jx_codec_rate_id(), jx_token_bits(), idx, 4, 4, 192,
                           (uint32_t)i, (i == 7) ? 1u : 0u);
      if (rc != 0)
        {
          break;
        }

      usleep(100 * 1000);
    }

  printf("[streamtx] probe worker rc=%d\n", rc);
  return NULL;
}

int jx_streamtx_probe(const char *ip, int port)
{
  struct jx_streamtx_ctx ctx;
  pthread_t worker;
  pthread_attr_t attr;

  memset(&ctx, 0, sizeof(ctx));
  ctx.port = port;
  strncpy(ctx.ip, ip, sizeof(ctx.ip) - 1);
  if (pthread_attr_init(&attr) != 0 ||
      pthread_attr_setstacksize(&attr, 32768) != 0 ||
      pthread_create(&worker, &attr, jx_streamtx_probe_worker, &ctx) != 0)
    {
      pthread_attr_destroy(&attr);
      return -EIO;
    }

  pthread_attr_destroy(&attr);
  pthread_join(worker, NULL);
  return 0;
}

static int jx_streamtx_run_common(const char *ip, int port, int secs, int use_lora)
{
  struct jx_streamtx_ctx ctx;
  pthread_t worker;
  pthread_t sender;
  pthread_attr_t attr;
  int ret;
  int ms;

  if (ip == NULL || port <= 0)
    {
      return -EINVAL;
    }
  if (secs < 1) { secs = 1; }
  if (secs > AI_VOICE_MAX_SECONDS) { secs = AI_VOICE_MAX_SECONDS; }

  memset(&ctx, 0, sizeof(ctx));
  ctx.port = port;
  ctx.use_lora = use_lora;
  strncpy(ctx.ip, ip, sizeof(ctx.ip) - 1);
  /* Keep the whole utterance in the ring; with after_rec the worker does
   * not drain it until recording ends. */

  /* 2026-10-21: 2s 余量在 3s 段 + 96KB 内置人声表时会把 PSRAM 顶满，
     * 编码线程直接卡死。采集 3s、编码落后 1.3s，1s 余量足够。 */
  ctx.cap = (secs + 1) * AI_VOICE_AI_RATE;
  {
    const char *ar = getenv("JX_TX_AFTER_REC");

    /* 2026-10-30: 所有高频 getenv 在这里读一次。编码线程、录音回调
     * 之后只读 ctx 字段，避免 NuttX getenv 全局锁和 ctx->lock 互锁。 */
    ctx.trace_tx = (getenv("JX_TRACE_TX") != NULL);
    ctx.minpkt = 4;
    {const char *v=getenv("JX_A_STEADY");ctx.steady_batch=(v==NULL || v[0]!='0');}
    ctx.maxpkt = 8;
    {
      const char *mp = getenv("JX_TX_MINPKT");
      const char *np = getenv("JX_TX_NPKT");
      const char *ge = getenv("JX_GATE");
      const char *le = getenv("JX_ALPF");

      if (mp != NULL)
        {
          int v = atoi(mp);
          if (v > 0 && v <= JX_STREAMTX_BATCH) { ctx.minpkt = v; }
        }
      if (np != NULL)
        {
          int v = atoi(np);
          if (v > 0 && v <= JX_STREAMTX_BATCH) { ctx.maxpkt = v; }
        }
      /* Gate and low-pass history are zeroed with ctx. Cache gain before
       * capture so the callback never acquires an environment lock. */
      const char *tg = getenv("JX_TX_GAIN");
      ctx.capture_gain = 1.0f;
      ctx.capture_txgain = (tg != NULL && tg[0] != '\0') ? (float)atof(tg) : 2.0f;
      if (ctx.capture_txgain < 1.0f) { ctx.capture_txgain = 1.0f; }
      if (ctx.capture_txgain > 32.0f) { ctx.capture_txgain = 32.0f; }
      ctx.gate_th = (ge != NULL) ? (float)atof(ge) : 0.01f;
      ctx.alpf_on = !(le != NULL && le[0] == (char)48);
    }
    ctx.after_rec = (ar != NULL && ar[0] != '\0' && strcmp(ar, "0") != 0);
  }
  ctx.ring = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.cap);
  ctx.batch = (int16_t *)malloc(sizeof(int16_t) * JX_STREAMTX_BATCH * JX_PKT_SAMPLES);
  ctx.tok = (int64_t *)malloc(sizeof(int64_t) * JX_STREAMTX_BATCH * JX_PKT_FRAMES);
  ctx.pcm = (float *)malloc(sizeof(float) * JX_STREAMTX_BATCH * JX_PKT_SAMPLES);
  ctx.dumpbuf = NULL;
  ctx.dumpn = 0;
  ctx.voiced_n = 0;
  ctx.dumpcap = 0;
  if (jx_serial_pcm_dump_enabled())
    {
      ctx.dumpcap = secs * AI_VOICE_AI_RATE;
      ctx.dumpbuf = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.dumpcap);
      if (ctx.dumpbuf == NULL) { ctx.dumpcap = 0; }
    }
  if (ctx.ring == NULL || ctx.batch == NULL || ctx.tok == NULL || ctx.pcm == NULL)
    {
      printf("[streamtx] buffer allocation failed\n");
      free(ctx.ring); free(ctx.batch); free(ctx.tok); free(ctx.pcm);
      return -ENOMEM;
    }

  if (pthread_mutex_init(&ctx.lock, NULL) != 0 || sem_init(&ctx.wake, 0, 0) != 0 ||
      pthread_mutex_init(&ctx.send_lock, NULL) != 0 ||
      sem_init(&ctx.send_wake, 0, 0) != 0 ||
      sem_init(&ctx.send_space, 0, JX_STREAMTX_QUEUE) != 0)
    {
      printf("[streamtx] sync init failed\n");
      free(ctx.ring); free(ctx.batch); free(ctx.tok); free(ctx.pcm);
      return -EIO;
    }

  if (pthread_attr_init(&attr) != 0 ||
      pthread_attr_setstacksize(&attr, 32768) != 0 ||
      pthread_create(&sender, &attr, jx_streamtx_sender, &ctx) != 0)
    {
      printf("[streamtx] sender create failed\n");
      pthread_attr_destroy(&attr);
      pthread_mutex_destroy(&ctx.lock);
      sem_destroy(&ctx.wake);
      pthread_mutex_destroy(&ctx.send_lock);
      sem_destroy(&ctx.send_wake);
      sem_destroy(&ctx.send_space);
      free(ctx.ring); free(ctx.batch); free(ctx.tok); free(ctx.pcm);
      return -EIO;
    }
  pthread_attr_destroy(&attr);

  if (pthread_attr_init(&attr) != 0 ||
      pthread_attr_setstacksize(&attr, 65536) != 0 ||
      pthread_create(&worker, &attr, jx_streamtx_worker, &ctx) != 0)
    {
      printf("[streamtx] worker create failed\n");
      pthread_attr_destroy(&attr);
      pthread_mutex_destroy(&ctx.lock);
      sem_destroy(&ctx.wake);
      pthread_mutex_destroy(&ctx.send_lock);
      sem_destroy(&ctx.send_wake);
      sem_destroy(&ctx.send_space);
      free(ctx.ring); free(ctx.batch); free(ctx.tok); free(ctx.pcm);
      return -EIO;
    }
  pthread_attr_destroy(&attr);

  {
    /* 2026-10-21: 最终时序 —— rearm(整片软上电) -> prearm 起振 ->
     * 播提示音 -> 关 DAC + 重写 ADC 模拟通路 -> 再等 250ms -> 录音。
     * rearm 必须早于提示音（提示音的 DAC 会把 ADC 压掉），但提示音
     * 之后必须再做一次模拟通路恢复，否则 probe 时好时坏。 */
    extern int voice_agent_rearm_adc(void);
    extern int voice_agent_kick_adc(void);
    extern int voice_agent_set_dac_mute(int mute);
    extern int ai_voice_init(void);
    extern void ai_voice_reset_init(void);
    const char *pm = getenv("JX_PREARM_MS");
    int prearm_ms = (pm != NULL) ? atoi(pm) : 2000;
    if (prearm_ms < 0) { prearm_ms = 0; }

    /* 2026-10-21: nsh 里 jixun 是同一进程空间，第二次执行时
     * ai_voice_init/voice_agent_init 的幂等锁还在，整条 I2S/码片
     * 初始化被跳过，ADC 直接全 0。每次进来先解锁，强制完整重配。 */
    /* 2026-10-22: do NOT reset/re-init on every command by default.
     * The 10/11 working version initialized the codec once and kept the
     * ADC powered. Re-running es8311_init writes REG00=0x80 (soft reset)
     * and leaves ASDOUT dead. set JX_AI_REINIT 1 to restore. */
    {
      const char *ri = getenv("JX_AI_REINIT");
      if (ri != NULL && atoi(ri) != 0)
        {
          ai_voice_reset_init();
        }
    }
    (void)ai_voice_init();

    /* 2026-10-21: 恢复 141427 的原始时序——先 rearm、等足 2000ms 让
     * 模拟前端稳定，最后才播提示音。之前把 prearm 砍到 200ms，probe
     * 从 4723 掉到 1278，听起来就是小声、失真。 */
    const char *pcmf0 = getenv("JX_TX_PCM_FILE");
    const char *tpb0  = getenv("JX_TX_PCM_BLOB");
    const char *ttg0  = getenv("JX_TX_TESTTONE");
    const char *cue_test=getenv("JX_CUE_WITH_BLOB");
    if ((cue_test && cue_test[0]=='1') || ((pcmf0 == NULL || pcmf0[0] == '\0' || strcmp(pcmf0, "0") == 0) &&
        (tpb0  == NULL || tpb0[0]  == '\0' || strcmp(tpb0,  "0") == 0) &&
        (ttg0  == NULL || ttg0[0]  == '\0' || strcmp(ttg0,  "0") == 0)))
      {
    if(!ai_voice_record_cue_ready())(void)voice_agent_rearm_adc();
    /* 2026-10-21: prearm 低于 300ms 时模拟 PGA 还没起振，probe 只有几百，
     * 录音全程又小又炸。给一个硬下限，环境变量再小也不认。 */
    /* 2026-10-29: 提示音会立刻把 ADC 压掉，这段 prearm 纯属白等。
     * 允许 0（跳过），真正起效的是提示音之后的整片 rearm + 300ms。 */
    if (prearm_ms < 0) { prearm_ms = 0; }
    if (prearm_ms > 0) { usleep((useconds_t)prearm_ms * 1000); }
    printf("[streamtx] ADC prearmed %dms ticks=%lu\n", prearm_ms,
           (unsigned long)clock_systime_ticks());

    const char *bv = getenv("JX_BEEP");
    if (bv == NULL || bv[0] == '\0' || strcmp(bv, "0") != 0)
      {
        printf("  提示音: 两声短音，响完立刻开始说\n");
        /* Keep both tones and their silent gap in one I2S TX session.
         * Restarting/muting the DAC between separate tone calls can swallow
         * one of the cues. The ADC is restored only after the entire pair. */
        int cue_rc = ai_voice_play_record_cue();
        printf("[cue] pair complete rc=%d\n", cue_rc);
        if (cue_rc < 0)
          {
            printf("[cue] playback failed; recording will still start\n");
          }
      }

    /* 2026-10-21: 提示音由 DAC 播放，DAC 与 ADC 共用模拟参考，播完之后
     * ADC 会被压 0.6~1.7s。显式关 DAC 并重写 ADC 模拟通路，再留 250ms
     * 恢复，否则 probe 时好时坏，就是用户看到的“有时能用”。 */
    if (bv == NULL || bv[0] == '\0' || strcmp(bv, "0") != 0)
      {
        (void)voice_agent_set_dac_mute(1);
        (void)voice_agent_rearm_adc();   /* 轻量 kick 救不回被 DAC 压掉的 PGA，整片重上电 */
        usleep(300 * 1000);
        printf("[streamtx] ADC full rearm after beep done\n");
      }
      }
  }

  /* REG44 mode 5 sends ADC on left and DAC feedback on right. Recording
   * needs ADC in both slots; keep digital feedback out of the mic path. */
  {
    const char *route = getenv("JX_ADC_ROUTE");
    uint8_t value = (route != NULL) ? (uint8_t)strtoul(route, NULL, 0) : 0x08;
    int route_rc = voice_agent_set_reg44(value);
    printf("[capture] ADC route REG44=%02x rc=%d\n", value, route_rc);
  }

  jx_ui_set_status("RECORDING");
  printf("[streamtx] recording %d s, batch=%d packets, ring=%d samples\n",
         secs, JX_STREAMTX_BATCH, ctx.cap);
  printf("  录音中 ...\n");
  {
    const char *pcmf = getenv("JX_TX_PCM_FILE");
    const char *tpb = getenv("JX_TX_PCM_BLOB");
    const char *ttg = getenv("JX_TX_TESTTONE");
    if (tpb != NULL && tpb[0] != '\0' && strcmp(tpb, "0") != 0)
      {
        int want = secs * AI_VOICE_AI_RATE;
        int off = 0;
        printf("[streamtx] pcm blob fed %d samples (%d s), realtime\n",
               jx_testpcm_len, secs);
        ctx.t_cap0 = jx_tx_now_ms();
        while (off < want)
          {
            int n = want - off;
            double target;

            if (n > JX_PKT_SAMPLES) { n = JX_PKT_SAMPLES; }
            target = ((double)off * 1000.0) / (double)AI_VOICE_AI_RATE;
            {
              double now = jx_tx_now_ms() - ctx.t_cap0;
              if (target > now)
                {
                  usleep((useconds_t)((target - now) * 1000.0));
                }
            }
            if (off < jx_testpcm_len)
              {
                int m = jx_testpcm_len - off;
                if (m > n) { m = n; }
                jx_streamtx_record_cb(&jx_testpcm[off], (size_t)m, &ctx);
                if (m < n)
                  {
                    static const int16_t zero[1024] = {0};
                    jx_streamtx_record_cb(zero, (size_t)(n - m), &ctx);
                  }
              }
            else
              {
                static const int16_t zero[1024] = {0};
                jx_streamtx_record_cb(zero, (size_t)n, &ctx);
              }
            off += n;
          }
        ms = secs * 1000;
      }
    else if (ttg != NULL && ttg[0] != '\0' && strcmp(ttg, "0") != 0)
      {
        int want = secs * AI_VOICE_AI_RATE;
        int16_t *fbuf = (int16_t *)malloc(sizeof(int16_t) * (size_t)want);
        if (fbuf == NULL)
          {
            printf("[streamtx] testtone malloc failed\n");
            ms = -1;
          }
        else
          {
            int off = 0;

            jx_streamtx_gen_testtone(fbuf, want, AI_VOICE_AI_RATE);
            printf("[streamtx] testtone fed %d samples (%d s), realtime\n",
                   want, secs);
            ctx.t_cap0 = jx_tx_now_ms();
            while (off < want)
              {
                int n = want - off;
                double target;

                if (n > JX_PKT_SAMPLES) { n = JX_PKT_SAMPLES; }
                target = ((double)off * 1000.0) / (double)AI_VOICE_AI_RATE;
                {
                  double now = jx_tx_now_ms() - ctx.t_cap0;
                  if (target > now)
                    {
                      usleep((useconds_t)((target - now) * 1000.0));
                    }
                }
                jx_streamtx_record_cb(fbuf + off, (size_t)n, &ctx);
                off += n;
              }
            free(fbuf);
            ms = secs * 1000;
          }
      }
    else if (pcmf != NULL && pcmf[0] != '\0')
      {
        FILE *f = fopen(pcmf, "rb");
        int want = secs * AI_VOICE_AI_RATE;
        int16_t *fbuf = (int16_t *)malloc(sizeof(int16_t) * (size_t)want);
        if (f == NULL || fbuf == NULL)
          {
            printf("[streamtx] pcm file open failed: %s\n", pcmf);
            ms = -1;
          }
        else
          {
            int got = (int)fread(fbuf, sizeof(int16_t), (size_t)want, f);
            if (got < want)
              {
                memset(fbuf + got, 0, sizeof(int16_t) * (size_t)(want - got));
              }
            jx_streamtx_record_cb(fbuf, (size_t)want, &ctx);
            fclose(f);
            free(fbuf);
            ms = secs * 1000;
            printf("[streamtx] pcm file fed: %d samples\n", want);
          }
      }
    else
      {
        ctx.t_cap0 = jx_tx_now_ms();
        if (jx_ptt_enabled())
          {
            pthread_t ptt_th;
            int thr_ok;

            g_ptt_watch_stop = 0;
            g_ptt_released    = 0;
            esp32s3_configgpio(0, INPUT | PULLUP);
            thr_ok = (pthread_create(&ptt_th, NULL, jx_ptt_watch, NULL) == 0);
            printf("[ptt] watch %s\n", thr_ok ? "on" : "FAIL");

            ms = ai_voice_stream_record(jx_streamtx_record_cb, &ctx, secs, 0);

            g_ptt_watch_stop = 1;
            if (thr_ok)
              {
                pthread_join(ptt_th, NULL);
              }

            printf("[ptt] take %d ms (released=%d)\n", ms, g_ptt_released);
          }
        else
          {
            ms = ai_voice_stream_record(jx_streamtx_record_cb, &ctx, secs, 0);
          }
      }
  }
  /* 2026-11-02: BOOT 3 秒固定发满。VAD 尾部裁剪在现场把 3 秒
   * 裁成 1.8 秒，K7 只听到半截；这里不再裁剪。 */
  jx_ui_set_status("ENCODE TX");
  printf("[streamtx] recording done: %d ms, queued=%d samples\n", ms, ctx.count);

  /* 2026-10-29: ADC 闩死时 ai_voice_stream_record 返回 -5 ms、queued=0。
   * 标志成 failed 让 jx_ui_lora_send_task 自动补录一次，而不是把
   * 5 包静音当成一次成功发送。 */
  if (ms <= 0)
    {
      ctx.failed = 1;
      /* 不要补 tail pad 再编码：否则这一轮会先给对端发 5 包纯静音，
       * 用户听到的就是「先滋一下、再正常」。直接跳到收尾，
       * 由 jx_ui_lora_send_task 自动重录一次。 */
      printf("[streamtx] capture failed, skip encode/tx\n");
      goto finish;
    }

  {
    extern int voice_agent_power_off(void);
    const char *po = getenv("JX_ADC_POWEROFF");

    /* 2026-10-22: keep codec powered by default; powering down after every
     * take is what makes the next take cold and unstable. */
    /* 2026-10-22 v31: default KEEP the codec powered.  Powering down after
     * every take forces a cold start next time; the ES8311 analog front-end
     * then fails to bias up and the ADC emits only a few tens of LSB, which
     * the 3 kbps encoder amplifies into continuous noise.  Set
     * JX_ADC_POWEROFF=1 only when a real cold start is wanted. */
    if (po != NULL && atoi(po) != 0)
      {
        (void)voice_agent_power_off();
        printf("[streamtx] ES8311 powered off after take\n");
      }
    else
      {
        printf("[streamtx] ES8311 kept powered after take\n");
      }
  }

  ctx.capture_stop = 1;

  {
    int pad_ms = JX_STREAMTX_PAD_MS;
    const char *pm = getenv("JX_PAD_MS");

    if (pm != NULL && pm[0] != '\0')
      {
        pad_ms = atoi(pm);
        if (pad_ms < 0) { pad_ms = 0; }
      }
    if (pad_ms > 0)
      {
        int pad_samples = (AI_VOICE_AI_RATE * pad_ms) / 1000;
        const char *wv=getenv("JX_WIRE_V2");
        if((!wv || atoi(wv)!=0) && pad_ms==JX_STREAMTX_PAD_MS && pad_samples<4118)
          pad_samples=4118;
        printf("[streamtx] tail pad: %d ms (%d samples)\n",
               pad_ms, pad_samples);
        if (getenv("JX_TRACE_TX"))
          {
            printf("[streamtx] pad before count=%d samples=%d\n",
                   ctx.count, pad_samples);
          }
        jx_streamtx_pad_silence(&ctx, pad_samples);
        if (getenv("JX_TRACE_TX"))
          {
            printf("[streamtx] pad after count=%d\n", ctx.count);
          }
      }
  }

  /* Do not keep RX DMA and HPWORK allocating DMA containers while the
   * encoder and network drain the captured ring.  A later read_slot starts
   * the prefetch chain again automatically.
   */

  {
    /* 2026-10-29：JX_RXIDLE 同时被 LoRa 收包超时使用（脚本里设 8000ms），
     * 不能再拿它当“录音结束停 RX 预读链”的开关：8000 被 atoi 成非 0，
     * 每轮结束都停链，第二次录音 ADC 就全 0。
     * 这里改用独立开关 JX_RX_IDLE_STOP，默认不停链。 */
    const char *ri = getenv("JX_RX_IDLE_STOP");
    if (ri != NULL && atoi(ri) != 0)
      {
        hal_i2s_rx_idle();
      }
  }

finish:
  pthread_mutex_lock(&ctx.lock);
  ctx.done = 1;
  pthread_mutex_unlock(&ctx.lock);
  /* 2026-10-30: worker 启动等待改成轮询，不再消费这里的 post；
   * 这样录音失败（ADC 全 0、没有回调 post）时，下面这一条也能把
   * worker 的第二层循环唤醒并退出，pthread_join 不会死锁。 */
  sem_post(&ctx.wake);
  pthread_join(worker, NULL);
  pthread_mutex_lock(&ctx.send_lock);
  ctx.worker_done = 1;
  pthread_mutex_unlock(&ctx.send_lock);
  sem_post(&ctx.send_wake);
  pthread_join(sender, NULL);

  /* 2026-10-29: PCM dump 必须放在 LoRa 发送全部结束之后。
   * 之前是「录音一停就 dump、sender 还在发」，115200 下 96 KB 的
   * hex 输出把控制台写满，sender 超时 => send failed rc=-1，
   * 紧接着整机 panic。挪到 join 之后，只影响输出时间，不影响音频内容。 */
  if (ctx.dumpbuf != NULL)
    {
      (void)jx_serial_send_pcm_i16("raw", ctx.dumpbuf, ctx.dumpn,
                                   AI_VOICE_AI_RATE);
    }
  ret = ctx.failed ? -EIO : 0;
  if (getenv("JX_STREAM_PROF") != NULL)
    {
      extern void jx_prof_dump(void);
      extern void jx_ct_dump(void);
      jx_prof_dump();
      jx_ct_dump();
    }
  pthread_mutex_destroy(&ctx.lock);
  sem_destroy(&ctx.wake);
  pthread_mutex_destroy(&ctx.send_lock);
  sem_destroy(&ctx.send_wake);
  sem_destroy(&ctx.send_space);
  free(ctx.ring); free(ctx.batch); free(ctx.tok); free(ctx.pcm);
  free(ctx.dumpbuf);
  if (use_lora) { jx_lora_release(); }
  return ret;
}

int jx_streamtx_run(const char *ip, int port, int secs)
{
  return jx_streamtx_run_common(ip, port, secs, 15);
}

extern void jx_lora_set_plan(int samples);
int jx_streamtx_run_lora(int secs)
{
  jx_lora_set_plan(secs*16000);
  return jx_streamtx_run_common("", 1, secs, 1);
}
