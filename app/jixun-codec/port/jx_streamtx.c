/* Streaming record -> encode -> UDP send.
 *
 * The audio callback only appends PCM to a ring buffer.  A worker thread
 * encodes complete batches while recording is still running.
 */

#include <nuttx/config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <unistd.h>

#include "model_config.h"
#include "ai_voice.h"
#include "jx_codec.h"
#include "jx_net.h"

extern void hal_i2s_rx_idle(void);

#define JX_STREAMTX_BATCH 64
#define JX_STREAMTX_RING_SECONDS 2
#define JX_STREAMTX_QUEUE 8

struct jx_streamtx_item
{
  int64_t  tok[JX_STREAMTX_BATCH * JX_PKT_FRAMES];
  int      ntok;
  int      tt;
  int      orig_t;
  uint32_t chunk;
  uint32_t last;
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
  int             count;
  int             done;
  int             failed;
  int             capture_started;
  int             port;
  char            ip[32];
  int16_t        *batch;
  int64_t        *tok;
  float          *pcm;
};

static void jx_streamtx_record_cb(FAR const int16_t *pcm16, size_t samples,
                                  void *arg)
{
  struct jx_streamtx_ctx *ctx = (struct jx_streamtx_ctx *)arg;
  size_t i;

  if (!ctx->capture_started)
    {
      ctx->capture_started = 1;
      printf("[streamtx] capture started\n");
    }

  pthread_mutex_lock(&ctx->lock);
  for (i = 0; i < samples; i++)
    {
      if (ctx->count >= ctx->cap)
        {
          ctx->tail = (ctx->tail + 1) % ctx->cap;
          ctx->count--;
        }

      ctx->ring[ctx->head] = pcm16[i];
      ctx->head = (ctx->head + 1) % ctx->cap;
      ctx->count++;
    }
  pthread_mutex_unlock(&ctx->lock);
  sem_post(&ctx->wake);
}

static void *jx_streamtx_sender(void *arg)
{
  struct jx_streamtx_ctx *ctx = (struct jx_streamtx_ctx *)arg;
  struct sched_param sp;

  sp.sched_priority = 200;
  (void)sched_setscheduler(0, SCHED_FIFO, &sp);

  for (;;)
    {
      int done = 0;

      sem_wait(&ctx->send_wake);
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

          rc = jx_net_send_idx(ctx->ip, ctx->port, 2, 18,
                               item.tok, item.ntok, item.tt, item.orig_t,
                               item.chunk, item.last);
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
  int noenc = (getenv("JX_STX_NOENC") != NULL);
  double prev_conv = 0.0;
  double prev_attn = 0.0;
  double prev_dec = 0.0;
  struct sched_param sp;

  sp.sched_priority = 50;
  (void)sched_setscheduler(0, SCHED_FIFO, &sp);

  /* Do not initialize the codec while I2S and WiFi are still coming up.
   * The first capture callback is the safe point: DMA is already running,
   * and the small ring gives the encoder enough time to allocate its cache.
   */

  for (;;)
    {
      int available;

      sem_wait(&ctx->wake);
      pthread_mutex_lock(&ctx->lock);
      available = ctx->count;
      pthread_mutex_unlock(&ctx->lock);
      if (available > 0 || ctx->done)
        {
          break;
        }
    }

  setenv("JX_SKIP_DEC", "1", 1);
  setenv("JX_PF", "1", 1);
  setenv("JX_PFCORE", "0", 1);
  setenv("JX_PFK", "1", 1);
  if (!noenc)
    {
      st = jx_stream_open(512);
      if (st == NULL)
        {
          ctx->failed = 1;
          return NULL;
        }
    }

  for (;;)
    {
      int stop = 0;

      sem_wait(&ctx->wake);
      for (;;)
        {
          int available;
          int done;
          int npkt;
          int need;
          int take;
          int last;
          int i;

          pthread_mutex_lock(&ctx->lock);
          available = ctx->count;
          done = ctx->done;
          npkt = available / JX_PKT_SAMPLES;
          if (npkt > JX_STREAMTX_BATCH)
            {
              npkt = JX_STREAMTX_BATCH;
            }
          if (npkt == 0 && done && available > 0)
            {
              npkt = 1;
            }
          if (npkt == 0)
            {
              stop = (done && available == 0);
              pthread_mutex_unlock(&ctx->lock);
              break;
            }

          need = npkt * JX_PKT_SAMPLES;
          take = need < available ? need : available;
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
              rc = jx_stream_feed_batch(st, ctx->batch, npkt, ctx->tok,
                                        ctx->pcm);
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
              printf("[streamtx] timing chunk=%d conv=%.1f attn=%.1f "
                     "dec=%.1f deq=%.1f endec=%.1f decnn=%.1f\n",
                     chunk, conv - prev_conv, attn - prev_attn, dec - prev_dec,
                     deq, endec, decnn);
              prev_conv = conv;
              prev_attn = attn;
              prev_dec = dec;
            }

          sem_wait(&ctx->send_space);
          pthread_mutex_lock(&ctx->send_lock);
          {
            struct jx_streamtx_item *item = &ctx->queue[ctx->qtail];

            memcpy(item->tok, ctx->tok,
                   sizeof(int64_t) * (size_t)npkt * JX_PKT_FRAMES);
            item->ntok = npkt * JX_PKT_FRAMES;
            item->tt = npkt * JX_PKT_FRAMES;
            item->orig_t = need;
            item->chunk = (uint32_t)chunk;
            item->last = (uint32_t)last;
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
      rc = jx_net_send_idx(ctx->ip, ctx->port, 2, 18, idx, 4, 4, 192,
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

int jx_streamtx_run(const char *ip, int port, int secs)
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
  strncpy(ctx.ip, ip, sizeof(ctx.ip) - 1);
  ctx.cap = JX_STREAMTX_RING_SECONDS * AI_VOICE_AI_RATE;
  ctx.ring = (int16_t *)malloc(sizeof(int16_t) * (size_t)ctx.cap);
  ctx.batch = (int16_t *)malloc(sizeof(int16_t) * JX_STREAMTX_BATCH * JX_PKT_SAMPLES);
  ctx.tok = (int64_t *)malloc(sizeof(int64_t) * JX_STREAMTX_BATCH * JX_PKT_FRAMES);
  ctx.pcm = (float *)malloc(sizeof(float) * JX_STREAMTX_BATCH * JX_PKT_SAMPLES);
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

  printf("[streamtx] recording %d s, batch=%d packets, ring=%d samples\n",
         secs, JX_STREAMTX_BATCH, ctx.cap);
  printf("  录音中 ...\n");
  ms = ai_voice_stream_record(jx_streamtx_record_cb, &ctx, secs, 0);
  printf("[streamtx] recording done: %d ms, queued=%d samples\n", ms, ctx.count);

  /* Do not keep RX DMA and HPWORK allocating DMA containers while the
   * encoder and network drain the captured ring.  A later read_slot starts
   * the prefetch chain again automatically.
   */

  hal_i2s_rx_idle();

  pthread_mutex_lock(&ctx.lock);
  ctx.done = 1;
  pthread_mutex_unlock(&ctx.lock);
  sem_post(&ctx.wake);
  pthread_join(worker, NULL);
  pthread_mutex_lock(&ctx.send_lock);
  ctx.worker_done = 1;
  pthread_mutex_unlock(&ctx.send_lock);
  sem_post(&ctx.send_wake);
  pthread_join(sender, NULL);
  ret = ctx.failed ? -EIO : 0;
  pthread_mutex_destroy(&ctx.lock);
  sem_destroy(&ctx.wake);
  pthread_mutex_destroy(&ctx.send_lock);
  sem_destroy(&ctx.send_wake);
  sem_destroy(&ctx.send_space);
  free(ctx.ring); free(ctx.batch); free(ctx.tok); free(ctx.pcm);
  return ret;
}
