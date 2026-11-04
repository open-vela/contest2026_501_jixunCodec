/* stream_lora_live.c - live 3kbps light-A stream codec over SX1262 LoRa.
 *
 *   tx <seconds> <mic_dev> [gain]
 *   rx <seconds> <play_dev> [gain]
 *
 * TX: ALSA capture 16k mono -> 384-sample (24ms) chunks -> 4 tokens each
 *     -> 4 chunks aggregated into one 52-byte LoRa packet every 96ms.
 * RX: one packet -> 16 tokens -> 1536 samples (96ms) -> ALSA playback.
 *
 * Capture/encode and radio TX run in separate threads; radio RX and decode
 * run in separate threads, so codec CPU time overlaps with air time.
 */
#define _GNU_SOURCE
#include "native_runtime.h"
#include <alsa/asoundlib.h>
#include <math.h>
#include <signal.h>
#include <stdatomic.h>
static atomic_int g_test_io_error;
static atomic_int g_audio_stop;
static volatile sig_atomic_t g_cancelled;
static void cancel_test(int sig) { (void)sig; g_cancelled=1; }
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "stream_codec.h"
#include "jx_neon.h"
#include "rf/sx1262.h"
#include "rf/k7_gpio.h"
#include "voice_aec.h"

#if defined(SQM_RATE_1K)
#define CODEC_MODE 1
#define CODEC_HOP 288
#define LORA_BW 0x04
#define MIN_TX_MS 60.0
#define REPLY_DELAY 0.020
#define INITIAL_LEAD 0.35
#else
#define CODEC_MODE 3
#define CODEC_HOP 96
#define LORA_BW 0x06
#define MIN_TX_MS 20.0
#define REPLY_DELAY 0.005
#define INITIAL_LEAD 0.15
#endif
#define FRAME_VERSION 0x45 /* A261 continuous-call format, protected clock header. */
#define STEP_SAMPLES (4 * CODEC_HOP)                 /* 384 samples = 24 ms */
#define CHUNKS_PER_PKT 4
#define TOKENS_PER_PKT (CHUNKS_PER_PKT * 4)    /* 16 */
#define TOK_BITS 18
#define PAYLOAD_BYTES ((TOKENS_PER_PKT * TOK_BITS + 7) / 8) /* 36 */
#define HEADER_BYTES 16
#define PKT_BYTES (HEADER_BYTES + PAYLOAD_BYTES)             /* 52 */
#define PERIOD_S (TOKENS_PER_PKT*CODEC_HOP/16000.0)
#define QDEPTH 64

/* dup_dec_thread alone has a 16,000-byte compiled frame. The native
 * pthread default is 8 KiB, so give codec workers explicit headroom. */
static int voice_worker_start(pthread_t *thread, void *(*entry)(void *), void *arg) {
    pthread_attr_t attr;
    int r = pthread_attr_init(&attr);
    if (r) return r;
    r = pthread_attr_setstacksize(&attr, 64 * 1024);
    if (!r) r = pthread_create(thread, &attr, entry, arg);
    pthread_attr_destroy(&attr);
    if (r) fprintf(stderr,"[voice thread_start_failed rc=%d]\n",r);
    return r;
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double epoch_s(void) { struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);return ts.tv_sec+ts.tv_nsec*1e-9; }

static uint16_t crc16(const uint8_t *d, int n) {
    uint16_t c = 0xFFFF;
    for (int i = 0; i < n; i++) {
        c ^= (uint16_t)d[i] << 8;
        for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    return c;
}

static void pack_tokens(const int64_t *tok, int nt, uint8_t *out, int bytes) {
    memset(out, 0, (size_t)bytes);
    int bit = 0;
    for (int i = 0; i < nt; i++) {
        uint32_t v = (uint32_t)tok[i] & ((1u << TOK_BITS) - 1u);
        for (int b = 0; b < TOK_BITS; b++, bit++)
            if ((v >> b) & 1u) out[bit >> 3] |= (uint8_t)(1u << (bit & 7));
    }
}

static void unpack_tokens(const uint8_t *in, int nt, int64_t *tok) {
    int bit = 0;
    for (int i = 0; i < nt; i++) {
        uint32_t v = 0;
        for (int b = 0; b < TOK_BITS; b++, bit++)
            if ((in[bit >> 3] >> (bit & 7)) & 1u) v |= 1u << b;
        tok[i] = (int64_t)v;
    }
}

/* ---------------- token queue ---------------- */
typedef struct {
    int64_t tok[TOKENS_PER_PKT];
    uint32_t seq;
} tok_pkt;

typedef struct {
    tok_pkt item[QDEPTH];
    int head, tail, count;
    int stop;
    pthread_mutex_t mu;
    pthread_cond_t not_empty, not_full;
    unsigned long pushed, dropped;
} tok_q;

static void q_init(tok_q *q) {
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static int q_push(tok_q *q, const tok_pkt *p, int block) {
    pthread_mutex_lock(&q->mu);
    if (q->count == QDEPTH) {
        if (!block) { q->dropped++; pthread_mutex_unlock(&q->mu); return -1; }
        while (q->count == QDEPTH && !q->stop) pthread_cond_wait(&q->not_full, &q->mu);
    }
    if (q->stop) { pthread_mutex_unlock(&q->mu); return -1; }
    q->item[q->tail] = *p;
    q->tail = (q->tail + 1) % QDEPTH;
    q->count++; q->pushed++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
    return 0;
}

static int q_pop(tok_q *q, tok_pkt *p) {
    pthread_mutex_lock(&q->mu);
    while (q->count == 0 && !q->stop) pthread_cond_wait(&q->not_empty, &q->mu);
    if (q->count == 0) { pthread_mutex_unlock(&q->mu); return -1; }
    *p = q->item[q->head];
    q->head = (q->head + 1) % QDEPTH;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
    return 0;
}

static void q_stop(tok_q *q) {
    pthread_mutex_lock(&q->mu);
    q->stop = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static int jx_pending_n;
/* ---------------- ALSA ---------------- */
static snd_pcm_t *open_pcm(const char *dev, int capture, unsigned period, unsigned periods, int *err) {
    snd_pcm_t *p = NULL;
    int r = snd_pcm_open(&p, capture ? "pcm_in0" : "pcm0", capture ? SND_PCM_STREAM_CAPTURE : SND_PCM_STREAM_PLAYBACK, 0);
    if (r < 0) { *err = r; return NULL; }
    snd_pcm_hw_params_t *hw;
    snd_pcm_hw_params_alloca(&hw);
    if ((r = snd_pcm_hw_params_any(p, hw)) < 0) goto fail;
    if ((r = snd_pcm_hw_params_set_access(p, hw, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) goto fail;
    if ((r = snd_pcm_hw_params_set_format(p, hw, SND_PCM_FORMAT_S16_LE)) < 0) goto fail;
    if ((r = snd_pcm_hw_params_set_channels(p, hw, 2)) < 0) goto fail;
    unsigned rate = 16000;
    if ((r = snd_pcm_hw_params_set_rate_near(p, hw, &rate, 0)) < 0) goto fail;
    if (rate != 16000) { fprintf(stderr, "[alsa] rate got %u\n", rate); }
    snd_pcm_uframes_t ps = period;
    if ((r = snd_pcm_hw_params_set_period_size_near(p, hw, &ps, 0)) < 0) goto fail;
    unsigned np = periods;
    if ((r = snd_pcm_hw_params_set_periods_near(p, hw, &np, 0)) < 0) goto fail;
    if ((r = snd_pcm_hw_params(p, hw)) < 0) goto fail;
    snd_pcm_uframes_t got_ps = 0, got_bs = 0;
    snd_pcm_hw_params_get_period_size(hw, &got_ps, 0);
    snd_pcm_hw_params_get_buffer_size(hw, &got_bs);
    fprintf(stderr, "[alsa] %s %s period=%lu buffer=%lu (%.0f ms)\n", dev,
            capture ? "cap" : "play", (unsigned long)got_ps, (unsigned long)got_bs,
            1000.0 * (double)got_bs / 16000.0);
    if (capture) {
        snd_pcm_prepare(p);
        int sr = snd_pcm_start(p);
        if (sr < 0) fprintf(stderr, "[warn] cap start: %s\n", snd_strerror(sr));
    }
    *err = 0;
    return p;
fail:
    *err = r;
    snd_pcm_close(p);
    return NULL;
}

/* Both capture/playback use the same two-slot I2S frame. A mono
 * capture configured before stereo playback runs at the wrong physical pace
 * on this board. Select one channel only AFTER reading full stereo frames. */
static int read_full(snd_pcm_t *p,int16_t *buf,int n) {
 int16_t stereo[STEP_SAMPLES*2];int got=0;
 const char*v=getenv("PAIR_MIC_CHANNEL");int channel=v&&v[0]=='1'?1:0;
 while(got<n&&!g_cancelled&&!atomic_load(&g_audio_stop)) {
  int take=n-got;if(take>STEP_SAMPLES)take=STEP_SAMPLES;
  snd_pcm_sframes_t r=snd_pcm_readi(p,stereo,(snd_pcm_uframes_t)take);
  if(r<0){fprintf(stderr,"[cap] readi: %s\n",snd_strerror((int)r));snd_pcm_recover(p,(int)r,1);snd_pcm_start(p);continue;}
  for(int i=0;i<r;i++)buf[got+i]=stereo[2*i+channel];got+=(int)r;
 }
 return got==n?got:-1;
}

/* The ES8388 playback path on these boards consumes frames at 2x the
 * declared rate (VOIP used --play-x2 for the same reason).  Duplicate each
 * sample so pitch and duration stay correct; 4608 content samples become
 * 9216 wire frames = 288 ms of real time. */
static int write_full(snd_pcm_t *p, const int16_t *buf, int n) {
    static int16_t dup[TOKENS_PER_PKT * CODEC_HOP * 2];
    static int16_t pending[TOKENS_PER_PKT * CODEC_HOP * 2];
    if (n > TOKENS_PER_PKT * CODEC_HOP) n = TOKENS_PER_PKT * CODEC_HOP;
    for (int i = 0; i < n; i++) { dup[2 * i] = buf[i]; dup[2 * i + 1] = buf[i]; }
    int frames = n, put = 0;
    while (put < frames&&!g_cancelled&&!atomic_load(&g_audio_stop)) {
        snd_pcm_sframes_t r = snd_pcm_writei(p, dup + 2*put, (snd_pcm_uframes_t)(frames - put));
        if (r < 0) {
            fprintf(stderr, "[play] writei: %s\n", snd_strerror((int)r));
            snd_pcm_recover(p, (int)r, 1);
            snd_pcm_prepare(p);
            continue;
        }
        put += (int)r;
    }
    if(put<frames)return -1;
    if(voice_aec.echo) {
        snd_pcm_sframes_t delay=0;
        if(snd_pcm_delay(p,&delay)<0)return -1;
        if(snd_pcm_state(p)==SND_PCM_STATE_RUNNING) {
            double start=now_s()+(double)delay/16000.0-(double)n/16000.0;
            if(jx_pending_n) {aec_reference(pending,jx_pending_n,start-(double)jx_pending_n/16000.0);jx_pending_n=0;}
            aec_reference(buf,n,start);
        } else if(jx_pending_n+n<=(int)(sizeof(pending)/sizeof(pending[0]))) {
            memcpy(pending+jx_pending_n,buf,(size_t)n*sizeof(int16_t));jx_pending_n+=n;
        }
    }
    return put;
}

/* ---------------- radio ---------------- */
static int radio_config_only(sx1262_t *d, int rx_mode);
static int radio_init(sx1262_t *d, int rx_mode) {
    static k7_line_t oe;
    if (k7_line_open(&oe, "gpio3", 21, 1, "stream-oe") != 0) return -1; /* pin17 */
    k7_line_set(&oe, 1);
    if (sx1262_init_spi(d, "/dev/spidev4.0", 1000000, "gpio3:8", "gpio3:22", NULL) != 0) return -1;
    sx1262_reset(d);
    struct timespec ts = {0, 60 * 1000 * 1000}; nanosleep(&ts, NULL);
    if (sx1262_bind_sw(d, NULL, "gpio2:31", 1) != 0) return -1; /* TXEN via DIO2, RXEN physical pin9 */
    return radio_config_only(d, rx_mode);
}

static int call_tx_parameter_dbm;
static const char *call_pa_config="unknown";
static int g_tx_dbm = 14;
static int radio_config_only(sx1262_t *d, int rx_mode) {
    if (sx1262_config_lora(d, 433175000, g_tx_dbm, 7, LORA_BW, 1, 0) != 0) return -1;
    /* Match the admission path and Semtech Rev1.2 Table13-21.
     * SetTxParams=22 alone with PA02/02 selects the 14 dBm operating mode.
     * Apply this inside radio_config_only so recovery reapplies it too. */
    if (g_tx_dbm == 22) {
        uint8_t pa[4] = {0x04, 0x07, 0x00, 0x01};
        uint8_t tx[2] = {22, 0x07};
        if (sx1262_cmd_write_public(d, SX_CMD_SET_PA_CFG, pa, sizeof(pa)) != 0 ||
            sx1262_cmd_write_public(d, SX_CMD_SET_TX_PARAMS, tx, sizeof(tx)) != 0) {
            fprintf(stderr,"[radio-power configuration_failed requested_dbm=22]\n");
            return -1;
        }
        call_pa_config = "04/07/00/01";
    } else {
        call_pa_config = "02/02/00/01";
    }
    call_tx_parameter_dbm = g_tx_dbm;
    fprintf(stderr,"[radio-power tx_parameter_dbm=%d pa=%s configured=1]\n",
            call_tx_parameter_dbm,call_pa_config);

    if (sx1262_set_lora_pkt_params(d, 8, 0, PKT_BYTES, 1, 0) != 0) return -1;
    if (sx1262_set_dio2_rfswitch(d, 1) != 0) return -1;
    sx1262_clear_device_errors(d);
    if (rx_mode) {
        sx1262_sw_rx(d);
        sx1262_clear_irq(d, 0xFFFF);
        if (sx1262_start_rx(d, 0) != 0) return -1;
    }
    return 0;
}

static int radio_full_reinit(sx1262_t *d) {
    sx1262_reset(d);
    struct timespec ts = {0, 60 * 1000 * 1000}; nanosleep(&ts, NULL);
    int r = radio_config_only(d, 0);
    fprintf(stderr, "[recover full-reinit r=%d]\n", r);
    return r;
}


static void radio_recover(sx1262_t *d, const char *why) {
    fprintf(stderr, "[recover %s full-config]\n", why);
    sx1262_set_standby(d);
    sx1262_clear_device_errors(d);
    radio_config_only(d, 0);   /* 重写全部配置(含PA/频点/IRQ), 修复可能的寄存器损坏 */
}

static void dump_chip_state(sx1262_t *d, const char *tag, uint32_t seq) {
    uint8_t st = 0; uint16_t de = 0;
    int r1 = sx1262_get_status(d, &st);
    int r2 = sx1262_get_device_errors(d, &de);
    int vt = -2, vr = -2;
    if (d->has_sw_txen) k7_line_get(&d->sw_txen, &vt);
    if (d->has_sw_rxen) k7_line_get(&d->sw_rxen, &vr);
    fprintf(stderr, "[chip %s seq=%u st_r=%d st=0x%02x de_r=%d de=0x%04x txen=%d rxen=%d]\n",
            tag, seq, r1, st, r2, de, vt, vr);
}

#include "call_mode.h"

static int send_packet(sx1262_t *d, uint32_t seq, uint32_t tok_idx, const int64_t *tok) {
    uint8_t pkt[PKT_BYTES];
    memset(pkt, 0, sizeof(pkt));
    pkt[0] = 'J'; pkt[1] = 'X'; pkt[2] = FRAME_VERSION; pkt[3] = 0;
    memcpy(pkt + 4, &seq, 4); memcpy(pkt + 8, &tok_idx, 4);
    uint16_t nt16 = TOKENS_PER_PKT; memcpy(pkt + 12, &nt16, 2);
    pack_tokens(tok, TOKENS_PER_PKT, pkt + HEADER_BYTES, PAYLOAD_BYTES);
    uint16_t c;
    mode_fill(pkt);
    c=call_crc(pkt);memcpy(pkt+14,&c,2);
    sx1262_sw_tx(d);
    if (sx1262_set_lora_payload_len(d, PKT_BYTES) != 0) return -1;
    if (sx1262_write_buffer(d, 0, pkt, PKT_BYTES) != 0) return -1;
    sx1262_clear_irq(d, 0xFFFF);
    if (sx1262_start_tx(d, 500) != 0) { dump_chip_state(d, "starttx-fail", seq); return -1; }
    int vt0 = -2, vr0 = -2;
    if (d->has_sw_txen) k7_line_get(&d->sw_txen, &vt0);
    if (d->has_sw_rxen) k7_line_get(&d->sw_rxen, &vr0);
    if ((d->has_sw_txen && vt0 != 1) || vr0 != 0)
        fprintf(stderr, "[chip txpin-bad seq=%u txen=%d rxen=%d]\n", seq, vt0, vr0);
    double t0 = now_s();
    while (now_s() - t0 < 0.8) {
        uint16_t irq = 0;
        if (sx1262_get_irq(d, &irq) != 0) break;
        if (irq & SX_IRQ_TX_DONE) {
            sx1262_clear_irq(d, 0xFFFF);
            double el = (now_s() - t0) * 1000.0;
            if (el < MIN_TX_MS) {
                fprintf(stderr, "[chip txdone-early seq=%u el=%.0fms]\n", seq, el);
                dump_chip_state(d, "txdone-early", seq);
                radio_recover(d, "txdone-early");
                return -1;
            }
            return 0;
        }
        if (irq & SX_IRQ_TIMEOUT) break;
    }
    fprintf(stderr, "[chip txfail seq=%u el=%.0fms]\n", seq, (now_s() - t0) * 1000.0);
    dump_chip_state(d, "txfail", seq);
    radio_recover(d, "txfail");
    return -1;
}

/* ---------------- tx ---------------- */
typedef struct {
    sx1262_t *radio;
    tok_q *q;
    double seconds;
    int sent, fail;
    uint32_t seq;
} tx_ctx;

static void *tx_thread(void *arg) {
    tx_ctx *c = (tx_ctx *)arg;
    double next = 0;
    long packets = (long)(c->seconds / PERIOD_S);
    uint32_t tok_idx = 0;
    for (long i = 0; i < packets; i++) {
        tok_pkt p;
        if (q_pop(c->q, &p) != 0) break;
        if (next == 0) next = now_s();
        double wait = next - now_s();
        if (wait > 0) { struct timespec w = {0, (long)(wait * 1e9)}; nanosleep(&w, NULL); }
        if (send_packet(c->radio, c->seq++, tok_idx, p.tok) == 0) c->sent++;
        else c->fail++;
        tok_idx += TOKENS_PER_PKT;
        next += PERIOD_S;
        if (now_s() > next + PERIOD_S) next = now_s(); /* do not build backlog */
    }
    c->seq = tok_idx / TOKENS_PER_PKT;
    return NULL;
}

static int run_tx(double seconds, const char *dev, float gain) {
    int err = 0;
    snd_pcm_t *cap = open_pcm(dev, 1, STEP_SAMPLES, 8, &err);
    if (!cap) { fprintf(stderr, "capture open %s: %s\n", dev, snd_strerror(err)); return 1; }
    sx1262_t d;
    if (radio_init(&d, 0) != 0) { fprintf(stderr, "radio init failed\n"); return 1; }
    SEncState *enc = senc_new();
    tok_q q; q_init(&q);
    tx_ctx tc = { .radio = &d, .q = &q, .seconds = seconds };
    pthread_t th; pthread_create(&th, NULL, tx_thread, &tc);

    int16_t pcm[STEP_SAMPLES];
    float x[STEP_SAMPLES];
    int64_t tok[4];
    double t0 = now_s();
    unsigned long chunks = 0, enc_us = 0;
    double in_sq = 0; unsigned long in_n = 0;
    tok_pkt pend; int fill = 0; uint32_t pseq = 0;
    while (now_s() - t0 < seconds) {
        if(read_full(cap,pcm,STEP_SAMPLES)!=STEP_SAMPLES)break;
        for (int i = 0; i < STEP_SAMPLES; i++) {
            float v = (float)pcm[i] * (1.0f / 32768.0f) * gain;
            if (v > 0.999f) v = 0.999f; if (v < -0.999f) v = -0.999f;
            x[i] = v;
        }
        for (int i = 0; i < STEP_SAMPLES; i++) { in_sq += (double)x[i] * x[i]; in_n++; }
        double e0 = now_s();
        int nt = senc_step(enc, x, STEP_SAMPLES, tok);
        enc_us += (unsigned long)((now_s() - e0) * 1e6);
        if (nt != 4) { fprintf(stderr, "encoder got %d tokens\n", nt); break; }
        memcpy(pend.tok + fill, tok, 4 * sizeof(int64_t));
        fill += 4;
        chunks++;
        if (fill == TOKENS_PER_PKT) {
            pend.seq = pseq++;
            q_push(&q, &pend, 0);
            fill = 0;
        }
    }
    q_stop(&q);
    pthread_join(th, NULL);
    fprintf(stderr, "[tx] chunks=%lu packets_ok=%d fail=%d avg_enc=%.1fms tokens=%u dropped=%lu rms_in=%.4f\n",
            chunks, tc.sent, tc.fail, chunks ? enc_us / 1000.0 / chunks : 0.0,
            tc.seq * TOKENS_PER_PKT, q.dropped,
            in_n ? sqrt(in_sq / (double)in_n) : 0.0);
    sx1262_set_standby(&d); sx1262_deinit(&d); senc_free(enc);
    snd_pcm_close(cap);
    return tc.fail ? 1 : 0;
}

/* ---------------- rx ---------------- */
typedef struct {
    sx1262_t *radio;
    tok_q *q;
    double seconds;
    long pkts, bad, gaps;
    int rssi_min, rssi_max;
    long rssi_sum, rssi_n;
} rx_ctx;

static void *rx_thread(void *arg) {
    rx_ctx *c = (rx_ctx *)arg;
    uint32_t expect = 0;
    double start = now_s();
    c->rssi_min = 999; c->rssi_max = -999;
    int64_t tok[TOKENS_PER_PKT];
    while (now_s() - start < c->seconds) {
        uint16_t irq = 0;
        if (sx1262_get_irq(c->radio, &irq) != 0) break;
        if (!irq) { struct timespec w = {0, 200 * 1000}; nanosleep(&w, NULL); continue; }
        if (irq & SX_IRQ_RX_DONE) {
            uint8_t len = 0, st = 0; int16_t rssi = 0;
            sx1262_get_rx_info(c->radio, &len, &st, &rssi);
        uint8_t packet_status[3]={0};
        sx1262_cmd_read_public(c->radio,0x14,NULL,0,1,packet_status,3);
        int snr_quarters=(int8_t)packet_status[1];
            sx1262_clear_irq(c->radio, 0xFFFF);
            sx1262_start_rx(c->radio, 0);   /* re-arm before decode/queue */
            uint8_t pkt[128];
            if (len == PKT_BYTES && sx1262_read_rx_payload(c->radio, st, pkt, len) == 0 &&
                pkt[0] == 'J' && pkt[1] == 'X' && pkt[2] == FRAME_VERSION) {
                uint16_t got_crc, nt16;
                memcpy(&got_crc, pkt + 14, 2); memcpy(&nt16, pkt + 12, 2);
                if (got_crc == call_crc(pkt) && nt16 == TOKENS_PER_PKT && (pkt[3]&1)!=(unsigned)call_role) {
                    uint32_t seq; memcpy(&seq, pkt + 4, 4);
                    if (seq != expect) { unsigned gap=(seq > expect)?seq-expect:1;c->gaps+=gap;atomic_fetch_add(&call_gap_count,gap);expect=seq; }
                    expect++;
                    unpack_tokens(pkt + HEADER_BYTES, TOKENS_PER_PKT, tok);
                    tok_pkt p; memcpy(p.tok, tok, sizeof(tok)); p.seq = seq;
                    q_push(c->q, &p, 1);
                    c->pkts++;
                    if (rssi < c->rssi_min) c->rssi_min = rssi;
                    if (rssi > c->rssi_max) c->rssi_max = rssi;
                    c->rssi_sum += rssi; c->rssi_n++;
                } else c->bad++;
            } else c->bad++;
        } else {
            sx1262_clear_irq(c->radio, 0xFFFF);
        }
    }
    q_stop(c->q);
    return NULL;
}

/* Prime the playback buffer with two silent 288 ms packets.  Without this
 * the first real packet starts the DAC exactly at the underrun boundary and
 * every following write returns EPIPE. */
static void prefill_silence(snd_pcm_t *p) {
    int16_t zeros[TOKENS_PER_PKT * CODEC_HOP];
    memset(zeros, 0, sizeof(zeros));
    write_full(p, zeros, TOKENS_PER_PKT * CODEC_HOP);
    write_full(p, zeros, TOKENS_PER_PKT * CODEC_HOP);
}

static int run_tone(double seconds, const char *dev) {
    int err = 0;
    snd_pcm_t *play = open_pcm(dev, 0, STEP_SAMPLES, 32, &err);
    if (!play) { fprintf(stderr, "playback open %s: %s\n", dev, snd_strerror(err)); return 1; }
    prefill_silence(play);
    int16_t out[TOKENS_PER_PKT * CODEC_HOP];
    double next = now_s() + 0.2;
    long n = 0;
    for (double t = 0; t < seconds; t += PERIOD_S) {
        for (int i = 0; i < TOKENS_PER_PKT * CODEC_HOP; i++) {
            long k = (long)(t * 16000.0) + i;
            out[i] = (int16_t)(9000.0 * sin(2.0 * M_PI * 1000.0 * (double)k / 16000.0));
        }
        double wait = next - now_s();
        if (wait > 0) { struct timespec w = {0, (long)(wait * 1e9)}; nanosleep(&w, NULL); }
        write_full(play, out, TOKENS_PER_PKT * CODEC_HOP);
        next += PERIOD_S; n++;
    }
    fprintf(stderr, "[tone] bursts=%ld dev=%s\n", n, dev);
    snd_pcm_close(play);
    return 0;
}

static int run_rx(double seconds, const char *dev, float gain) {
    int err = 0;
    snd_pcm_t *play = open_pcm(dev, 0, STEP_SAMPLES, 32, &err);
    if (!play) { fprintf(stderr, "playback open %s: %s\n", dev, snd_strerror(err)); return 1; }
    sx1262_t d;
    if (radio_init(&d, 1) != 0) { fprintf(stderr, "radio init failed\n"); return 1; }
    tok_q q; q_init(&q);
    rx_ctx rc = { .radio = &d, .q = &q, .seconds = seconds };
    pthread_t th; pthread_create(&th, NULL, rx_thread, &rc);
    /* sdec_new() allocates ~24 MB and takes a few seconds; run it after the
     * RX thread is already queueing packets so the link does not lose the
     * first ~3 s of audio at startup. */
    SDecState *dec = sdec_new();
    prefill_silence(play);

    float wav[TOKENS_PER_PKT * CODEC_HOP];
    int16_t out[TOKENS_PER_PKT * CODEC_HOP];
    double t0 = now_s();
    unsigned long decoded = 0, samples = 0, dec_us = 0;
    double out_sq = 0; unsigned long out_n = 0;
    tok_pkt p;
    while (now_s() - t0 < seconds + 1.0) {
        if (q_pop(&q, &p) != 0) break;
        double d0 = now_s();
        int ns = sdec_step(dec, p.tok, TOKENS_PER_PKT, wav);
        dec_us += (unsigned long)((now_s() - d0) * 1e6);
        for (int i = 0; i < ns; i++) {
            float v = wav[i] * gain;
            if (v > 1) v = 1; if (v < -1) v = -1;
            out[i] = (int16_t)(v * 32767.0f);
            out_sq += (double)v * v; out_n++;
        }
        write_full(play, out, ns);
        decoded++; samples += (unsigned long)ns;
    }
    pthread_join(th, NULL);
    fprintf(stderr, "[rx] packets=%ld bad=%ld gaps=%ld decoded=%lu samples=%lu (%.2fs) rssi min/avg/max=%d/%.0f/%d avg_dec=%.1fms rms_out=%.4f\n",
            rc.pkts, rc.bad, rc.gaps, decoded, samples, samples / 16000.0,
            rc.rssi_min, rc.rssi_n ? (double)rc.rssi_sum / rc.rssi_n : 0.0, rc.rssi_max,
            decoded ? dec_us / 1000.0 / decoded : 0.0,
            out_n ? sqrt(out_sq / (double)out_n) : 0.0);
    sx1262_set_standby(&d); sx1262_deinit(&d); sdec_free(dec); snd_pcm_close(play);
    return 0;
}


/* ================= duplex (TDD) =================
 * 96 ms cycle: role A transmits in the first 144 ms, role B transmits
 * 20 ms after it finishes receiving A's packet (A's TX ends at ~94 ms, so B
 * starts at ~114 ms).  Both directions send one 52-byte packet per cycle,
 * i.e. 288 ms of audio per 288 ms.  Role B derives its slot from the arrival
 * of A's packet, so no wall-clock synchronisation is needed.
 */
static double g_t0;
static double g_wall0;
static double wall_s(void) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (double)ts.tv_sec + ts.tv_nsec*1e-9 - g_wall0; }
#define DUP_CYCLE PERIOD_S
#define DUP_B_DELAY REPLY_DELAY
#define DUP_LEAD INITIAL_LEAD

static int q_trypop(tok_q *q, tok_pkt *p) {
    pthread_mutex_lock(&q->mu);
    if (q->count == 0) { pthread_mutex_unlock(&q->mu); return -1; }
    *p = q->item[q->head];
    q->head = (q->head + 1) % QDEPTH;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
    return 0;
}

typedef struct {
    snd_pcm_t *cap;
    tok_q *q;
    float gain;
    atomic_int *stop;
    unsigned long chunks;
    double in_sq;
    unsigned long in_n;
    unsigned long produced;
} dup_cap_ctx;

static void *dup_cap_thread(void *arg) {
    call_pin(4);
    dup_cap_ctx *c = (dup_cap_ctx *)arg;
    int16_t pcm[STEP_SAMPLES];
    float x[STEP_SAMPLES];
    int64_t tok[4];
    tok_pkt pend; int fill = 0; uint32_t seq = 0;
    SEncState *enc = senc_new();
    FILE *input=NULL,*audio=NULL,*meta=NULL;
    if(getenv("TEST_PCM_IN"))input=fopen(getenv("TEST_PCM_IN"),"rb");
    if(getenv("TEST_PCM_TX"))audio=fopen(getenv("TEST_PCM_TX"),"wb");
    if(getenv("TEST_TX_META"))meta=fopen(getenv("TEST_TX_META"),"w");
    if((getenv("TEST_PCM_IN")&&!input)||(getenv("TEST_PCM_TX")&&!audio)||(getenv("TEST_TX_META")&&!meta)) {
        fprintf(stderr,"[test_io_error capture_files]\n");atomic_store(&g_test_io_error,1);*c->stop=1;
        if(input)fclose(input);if(audio)fclose(audio);if(meta)fclose(meta);senc_free(enc);return NULL;
    }
    if(meta)fprintf(meta,"seq,capture_complete_epoch\n");
    /* Restart capture after model allocation so startup backlog is not encoded as live audio. */
    {int64_t at=(int64_t)call_anchor_us-atomic_load(&call_offset_us)-200000;
     if(at>(int64_t)call_us()){struct timespec t={at/1000000,(at%1000000)*1000};clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&t,NULL);}}
    snd_pcm_drop(c->cap);snd_pcm_prepare(c->cap);snd_pcm_start(c->cap);
    while (!*c->stop) {
        if(read_full(c->cap,pcm,STEP_SAMPLES)!=STEP_SAMPLES)break;
        snd_pcm_sframes_t backlog=0;
        if(snd_pcm_delay(c->cap,&backlog)<0) { *c->stop=1;atomic_store(&g_test_io_error,1);break; }
        double capture_start=now_s()-(double)(backlog+STEP_SAMPLES)/16000.0;
        if(input) {
            size_t n=fread(pcm,sizeof(int16_t),STEP_SAMPLES,input);
            if(n<STEP_SAMPLES)memset(pcm+n,0,(STEP_SAMPLES-n)*sizeof(int16_t));
        }
        aec_capture(pcm,STEP_SAMPLES,capture_start);
        aec_denoise(pcm,STEP_SAMPLES);
        for (int i = 0; i < STEP_SAMPLES; i++) {
            float v = (float)pcm[i] * (1.0f / 32768.0f) * c->gain;
            if (v > 0.999f) v = 0.999f;
            if (v < -0.999f) v = -0.999f;
            x[i] = v;
            c->in_sq += (double)v * v; c->in_n++;
        }
        if(audio&&fwrite(x,sizeof(float),STEP_SAMPLES,audio)!=STEP_SAMPLES)atomic_store(&g_test_io_error,1);
        int nt = senc_step(enc, x, STEP_SAMPLES, tok);
        if (nt != 4) break;
        memcpy(pend.tok + fill, tok, 4 * sizeof(int64_t));
        fill += 4; c->chunks++;
        if (fill == TOKENS_PER_PKT) {
            pend.seq = seq++;
            if(meta)fprintf(meta,"%u,%.9f\n",pend.seq,epoch_s());
            q_push(c->q, &pend, 0);
            c->produced++;
            if((c->produced%16)==0)fprintf(stderr,"[voice-progress encoded=%lu rms_in=%.5f]\n",c->produced,c->in_n?sqrt(c->in_sq/c->in_n):0.0);
            fill = 0;
        }
    }
    if(input)fclose(input);if(audio)fclose(audio);if(meta)fclose(meta);
    senc_free(enc);
    return NULL;
}

typedef struct {
    sx1262_t *radio;
    pthread_mutex_t *mu;
    tok_q *q;
    double seconds;
    int role;                     /* 0 = A(master), 1 = B(slave) */
    volatile double *b_tx_at;
    atomic_int *stop;
    int sent, fail, skipped, stale_dropped;
    double tx_ms_sum;
} dup_tx_ctx;

static void *dup_tx_thread(void *arg) {
 dup_tx_ctx*c=arg;call_pin(6);uint64_t index=0;
 while(!*c->stop){
  int64_t off=atomic_load(&call_offset_us);
  uint64_t master_target=call_anchor_us+index*96000u+(c->role?48000u:0u);
  int64_t local_target=(int64_t)master_target-off;int64_t now=(int64_t)call_us();
  if(local_target<now-5000){index++;c->skipped++;continue;}
  if(local_target>now){struct timespec ts={local_target/1000000,(local_target%1000000)*1000};int r;
   do{r=clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&ts,NULL);}while(r==EINTR&&!*c->stop);
  }
  if(*c->stop)break;index++;
  tok_pkt p;if(q_trypop(c->q,&p)){c->skipped++;continue;}
  /* Keep causal token order. Do not discard every pending packet to chase the
   * most recent one: that silently removes speech from a continuous call. */
  double begin=now_s();pthread_mutex_lock(c->mu);sx1262_set_standby_xosc(c->radio);
  call_tx_stamp=(uint32_t)(call_us()+atomic_load(&call_offset_us));
  int rc=send_packet(c->radio,p.seq,p.seq*TOKENS_PER_PKT,p.tok);
  sx1262_set_fallback(c->radio,0x30);sx1262_set_standby_xosc(c->radio);sx1262_sw_rx(c->radio);sx1262_clear_irq(c->radio,0xffff);
  int rearm=sx1262_start_rx(c->radio,500);pthread_mutex_unlock(c->mu);
  double elapsed=(now_s()-begin)*1000;c->tx_ms_sum+=elapsed;
  if(!rc){c->sent++;mode_transmitted(p.seq);}else c->fail++;
  fprintf(stderr,"[dtx w=%.3f seq=%u r=%d txd=%.1fms ra=%d slot_index=%"PRIu64"]\n",wall_s(),p.seq,rc,elapsed,rearm,index-1);
 }
 return NULL;
}


typedef struct {
    sx1262_t *radio;
    pthread_mutex_t *mu;
    tok_q *q;
    double seconds;
    int role;
    volatile double *b_tx_at;
    atomic_int *stop;
    long pkts, bad, gaps;
    int rssi_min, rssi_max;
    long rssi_sum, rssi_n;
} dup_rx_ctx;

static void *dup_rx_thread(void *arg) {
    call_pin(7);
    dup_rx_ctx *c = (dup_rx_ctx *)arg;
    uint32_t expect = 0;
    int consec_to = 0, storms = 0, reinits = 0, sniffed = 0;
    double start = now_s();
    double hb = start;
    double last_irq = start;
    c->rssi_min = 999; c->rssi_max = -999;
    int64_t tok[TOKENS_PER_PKT];
    while (now_s() - start < c->seconds && !*c->stop) {
        if (now_s() - hb > 5.0) { hb = now_s(); fprintf(stderr, "[rx-alive t=%.1f]\n", now_s()-g_t0); }
        uint16_t irq = 0;
        pthread_mutex_lock(c->mu);
        int gr = sx1262_get_irq(c->radio, &irq);
        if (irq) last_irq = now_s();
        if (!irq && now_s() - last_irq > 3.0) {
            fprintf(stderr, "[rx-watchdog no-irq %.1fs]\n", now_s() - last_irq);
            dump_chip_state(c->radio, "watchdog", expect);
            if (storms == 0) radio_recover(c->radio, "watchdog"); else { radio_full_reinit(c->radio); reinits++; }
            sx1262_set_fallback(c->radio, 0x30);
            storms++;
            sx1262_set_standby_xosc(c->radio); sx1262_sw_rx(c->radio);
            int rw = sx1262_start_rx(c->radio, 500);
            fprintf(stderr, "[rx-watchdog rearm r=%d]\n", rw);
            pthread_mutex_unlock(c->mu);
            last_irq = now_s();
            continue;
        }
        if (gr != 0 || !irq) { pthread_mutex_unlock(c->mu); struct timespec w = {0, 200 * 1000}; nanosleep(&w, NULL); continue; }
        if (irq & SX_IRQ_RX_DONE) { consec_to = 0; storms = 0; sniffed = 0; }
        if (!(irq & SX_IRQ_RX_DONE)) {
            if (irq & SX_IRQ_TIMEOUT) { fprintf(stderr, "[dirq w=%.3f TIMEOUT #%d]\n", wall_s(), consec_to + 1); if (++consec_to == 1 || (consec_to % 20) == 0) dump_chip_state(c->radio, "rx-timeout", expect); if (consec_to >= 20) { if (storms == 0) radio_recover(c->radio, "rx-storm"); else { radio_full_reinit(c->radio); reinits++; } sx1262_set_fallback(c->radio, 0x30); storms++; consec_to = 0; }
            if (storms >= 2 && !sniffed) {
                sniffed = 1;
                sx1262_sw_rx(c->radio);
                sx1262_start_rx(c->radio, 0);
                int mn = 999, mx = -999; long sum = 0;
                for (int si = 0; si < 40; si++) {
                    struct timespec sw2 = {0, 50 * 1000 * 1000}; nanosleep(&sw2, NULL);
                    int dbm = -999;
                    if (sx1262_get_rssi_inst(c->radio, &dbm) == 0) {
                        if (dbm < mn) mn = dbm; if (dbm > mx) mx = dbm; sum += dbm;
                    }
                }
                fprintf(stderr, "[sniff rssi min=%d max=%d avg=%ld]\n", mn, mx, sum / 40);
                sx1262_set_standby(c->radio);
            } sx1262_set_standby_xosc(c->radio); sx1262_sw_rx(c->radio); int rto = sx1262_start_rx(c->radio, 500); if (rto != 0) fprintf(stderr, "[rearm-timeout r=%d]\n", rto); }
            sx1262_clear_irq(c->radio, 0xFFFF);
            pthread_mutex_unlock(c->mu); continue; }
        uint8_t len = 0, st = 0; int16_t rssi = 0;
        sx1262_get_rx_info(c->radio, &len, &st, &rssi);
        uint8_t packet_status[3]={0};
        sx1262_cmd_read_public(c->radio,0x14,NULL,0,1,packet_status,3);
        int snr_quarters=(int8_t)packet_status[1];
        sx1262_clear_irq(c->radio, 0xFFFF);
        uint8_t pkt[128];
        if (len > sizeof(pkt)) len = sizeof(pkt);   /* 防止损坏的长度字节栈溢出 */
        int rr = sx1262_read_rx_payload(c->radio, st, pkt, len);
        sx1262_start_rx(c->radio, 500);
        pthread_mutex_unlock(c->mu);
        double t_done = now_s();
        if (len == PKT_BYTES && rr == 0 && pkt[0]=='J' && pkt[1]=='R' && pkt[2]==1) {
            uint16_t control_crc=(uint16_t)(pkt[PKT_BYTES-2]|((uint16_t)pkt[PKT_BYTES-1]<<8));
            if(control_crc==crc16(pkt,PKT_BYTES-2))continue; /* repeated admission confirmation */
        }
        if (len == PKT_BYTES && rr == 0 && pkt[0] == 'J' && pkt[1] == 'X' && pkt[2] == FRAME_VERSION) {
            uint16_t got_crc, nt16;
            memcpy(&got_crc, pkt + 14, 2); memcpy(&nt16, pkt + 12, 2);
            if (got_crc == call_crc(pkt) && nt16 == TOKENS_PER_PKT && (pkt[3]&1)!=(unsigned)call_role) {
                uint32_t seq; memcpy(&seq, pkt + 4, 4);
                if (seq != expect) { unsigned gap=(seq > expect)?seq-expect:1;c->gaps+=gap;atomic_fetch_add(&call_gap_count,gap);expect=seq; }
                expect++;
                mode_received(pkt,seq,rssi,snr_quarters);
                unpack_tokens(pkt + HEADER_BYTES, TOKENS_PER_PKT, tok);
                tok_pkt p; memcpy(p.tok, tok, sizeof(tok)); p.seq = seq;
                q_push(c->q, &p, 1);
                c->pkts++;
                if (rssi < c->rssi_min) c->rssi_min = rssi;
                if (rssi > c->rssi_max) c->rssi_max = rssi;
                c->rssi_sum += rssi; c->rssi_n++;
                fprintf(stderr, "[drx w=%.3f seq=%u rssi=%d]\n", wall_s(), seq, (int)rssi);
                fprintf(stderr,"[rfq w=%.3f seq=%u rssi=%d snr_db=%.2f]\n",wall_s(),seq,(int)rssi,snr_quarters/4.0);
                if (c->role == 1) *c->b_tx_at = t_done + DUP_B_DELAY;
            } else c->bad++;
        } else c->bad++;
    }
    *c->stop = 1;
    q_stop(c->q);
    return NULL;
}

typedef struct {
    tok_q *q;
    SDecState *dec;
    snd_pcm_t *play;
    float gain;
    atomic_int *stop;
    unsigned long decoded, samples;
    double out_sq; unsigned long out_n;
    double dec_us;
} dup_dec_ctx;

static void *dup_dec_thread(void *arg) {
    call_pin(5);
    dup_dec_ctx *c = (dup_dec_ctx *)arg;
    float wav[TOKENS_PER_PKT * CODEC_HOP];
    int16_t out[TOKENS_PER_PKT * CODEC_HOP];
    tok_pkt p;
    int16_t buffered[2][TOKENS_PER_PKT*CODEC_HOP];
    int buffered_n[2],held=0,started=0;
    FILE *audio=NULL,*meta=NULL;unsigned long offset=0;
    if(getenv("TEST_PCM_RX"))audio=fopen(getenv("TEST_PCM_RX"),"wb");
    if(getenv("TEST_RX_META"))meta=fopen(getenv("TEST_RX_META"),"w");
    if((getenv("TEST_PCM_RX")&&!audio)||(getenv("TEST_RX_META")&&!meta)) {
        fprintf(stderr,"[test_io_error decode_files]\n");atomic_store(&g_test_io_error,1);*c->stop=1;
        if(audio)fclose(audio);if(meta)fclose(meta);return NULL;
    }
    if(meta)fprintf(meta,"seq,samples,offset,decode_complete_epoch,decode_ms,concealed\n");
    tok_pkt pending,previous;int pending_valid=0,have_previous=0;uint32_t expected=0;
    unsigned long concealed_packets=0;int trim_remaining=4118;
    while (!*c->stop&&!g_cancelled) {
        if(!pending_valid) {if(q_pop(c->q,&pending))break;pending_valid=1;}
        if(have_previous&&pending.seq<expected) {pending_valid=0;continue;}
        int concealed=have_previous&&pending.seq>expected;
        if(concealed) {p=previous;p.seq=expected;concealed_packets++;}
        else {p=pending;pending_valid=0;}
        previous=p;have_previous=1;expected=p.seq+1;
        double d0 = now_s();
        int ns = sdec_step(c->dec, p.tok, TOKENS_PER_PKT, wav);
        double decode_ms=(now_s()-d0)*1000;
        c->dec_us += decode_ms*1000;
        if(ns<0||ns>TOKENS_PER_PKT*CODEC_HOP){atomic_store(&g_test_io_error,1);*c->stop=1;break;}
        {int cut=trim_remaining<ns?trim_remaining:ns;trim_remaining-=cut;ns-=cut;if(cut)memmove(wav,wav+cut,(size_t)ns*sizeof(float));}
        if(!ns){c->decoded++;continue;}
        for (int i = 0; i < ns; i++) {
            float v = wav[i] * c->gain;
            if (v > 1) v = 1;
            if (v < -1) v = -1;
            out[i] = (int16_t)(v * 32767.0f);
            wav[i]=v;
            c->out_sq += (double)v * v; c->out_n++;
        }
        if(audio&&fwrite(wav,sizeof(float),(size_t)ns,audio)!=(size_t)ns)atomic_store(&g_test_io_error,1);
        if(meta)fprintf(meta,"%u,%d,%lu,%.9f,%.3f,%d\n",p.seq,ns,offset,epoch_s(),decode_ms,concealed);
        offset+=(unsigned long)ns;
        if(!started) {
            memcpy(buffered[held],out,(size_t)ns*sizeof(int16_t));buffered_n[held++]=ns;
            if(held==2) {
                fprintf(stderr,"[voice plc concealed_packets=%lu]\n",concealed_packets);
    for(int i=0;i<held;i++)write_full(c->play,buffered[i],buffered_n[i]);
                held=0;started=1;
                fprintf(stderr,"[voice playback_started buffered_packets=2 epoch=%.9f]\n",epoch_s());
            }
        } else write_full(c->play, out, ns);
        c->decoded++; c->samples += (unsigned long)ns;
        if((c->decoded%16)==0) fprintf(stderr,"[voice-progress decoded=%lu audio_s=%.3f rms_out=%.5f]\n",c->decoded,c->samples/16000.0,c->out_n?sqrt(c->out_sq/c->out_n):0.0);
    }
    fprintf(stderr,"[voice plc concealed_packets=%lu]\n",concealed_packets);
    if(!*c->stop&&!g_cancelled)
        for(int i=0;i<held;i++)write_full(c->play,buffered[i],buffered_n[i]);
    if(audio)fclose(audio);if(meta)fclose(meta);
    return NULL;
}

#include "lora_join.h"

static int run_duplex(double seconds, const char *dev, float gain, int role, int notx, int norx) {
    int err = 0;
    sx1262_t d;
    if (radio_init(&d, norx ? 0 : 1) != 0) { fprintf(stderr, "radio init failed\n"); return 1; }
    if (getenv("REQUIRE_JOIN") && atoi(getenv("REQUIRE_JOIN"))) {
        int joined=lora_join(&d,role,8);
        if(joined) { fprintf(stderr,"[voice blocked join_error=%d]\n",joined);sx1262_reset(&d);sx1262_deinit(&d);return joined; }
        fprintf(stderr,"[voice admitted role=%c cookie=%08x]\n",role?'B':'A',join_cookie);
    }
    mode_init(role);
    snd_pcm_t *cap = open_pcm(dev, 1, STEP_SAMPLES, 16, &err);
    if (!cap) { fprintf(stderr,"capture open: %s\n",snd_strerror(err));sx1262_deinit(&d);return 1; }
    snd_pcm_t *play = open_pcm(dev, 0, STEP_SAMPLES, 32, &err);
    if (!play) { fprintf(stderr,"playback open: %s\n",snd_strerror(err));snd_pcm_close(cap);sx1262_deinit(&d);return 1; }
    snd_pcm_sw_params_t *sw;snd_pcm_sw_params_alloca(&sw);
    int sw_result=snd_pcm_sw_params_current(play,sw);
    if(sw_result>=0)sw_result=snd_pcm_sw_params_set_start_threshold(play,sw,TOKENS_PER_PKT*CODEC_HOP*2);
    if(sw_result>=0)sw_result=snd_pcm_sw_params(play,sw);
    if(sw_result<0) {fprintf(stderr,"[voice playback_params_failed %s]\n",snd_strerror(sw_result));snd_pcm_close(cap);snd_pcm_close(play);sx1262_deinit(&d);return 1;}
    fprintf(stderr,"[voice audio_opened role=%c]\n",role?'B':'A');
    if(aec_init()) {
        fprintf(stderr,"[aec fatal initialization_failed]\n");
        snd_pcm_close(cap);snd_pcm_close(play);sx1262_deinit(&d);return 9;
    }
    sx1262_set_fallback(&d, 0x30);   /* TX/RX 完成后回落 STDBY_XOSC, 晶体不停 */
    sx1262_start_rx(&d, 500);
    g_t0 = now_s();
    { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); g_wall0 = (double)ts.tv_sec + ts.tv_nsec*1e-9; }
    fprintf(stderr, "[start mono0 wall0]\n");
    tok_q txq, rxq; q_init(&txq); q_init(&rxq);
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    atomic_int stop = 0;
    volatile double b_tx_at = 0;
    dup_rx_ctx rc = { .radio=&d, .mu=&mu, .q=&rxq, .seconds=seconds+2.0, .role=role, .b_tx_at=&b_tx_at, .stop=&stop };
    pthread_t thrx, thcap, thtx, thdec;
    int rx_started=0, cap_started=0, tx_started=0, dec_started=0;
    int startup_error=0, switch_mode=0;
    SDecState *dec = sdec_new();
    /* Playback starts from real decoded audio, not silence that drains while joining. */
    dup_cap_ctx cc = { .cap=cap, .q=&txq, .gain=gain, .stop=&stop };
    dup_tx_ctx tc = { .radio=&d, .mu=&mu, .q=&txq, .seconds=seconds, .role=role, .b_tx_at=&b_tx_at, .stop=&stop };
    dup_dec_ctx dc = { .q=&rxq, .dec=dec, .play=play, .gain=1.0f, .stop=&stop };
    if (!dec) {startup_error=1;goto threads_stop;}
    if (!norx) {
        if (voice_worker_start(&thrx,dup_rx_thread,&rc)) {startup_error=1;goto threads_stop;}
        rx_started=1;
    }
    if (voice_worker_start(&thdec,dup_dec_thread,&dc)) {startup_error=1;goto threads_stop;}
    dec_started=1;
    if (voice_worker_start(&thcap,dup_cap_thread,&cc)) {startup_error=1;goto threads_stop;}
    cap_started=1;
    if (!notx) {
        if (voice_worker_start(&thtx,dup_tx_thread,&tc)) {startup_error=1;goto threads_stop;}
        tx_started=1;
    }
    double deadline = now_s() + seconds;
    while (!stop && !g_cancelled && now_s() < deadline) {
        switch_mode=mode_tick();if(switch_mode)break;
        struct timespec w = {0, 50 * 1000 * 1000};
        nanosleep(&w, NULL);
    }
threads_stop:
    stop = 1;
    atomic_store(&g_audio_stop,1);
    q_stop(&txq); q_stop(&rxq);
    if (tx_started) pthread_join(thtx, NULL);
    if (cap_started) pthread_join(thcap, NULL);
    if (rx_started) pthread_join(thrx, NULL);
    if (dec_started) pthread_join(thdec, NULL);
    fprintf(stderr, "[dup %s] role=%c tx ok=%d fail=%d skip=%d avg_tx=%.1fms | rx pkt=%ld bad=%ld gaps=%ld rssi=%d/%.0f/%d | dec=%lu %.2fs rms_in=%.4f rms_out=%.4f avg_dec=%.1fms\n",
            dev, role ? 'B' : 'A', tc.sent, tc.fail, tc.skipped,
            tc.sent ? tc.tx_ms_sum / tc.sent : 0.0,
            rc.pkts, rc.bad, rc.gaps, rc.rssi_min,
            rc.rssi_n ? (double)rc.rssi_sum / rc.rssi_n : 0.0, rc.rssi_max,
            dc.decoded, dc.samples / 16000.0,
            cc.in_n ? sqrt(cc.in_sq / (double)cc.in_n) : 0.0,
            dc.out_n ? sqrt(dc.out_sq / (double)dc.out_n) : 0.0,
            dc.decoded ? dc.dec_us / 1000.0 / dc.decoded : 0.0);
    fprintf(stderr,"[voice queues stale_dropped=%d tx_overflow=%lu rx_overflow=%lu]\n",tc.stale_dropped,txq.dropped,rxq.dropped);
    pthread_cond_destroy(&txq.not_empty); pthread_cond_destroy(&txq.not_full);
    pthread_mutex_destroy(&txq.mu);
    pthread_cond_destroy(&rxq.not_empty); pthread_cond_destroy(&rxq.not_full);
    pthread_mutex_destroy(&rxq.mu); pthread_mutex_destroy(&mu);
    sx1262_set_standby(&d); sx1262_deinit(&d); sdec_free(dec);
    snd_pcm_close(cap); snd_pcm_close(play);
    aec_destroy();
    pthread_mutex_destroy(&voice_aec.mu);
    return startup_error||atomic_load(&g_test_io_error)?9:(switch_mode?40+switch_mode:0);
}


static int run_rf_probe(int transmit,int count) {
    sx1262_t d;if(radio_init(&d,!transmit))return 2;
    int sent=0,received=0,bad=0;int64_t tokens[TOKENS_PER_PKT]={0};
    if(transmit) {
        for(int i=0;i<count&&!g_cancelled;i++) {
            sx1262_set_standby_xosc(&d);
            if(send_packet(&d,i,i*TOKENS_PER_PKT,tokens))bad++;else sent++;
            struct timespec gap={0,65000000};nanosleep(&gap,NULL);
        }
    } else {
        double end=now_s()+count;
        while(now_s()<end&&!g_cancelled) {
            uint16_t irq=0;if(sx1262_get_irq(&d,&irq)){bad++;break;}
            if(irq&SX_IRQ_RX_DONE) {
                uint8_t len=0,off=0,pkt[128];int16_t rssi=0;
                sx1262_get_rx_info(&d,&len,&off,&rssi);
                int r=len<=sizeof(pkt)?sx1262_read_rx_payload(&d,off,pkt,len):-1;
                uint16_t crc=0;if(len==PKT_BYTES)memcpy(&crc,pkt+14,2);
                if(!r&&len==PKT_BYTES&&pkt[2]==FRAME_VERSION&&crc==call_crc(pkt)) {
                    extern int g_sx1262_last_snr_q4;
                    received++;fprintf(stderr,"[rf_probe rx=%d seq=%u rssi=%d snr_db=%.2f]\n",received,call_rd32(pkt+4),rssi,g_sx1262_last_snr_q4/4.0);
                } else bad++;
            }
            if(irq&(SX_IRQ_RX_DONE|SX_IRQ_TIMEOUT|SX_IRQ_CRC_ERR|SX_IRQ_HEADER_ERR)) {
                sx1262_clear_irq(&d,0xffff);sx1262_start_rx(&d,500);
            }
            struct timespec gap={0,500000};nanosleep(&gap,NULL);
        }
    }
    fprintf(stderr,"[rf_probe tx=%d rx=%d bad=%d]\n",sent,received,bad);
    sx1262_set_standby(&d);sx1262_deinit(&d);return bad?3:0;
}

static int run_codec_bench(int chunks,int lane) {
    if(chunks<8||chunks>128||chunks%4)return 2;
    call_pin(lane);
    SEncState *enc=senc_new();SDecState *dec=sdec_new();
    if(!enc||!dec){if(enc)senc_free(enc);if(dec)sdec_free(dec);return 9;}
    float x[STEP_SAMPLES],wav[TOKENS_PER_PKT*CODEC_HOP];int64_t tokens[TOKENS_PER_PKT];
    for(int i=0;i<STEP_SAMPLES;i++)x[i]=0.1f*sinf((float)i*0.05f);
    double enc_ms=0,dec_ms=0;int produced=0;uint64_t digest=1469598103934665603ull;
    for(int i=0;i<chunks&&!g_cancelled;i++) {
        double t=now_s();int n=senc_step(enc,x,STEP_SAMPLES,tokens+(i%4)*4);enc_ms+=(now_s()-t)*1000;
        if(n!=4){senc_free(enc);sdec_free(dec);return 9;}
        if(i%4==3) {
            for(int j=0;j<TOKENS_PER_PKT;j++){digest^=(uint64_t)tokens[j];digest*=1099511628211ull;}
            t=now_s();n=sdec_step(dec,tokens,TOKENS_PER_PKT,wav);dec_ms+=(now_s()-t)*1000;produced++;
            if(n!=TOKENS_PER_PKT*CODEC_HOP){senc_free(enc);sdec_free(dec);return 9;}
        }
    }
    fprintf(stderr,"[codec-bench chunks=%d encoded_audio_ms=%.1f encode_ms=%.3f encode_chunk_ms=%.3f decode_ms=%.3f decode_packet_ms=%.3f token_hash=%016"PRIx64"]\n",
            chunks,chunks*STEP_SAMPLES/16.0,enc_ms,enc_ms/chunks,dec_ms,produced?dec_ms/produced:0,digest);
    senc_free(enc);sdec_free(dec);
    return chunks==32&&digest!=UINT64_C(0x7c5ab4afb3b67a3a)?10:0;
}

struct codec_bench_job {int chunks,lane,result;};
static void *codec_bench_worker(void *opaque) {
    struct codec_bench_job *job=opaque;
    job->result=run_codec_bench(job->chunks,job->lane);return NULL;
}
static int run_codec_dual_bench(int chunks) {
    struct codec_bench_job jobs[2]={{chunks,4,-1},{chunks,5,-1}};
    pthread_t threads[2];int started=0,error=0;
    for(int i=0;i<2;i++) {
        error=voice_worker_start(&threads[i],codec_bench_worker,&jobs[i]);
        if(error)break;started++;
    }
    for(int i=0;i<started;i++)pthread_join(threads[i],NULL);
    fprintf(stderr,"[codec-dual-bench started=%d results=%d,%d]\n",started,jobs[0].result,jobs[1].result);
    return error||started!=2||jobs[0].result||jobs[1].result?10:0;
}

static int jx_voice_run(int argc, char **argv) {
    signal(SIGINT,cancel_test);signal(SIGTERM,cancel_test);
    if (argc < 4) {
        fprintf(stderr, "usage: %s tx|rx <seconds> <alsa_dev> [gain]\n", argv[0]);
        return 2;
    }
    if(!strcmp(argv[1],"bench"))return run_codec_bench(atoi(argv[2]),4);
    if(!strcmp(argv[1],"benchdual"))return run_codec_dual_bench(atoi(argv[2]));
    if (!strcmp(argv[1],"rftx")||!strcmp(argv[1],"rfrx")) {g_tx_dbm=22;return run_rf_probe(!strcmp(argv[1],"rftx"),atoi(argv[2]));}
    if (!strcmp(argv[1],"register")) {
        int role=argv[3][0]=='b';sx1262_t d;
        const char *power=getenv("TX_DBM");if(power)g_tx_dbm=atoi(power);
        if(radio_init(&d,1))return 2;
        int rc=lora_join(&d,role,atof(argv[2]));
        sx1262_reset(&d);sx1262_sw_rx(&d);sx1262_deinit(&d);return rc;
    }
    float gain = argc > 4 ? (float)atof(argv[4]) : 1.0f;
    if (!strcmp(argv[1], "tx")) return run_tx(atof(argv[2]), argv[3], gain);
    if (!strcmp(argv[1], "rx")) return run_rx(atof(argv[2]), argv[3], gain);
    if (!strcmp(argv[1], "tone")) return run_tone(atof(argv[2]), argv[3]);
    { const char *e = getenv("TX_DBM"); if (e) g_tx_dbm = atoi(e); }
    if (!strcmp(argv[1], "duplex")) {
        if (argc < 6 || (argv[5][0] != 'a' && argv[5][0] != 'b')) {
            fprintf(stderr, "usage: %s duplex <seconds> <alsa_dev> <gain> a|b [notx|norx]\n", argv[0]);
            return 2;
        }
        int notx = argc > 6 && !strcmp(argv[6], "notx");
        int norx = argc > 6 && !strcmp(argv[6], "norx");
        return run_duplex(atof(argv[2]), argv[3], gain, argv[5][0] == 'b', notx, norx);
    }
    return 2;
}

int k7_voice_main(int argc,char **argv) {
 if(k7_native_claim())return 1;
#ifdef __NuttX__
 if(wt_names_begin()){k7_native_release();return 9;}
#endif
 g_cancelled=0;atomic_store(&g_test_io_error,0);atomic_store(&g_audio_stop,0);jx_pending_n=0;g_tx_dbm=22;
#ifdef JX_NEON_ENABLED
 fprintf(stderr,"[native codec_helpers=%d]\n",jx_parallel_start());
#endif
 int rc=jx_voice_run(argc,argv);
#ifdef JX_NEON_ENABLED
 jx_parallel_stop();
#endif
#ifdef __NuttX__
 wt_names_end();
#endif
 k7_native_release();return rc;
}
