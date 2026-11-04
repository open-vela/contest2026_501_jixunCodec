/* Receiver-owned nonblocking input queue. Decoder math/weights are unchanged. */
#include <errno.h>
#include <fcntl.h>
#include <ctype.h>

static int g_dec_enabled, g_dec_failed, g_early_play;
static char g_dec_input[65536];
static size_t g_dec_used, g_dec_written;
static unsigned g_dec_rows;
static double g_dec_started;
static char g_token_path[320] = "/tmp/rx_tokens.txt";
static char g_pcm_path[320] = "/tmp/jxplay.pcm";
static char g_stereo_path[320] = "/tmp/jxplay_stereo.pcm";
static char g_live_path[320] = "/tmp/rx_live.pcm";
static char g_abort_path[320] = "/tmp/jxplay.abort";
static char g_sample_path[320] = "/tmp/jxplay.samples";

static void chain_mark(const char *event)
{
  fprintf(stderr, "[chain] event=%s mono=%.6f epoch=%.6f\n", event, now_s(), wall_s());
}

static void chain_paths(void)
{
  const char *dir = getenv("JX_CHAIN_DIR");
  if (!dir || !*dir) dir = "/tmp";
  if (dir[0] != '/' || strlen(dir) > 240) abort();
  for (const char *p = dir; *p; p++)
    if (!isalnum((unsigned char)*p) && !strchr("/_.-", *p)) abort();
  snprintf(g_token_path, sizeof(g_token_path), "%s/rx_tokens.txt", dir);
  snprintf(g_pcm_path, sizeof(g_pcm_path), "%s/jxplay.pcm", dir);
  snprintf(g_stereo_path, sizeof(g_stereo_path), "%s/jxplay_stereo.pcm", dir);
  snprintf(g_live_path, sizeof(g_live_path), "%s/rx_live.pcm", dir);
  snprintf(g_abort_path, sizeof(g_abort_path), "%s/jxplay.abort", dir);
  snprintf(g_sample_path, sizeof(g_sample_path), "%s/jxplay.samples", dir);
}

static int chain_sample_limit(uint32_t samples) {
  char tmp[336];snprintf(tmp,sizeof(tmp),"%s.tmp",g_sample_path);
  FILE *f=fopen(tmp,"w");if(!f)return -1;
  int bad=fprintf(f,"%u\n",samples)<0;
  if(fclose(f))bad=1;
  if(bad){unlink(tmp);return -1;}
  return rename(tmp,g_sample_path);
}

static void chain_abort(void)
{
  if(!g_early_play)return;
  int fd=open(g_abort_path,O_WRONLY|O_CREAT|O_TRUNC,0600);
  if(fd>=0)close(fd);
}

static void chain_command(char *cmd, size_t size, int batch)
{
  snprintf(cmd, size,
      "JX_SAMPLE_LIMIT_FILE='%s' JX_PROFILE_HALO=%d JX_DECODE_PCM_OUT='%s' JX_EARLY_PLAY=%d JX_EARLY_STEREO_OUT='%s' JX_EARLY_ABORT_FILE='%s' /userdata/tomatod/jixun_light_a_port_20261103/tokdec_profile_stream_a.sh 1.0 > '%s'%s%s%s",
      g_sample_path, g_profile_halo<0 ? 5 : g_profile_halo, g_live_path, g_early_play, g_stereo_path, g_abort_path, g_pcm_path, batch ? " < '" : "", batch ? g_token_path : "", batch ? "'" : "");
}

static void chain_drain(void)
{
  if (!g_play_pipe || g_dec_failed) return;
  while (g_dec_written < g_dec_used)
    {
      ssize_t n = write(fileno(g_play_pipe), g_dec_input + g_dec_written,
                        g_dec_used - g_dec_written);
      if (n > 0) g_dec_written += (size_t)n;
      else if (n < 0 && errno == EINTR) continue;
      else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
      else { g_dec_failed = 1; break; }
    }
}

static void chain_feed(const int64_t *row, int count)
{
  if (!g_dec_enabled || g_dec_failed) return;
  if (!g_play_pipe)
    {
      char cmd[1600];
      chain_command(cmd, sizeof(cmd), 0);
      g_dec_started = now_s();
      g_play_pipe = popen(cmd, "w");
      if (!g_play_pipe) { g_dec_failed = 1; return; }
      int fd = fileno(g_play_pipe), flags = fcntl(fd, F_GETFL, 0);
      if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
        { g_dec_failed = 1; return; }
      chain_mark("decoder_open");
    }
  char line[192];
  size_t used = 0;
  for (int i = 0; i < count; i++)
    {
      int n = snprintf(line + used, sizeof(line) - used, "%lld ", (long long)row[i]);
      if (n < 0 || (size_t)n >= sizeof(line) - used) { g_dec_failed = 1; return; }
      used += (size_t)n;
    }
  line[used++] = '\n';
  if (g_dec_used + used > sizeof(g_dec_input)) { g_dec_failed = 1; return; }
  memcpy(g_dec_input + g_dec_used, line, used);
  g_dec_used += used; g_dec_rows++;
  chain_drain();
}

/* 0: successful incremental decode; 1: safely fall back to full token file. */
static int chain_finish(void)
{
  if (!g_dec_enabled) return 1;
  double tail = now_s(), deadline = tail + 2.;
  while (g_play_pipe && !g_dec_failed && g_dec_written < g_dec_used && now_s() < deadline)
    { chain_drain(); if (g_dec_written < g_dec_used) usleep(1000); }
  if (g_dec_written != g_dec_used) g_dec_failed = 1;
  if(g_dec_failed)chain_abort();
  int rc = g_play_pipe ? pclose(g_play_pipe) : -1;
  g_play_pipe = NULL;
  fprintf(stderr, "[inc] finish rows=%u written=%zu queued=%zu failed=%d rc=%d duration_ms=%.3f tail_ms=%.3f\n",
          g_dec_rows, g_dec_written, g_dec_used, g_dec_failed, rc,
          (now_s()-g_dec_started)*1000., (now_s()-tail)*1000.);
  return g_dec_failed || rc != 0;
}

static void chain_reset(void)
{
  if (g_play_pipe) { pclose(g_play_pipe); g_play_pipe = NULL; }
  const char *enabled = getenv("JX_K7_INCREMENTAL");
  g_dec_enabled = enabled && enabled[0] == '1';
  const char *early=getenv("JX_EARLY_PLAY");
  g_early_play=g_dec_enabled && early && early[0]=='1';
  g_dec_failed = 0; g_dec_used = g_dec_written = 0; g_dec_rows = 0; g_dec_started = 0.;
  chain_paths();
  unlink(g_abort_path);
  unlink(g_sample_path);
}
