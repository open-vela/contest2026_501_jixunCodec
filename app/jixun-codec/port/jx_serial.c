/****************************************************************************
 * apps/jixun-codec/port/jx_serial.c
 *
 * Line-oriented USB transport for the SAFE DEMO mode.
 *
 * Wire line:
 *   @@JXTOK1:<hex of jx_net_hdr_t + token payload>\r\n
 *
 * The PC relay parses and displays the header, but forwards the original
 * token frame to the receiver.  It never decodes or forwards PCM.
 ****************************************************************************/

#include <nuttx/config.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "jx_net.h"
#include "jx_serial.h"

#define JX_SERIAL_PREFIX "@@JXTOK1:"
#define JX_SERIAL_LINE_MAX 512
#define JX_SERIAL_PAYLOAD_MAX 192
#define JX_SERIAL_FRAG_MAX 8

static uint32_t jx_serial_crc32(const uint8_t *p, int n)
{
  uint32_t crc = 0xffffffffu;
  int i;
  int b;

  for (i = 0; i < n; i++)
    {
      crc ^= p[i];
      for (b = 0; b < 8; b++)
        {
          crc = (crc >> 1) ^
                (0xedb88320u & (uint32_t)(-(int32_t)(crc & 1u)));
        }
    }

  return crc ^ 0xffffffffu;
}

static int jx_serial_write_all(const uint8_t *buf, int len)
{
  int done = 0;

  while (done < len)
    {
      ssize_t n = write(STDOUT_FILENO, buf + done, (size_t)(len - done));

      if (n < 0)
        {
          if (errno == EINTR)
            {
              continue;
            }

          return -errno;
        }

      if (n == 0)
        {
          return -EIO;
        }

      done += (int)n;
    }

  return 0;
}

int jx_serial_enabled(void)
{
  const char *v = getenv("JX_USB");

  return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}

int jx_serial_pcm_dump_enabled(void)
{
  const char *v;

  if (!jx_serial_enabled())
    {
      return 0;
    }

  v = getenv("JX_PCM_DUMP");
  return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}

static int jx_serial_send_pcm_bytes(const char *kind, const uint8_t *pcm,
                                    int nbytes, int rate)
{
  static const char hex[] = "0123456789abcdef";
  char line[JX_SERIAL_LINE_MAX];
  int seq = 0;
  int off;

  if (kind == NULL || pcm == NULL || nbytes <= 0)
    {
      return -EINVAL;
    }

  for (off = 0; off < nbytes; off += 128)
    {
      int len = nbytes - off;
      int last;
      int i;
      int pos;

      if (len > 128)
        {
          len = 128;
        }

      last = (off + len >= nbytes) ? 1 : 0;
      pos = snprintf(line, sizeof(line), "@@JXPCM1:%s:%d:%d:%d:",
                     kind, seq, last, rate);
      if (pos < 0 || pos >= (int)sizeof(line) - 2 * 128 - 3)
        {
          return -EINVAL;
        }

      for (i = 0; i < len; i++)
        {
          uint8_t v = pcm[off + i];
          line[pos++] = hex[v >> 4];
          line[pos++] = hex[v & 0x0f];
        }

      line[pos++] = '\r';
      line[pos++] = '\n';
      if (jx_serial_write_all((const uint8_t *)line, pos) < 0)
        {
          return -EIO;
        }

      seq++;
    }

  printf("[usb] pcm dump kind=%s bytes=%d rate=%d chunks=%d\n",
         kind, nbytes, rate, seq);
  return 0;
}

int jx_serial_send_pcm_i16(const char *kind, const int16_t *pcm,
                           int nsamples, int rate)
{
  if (pcm == NULL || nsamples <= 0)
    {
      return -EINVAL;
    }

  return jx_serial_send_pcm_bytes(kind, (const uint8_t *)pcm,
                                  nsamples * (int)sizeof(int16_t), rate);
}

int jx_serial_send_pcm_f32(const char *kind, const float *pcm,
                           int nsamples, int rate)
{
  if (pcm == NULL || nsamples <= 0)
    {
      return -EINVAL;
    }

  return jx_serial_send_pcm_bytes(kind, (const uint8_t *)pcm,
                                  nsamples * (int)sizeof(float), rate);
}

int jx_serial_send_idx(int rate_id, int bits, const int64_t *idx,
                       int ntok, int tt, int orig_t,
                       uint32_t chunk, uint32_t last)
{
  static const char hex[] = "0123456789abcdef";
  uint8_t bitsbuf[JX_NET_CHUNK + 8];
  uint8_t frame[JX_NET_HDR_LEN + JX_NET_CHUNK];
  char line[JX_SERIAL_LINE_MAX];
  jx_net_hdr_t h;
  int nbits;
  int paylen;
  int nchunk;
  int c;
  int pos;

  if (ntok <= 0 || bits <= 0 || bits > 63 ||
      (size_t)ntok > (sizeof(bitsbuf) * 8u) / (size_t)bits)
    {
      printf("[usb] token frame parameter error ntok=%d bits=%d\n", ntok, bits);
      return -EFBIG;
    }

  nbits = jx_net_pack_bits(idx, ntok, bits, bitsbuf, (int)sizeof(bitsbuf));
  paylen = (nbits + 7) / 8;
  nchunk = (paylen + JX_SERIAL_PAYLOAD_MAX - 1) / JX_SERIAL_PAYLOAD_MAX;

  if (nchunk <= 0)
    {
      return -EINVAL;
    }

  memset(&h, 0, sizeof(h));
  h.magic = JX_NET_MAGIC;
  h.ver = JX_NET_VER;
  h.rate_id = (uint16_t)rate_id;
  h.bits = (uint16_t)bits;
  h.ntok = (uint32_t)ntok;
  h.tt = (uint32_t)tt;
  h.orig_t = (uint32_t)orig_t;
  h.ttl = (uint32_t)nchunk;
  h.chunk = chunk;
  h.last = last;

  for (c = 0; c < nchunk; c++)
    {
      int off = c * JX_SERIAL_PAYLOAD_MAX;
      int len = paylen - off;
      int i;

      if (len > JX_SERIAL_PAYLOAD_MAX)
        {
          len = JX_SERIAL_PAYLOAD_MAX;
        }

      h.seq = (uint32_t)c;
      h.payload = (uint32_t)len;
      h.crc = jx_serial_crc32(bitsbuf + off, len);
      memcpy(frame, &h, JX_NET_HDR_LEN);
      memcpy(frame + JX_NET_HDR_LEN, bitsbuf + off, (size_t)len);

      memcpy(line, JX_SERIAL_PREFIX, strlen(JX_SERIAL_PREFIX));
      pos = (int)strlen(JX_SERIAL_PREFIX);
      for (i = 0; i < JX_NET_HDR_LEN + len; i++)
        {
          uint8_t v = frame[i];
          line[pos++] = hex[v >> 4];
          line[pos++] = hex[v & 0x0f];
        }

      line[pos++] = '\r';
      line[pos++] = '\n';
      if (jx_serial_write_all((const uint8_t *)line, pos) < 0)
        {
          return -EIO;
        }
    }

  printf("[usb] token frame chunk=%u tokens=%d bits=%d payload=%d bytes%s\n",
         (unsigned)chunk, ntok, bits, paylen, last ? " last" : "");
  return 0;
}

static int jx_serial_hexval(char c)
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
  return -1;
}

static int jx_serial_get_line(char *line, int cap)
{
  static char *rxbuf;
  static int rxlen;

  if (rxbuf == NULL)
    {
      rxbuf = (char *)malloc(JX_SERIAL_LINE_MAX);
      if (rxbuf == NULL)
        {
          return -ENOMEM;
        }
    }

  for (;;)
    {
      char *nl = (char *)memchr(rxbuf, '\n', (size_t)rxlen);
      int copylen;
      int used;

      if (nl != NULL)
        {
          used = (int)(nl - rxbuf) + 1;
          copylen = used;
          if (copylen > cap - 1)
            {
              copylen = cap - 1;
            }

          memcpy(line, rxbuf, (size_t)copylen);
          line[copylen] = '\0';
          memmove(rxbuf, rxbuf + used, (size_t)(rxlen - used));
          rxlen -= used;
          return copylen;
        }

      if (rxlen >= JX_SERIAL_LINE_MAX)
        {
          rxlen = 0;
        }

      {
        ssize_t n = read(STDIN_FILENO, rxbuf + rxlen,
                         (size_t)(JX_SERIAL_LINE_MAX - rxlen));

        if (n > 0)
          {
            rxlen += (int)n;
            continue;
          }

        if (n == 0 || (n < 0 && (errno == EINTR || errno == EAGAIN)))
          {
            usleep(10 * 1000);
            continue;
          }

        return -errno;
      }
    }
}

int jx_serial_recv_idx(int timeout_s, int *rate_id, int *bits,
                       int64_t **out_idx, int *out_ntok, int *out_tt,
                       int *out_orig_t, uint32_t *out_chunk,
                       uint32_t *out_last)
{
  uint8_t frame[JX_NET_HDR_LEN + JX_NET_CHUNK];
  char line[JX_SERIAL_LINE_MAX];
  jx_net_hdr_t h;
  static uint8_t *asm_payload;
  static uint8_t asm_seen[JX_SERIAL_FRAG_MAX];
  static uint32_t asm_chunk = 0xffffffffu;
  static uint32_t asm_ttl;
  static uint32_t asm_got;
  int i;
  int hlen;

  (void)timeout_s;

  for (;;)
    {
      char *p;
      int nhex;
      int nline;

      nline = jx_serial_get_line(line, (int)sizeof(line));
      if (nline < 0)
        {
          return nline;
        }

      if (nline == 0)
        {
          continue;
        }

      p = strstr(line, JX_SERIAL_PREFIX);
      if (p == NULL)
        {
          continue;
        }

      p += strlen(JX_SERIAL_PREFIX);
      nhex = 0;
      while (jx_serial_hexval(p[nhex]) >= 0)
        {
          nhex++;
        }

      if (nhex < JX_NET_HDR_LEN * 2 || (nhex & 1) != 0 ||
          nhex / 2 > (int)sizeof(frame))
        {
          continue;
        }

      hlen = nhex / 2;
      for (i = 0; i < hlen; i++)
        {
          int hi = jx_serial_hexval(p[i * 2]);
          int lo = jx_serial_hexval(p[i * 2 + 1]);
          frame[i] = (uint8_t)((hi << 4) | lo);
        }

      memcpy(&h, frame, JX_NET_HDR_LEN);
      if (h.magic != JX_NET_MAGIC || h.ver != JX_NET_VER ||
          h.payload > JX_SERIAL_PAYLOAD_MAX ||
          hlen < JX_NET_HDR_LEN + (int)h.payload)
        {
          continue;
        }

      if (h.crc != jx_serial_crc32(frame + JX_NET_HDR_LEN,
                                   (int)h.payload))
        {
          printf("[usb] token CRC error chunk=%u seq=%u\n",
                 (unsigned)h.chunk, (unsigned)h.seq);
          continue;
        }

      if (h.ttl == 0 || h.ttl > JX_SERIAL_FRAG_MAX || h.seq >= h.ttl)
        {
          continue;
        }

      if (asm_chunk != h.chunk)
        {
          if (h.seq != 0)
            {
              asm_chunk = 0xffffffffu;
              continue;
            }

          asm_chunk = h.chunk;
          asm_ttl = h.ttl;
          asm_got = 0;
          if (asm_payload == NULL)
            {
              asm_payload = (uint8_t *)malloc(JX_NET_CHUNK);
              if (asm_payload == NULL)
                {
                  return -ENOMEM;
                }
            }
          memset(asm_seen, 0, sizeof(asm_seen));
          memset(asm_payload, 0, JX_NET_CHUNK);
        }

      if (h.ttl != asm_ttl)
        {
          asm_chunk = 0xffffffffu;
          continue;
        }

      if (!asm_seen[h.seq])
        {
          int off = (int)h.seq * JX_SERIAL_PAYLOAD_MAX;

          if (off + (int)h.payload > JX_NET_CHUNK)
            {
              free(asm_payload);
              asm_payload = NULL;
              asm_chunk = 0xffffffffu;
              continue;
            }

          memcpy(asm_payload + off, frame + JX_NET_HDR_LEN,
                 (size_t)h.payload);
          asm_seen[h.seq] = 1;
          asm_got++;
        }

      if (asm_got < asm_ttl)
        {
          continue;
        }

      {
        int64_t *idx;
        int ntok = (int)h.ntok;
        int nbits_total = ntok * (int)h.bits;
        int bytes = (nbits_total + 7) / 8;

        if (ntok <= 0 || h.bits == 0 || h.bits > 63 ||
            bytes <= 0 || bytes > JX_NET_CHUNK)
          {
            free(asm_payload);
            asm_payload = NULL;
            asm_chunk = 0xffffffffu;
            continue;
          }

        idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)ntok);
        if (idx == NULL)
          {
            return -ENOMEM;
          }

        *out_ntok = jx_net_unpack_bits(asm_payload, nbits_total,
                                       (int)h.bits, idx, ntok);
        *out_idx = idx;
        *rate_id = (int)h.rate_id;
        *bits = (int)h.bits;
        *out_tt = (int)h.tt;
        *out_orig_t = (int)h.orig_t;
        *out_chunk = h.chunk;
        *out_last = h.last;
      }

      printf("[usb] received chunk=%u tokens=%d bits=%d bytes=%d frags=%u%s\n",
             (unsigned)h.chunk, (int)h.ntok, (int)h.bits,
             ((int)h.ntok * (int)h.bits + 7) / 8, (unsigned)h.ttl,
             h.last ? " last" : "");
      free(asm_payload);
      asm_payload = NULL;
      asm_chunk = 0xffffffffu;
      asm_got = 0;
      return 0;
    }
}
