/****************************************************************************
 * apps/jixun-codec/port/jx_net.c
 *
 * 极讯 AI Codec —— 双板 WiFi/UDP 话音 index 传输（见 jx_net.h）
 ****************************************************************************/

#include <nuttx/config.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <netinet/arp.h>
#include <net/if.h>
#include <arpa/inet.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "jx_net.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"
#include "jx_serial.h"

/****************************************************************************
 * CRC32（按位实现，够快也够小；载荷最大几 KB）
 ****************************************************************************/

static uint32_t jx_crc32(const uint8_t *p, int n)
{
  uint32_t crc = 0xffffffffu;
  int i, b;

  for (i = 0; i < n; i++)
    {
      crc ^= p[i];
      for (b = 0; b < 8; b++)
        {
          crc = (crc >> 1) ^ (0xedb88320u & (uint32_t)(-(int32_t)(crc & 1u)));
        }
    }

  return crc ^ 0xffffffffu;
}

/****************************************************************************
 * 位打包：MSB-first
 ****************************************************************************/

int jx_net_pack_bits(const int64_t *idx, int ntok, int bits,
                     uint8_t *out, int out_cap)
{
  int bit = 0;
  int i, b;
  int total = ntok * bits;

  memset(out, 0, (size_t)out_cap);
  if (total > out_cap * 8)
    {
      total = out_cap * 8;
    }

  for (i = 0; i < ntok; i++)
    {
      uint64_t v = (uint64_t)idx[i];

      for (b = bits - 1; b >= 0; b--)
        {
          if (bit >= total)
            {
              return bit;
            }

          if ((v >> b) & 1u)
            {
              out[bit >> 3] |= (uint8_t)(1u << (7 - (bit & 7)));
            }

          bit++;
        }
    }

  return bit;
}

int jx_net_unpack_bits(const uint8_t *in, int nbits, int bits,
                       int64_t *idx, int max_tok)
{
  int bit = 0;
  int t;

  for (t = 0; t < max_tok && bit + bits <= nbits; t++)
    {
      uint64_t v = 0;
      int b;

      for (b = 0; b < bits; b++)
        {
          v = (v << 1) | (uint64_t)((in[bit >> 3] >> (7 - (bit & 7))) & 1u);
          bit++;
        }

      idx[t] = (int64_t)v;
    }

  return t;
}


int jx_net_setarp_static(const char *ip, const char *mac)
{
  struct arpreq req;
  struct sockaddr_in *pa = (struct sockaddr_in *)&req.arp_pa;
  unsigned m[6];
  int fd;
  int ret;
  int i;

  if (ip == NULL || mac == NULL ||
      sscanf(mac, "%x:%x:%x:%x:%x:%x",
             &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]) != 6)
    {
      return -EINVAL;
    }

  memset(&req, 0, sizeof(req));
  pa->sin_family = AF_INET;
  pa->sin_addr.s_addr = inet_addr(ip);
  req.arp_ha.sa_family = ARPHRD_ETHER;
  for (i = 0; i < 6; i++)
    {
      req.arp_ha.sa_data[i] = (char)m[i];
    }
  req.arp_flags = ATF_PERM | ATF_COM;
  strncpy(req.arp_dev, "wlan0", sizeof(req.arp_dev) - 1);

  fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    {
      return -errno;
    }
  ret = ioctl(fd, SIOCSARP, (unsigned long)(uintptr_t)&req);
  if (ret < 0)
    {
      ret = -errno;
    }
  close(fd);
  return ret;
}

/****************************************************************************
 * 发送
 ****************************************************************************/

int jx_net_send_idx(const char *ip, int port, int rate_id, int bits,
                    const int64_t *idx, int ntok, int tt, int orig_t,
                    uint32_t chunk, uint32_t last)
{
  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_send_idx(rate_id, bits, idx, ntok, tt,
                                orig_t, chunk, last);
    }

  uint8_t bitsbuf[JX_NET_HDR_LEN + JX_NET_CHUNK + 8];
  uint8_t pkt[JX_NET_HDR_LEN + JX_NET_CHUNK];
  jx_net_hdr_t h;
  int nbits;
  int paylen;
  int nchunk;
  int c, rep;
  int fd;
  int sent = 0;
  struct sockaddr_in dst;

  if ((size_t)ntok * (size_t)bits / 8 + 8 > sizeof(bitsbuf))
    {
      printf("[net] 单包过大 (%d token x %d bit)\n", ntok, bits);
      return -EFBIG;
    }

  nbits  = jx_net_pack_bits(idx, ntok, bits, bitsbuf, (int)sizeof(bitsbuf));
  paylen = (nbits + 7) / 8;
  nchunk = (paylen + JX_NET_CHUNK - 1) / JX_NET_CHUNK;

  printf("[net] 载荷 %d token x %d bit = %d bit (%d 字节) -> %d 个 UDP 分片\n",
         ntok, bits, nbits, paylen, nchunk);

  fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    {
      printf("[net] socket 失败: %d\n", errno);
      return -errno;
    }

  printf("[net] TX_PROBE_WINDOW\n");
  usleep(300 * 1000);
  memset(&dst, 0, sizeof(dst));
  dst.sin_family      = AF_INET;
  dst.sin_port        = htons((uint16_t)port);
  dst.sin_addr.s_addr = inet_addr(ip);
  if (connect(fd, (struct sockaddr *)&dst, sizeof(dst)) < 0)
    {
      printf("[net] UDP connect 失败: %d\n", errno);
      close(fd);
      return -errno;
    }

  memset(&h, 0, sizeof(h));
  h.magic   = JX_NET_MAGIC;
  h.ver     = JX_NET_VER;
  h.rate_id = (uint16_t)rate_id;
  h.bits    = (uint16_t)bits;
  h.ntok    = (uint32_t)ntok;
  h.tt      = (uint32_t)tt;
  h.orig_t  = (uint32_t)orig_t;
  h.ttl     = (uint32_t)nchunk;
  h.chunk   = chunk;
  h.last    = last;

  for (rep = 0; rep < JX_NET_REPEAT; rep++)
    {
      for (c = 0; c < nchunk; c++)
        {
          int off = c * JX_NET_CHUNK;
          int len = paylen - off;

          if (len > JX_NET_CHUNK)
            {
              len = JX_NET_CHUNK;
            }

          h.seq     = (uint32_t)c;
          h.payload = (uint32_t)len;
          h.crc     = jx_crc32(bitsbuf + off, len);

          memcpy(pkt, &h, JX_NET_HDR_LEN);
          memcpy(pkt + JX_NET_HDR_LEN, bitsbuf + off, (size_t)len);

          if (send(fd, pkt, (size_t)(JX_NET_HDR_LEN + len), 0) < 0)
            {
              printf("[net] sendto 失败: %d\n", errno);
            }
          else
            {
              sent++;
            }

          usleep(20 * 1000);
        }
    }

  usleep(50 * 1000);
  close(fd);
  printf("[net] 已发 %d 个包（%d 分片 x %d 次重复）-> %s:%d\n",
         sent, nchunk, JX_NET_REPEAT, ip, port);
  if (!last)
    {
      usleep(500 * 1000);
    }

  return sent > 0 ? 0 : -EIO;
}

/****************************************************************************
 * 接收
 ****************************************************************************/

/****************************************************************************
 * Name: jx_rx_socket
 *
 * 2026-09-14 第三十轮（分块流水）：接收套接字必须跨块常驻。
 * 原来每收一段就 socket()/bind()/close()，而"解上一块"要花好几秒，
 * 这几秒里下一块的 UDP 包全被丢掉 -> 流水必然断。这里做成静态常驻，
 * 并把接收缓冲拉大，让解码期间到达的包先排队。
 ****************************************************************************/

static int jx_rx_socket(int port)
{
  static int fd = -1;
  static int bound = -1;
  struct sockaddr_in me;
  int rcvbuf = 128 * 1024;

  if (fd >= 0 && bound == port)
    {
      return fd;
    }

  if (fd >= 0)
    {
      close(fd);
      fd = -1;
    }

  fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    {
      printf("[net] socket 失败: %d\n", errno);
      return -1;
    }

  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

  memset(&me, 0, sizeof(me));
  me.sin_family      = AF_INET;
  me.sin_port        = htons((uint16_t)port);
  me.sin_addr.s_addr = htonl(INADDR_ANY);

  if (bind(fd, (struct sockaddr *)&me, sizeof(me)) < 0)
    {
      printf("[net] bind %d 失败: %d\n", port, errno);
      close(fd);
      fd = -1;
      return -1;
    }

  bound = port;
  return fd;
}

int jx_net_recv_idx(int port, int timeout_s, int *rate_id, int *bits,
                    int64_t **out_idx, int *out_ntok, int *out_tt,
                    int *out_orig_t, uint32_t *out_chunk, uint32_t *out_last)
{
  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  if (jx_serial_enabled())
    {
      return jx_serial_recv_idx(timeout_s, rate_id, bits, out_idx,
                                out_ntok, out_tt, out_orig_t,
                                out_chunk, out_last);
    }

  uint8_t pkt[JX_NET_HDR_LEN + JX_NET_CHUNK + 64];
  uint8_t *bitsbuf = NULL;
  uint8_t *seen    = NULL;
  int64_t *idx     = NULL;
  int fd;
  int got = 0;
  int ttl = 0;
  int nbits_total = 0;
  int ntok = 0;
  int first = 1;
  int idle = 0;
  struct timeval tv;

  fd = jx_rx_socket(port);
  if (fd < 0)
    {
      return -EIO;
    }

  tv.tv_sec  = 1;
  tv.tv_usec = 0;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  for (;;)
    {
      jx_net_hdr_t h;
      int n;

      /* SO_RCVTIMEO = 1s：连续 timeout_s 次收不到东西就放弃本段 */
      n = recvfrom(fd, pkt, sizeof(pkt), 0, NULL, NULL);
      if (n < JX_NET_HDR_LEN)
        {
          /* RECV BACKOFF: NuttX can return EAGAIN immediately even with
           * SO_RCVTIMEO set; sleep so timeout_s really means seconds. */
          if (n <= 0)
            {
              usleep(1000 * 1000);
            }

          if (++idle > timeout_s)
            {
              break;
            }

          continue;
        }

      idle = 0;

      memcpy(&h, pkt, JX_NET_HDR_LEN);
      if (h.magic != JX_NET_MAGIC || h.payload > JX_NET_CHUNK ||
          n < JX_NET_HDR_LEN + (int)h.payload)
        {
          continue;
        }

      if (first)
        {
          int bytes;

          first   = 0;
          ttl     = (int)h.ttl;
          ntok    = (int)h.ntok;
          bytes   = (ntok * (int)h.bits + 7) / 8;
          nbits_total = ntok * (int)h.bits;

          if (ttl <= 0 || ttl > 4096 || bytes <= 0 || bytes > 8 * 1024 * 1024)
            {
              printf("[net] 头部异常 ttl=%d ntok=%d，丢弃\n", ttl, ntok);
              first = 1;
              continue;
            }

          bitsbuf = (uint8_t *)malloc((size_t)bytes);
          seen    = (uint8_t *)calloc((size_t)ttl, 1);
          if (bitsbuf == NULL || seen == NULL)
            {
              printf("[net] 接收缓冲分配失败 (%d 字节)\n", bytes);
              free(bitsbuf); free(seen);
              return -ENOMEM;
            }

          memset(bitsbuf, 0, (size_t)bytes);
          *rate_id    = (int)h.rate_id;
          *bits       = (int)h.bits;
          *out_tt     = (int)h.tt;
          *out_orig_t = (int)h.orig_t;
          *out_chunk  = h.chunk;
          *out_last   = h.last;

          printf("[net] 收到块 %u%s: %d token / %d bit / Tt=%d / %d 分片\n",
                 (unsigned)h.chunk, h.last ? "(末)" : "", ntok, (int)h.bits,
                 (int)h.tt, ttl);
        }

      if (h.seq < (uint32_t)ttl && seen[h.seq] == 0)
        {
          int off = (int)h.seq * JX_NET_CHUNK;

          /* 重复包很多（JX_NET_REPEAT 次），CRC 只在首见时校验，
           * 免得重复包刷一屏 "CRC 错" */

          if (h.crc != jx_crc32(pkt + JX_NET_HDR_LEN, (int)h.payload))
            {
              printf("[net] 分片 %d CRC 错，丢弃\n", (int)h.seq);
              continue;
            }

          memcpy(bitsbuf + off, pkt + JX_NET_HDR_LEN, h.payload);
          seen[h.seq] = 1;
          got++;
        }

      if (got >= ttl)
        {
          break;
        }
    }

  if (bitsbuf == NULL || got < ttl)
    {
      printf("[net] 未收全（%d/%d）\n", got, ttl <= 0 ? 0 : ttl);
      free(bitsbuf); free(seen);
      return -ETIMEDOUT;
    }

  idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)ntok);
  if (idx == NULL)
    {
      free(bitsbuf); free(seen);
      return -ENOMEM;
    }

  *out_ntok = jx_net_unpack_bits(bitsbuf, nbits_total, *bits, idx, ntok);
  *out_idx  = idx;

  free(bitsbuf);
  free(seen);
  return 0;
}
