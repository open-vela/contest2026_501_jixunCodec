/****************************************************************************
 * apps/jixun-codec/port/jx_net.h
 *
 * 极讯 AI Codec —— 双板语音通信（WiFi/UDP）
 *
 * 数据面：编码器输出的量化索引 idx[]（每 token VQ_TOTAL_BITS 位）
 *         = 训练好的"话音语义 index"，直接当空口载荷。
 *         2 秒 3kbps 音频 ≈ 334 token × 18 bit = 752 字节，一个 UDP 包就能装下。
 *
 * 包格式（定长 40 字节小端头 + 载荷分片，分片大小 JX_NET_CHUNK）：
 *   magic 'J','X','C','1' | ver | rate | bits | ntok | tt | origT |
 *   seq | ttl | payload_bytes | crc32(payload)
 *
 * 可靠性：局域网内不做重传协议，直接整包重复 JX_NET_REPEAT 次（20ms 间隔），
 *         接收端按 seq 去重。丢包率 <1% 的场景够用，也便于现场看日志。
 ****************************************************************************/

#ifndef __APPS_JIXUN_CODEC_PORT_JX_NET_H
#define __APPS_JIXUN_CODEC_PORT_JX_NET_H

#include <stdint.h>

#define JX_NET_MAGIC      0x3143584au   /* "JXC1" 小端 */
#define JX_NET_VER        2
#define JX_NET_PORT_DEF   45678
#define JX_NET_CHUNK      1024          /* 每次 sendto 的载荷字节（含头 <1500 MTU） */
#define JX_NET_REPEAT     3
#define JX_NET_HDR_LEN    48

/* 包头（必须打包成 40 字节，字段顺序即线序） */
typedef struct
{
  uint32_t magic;         /* 0  JX_NET_MAGIC */
  uint16_t ver;           /* 4  JX_NET_VER  */
  uint16_t rate_id;       /* 6  0=1k 1=3k 2=6k */
  uint16_t bits;          /* 8  每 token 位数 */
  uint16_t reserved;      /* 10 */
  uint32_t ntok;          /* 12 token 总数 */
  uint32_t tt;            /* 16 帧数 Tt（padded/ HOP_LENGTH） */
  uint32_t orig_t;        /* 20 原始采样数（解码输出长度） */
  uint32_t seq;           /* 24 本片序号 */
  uint32_t ttl;           /* 28 分片总数 */
  uint32_t payload;       /* 32 本片载荷字节 */
  uint32_t crc;           /* 36 本片载荷 CRC32 */
  uint32_t chunk;         /* 40 分块序号（0..N-1），整段单发时为 0 */
  uint32_t last;          /* 44 1 = 本段最后一块 */
} jx_net_hdr_t;

/* 极讯 index 的位打包（MSB-first，与 PC 侧一致；返回位数） */
int jx_net_pack_bits(const int64_t *idx, int ntok, int bits,
                     uint8_t *out, int out_cap);
int jx_net_unpack_bits(const uint8_t *in, int nbits, int bits,
                       int64_t *idx, int max_tok);

/* 编码端：idx -> UDP 发到 ip:port。成功返回 0，负 errno。 */
int jx_net_setarp_static(const char *ip, const char *mac);
int jx_net_send_idx(const char *ip, int port, int rate_id, int bits,
                    const int64_t *idx, int ntok, int tt, int orig_t,
                    uint32_t chunk, uint32_t last);

/* 解码端：UDP 收 -> idx[]（调用方 free）。timeout_s 为最长等待秒数。
 * 成功返回 0，负 errno（-ETIMEDOUT 表示没收到完整包）。 */
int jx_net_recv_idx(int port, int timeout_s, int *rate_id, int *bits,
                    int64_t **out_idx, int *out_ntok, int *out_tt,
                    int *out_orig_t, uint32_t *out_chunk, uint32_t *out_last);

#endif /* __APPS_JIXUN_CODEC_PORT_JX_NET_H */
