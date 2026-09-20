/* jx_dcp.h -- ESP32-S3 DCache 硬件预载（manual preload）封装
 *
 * 背景（详见 _REALTIME_R112_20260916.md）：
 *   PSRAM 顺序读只有 85 MB/s，而且**一个核就已经打满**（两核只多 5%），
 *   所以整机耗时 T = M(访存串行时间) + C/n(计算)。
 *   这台预载引擎可以在 CPU 继续算的同时把后面的数据拉进 L1，
 *   把式子变成 T = max(M, C/n)。实测（板端 [27]d）：同一段「读+空转」
 *   135.6 ms -> 101.9 ms，其中读的代价 50.0 -> 15.6 ms，等待仅 0.7 ms。
 *
 * 硬件约束：
 *   - 一次只能预载一个**连续**区间，长度上限 = 64KB；
 *   - SIZE 寄存器单位 = 64 字节/个（实测标定），超出部分被裁剪；
 *   - ENA 写 1 后异步执行，完成时 CTRL 从 0x1 变成 0x2（ENA 清零、DONE 置位）；
 *   - 全片共一个描述符，两核共用：忙时直接跳过（非阻塞）。
 *
 * 开关：环境变量 JX_DCP=0 关闭（1 或未设 = 开）。
 * 主机回归构建（非 Xtensa）下全部退化为空操作。
 */
#ifndef JX_DCP_H
#define JX_DCP_H

#include <stdint.h>
#include <stdlib.h>
#include "jx_percore.h"

/* 一次预载的最大字节数（= DCache 容量） */
#define JX_DCP_MAX   65536u
/* 预载描述符（一个对象一次能拉的最大字节数，太大会把 L1 全换出） */
#define JX_DCP_TILE  16384u

static inline int jx_dcp_on(void)
{
  static int v = -1;
  if (v < 0)
    {
      const char *e = getenv("JX_DCP");
      v = (e != NULL && e[0] != (char)48);   /* r112：默认关闭（等价于 r107 基线） */
    }
  return v;
}

/* ---- 可调参数（getenv 只读一次），主机/板端都有 ----
 *   JX_DCPT  每次预载的字节数（默认 16384）
 *   JX_DCPC  允许发起预载的核掩码（默认 3 = 两核都可）
 *   JX_DCPB  算子分组掩码（位 0 = 逐元素/流式，位 1 = int8 矩阵，默认 3）
 */
static inline unsigned jx_dcp_tile(void)
{
  static long v = -1;
  if (v < 0) { const char *e = getenv("JX_DCPT"); v = (e != NULL) ? atol(e) : 16384; }
  if (v < 512) { v = 512; } if (v > 65536) { v = 65536; }
  return (unsigned)v;
}
static inline unsigned jx_dcp_cmask(void)
{
  static long v = -1;
  if (v < 0) { const char *e = getenv("JX_DCPC"); v = (e != NULL) ? atol(e) : 3; }
  return (unsigned)v;
}
static inline unsigned jx_dcp_bmask(void)
{
  static long v = -1;
  if (v < 0) { const char *e = getenv("JX_DCPB"); v = (e != NULL) ? atol(e) : 3; }
  return (unsigned)v;
}
static inline int jx_dcp_allow(unsigned grp)
{
  return ((jx_dcp_bmask() >> grp) & 1u) != 0u;
}

extern volatile unsigned g_dcp_kick, g_dcp_skip, g_dcp_pend;

#if defined(__XTENSA__)

#define JX_DCP_REG(o) (*(volatile unsigned *)(uintptr_t)(0x600C4000u + (unsigned)(o)))

static inline int jx_dcp_busy(void)
{
  unsigned v = JX_DCP_REG(0x40u);
  return ((v & 1u) != 0u) || ((v & 2u) == 0u);
}

/* 非阻塞下发：引擎忙或本核无权就跳过。 */
static inline void jx_dcp_pf(const void *p, unsigned bytes)
{
  if (!jx_dcp_on() || p == NULL || bytes < 64u) { return; }
  if ((jx_dcp_cmask() & (1u << (unsigned)(up_cpu_index() & 1))) == 0u) { return; }
  { unsigned tl = jx_dcp_tile(); if (bytes > tl) { bytes = tl; } }
  if (bytes > JX_DCP_MAX) { bytes = JX_DCP_MAX; }
  if (jx_dcp_busy()) { g_dcp_skip++; return; }
  JX_DCP_REG(0x44u) = (unsigned)(uintptr_t)p;
  JX_DCP_REG(0x48u) = (bytes + 63u) >> 6;
  JX_DCP_REG(0x40u) = 1u;
  g_dcp_kick++;
}

/* 等待当前预载完成（仅供探针/收尾用，热路径不要调） */
static inline void jx_dcp_wait(void)
{
  unsigned spin = 0;
  while (jx_dcp_busy() && spin < 4000000u) { spin++; }
}

#else  /* 主机回归：空操作 */

static inline int  jx_dcp_busy(void) { return 0; }
static inline void jx_dcp_pf(const void *p, unsigned bytes) { (void)p; (void)bytes; }
static inline void jx_dcp_wait(void) { }

#endif

/* 组 0 = 逐元素/流式算子，组 1 = int8 矩阵算子（主机也要有） */
#ifndef JX_DCP_ENABLE
/* 当前模型实测 DCP 下发数为 0，保留调用只会给双核热路径增加环境查询和
 * CPU 查询开销，并已在 live TX + 解码并发时触发 NuttX rmutex 断言。
 * 默认编译成空操作；需要重新做 DCP 实验时定义 JX_DCP_ENABLE。 */
static inline void jx_dcp_pf_ew(const void *p, unsigned bytes) { (void)p; (void)bytes; }
static inline void jx_dcp_pf_i8(const void *p, unsigned bytes) { (void)p; (void)bytes; }
#else
static inline void jx_dcp_pf_ew(const void *p, unsigned bytes)
{
  if (jx_dcp_allow(0u)) { jx_dcp_pf(p, bytes); }
}
static inline void jx_dcp_pf_i8(const void *p, unsigned bytes)
{
  if (jx_dcp_allow(1u)) { jx_dcp_pf(p, bytes); }
}
#endif

#endif /* JX_DCP_H */
