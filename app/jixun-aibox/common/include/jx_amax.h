#ifndef JX_AMAX_H
#define JX_AMAX_H

/* ============================================================================
 * jx_amax.h — "逐位置最大幅值"边车（P0 / 2026-09-17 第四十五轮）
 *
 * 问题：conv/linear 的 int8 快路径在点积之前必须先把输入激活量化成 int8，
 * 而逐位置量化要用到"该位置所有输入通道 |x| 的最大值"（Pass A）。Pass A 只能
 * 把整张输入激活从头读一遍 —— 板端实测单核 14~17 周期/元素，而 4 字节/元素
 * 在 240 MHz + ~80 MB/s PSRAM 下的带宽下限就是 12 周期/元素：这一趟读已经跑
 * 在内存带宽上，纯粹是被多读了一遍（整机 ≈ 282 ms / 3570 ms）。
 *
 * 关键观察：产生这张激活的上一个算子，在它的 epilogue 里**手里正握着每个
 * 输出元素**。让它顺手把 |out| 折进 amax[t]（一次 SRAM 读改写，~2 周期/
 * 元素），消费者就可以整趟跳过 Pass A。
 *
 * 正确性：
 *   - 边车带 (ptr, C, T) 键：只有消费者手里的输入指针/通道数/长度都对得上才用；
 *     生产者的 C 可以**小于**消费者的 C（例如 81 = 80 + 1 的拼接），此时只覆盖
 *     前 C 条通道，剩下的通道消费者照旧自己扫。
 *   - 再加一层抽样复核（JX_AMAXV，默认 4 个位置全通道重算 max 比对），
 *     防止缓冲区被复用后键偶然相同。
 *   - 键不命中 / 复核不过 / JX_AMAX=0 -> 完全退回原来的 Pass A，结果逐位一致。
 *
 * 缓冲是 ping-pong 两片：同一个算子既是消费者又是生产者，不能让 arm() 把
 * 正在读的那片清掉。
 * ========================================================================== */

extern unsigned *jx_amax_cur;            /* 武装期间指向本张量(生产者)的 amax 缓冲 */

extern unsigned long jx_amax_prod_pos;   /* 生产侧：被边车记录的位置数 */
extern unsigned long jx_amax_prod_call;  /* 生产侧：武装次数 */
extern unsigned long jx_amax_tot_pos;    /* 消费侧：Pass A 处理过的总位置数 */
extern unsigned long jx_amax_hit_pos;    /* 消费侧：命中边车而免掉 Pass A 的位置数 */
extern unsigned long jx_amax_hit_elem;   /* 消费侧：命中边车而免读的 (ci,pos) 对数 */
extern unsigned long jx_amax_calls;      /* 消费侧：probe 次数 */
extern unsigned long jx_amax_rej;        /* 消费侧：键不匹配 / 复核不过次数 */

void jx_amax_arm(const float *dst, int C, int T);
void jx_amax_pub(void);
void jx_amax_kill(void);
const unsigned *jx_amax_probe(const float *src, int C, int T, int *have);
void jx_amax_dump(void);

/* 生产者 epilogue：把 |v| 折进 buf[t]（与 Pass A 的 jx_uabs 完全同一套位运算） */
static inline void jx_amax_put(unsigned *buf, int t, float v)
{
  unsigned u;
  __builtin_memcpy(&u, &v, sizeof(u));
  u &= 0x7fffffffu;
  if (u > buf[t]) { buf[t] = u; }
}

#endif /* JX_AMAX_H */
