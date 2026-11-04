#ifndef JX_PERCORE_H
#define JX_PERCORE_H

#include <stdlib.h>

/* ============================================================================
 * jx_percore.h — 分核（per-core）静态状态
 *
 * 单核构建（CONFIG_SMP=n / SMP_NCPUS=1）时所有数组退化成 1 份，行为与原来
 * 完全一致；双核构建时按 up_cpu_index() 取各自的一份，编码核与解码核互不
 * 干扰，从而不需要加锁就能并行。
 *
 * 同一份源码要同时喂两个构建：
 *   - 板端 NuttX 构建：有 <nuttx/arch.h>，能拿到 CONFIG_SMP
 *   - 主机 gcc 对照构建：没有 NuttX 头，必须退化成单核
 * 注意必须探 <nuttx/arch.h>：主机环境里存在只含 config.h 的 nuttx 桩目录，
 * 用 <nuttx/config.h> 探测会误判（踩过）。
 * ========================================================================== */

#if defined(__has_include)
#  if __has_include(<nuttx/arch.h>)
#    include <nuttx/config.h>
#    include <nuttx/arch.h>
#    define JX_HAVE_NUTTX 1
#  endif
#endif

#if defined(JX_HAVE_NUTTX) && defined(CONFIG_SMP) && CONFIG_SMP_NCPUS > 1
#  define JX_NCORE   CONFIG_SMP_NCPUS
/* r164（2026-09-17 双核不确定性诊断）：JX_COREF=<n> 把 jx_core_id() 强制成 n。
 * 用来回答"双核与单核结果不同，是分核静态缓冲（per-core）在 CPU1 上没被正确
 * 初始化，还是切分本身有问题"。-2 = 还没读环境变量，-1 = 不覆盖。 */
extern int jx_core_force;
static inline int jx_core_id(void)
{
  if (jx_core_force == -2)
    {
      const char *e = getenv("JX_COREF");
      jx_core_force = (e != NULL) ? atoi(e) : -1;
      if (jx_core_force < -1) { jx_core_force = -1; }
    }
  if (jx_core_force >= 0) { return jx_core_force; }
  return up_cpu_index();
}
#else
#  define JX_NCORE   1
extern int jx_core_force;
static inline int jx_core_id(void) { return 0; }
#endif

#endif /* JX_PERCORE_H */
