/* ============================================================================
 * nn_ops.c — implementations of l3ac/layers.py primitives.
 * See nn_ops.h for the Python-aligned documentation.
 * weights_data.h (the exported weight table) is included here exactly once so
 * the tensor/weight lookups have a single home, matching the original layout.
 * ========================================================================== */
#include <syslog.h>

/* 本 defconfig 的 syslog 级别宏不在 <syslog.h> 里，直接用数值级别 3(ERR)。
 * 用 syslog() 而不是 printf()：这里可能两个线程同时进来，printf() 互相冲毁会丢现场。 */
#define JX_LOG_ERR 3
#include "nn_ops.h"
#include "weights_data.h"
#include "jxprof.h"
#include "jx_percore.h"
#include "jx_dcp.h"        /* DCache 硬件预载 */
#include "jx_amax.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include <unistd.h>

/* ==================== r56：双核并行 for ====================
 * 背景（_REALTIME_ROUND34 路线 1）：FPU 与 PIE 都是 **per-core** 资源，而所有
 * 热循环都挤在单核上（epilogue 实测 ~48-57 周期/元素，IPC 只有 0.4~0.5）。
 * 之前那次"双核 0.94x"是"编码线程 ∥ 解码线程"——两条完整流水线互相抢 PSRAM，
 * 与"同一个算子按行切分"是完全不同的场景，不能作为否决依据。
 *
 * 设计：
 *   - 一个常驻 worker 线程，钉在 CPU1；调用方钉在 CPU0。这样 jx_core_id()
 *     在两侧都是确定的，per-core 的 scratch / acc 缓冲不会互相踩。
 *   - 每颗核只跑**自己那一半行**，行与行完全独立 -> 结果逐位相同。
 *   - 只切一次（二分），同步成本 = 2 次信号量往返。
 *   - 任何一步失败（线程建不起来 / 栈不够 / JX_PF=0）都退回串行，绝不改变结果。
 * ============================================================ */
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <unistd.h>

typedef struct {
  void (*fn)(void *arg, int lo, int hi);
  void  *arg;
  int    lo, hi;
} jx_pfjob_t;

/* r164：jx_core_id() 的强制覆盖（诊断用，见 include/jx_percore.h）。
 * -2 = 还没读环境变量，-1 = 不覆盖，>=0 = 恒返回该核号。 */
int jx_core_force = -2;

/* ---- r91：并行框架核算（诊断，JX_NO_PROFILE 下编译成空）-----------------
 * 目的：把"两颗核各自在并行区里干了多少、等了多久"讲清楚。
 *   w0 = CPU0 在并行区里干活的时间   wt = CPU0 等 worker 的时间（空转）
 *   w1 = CPU1（worker）干活的时间
 * 同一张表按 worker 函数指针索引（最多 24 项），打印十六进制地址，
 * 用 xtensa-esp32s3-elf-nm 反查函数名。
 * 只做统计，不加锁（诊断用途，允许少量丢失）。 */
typedef struct {
  void (*fn)(void *, int, int);
  unsigned long long w0, wt, w1;
  unsigned long n;
} jx_pfst_t;
static jx_pfst_t g_pfst[24];
static int g_pfst_n = 0;
unsigned long long jx_pfw_work = 0, jx_pfw_wait = 0, jx_pfs_work = 0;
unsigned long long jx_pfw1_work = 0;
unsigned long jx_pfw_n = 0, jx_pfs_n = 0;

#ifdef JX_NO_PROFILE
#  define jx_pfcc() 0u
#else
static inline unsigned jx_pfcc(void)
{
  unsigned c;
  __asm__ __volatile__("rsr %0, ccount" : "=r"(c));
  return c;
}
#endif

static jx_pfst_t *jx_pfst_get(void (*fn)(void *, int, int))
{
  int i;
  for (i = 0; i < g_pfst_n; i++) { if (g_pfst[i].fn == fn) { return &g_pfst[i]; } }
  if (g_pfst_n >= 24) { return NULL; }
  i = g_pfst_n; g_pfst_n = i + 1;
  g_pfst[i].fn = fn; g_pfst[i].w0 = g_pfst[i].wt = g_pfst[i].w1 = 0; g_pfst[i].n = 0;
  return &g_pfst[i];
}

void jx_pfstat_dump(void);

static jx_pfjob_t g_pf_job;
static sem_t      g_pf_go, g_pf_done;
static sem_t      g_pf_start;      /* worker 完成核绑定后的回报 */
static volatile int g_pf_wcpu = -1;  /* worker 实际落到的核号 */
static pthread_t  g_pf_th;
static int        g_pf_state = 0;   /* 0=未试  1=就绪  -1=不可用 */
unsigned long     jx_pf_stackaddr = 0;

/* r57g：核绑定。sched_setaffinity 只改亲和掩码，真正的迁移发生在下一个
 * 调度点；实测立刻读 up_cpu_index() 还是旧核号，所以必须让出后再校验。
 * 返回最终落到的核号（-1 表示绑定调用失败）。 */
static int g_pf_pinret = 0;
static int jx_pf_pin(int cpu)
{
#if defined(CONFIG_SMP)
  cpu_set_t s;
  CPU_ZERO(&s); CPU_SET(cpu, &s);
  g_pf_pinret = sched_setaffinity(0, sizeof(s), &s);
#endif
  for (int i = 0; i < 500; i++)
    {
      if (jx_core_id() == cpu) { break; }
      usleep(1000);
    }
  return jx_core_id();
}

static int jx_pf_worker_core(void)
{
  const char *e = getenv("JX_PFCORE");
  int v = (e != NULL) ? atoi(e) : 0;
  return (v == 0) ? 0 : 1;
}

static void *jx_pf_worker(void *p)
{
  (void)p;
  /* r76：探针 —— worker 自己的栈落在哪。内部 DRAM=0x3FC8xxxx~0x3FCFxxxx，
   * PSRAM=0x3Cxxxxxx~0x3Exxxxxx。只打印一次。 */
  {
    volatile char loc = 0;
    jx_pf_stackaddr = (unsigned long)(uintptr_t)&loc;
  }
  g_pf_wcpu = jx_pf_pin(jx_pf_worker_core());
  sem_post(&g_pf_start);
  for (;;)
    {
      jx_pfst_t *st;
      unsigned long _t0;
      sem_wait(&g_pf_go);
      jx_pie_enable();          /* CPENABLE 随任务保存/恢复，worker 每次都得补 */
      st = jx_pfst_get(g_pf_job.fn);
      _t0 = jx_pfcc();
      g_pf_job.fn(g_pf_job.arg, g_pf_job.lo, g_pf_job.hi);
      {
        unsigned long _d = jx_pfcc() - _t0;
        jx_pfw1_work += (unsigned long long)_d;
        if (st != NULL) { st->w1 += (unsigned long long)_d; }
      }
      __sync_synchronize();
      sem_post(&g_pf_done);
    }
  return NULL;
}

int jx_pf_on(void)
{
  static int v = -1;
  if (v < 0) { const char *e = getenv("JX_PF"); v = !(e != NULL && e[0] == (char)48); }
  return v;
}

static int jx_pf_minrows(void)
{
  static int v = -1;
  if (v < 0) { const char *e = getenv("JX_PFMIN"); v = (e != NULL) ? atoi(e) : 64; if (v < 1) { v = 1; } }
  return v;
}

/* r138：并行阈值全局覆盖。
 * 板端 [pf] 表显示 serial-fallback 345 ms / n=64 —— 约 9.6% 的时间只跑在
 * 单核上，仅仅因为那些调用点的 n 小于各自的 minn（64/256 等）。这个 knobs
 * 把所有调用点的阈值 min() 到一个统一上限，用来量化"把这些小块也并行掉"
 * 到底值多少。0/未设 = 保持原行为（默认）。 */
static int jx_pf_mincap(void)
{
  static int v = -2;
  if (v == -2) { const char *e = getenv("JX_PFMIN2"); v = (e != NULL) ? atoi(e) : 0; if (v < 0) { v = 0; } }
  return v;
}

static int jx_pf_ready(void)
{
#if JX_NCORE < 2
  /* 主机回归构建：jx_core_id() 恒为 0，两线程会共用同一份 per-core
   * scratch/acc，必须串行；只有板上才允许并行。 */
  return 0;
#else
  if (!jx_pf_on()) { return 0; }
  if (g_pf_state != 0)
    {
      /* The worker is a pthread owned by the current NSH process.  A later
       * `jixun ...` invocation has the same app globals but the old worker is
       * gone, so rebuild the worker/semaphores instead of waiting forever. */
      if (g_pf_state == 1 && g_pf_wcpu != (int)getpid())
        {
          (void)sem_destroy(&g_pf_go);
          (void)sem_destroy(&g_pf_done);
          (void)sem_destroy(&g_pf_start);
          g_pf_state = 0;
        }
      else
        {
          return (g_pf_state == 1);
        }
    }
  g_pf_state = -1;
  if (sem_init(&g_pf_go, 0, 0) != 0) { return 0; }
  if (sem_init(&g_pf_done, 0, 0) != 0) { return 0; }
  if (sem_init(&g_pf_start, 0, 0) != 0) { return 0; }
  {
    pthread_attr_t at;
    if (pthread_attr_init(&at) != 0) { return 0; }
    /* r76：栈尺寸可调。内部 DRAM 未用空间很小，96KB 栈必然落到 PSRAM；
     * 小的未必。JX_PFSZ 用来 A/B（< 4096 视为无效）。 */
    {
      const char *e = getenv("JX_PFSZ");
      int sz = (e != NULL) ? atoi(e) : 0;
      if (sz < 4096) { sz = 98304; }
      (void)pthread_attr_setstacksize(&at, (size_t)sz);
    }
    if (pthread_create(&g_pf_th, &at, jx_pf_worker, NULL) != 0)
      { (void)pthread_attr_destroy(&at); printf("  [pf] worker 线程创建失败 -> 单核\n"); return 0; }
    (void)pthread_attr_destroy(&at);
  }
  { int wcore = jx_pf_worker_core();
    int ccore = 1 - wcore;
    int c0 = jx_pf_pin(ccore);
    sem_wait(&g_pf_start);          /* 等 worker 报它真正落到的核号 */
    if (g_pf_wcpu != wcore || c0 != ccore)
      {
        printf("  [pf] worker=%d caller=%d pinret=%d -> 单核\n",
               (int)g_pf_wcpu, c0, g_pf_pinret);
        return 0;
      }
  }
  g_pf_state = 1;
  printf("  [pf] 双核并行已启用 (worker=CPU%d / caller=CPU%d)\n",
         (int)g_pf_wcpu, 1 - (int)g_pf_wcpu);
  /* Reuse this otherwise-debug field to tag the owning process. */
  g_pf_wcpu = (int)getpid();
  { volatile char loc = 0;
    printf("  [pf] 栈地址 caller=0x%08lx worker=0x%08lx\n",
           (unsigned long)(uintptr_t)&loc, jx_pf_stackaddr); }
  return 1;
#endif /* JX_NCORE < 2 */
}

/* 把 [0,n) 二分：上半给 worker，下半本核跑。返回 1 表示真的并行了。 */
/* r67: 嵌套保护。并行区内再调 jx_pf_run_min 会覆盖同一个 g_pf_job
 * 并和已唤醒的 worker 抢信号量，必须退化成串行。调用方只在 CPU0 进入。 */
static volatile int g_pf_busy = 0;

/* r165（诊断）：给每个并行点（按 worker 函数指针）编号。
 *   JX_PFLOG=1  打印每个点首次被使用时的编号 / n / mid
 *   JX_PFSITE=k 只让第 k 号点真正切分，其余全部串行（默认 -1 = 全部按原样）
 *   JX_PFNOSYNC=1 两半都放在调用核上顺序跑（切分语义探针）
 *   JX_PFNOSYNC=2 只做 jx_pf_ready() 初始化，然后不切分整体串行跑
 *                （用来把"建 worker / 钉核"与"切分"的影响分开） */
#define JX_PFSITE_MAX 48
static void (*g_pfsite_fn[JX_PFSITE_MAX])(void *, int, int);
static int g_pfsite_n = 0;

static int jx_pf_envint(const char *nm, int dflt)
{
  const char *e = getenv(nm);
  return (e != NULL) ? atoi(e) : dflt;
}

/* Diagnostic bitmask: force the selected parallel sites back to serial. */
static int jx_pf_skipmask(void)
{
  static int v = -2;

  if (v == -2)
    {
      const char *e = getenv("JX_PFNO");
      v = (e != NULL) ? (int)strtol(e, NULL, 0) : 0;
    }

  return v;
}

static int jx_pf_site(void (*fn)(void *, int, int), int n, int mid)
{
  int i;
  for (i = 0; i < g_pfsite_n; i++) { if (g_pfsite_fn[i] == fn) { return i; } }
  if (g_pfsite_n >= JX_PFSITE_MAX) { return -1; }
  i = g_pfsite_n; g_pfsite_n = i + 1; g_pfsite_fn[i] = fn;
  if (jx_pf_envint("JX_PFLOG", 0) != 0)
    { printf("[PF] site %2d fn=%p n=%d mid=%d\n", i, (void *)fn, n, mid); }
  return i;
}

int jx_pf_run_min(int n, void (*fn)(void *, int, int), void *arg, int minn)
{
  int mid;
  if (n <= 0 || fn == NULL) { return 0; }
  { int _cap = jx_pf_mincap(); if (_cap > 0 && minn > _cap) { minn = _cap; } }
  if (g_pf_busy || n < minn || !jx_pf_ready() ||
      __sync_lock_test_and_set(&g_pf_busy, 1) != 0)
    {
      unsigned long _t0 = jx_pfcc();
      fn(arg, 0, n);
      jx_pfs_work += (unsigned long long)(jx_pfcc() - _t0);
      jx_pfs_n++;
      return 0;
    }
  mid = (n + 1) / 2;
  {
    int _ns = jx_pf_envint("JX_PFNOSYNC", 0);
    if (_ns == 2)
      { (void)jx_pf_site(fn, n, mid); fn(arg, 0, n); return 0; }
    {
      int _sel = jx_pf_envint("JX_PFSITE", -1);
      int _s = jx_pf_site(fn, n, mid);
      if ((_sel >= 0 && _s != _sel) ||
          (_s >= 0 && (_s < 31) && (jx_pf_skipmask() & (1 << _s))))
        { fn(arg, 0, n); jx_pfs_n++; return 0; }
    }
    if (_ns == 1)
      {
        fn(arg, 0, mid);
        fn(arg, mid, n);
        return 1;
      }
  }
  g_pf_job.fn = fn; g_pf_job.arg = arg; g_pf_job.lo = mid; g_pf_job.hi = n;
  __sync_synchronize();
  sem_post(&g_pf_go);
  {
    jx_pfst_t *st = jx_pfst_get(fn);
    unsigned long _t0 = jx_pfcc(), _t1, _t2, _dw, _dwq;
    fn(arg, 0, mid);
    _t1 = jx_pfcc();
    sem_wait(&g_pf_done);
    _t2 = jx_pfcc();
    _dw  = _t1 - _t0;
    _dwq = _t2 - _t1;
    jx_pfw_work += (unsigned long long)_dw;
    jx_pfw_wait += (unsigned long long)_dwq;
    jx_pfw_n++;
    if (st != NULL)
      { st->w0 += (unsigned long long)_dw; st->wt += (unsigned long long)_dwq; st->n++; }
  }
  __sync_lock_release(&g_pf_busy);
  return 1;
}

/* 默认阈值版本：行/块数少于 JX_PFMIN 就不值得同步，直接串行。 */
int jx_pf_run(int n, void (*fn)(void *, int, int), void *arg)
{
  return jx_pf_run_min(n, fn, arg, jx_pf_minrows());
}


/* ==================== 快速超越函数的系数表（第三十八轮）====================
 * 见 include/jx_fastmath.h 顶部注释：热循环里 K[0..3] 被 GCC 折叠成
 * flash literal pool 的 l32r + wfr，占掉 ~108 周期/元素。表内容改为运行期
 * 填充（本函数 noinline），GCC 只能按 lsi 从 SRAM 装进 f 寄存器。
 * 这里持有唯一的一份，全部 TU 共用（头文件里是 extern 声明）。 */
float jx_kt[64];

/* r112：DCache 预载计数器（诊断用，开销一次加法） */
volatile unsigned g_dcp_kick = 0, g_dcp_skip = 0, g_dcp_pend = 0;
float jx_glut[JX_GLUT_N];
float jx_ssq[JX_SSQ_N];   /* r96: sin^2 周期表（见 jx_fastmath.h） */
unsigned long long jx_sinsq_slow_n = 0ULL;
volatile int jx_kt_ready = 0;
__attribute__((noinline)) void jx_kt_fill(void)
{
  /* r96：sin^2 周期表。用 libm sinf 精确填充 N 项（一次性，~几十微秒）。
   * 表值 = sin^2(pi*i/N)，与 jx_sinsq_lut 的 idx 定义严格对应。 */
  { const float _step = JX_PI_F / (float)JX_SSQ_N;
    for (int i = 0; i < JX_SSQ_N; i++)
      { float _a = _step * (float)i; float _sv = sinf(_a); jx_ssq[i] = _sv * _sv; } }
  /* [0..3] sin² 快路径（= jx_sq_k[0..3]，一字不差） */
  jx_kt[0] = jx_sq_k[0];
  jx_kt[1] = jx_sq_k[1];
  jx_kt[2] = jx_sq_k[2];
  jx_kt[3] = jx_sq_k[3];
  /* [4..12] erf/gelu 的 9 个系数（与原来逐字符相同的次序） */
  jx_kt[4]  =  3.724282582e-04f;
  jx_kt[5]  = -6.735875499e-03f;
  jx_kt[6]  =  4.943256446e-02f;
  jx_kt[7]  = -1.832618300e-01f;
  jx_kt[8]  =  3.294201400e-01f;
  jx_kt[9]  = -1.361734042e-01f;
  jx_kt[10] = -3.334753242e-01f;
  jx_kt[11] = -5.409212864e-03f;
  jx_kt[12] =  1.128535984e+00f;
  /* [16..23] tanh 的 P/Q 系数 */
  jx_kt[16] = -5.858162427e-09f;
  jx_kt[17] =  6.994668850e-06f;
  jx_kt[18] =  2.673332299e-03f;
  jx_kt[19] =  1.268952674e-01f;
  jx_kt[20] =  9.999998926e-01f;
  jx_kt[21] =  1.938164396e-04f;
  jx_kt[22] =  2.275157832e-02f;
  jx_kt[23] =  4.602268718e-01f;
  /* [24..34] sin² 慢路径 11 阶系数（= jx_sq_k[5..15]，一字不差）。
   * 第三十九轮：原来这 11 个在 jx_sq_k 里，被 GCC 判定为"从未被写过的
   * static 数组"而常量折叠进 flash literal pool，每次用到都要 l32r+wfr。 */
  for (int i = 0; i < 11; i++) { jx_kt[24 + i] = jx_sq_k[5 + i]; }
  /* [40..44] 热循环常用常量 */
  jx_kt[JX_KT_HALF]   = 0.5f;
  jx_kt[JX_KT_ONE]    = 1.0f;
  jx_kt[JX_KT_R127]   = 127.0f;
  jx_kt[JX_KT_ISQ2]   = 0.70710678f;
  jx_kt[JX_KT_HALFPI] = JX_HALF_PI_F;
  jx_kt[JX_KT_INVPI]  = JX_INV_PI_F;
  jx_kt[JX_KT_PI]     = JX_PI_F;
  jx_kt[JX_KT_ZERO]   = 0.0f;
  /* r45b: gelu 查表用的常数（热循环里一次装载，避免 literal pool） */
  jx_kt[48] = 512.0f;   /* 8.0f * 64.0f */
  jx_kt[49] = 64.0f;    /* 1 / 步长 */
  jx_kt[50] = 8.0f;
  jx_kt[51] = -8.0f;
  /* 表内容用同一个 jx_gelu9 算，保证与多项式实现同源；
   * jx_kt[4..12] 已在上方填好，此处必须用 jx_gelu9（不能用 jx_gelu_fast）。 */
  { const jx_erf9 E0 = jx_erf9_load();
    for (int gi = 0; gi < JX_GLUT_N; gi++)
      { float gx = (float)gi * (1.0f / 64.0f) - 8.0f;
        jx_glut[gi] = jx_gelu9(gx, E0); }
    /* r100：gelu9(-8) = 0.5f*(-8)*(1.0f+(-1.0f)) 算出来是 -0.0f。旧实现在
     * x<=lo 时走提前返回返回 +0.0f、在 x∈(lo,..) 上因 b-a=+0.0f 也得 +0.0f，
     * 所以这里改成 +0.0f 对旧路径逐位无影响，而新的无分支实现要靠它把
     * 下饱和区也精确地给成 +0.0f（见 jx_gelu_fast 注释）。 */
    jx_glut[0] = 0.0f; }
  jx_kt_ready = 1;
}

/* ============================ 算子级计时探针 ============================ */
double   jx_prof_ms[JP_N];
unsigned jx_prof_cnt[JP_N];
int      jx_prof_off = 0;
double   jx_elems_snake = 0.0;
double   jx_elems_sinsq = 0.0;
double   jx_alloc_mb    = 0.0;
/* 2026-09-13 第十二轮：并行跑时的堆占用诊断（t_alloc/t_free 维护） */
double   jx_live_mb     = 0.0;
double   jx_peak_mb     = 0.0;
/* conv1d 因 realloc 失败而被跳过的次数。>0 说明输出是错的：
 * 十分钟的语音里少一次卷积听不出来，必须明报。 */
int      jx_conv_skip   = 0;
/* 主机端分布探测（JX_DIST=1）：统计 snake 的 |a*v| 分布，为快速路径定阈值。
 * __XTENSA__ 下整体编译为空，板端零开销。 */
#ifndef __XTENSA__
#include <math.h>
static int  jx_dist_on = -1;
static unsigned long long jx_dist_n, jx_dist_b1, jx_dist_b2, jx_dist_b3, jx_dist_b4;
static float jx_dist_max;
static void jx_dist_uv(float uv) {
    float a = fabsf(uv);
    jx_dist_n++;
    if (a > jx_dist_max) { jx_dist_max = a; }
    if (a <= 1.5707963f) { jx_dist_b1++; }
    else if (a <= 3.1415927f) { jx_dist_b2++; }
    else if (a <= 6.2831855f) { jx_dist_b3++; }
    else if (a <= 12.566371f) { jx_dist_b4++; }
}
static void jx_dist_report(void) {
    if (jx_dist_n) {
        unsigned long long hi = jx_dist_n - jx_dist_b1 - jx_dist_b2 - jx_dist_b3 - jx_dist_b4;
        fprintf(stderr,
          "[DIST] snake |a*v| n=%llu max=%.3f | <=pi/2 %.4f (pi/2,pi] %.4f (pi,2pi] %.4f (2pi,4pi] %.4f >4pi %.4f\n",
          jx_dist_n, (double)jx_dist_max,
          (double)jx_dist_b1 / jx_dist_n, (double)jx_dist_b2 / jx_dist_n,
          (double)jx_dist_b3 / jx_dist_n, (double)jx_dist_b4 / jx_dist_n,
          (double)hi / jx_dist_n);
    }
}
static void jx_dist_arm(void) {
    if (jx_dist_on < 0) {
        jx_dist_on = getenv("JX_DIST") ? 1 : 0;
        if (jx_dist_on) { atexit(jx_dist_report); }
    }
}
#define JX_DIST_ARM()      jx_dist_arm()
#define JX_DIST_UV(uv)     do { if (jx_dist_on) { jx_dist_uv(uv); } } while (0)
#else
#define JX_DIST_ARM()      ((void)0)
#define JX_DIST_UV(uv)     ((void)0)
#endif
   /* snake1d 累计处理元素数 */

static const char *const _jx_prof_label[JP_N] = {
    "TOTAL       (encode+decode 全程)",
    "encoder_forward          编码主干",
    "en_encoder_forward       编码注意力",
    "first_block              编码首块",
    "conv_unit                卷积单元",
    "enhance_block            增强块",
    "legacy_unit              解码尾块",
    "local_transformer        局部注意力",
    "en_decoder_forward       解码注意力",
    "decoder_forward          解码主干",
    "quantizer                FSQ 量化/反量化",
    "conv1d / conv1d_d        全部卷积",
    "linear                   全部全连接",
    "snake1d                  Snake 激活(sinf)",
    "gelu / 其它超越函数循环",
    "trend_pool               趋势池化",
    "upsampling_linear        线性上采样",
    "channel_norm/grn/instnorm",
    "permute / 搬运循环",
    "  FB.trendpool+branch",
    "  FB.conv_1 (20->80)",
    "  FB.conv_2 (81->16)",
    "  FB.cat 81ch 拼接",
    "  CU.dw_conv+in-permute",
    "  CU.norm+l1+snake+grn",
    "  CU.l2+out-permute",
    "  I8.quantize prologue",
    "  I8.matmul+epilogue",
    "  I8.setup",
    "  k1.im2col量化prologue",
    "  k1.dot点积批",
    "  MISC 分块拷贝/累加",
};

/* ==================== 按卷积形状的耗时直方图 ====================
 * 目的：conv1d 是最大的单项（~3.4 s），但它是 112 次不同形状的调用混在一起。
 * 只有按 (Ci,Co,k,g,dil) 分开统计 + 记录走的是哪条路径，才能知道该优化谁。
 * 表很小（64 项）、只在 conv 调用前后各取一次时钟。 */
#define JX_CT_N 96
typedef struct {
    int    Ci, Co, k, g, dil, Tin, Tout;
    long   n;
double ms, mac, inb, outb;   /* inb/outb = 累计访存字节（下界：输入一遍+输出一遍）*/
    int    path;       /* 最近一次路径 */
    long   npath[4];   /* 各路径的次数 */
} jx_ct_ent;
static jx_ct_ent g_ct[JX_CT_N];
static int       g_ct_used;
int jx_conv_path = -1;

void jx_ct_reset(void) { g_ct_used = 0; memset(g_ct, 0, sizeof(g_ct)); }

void jx_ct_add(int Ci, int Co, int k, int g, int dil, double ms, double mac, int Tin, int Tout) {
    static long _tr_n = 0;
    int i;
    int p = jx_conv_path;
    if (getenv("JX_CTTRACE") != NULL)
        printf("[CT] #%-3ld Ci=%-4d Co=%-4d k=%d g=%-3d dil=%-2d p=%d %7.2f ms %7.3f MMAC %6.2f cyc/MAC\n",
               _tr_n, Ci, Co, k, g, dil, p, ms, mac / 1e6, mac > 0.0 ? ms * 240000.0 / mac : 0.0);
    _tr_n++;
    for (i = 0; i < g_ct_used; i++) {
        if (g_ct[i].Ci == Ci && g_ct[i].Co == Co && g_ct[i].k == k &&
            g_ct[i].g == g && g_ct[i].dil == dil) break;
    }
    if (i == g_ct_used) {
        if (g_ct_used >= JX_CT_N) return;
        g_ct[i].Ci = Ci; g_ct[i].Co = Co; g_ct[i].k = k;
        g_ct[i].g = g; g_ct[i].dil = dil;
        g_ct_used++;
    }
g_ct[i].n++; g_ct[i].Tin = Tin; g_ct[i].Tout = Tout;
    { double _d = (double)Co * (double)(Ci / g) * (double)k * (double)Tout;
      double _B = (_d > 0.0) ? mac / _d : 1.0; if (_B < 0.5) { _B = 1.0; }
      g_ct[i].inb  += _B * (double)Ci * (double)Tin  * 4.0;
      g_ct[i].outb += _B * (double)Co * (double)Tout * 4.0; }
    g_ct[i].ms += ms; g_ct[i].mac += mac;
    g_ct[i].path = p;
    if (p >= 0 && p < 4) g_ct[i].npath[p]++;
}

void jx_ct_dump(void) {
    int order[JX_CT_N];
    int i, j;
    if (g_ct_used == 0) return;
    for (i = 0; i < g_ct_used; i++) order[i] = i;
    for (i = 1; i < g_ct_used; i++) {
        int v = order[i];
        for (j = i - 1; j >= 0 && g_ct[order[j]].ms < g_ct[v].ms; j--) order[j + 1] = order[j];
        order[j + 1] = v;
    }
    printf("  ---- conv1d 按形状（按耗时降序，最多 14 项） ----\n");
    for (i = 0; i < g_ct_used && i < 40; i++) {
        const jx_ct_ent *e = &g_ct[order[i]];
        printf("  Ci=%-3d Co=%-3d k=%d g=%-3d dil=%-2d  n=%-3ld %8.1f ms  %7.2f MMAC  %6.2f cyc/MAC  路径 fp32=%ld dense=%ld preq=%ld pert=%ld\n",
               e->Ci, e->Co, e->k, e->g, e->dil, e->n, e->ms, e->mac / 1e6,
               e->mac > 0.0 ? e->ms * 240000.0 / e->mac : 0.0,
               e->npath[0], e->npath[1], e->npath[2], e->npath[3]);
    }
    /* r52a：访存表（输入一遍 + 输出一遍的下界，按 in+out 降序）——判断"时间是否正比于字节数" */
    {
      int o2[JX_CT_N];
      int k2, j2;
      double tb = 0.0;
      for (k2 = 0; k2 < g_ct_used; k2++) { o2[k2] = k2; tb += g_ct[k2].inb + g_ct[k2].outb; }
      for (k2 = 1; k2 < g_ct_used; k2++) {
        int v = o2[k2];
        for (j2 = k2 - 1; j2 >= 0 && (g_ct[o2[j2]].inb + g_ct[o2[j2]].outb) < (g_ct[v].inb + g_ct[v].outb); j2--) o2[j2 + 1] = o2[j2];
        o2[j2 + 1] = v;
      }
      printf("  ---- \u8bbf\u5b58\u4e0b\u754c\uff08in+out\uff0cMB\uff09\u603b=%.1f MB  \u6309\u5b57\u8282\u964d\u5e8f ----\n", tb / 1e6);
      for (k2 = 0; k2 < g_ct_used && k2 < 24; k2++) {
        const jx_ct_ent *e = &g_ct[o2[k2]];
        double bb = e->inb + e->outb;
        printf("  Ci=%-3d Co=%-3d k=%d dil=%-2d T=%-6d n=%-3ld in=%7.2f out=%7.2f MB  %8.1f ms  eff=%6.1f MB/s  %6.2f ms/MB\n",
               e->Ci, e->Co, e->k, e->dil, e->Tout, e->n, e->inb / 1e6, e->outb / 1e6, e->ms,
               bb > 0.0 ? bb / 1e6 / (e->ms / 1e3) : 0.0, bb > 0.0 ? e->ms / (bb / 1e6) : 0.0);
      }
    }
}

void jx_prof_reset(void) {
    const char *e = getenv("JX_PROF");
    jx_prof_off = (e && e[0] == '0') ? 1 : 0;
    for (int i = 0; i < JP_N; i++) { jx_prof_ms[i] = 0.0; jx_prof_cnt[i] = 0; }
    jx_elems_snake = 0.0;
    jx_elems_sinsq = 0.0;
    jx_sinsq_slow_n = 0ULL;
    jx_alloc_mb = 0.0;
    jx_ct_reset();
    jx_pfw_work = jx_pfw_wait = jx_pfs_work = jx_pfw1_work = 0;
    jx_pfw_n = jx_pfs_n = 0;
    if (!jx_prof_off) { /* 诊断表按 run 重来 */ }
    g_pfst_n = 0;
    jx_mac_conv = jx_mac_linear = 0.0;
    jx_n_conv = jx_n_linear = 0;
    jx_conv_skip = 0;
#ifndef JX_NO_PROFILE
    /* 2026-09-15 第三十六轮：这些 cycle 计数器此前**从来没有被清零**，同一个上电
     * 周期里连跑两次 bench 就会把两次结果累加在一起——早期报告里的 "cyc/元素"
     * 读数可能就是这么来的。这里统一按"每次 run 重新计"处理，让读数自解释。
     * （烧录后只跑一次 bench 的场合，与历史读数一致。） */
    {
      extern unsigned long jx_cyc_mm, jx_cyc_ep, jx_cyc_q, jx_dcol_calls;
      extern double jx_epi_elems, jx_q_elems;
      extern double jx_fast_elems, jx_slow_elems;
      extern unsigned long jx_slow_calls, jx_fast_calls;
      extern unsigned long jx_dt_cyc[4];
      extern unsigned long jx_dt_pos, jx_dt_blk, jx_dt_calls;
      extern unsigned long jx_k1q_cyc, jx_k1n_cyc, jx_k1q_pos, jx_k1n_pos;
      extern unsigned long jx_cf_cyc[4], jx_cf_rows;
      extern double jx_cf_elems, jx_cf_in_elems;
      extern unsigned long jx_k1s_cyc[4];
      extern double jx_k1s_qe, jx_k1s_oe;

      jx_cyc_mm = jx_cyc_ep = jx_cyc_q = 0;
      jx_dcol_calls = 0;
      jx_epi_elems = jx_q_elems = 0.0; jx_fast_elems = jx_slow_elems = 0.0; jx_fast_calls = jx_slow_calls = 0;
      jx_dt_cyc[0] = jx_dt_cyc[1] = jx_dt_cyc[2] = jx_dt_cyc[3] = 0;
      jx_dt_pos = jx_dt_blk = jx_dt_calls = 0;
      jx_k1q_cyc = jx_k1n_cyc = jx_k1q_pos = jx_k1n_pos = 0;
      jx_cf_cyc[0] = jx_cf_cyc[1] = jx_cf_cyc[2] = jx_cf_cyc[3] = 0;
      jx_cf_rows = 0;
      jx_cf_elems = jx_cf_in_elems = 0.0;
      jx_k1s_cyc[0] = jx_k1s_cyc[1] = jx_k1s_cyc[2] = jx_k1s_cyc[3] = 0;
      jx_k1s_qe = jx_k1s_oe = 0.0;
      { extern int jx_kc_n; unsigned long jx_kc_calls; jx_kc_n = 0; jx_kc_calls = 0; }
    }
#endif
  }

void jx_pfstat_dump(void)
{
    const char *e = getenv("JX_PFSTAT");
    printf("  [pf] CPU0 parallel-work %.1f ms / wait %.1f ms (n=%lu) | serial-fallback %.1f ms (n=%lu) | CPU1 work %.1f ms\n",
           (double)jx_pfw_work / 240000.0, (double)jx_pfw_wait / 240000.0, jx_pfw_n,
           (double)jx_pfs_work / 240000.0, jx_pfs_n,
           (double)jx_pfw1_work / 240000.0);
    if (e == NULL || e[0] == (char)48) { return; }
    printf("  [pf] fn        CPU0work  CPU0wait  CPU1work      n   (ms)\n");
    printf("  [dcp] 预载下发=%u 跳过(忙)=%u tile=%u cmask=%u bmask=%u\n",
           g_dcp_kick, g_dcp_skip, jx_dcp_tile(), jx_dcp_cmask(), jx_dcp_bmask());
    for (int i = 0; i < g_pfst_n; i++)
      {
        printf("  [pf] 0x%08lx %9.1f %9.1f %9.1f %6lu\n",
               (unsigned long)(uintptr_t)g_pfst[i].fn,
               (double)g_pfst[i].w0 / 240000.0, (double)g_pfst[i].wt / 240000.0,
               (double)g_pfst[i].w1 / 240000.0, g_pfst[i].n);
      }
}

/* ============================================================================
 * r151：实测算力频率 —— 回答"CPU 到底跑在多少 MHz"
 *
 * 只看编译期配置（CONFIG_ESP32S3_DEFAULT_CPU_FREQ_240）不够：DFS/PM 或
 * 早期启动路径都可能把实际频率换掉，而手册里所有"周期 -> 时间"的换算
 * （比如 [pf] 里的 /240000.0）都硬编码 240 MHz，频率错了整张耗时地图就错了。
 *
 * 做法：CCOUNT 是 CPU 自己的周期计数（CPU 只要在跑就涨，即使被抢占也一样），
 * CLOCK_MONOTONIC 来自系统定时器（与 CPU 时钟无关）。让 CPU 忙等一段已知的
 * 实时区间，比较两者即可得到真实频率。同时把时钟配置寄存器原样打出来，
 * 按 TRM 17.3 SYSTEM_CPU_PER_CONF_REG / 17.20 SYSTEM_SYSCLK_CONF_REG 解码。
 * ========================================================================== */
/* r152：纯硬件交叉验证 —— CCOUNT vs SYSTIMER 原值。
 * TRM 11.3：SYSTIMER 的 CNT_CLK = XTAL_CLK/2.5 = 16 MHz，与 CPU 同源于同一颗
 * 40 MHz 晶振（CPU = PLL480/2 = 12xXTAL/2 = 6xXTAL）。因此
 *      CPU_CLK : CNT_CLK  必须严格等于 240:16 = 15.000，
 * 与晶振的绝对精度无关。比值 = 15.000 就证明 CPU 分频链（PLL480/2）完全正确；
 * 比值明显小于 15 才说明 CPU 真的没跑到 240 MHz。
 * 只用 LO 32 位：16 MHz 下 2^32 计数约 268 s，够用。 */
static unsigned long jx_systimer_lo(void)
{
  volatile unsigned *op = (volatile unsigned *)(uintptr_t)0x60023004u; /* UNIT0_OP */
  volatile unsigned *lo = (volatile unsigned *)(uintptr_t)0x60023044u; /* VALUE_LO */
  unsigned v;
  /* 完全照抄 NuttX tickless_getcounter 的握手（触发后硬件自会置 VALID），
   * 不做任何"先清 VALID"的动作，避免与其他核上的取值握手互相打断而死等。 */
  *op = (1u << 30);                    /* TIMER_UNIT0_UPDATE（WT，自清） */
  while ((*op & (1u << 29)) == 0) { }  /* 等 TIMER_UNIT0_VALUE_VALID */
  (void)*lo;
  v = *lo;
  return v;
}

int jx_clk_probe_cmd(void)
{
  volatile unsigned *cpuper = (volatile unsigned *)(uintptr_t)0x600C0010u;
  volatile unsigned *sysclk = (volatile unsigned *)(uintptr_t)0x600C0060u;
  unsigned cp = *cpuper, sc = *sysclk;
  unsigned cpupor = cp & 3u, pllsel = (cp >> 2) & 1u;
  unsigned socsel = (sc >> 10) & 3u, prediv = sc & 1023u, xtalf = (sc >> 12) & 0x7fu;

  printf("[clk] SYSTEM_CPU_PER_CONF(0x600C0010) = 0x%08x\n", cp);
  printf("[clk]   CPUPERIOD_SEL=%u  PLL_FREQ_SEL=%u  WAITI_FORCE_ON=%u  WAITI_DELAY=%u\n",
         cpupor, pllsel, (cp >> 3) & 1u, (cp >> 4) & 15u);
  printf("[clk] SYSTEM_SYSCLK_CONF(0x600C0060) = 0x%08x\n", sc);
  printf("[clk]   SOC_CLK_SEL=%u  PRE_DIV_CNT=%u  CLK_XTAL_FREQ=%u\n",
         socsel, prediv, xtalf);
  if (socsel == 1u)
    {
      const unsigned pll = pllsel ? 480u : 320u;
      const unsigned div = pllsel ? (cpupor == 0u ? 6u : (cpupor == 1u ? 3u : 2u))
                                  : (cpupor == 0u ? 4u : 2u);
      printf("[clk]   => TRM 表 7.2-2: PLL=%u MHz / %u = %u MHz\n", pll, div, pll / div);
    }
  else
    {
      printf("[clk]   => CPU 源不是 PLL（SOC_CLK_SEL=%u），按 TRM 表 7.2-2 不是 240 MHz 档\n", socsel);
    }

  {
    unsigned c0 = 0, c1 = 0;
    double t0, t1;
    __asm__ __volatile__("rsr %0, ccount" : "=r"(c0));
    t0 = jx_prof_now();
    do { t1 = jx_prof_now(); } while ((t1 - t0) < 1000.0);
    __asm__ __volatile__("rsr %0, ccount" : "=r"(c1));
    {
      unsigned dc = (unsigned)(c1 - c0);
      double dm = t1 - t0;
      printf("[clk] measured: %u cycles in %.2f ms  =>  %.3f MHz\n",
             dc, dm, (double)dc / (dm * 1000.0));
      printf("[clk] 与 240 MHz 之比 = %.4f  (10 MHz 的整数倍: %.1f)\n",
             (double)dc / (dm * 1000.0) / 240.0, (double)dc / (dm * 1000.0) / 10.0);
    }
  }

  /* ---- 纯硬件对拉：CPU 周期 vs SYSTIMER 计数（不经过任何软件时钟） ---- */
  for (int k = 0; k < 2; k++)
    {
      const unsigned long target = (k == 0) ? 24000000UL : 480000000UL; /* ~0.1 s / ~2 s */
      unsigned long c0 = 0, c1 = 0, s0, s1;
      double w0, w1;

      __asm__ __volatile__("rsr %0, ccount" : "=r"(c0));
      s0 = jx_systimer_lo();
      w0 = jx_prof_now();
      do { __asm__ __volatile__("rsr %0, ccount" : "=r"(c1)); }
      while ((unsigned long)(c1 - c0) < target);
      s1 = jx_systimer_lo();
      w1 = jx_prof_now();

      {
        unsigned long dc = c1 - c0, ds = s1 - s0;
        double dt = w1 - w0;
        printf("[clk] #%d  dc=%lu  ds=%lu  CPU:SYSTIMER = %.5f  (理论 15.000 = 240/16)\n",
               k + 1, dc, ds, (ds && ds < 1000000000UL) ? (double)dc / (double)ds : 0.0);
        if (ds == 0) { printf("[clk]    !! SYSTIMER 计数未变化，本次测量无效\n"); }
        printf("[clk]    占空 %.2f ms -> CPU %.3f MHz | SYSTIMER 实测 %.4f MHz (NuttX 按 16 MHz 计时间)\n",
               dt, dt > 0.0 ? (double)dc / (dt * 1000.0) : 0.0,
               dt > 0.0 ? (double)ds / (dt * 1000.0) : 0.0);
      }
    }

  /* 跑完再看一眼时钟寄存器，确认负载下没有被改频 */
  {
    unsigned cp2 = *cpuper, sc2 = *sysclk;
    printf("[clk] 复读: CPU_PER_CONF=0x%08x (原 0x%08x)  SYSCLK_CONF=0x%08x (原 0x%08x) %s\n",
           cp2, cp, sc2, sc, (cp2 == cp && sc2 == sc) ? "一致" : "!! 有变化");
  }
  return 0;
}

void jx_prof_dump(void) {
    if (jx_prof_off) { printf("  [prof] 已关闭 (JX_PROF=0)\n"); return; }
    jx_pfstat_dump();
    printf("  ---- 算子级耗时 (ms) ----\n");
    for (int i = 0; i < JP_N; i++) {
        if (jx_prof_cnt[i] == 0 && jx_prof_ms[i] == 0.0) continue;
        printf("  %-38s %10.1f ms  (调用 %u 次)\n",
               _jx_prof_label[i], jx_prof_ms[i], jx_prof_cnt[i]);
    }
    printf("  [elems] snake=%.0f  ->  %.1f cyc/elem\n",
           jx_elems_snake, jx_elems_snake > 0.0 ? (jx_prof_ms[JP_SNAKE] * 240000.0 / jx_elems_snake) : 0.0);
    printf("  [elems] fused-sinsq=%.0f  (占 I8MM 桶 %.1f cyc/elem，仅趋势参考)\n",
           jx_elems_sinsq, jx_elems_sinsq > 0.0 ? (jx_prof_ms[JP_I8MM] * 240000.0 / jx_elems_sinsq) : 0.0);
    printf("  [elems] sinsq 慢路径绕计=%.0f  (snake 元素的回退比例 %.4f)\n",
           (double)jx_sinsq_slow_n,
           jx_elems_snake > 0.0 ? (double)jx_sinsq_slow_n / jx_elems_snake : 0.0);
#ifndef JX_NO_PROFILE
    {
      extern unsigned long jx_cyc_mm, jx_cyc_ep, jx_cyc_q, jx_dcol_calls;
      extern double jx_epi_elems, jx_q_elems;
      extern double jx_fast_elems, jx_slow_elems;
      extern unsigned long jx_slow_calls, jx_fast_calls;
      printf("  [i8mm] 中段: matmul %.1f Mcyc / epilogue %.1f Mcyc\n",
             (double)jx_cyc_mm / 1e6, (double)jx_cyc_ep / 1e6);
      printf("  [i8mm] 量化prologue %.1f Mcyc / %.0f 元素 = %.2f cyc/元素\n",
             (double)jx_cyc_q / 1e6, jx_q_elems,
             jx_q_elems > 0.0 ? (double)jx_cyc_q / jx_q_elems : 0.0);
      printf("  [i8mm] epilogue %.1f Mcyc / %.0f 元素 = %.2f cyc/元素 (行批 %lu)\n",
             (double)jx_cyc_ep / 1e6, jx_epi_elems,
             jx_epi_elems > 0.0 ? (double)jx_cyc_ep / jx_epi_elems : 0.0,
             jx_dcol_calls);
      printf("  [i8mm] fast %.0f 元素/%lu 块   slow %.0f 元素/%lu 块\n", jx_fast_elems, jx_fast_calls, jx_slow_elems, jx_slow_calls);
      {
        extern unsigned long jx_dt_cyc[4];
        extern unsigned long jx_dt_pos, jx_dt_blk, jx_dt_calls;
        double dt = (double)(jx_dt_cyc[0] + jx_dt_cyc[1] + jx_dt_cyc[2] + jx_dt_cyc[3]);
        printf("  [dt] dense tile: %.1f Mcyc  pos=%lu blk=%lu calls=%lu\n",
               dt / 1e6, jx_dt_pos, jx_dt_blk, jx_dt_calls);
        printf("  [dt]   量化 %.1f 收集 %.1f 点积 %.1f epi %.1f Mcyc\n",
               (double)jx_dt_cyc[0] / 1e6, (double)jx_dt_cyc[1] / 1e6,
               (double)jx_dt_cyc[2] / 1e6, (double)jx_dt_cyc[3] / 1e6);
        if (jx_dt_pos > 0)
          printf("  [dt]   每输出位置 %.0f 周期 (收集 %.1f 点积 %.1f epi %.1f)\n",
                 dt / (double)jx_dt_pos,
                 (double)jx_dt_cyc[1] / (double)jx_dt_pos,
                 (double)jx_dt_cyc[2] / (double)jx_dt_pos,
                 (double)jx_dt_cyc[3] / (double)jx_dt_pos);
      }
      {
        extern unsigned long jx_k1q_cyc, jx_k1n_cyc, jx_k1q_pos, jx_k1n_pos;
        printf("  [k1] QACC 分支: %.1f Mcyc / %lu pos = %.0f cyc/位置\n",
               (double)jx_k1q_cyc / 1e6, jx_k1q_pos,
               jx_k1q_pos ? (double)jx_k1q_cyc / (double)jx_k1q_pos : 0.0);
        printf("  [k1] accx 分支: %.1f Mcyc / %lu pos = %.0f cyc/位置\n",
               (double)jx_k1n_cyc / 1e6, jx_k1n_pos,
               jx_k1n_pos ? (double)jx_k1n_cyc / (double)jx_k1n_pos : 0.0);
        {
          extern unsigned long jx_k1s_cyc[4];
          extern double jx_k1s_qe, jx_k1s_oe;
          printf("  [k1s] PassA求max %.1f / PassB量化 %.1f / 点积 %.1f / epilogue %.1f Mcyc\n",
                 (double)jx_k1s_cyc[0] / 1e6, (double)jx_k1s_cyc[1] / 1e6,
                 (double)jx_k1s_cyc[2] / 1e6, (double)jx_k1s_cyc[3] / 1e6);
          if (jx_k1s_qe > 0.0)
            printf("  [k1s] 量化 %.2f+%.2f = %.2f cyc/(ci,j)  (%.0f 元素对)\n",
                   (double)jx_k1s_cyc[0] / jx_k1s_qe,
                   (double)jx_k1s_cyc[1] / jx_k1s_qe,
                   (double)(jx_k1s_cyc[0] + jx_k1s_cyc[1]) / jx_k1s_qe, jx_k1s_qe);
          if (jx_k1s_oe > 0.0)
            printf("  [k1s] 输出 %.0f 元素: 点积 %.2f + epilogue %.2f = %.2f cyc/元素\n",
                   jx_k1s_oe, (double)jx_k1s_cyc[2] / jx_k1s_oe,
                   (double)jx_k1s_cyc[3] / jx_k1s_oe,
                   (double)(jx_k1s_cyc[2] + jx_k1s_cyc[3]) / jx_k1s_oe);
        }
      }
      {
        /* 第四十四轮：jx_conv1d_k1_cf 子阶段拆分（实测打印）*/
        extern int jx_kc_n;
        extern int jx_kc_sc5[][5];
        extern unsigned long jx_kc_scy[][6];
        extern unsigned long jx_kc_spos[], jx_kc_sblk[], jx_kc_calls;
        extern unsigned long jx_kc_sd[][3];
        extern unsigned long jx_kc_sepi[][3];
        extern int jx_kc_spost[];
        if (jx_kc_n > 0)
          {
            printf("  [kc] k1cf 子阶段 (calls=%lu):\n", jx_kc_calls);
            printf("  [kc]   Ci  Co  T_in   T_out  TB    blk      pos    A_max  scale  B_quant    dot    epi  total  post  cyc/pos\n");
            for (int i = 0; i < jx_kc_n; i++)
              {
                double pos = (jx_kc_spos[i] > 0) ? (double)jx_kc_spos[i] : 1.0;
                printf("  [kc] %4d%4d%7d%7d%5d%7lu%9lu  %7.1f%6.1f%8.1f%7.1f%7.1f  %7.1f%5d\n",
                       jx_kc_sc5[i][0], jx_kc_sc5[i][1], jx_kc_sc5[i][2],
                       jx_kc_sc5[i][3], jx_kc_sc5[i][4], jx_kc_sblk[i], jx_kc_spos[i],
                       (double)jx_kc_scy[i][0] / pos, (double)jx_kc_scy[i][1] / pos,
                       (double)jx_kc_scy[i][2] / pos, (double)jx_kc_scy[i][3] / pos,
                       (double)jx_kc_scy[i][4] / pos, (double)jx_kc_scy[i][5] / pos, jx_kc_spost[i]);
                printf("  [kc]  epi o0 %7.1f  o1 %7.1f  o2+ %7.1f  (cyc/pos)\n",
                       (double)jx_kc_sepi[i][0] / pos, (double)jx_kc_sepi[i][1] / pos,
                       (double)jx_kc_sepi[i][2] / pos);
              }
          }
      jx_amax_dump();
      }
      {
        /* 2026-09-15 第三十六轮：conv_unit l2（jx_linear_i8_cf）的周期拆分。
         * 这个函数此前一个计数器都没有，它在 JP_I8MM 里那 ~900 ms 只能猜。 */
        extern unsigned long jx_cf_cyc[4], jx_cf_rows;
        extern double jx_cf_elems, jx_cf_in_elems;
        printf("  [cf] l2 量化 %.1f / 点积 %.1f / epilogue %.1f / 转置 %.1f Mcyc\n",
               (double)jx_cf_cyc[0] / 1e6, (double)jx_cf_cyc[1] / 1e6,
               (double)jx_cf_cyc[2] / 1e6, (double)jx_cf_cyc[3] / 1e6);
        if (jx_cf_elems > 0.0)
          {
            printf("  [cf] epilogue %.1f Mcyc / %.0f 元素 = %.2f cyc/元素"
                   "  (每元素 4 条 FP: cvt+mul+mul+add)\n",
                   (double)jx_cf_cyc[2] / 1e6, jx_cf_elems,
                   (double)jx_cf_cyc[2] / jx_cf_elems);
            if (jx_cf_in_elems > 0.0)
              printf("  [cf] 量化 %.2f cyc/元素 (%.0f 元素) / %.0f 行 / 点积 %.0f cyc/位置\n",
                     (double)jx_cf_cyc[0] / jx_cf_in_elems, jx_cf_in_elems,
                     (double)jx_cf_rows,
                     jx_cf_rows ? (double)jx_cf_cyc[1] / (double)jx_cf_rows : 0.0);
          }
      }
    }
#endif
    printf("  [alloc] t_alloc 累计 %.1f MB\n", jx_alloc_mb);
    if (jx_conv_skip > 0)
      {
        printf("  !! [WARN] 有 %d 次 conv1d 因内存不足被跳过 —— 本次结果无效\n",
               jx_conv_skip);
      }
    jx_ct_dump();
    {
        /* 240 cycles/ms（240MHz 单核）。MMAC/s 与 周期/MAC 是判定"算力瓶颈
         * 还是访存瓶颈"的两个硬指标：PIE int8 纯寄存器上限 3012 MMAC/s。 */
        double cs = jx_prof_ms[JP_CONV1D], ls = jx_prof_ms[JP_LINEAR];
        printf("  [mac] conv  : %.1f MMAC  %6.0f ms  %8.1f MMAC/s  %6.2f cyc/MAC  (%ld calls)\n",
               jx_mac_conv / 1e6, cs, cs > 0.0 ? jx_mac_conv / cs / 1000.0 : 0.0,
               jx_mac_conv > 0.0 ? cs * 240000.0 / jx_mac_conv : 0.0, jx_n_conv);
        printf("  [mac] linear: %.1f MMAC  %6.0f ms  %8.1f MMAC/s  %6.2f cyc/MAC  (%ld calls)\n",
               jx_mac_linear / 1e6, ls, ls > 0.0 ? jx_mac_linear / ls / 1000.0 : 0.0,
               jx_mac_linear > 0.0 ? ls * 240000.0 / jx_mac_linear : 0.0, jx_n_linear);
    }
}

/* Accumulator type for the dense kernels (linear / conv1d). The upstream code
 * always accumulated in `double` for cross-language agreement, but ESP32-S3 has
 * no double-precision FPU, so every double op there becomes a libgcc software
 * routine. Default is single precision; define SQ_CODEC_DOUBLE_ACC to restore
 * the original double accumulation. */
#ifdef SQ_CODEC_DOUBLE_ACC
  typedef double acc_t;
#else
  typedef float  acc_t;
#endif

/* ============================ tensor helpers ============================ */

/* conv1d_core 内层 t 轴的分块宽度（见该函数注释）：512 个 float = 2KB，
 * 保证块内 y 常驻 L1。 */
#define JX_CONV_TB 512

/* linear 的输出维分块目标字节数：让 W 的一个子块常驻 32KB 数据缓存，
 * 从而对所有输入行复用（见 linear 里的注释）。 */
#define JX_LINEAR_OB_BYTES 16384

/* conv1d_core 的输出通道分块目标字节数：块内所有 co 复用同一批 x 行。 */
/* 板端扫描：fp32 卷积的 co 分块 16384->32768 略优（12 530 -> 12 380 ms），
 * 8192 明显更差（12 720）。 */
#define JX_CONV_CB_BYTES 32768

/* 分块大小可在运行期覆盖（方便一次烧录对比多组）：
 *   set JX_OB 0      -> linear 不分块
 *   set JX_OB 8192   -> linear 每块 8KB
 *   set JX_CB 0      -> conv 不分块
 *   set JX_WT 1      -> 权重从 flash XIP 搬到堆/PSRAM（需重新 jixun 一次）
 * 未设置时用上面的默认值。 */
static int jx_block_bytes(const char *env, int defval)
{
    const char *e = getenv(env);
    if (e == NULL) { return defval; }
    int v = atoi(e);
    return (v < 0) ? 0 : v;
}

/* ==================== r174：16 字节对齐的堆分配 ====================
 * PIE 的 ee.vld.128 / ee.vst.128 要求地址 16 字节对齐，而 malloc 只保证 8。
 * 之前所有张量都是裸 malloc：地址随堆布局变化 -> 同一份代码在
 * 不同配置下读到错位数据（整个 conv/linear 结果不一样）。这里把它们
 * 全部对齐到 16；返回的指针前面埋一个 16 字节头（原始指针/字节数/魔数），
 * 所以 free/realloc 行为与原来完全一致（非本分配器的指针照旧交给系统 free/realloc）。 */
typedef struct { void *raw; unsigned bytes; unsigned magic; unsigned pad; } jx_ahdr_t;
#define JX_AMAGIC 0x4a58a11cu
static jx_ahdr_t *jx_ahdr_of(void *p)
{ return (jx_ahdr_t *)(void *)((unsigned char *)p - sizeof(jx_ahdr_t)); }
void *jx_amalloc(size_t bytes)
{
  unsigned char *raw = (unsigned char *)malloc(bytes + 96u);
  unsigned char *al; jx_ahdr_t *h;
  if (raw == NULL) { return NULL; }
  al = (unsigned char *)(((uintptr_t)raw + 64u) & ~(uintptr_t)15u);
  h = (jx_ahdr_t *)(void *)(al - sizeof(jx_ahdr_t));
  h->raw = (void *)raw; h->bytes = (unsigned)bytes; h->magic = JX_AMAGIC; h->pad = 0u;
  return (void *)al;
}
void *jx_arealloc(void *p, size_t newbytes)
{
  jx_ahdr_t *h; void *n; size_t cp;
  if (p == NULL) { return jx_amalloc(newbytes); }
  h = jx_ahdr_of(p);
  if (h->magic != JX_AMAGIC) { return realloc(p, newbytes); }
  n = jx_amalloc(newbytes);
  if (n == NULL) { return NULL; }          /* 失败时保留原块，与 realloc 同语义 */
  cp = ((size_t)h->bytes < newbytes) ? (size_t)h->bytes : newbytes;
  memcpy(n, p, cp);
  free(h->raw);
  return n;
}
void jx_afree(void *p)
{
  jx_ahdr_t *h;
  if (p == NULL) { return; }
  h = jx_ahdr_of(p);
  if (h->magic == JX_AMAGIC) { free(h->raw); } else { free(p); }
}

T t_alloc(int ndim, const int *shape) {
    T t; t.ndim = ndim; int len = 1;
    for (int i = 0; i < ndim; i++) { t.shape[i] = shape[i]; len *= shape[i]; }
    t.len = len;
    /* 2026-09-13 第八轮（冲击 1s 级的最大单项）：这里原来是 calloc ——
     * 每次分配都要把整块内存清零。PSRAM 带宽实测只有 ~50 MB/s，而本模型的
     * 中间张量动辄 4~5 MB（conv_unit 的 p1、first_block 的 hc/h1），
     * 一次清零就是 80~100 ms；全模型累计的清零流量在 100 MB 量级。
     *
     * 所有调用点在读之前都会把张量写满（conv/linear/permute/epilogue），
     * 所以清零是纯浪费。改成 malloc。JX_ZALLOC=1 可退回 calloc 做 A/B。
     * 主机回归用 MALLOC_PERTURB_ 验证过：没有任何一处依赖零初始化。 */
    t.d = (float*)jx_amalloc((size_t)len * sizeof(float));
    if (t.d != NULL)
      {
        double mb = (double)((size_t)len * sizeof(float)) / 1.0e6;
        jx_live_mb += mb;
        if (jx_live_mb > jx_peak_mb) { jx_peak_mb = jx_live_mb; }
      }
    else
      {
        struct mallinfo mi = mallinfo();
        syslog(JX_LOG_ERR,
               "[OOM] t_alloc ret=%p ndim=%d shape=%d,%d,%d,%d len=%d "
               "req=%.2fMB live=%.2f peak=%.2f uord=%u ford=%u mx=%u\n",
               __builtin_return_address(0), ndim,
               (ndim > 0) ? t.shape[0] : 0, (ndim > 1) ? t.shape[1] : 0,
               (ndim > 2) ? t.shape[2] : 0, (ndim > 3) ? t.shape[3] : 0,
               len, (double)((size_t)len * sizeof(float)) / 1.0e6,
               jx_live_mb, jx_peak_mb,
               (unsigned)mi.uordblks, (unsigned)mi.fordblks,
               (unsigned)mi.mxordblk);
      }
    if (getenv("JX_ZALLOC") && t.d != NULL) { memset(t.d, 0, (size_t)len * sizeof(float)); }
    jx_alloc_mb += (double)len * sizeof(float) / 1.0e6;
    return t;
}
void t_free(T *t)
{
    if (t->d != NULL)
      {
        jx_live_mb -= (double)((size_t)t->len * sizeof(float)) / 1.0e6;
        if (jx_live_mb < 0.0) { jx_live_mb = 0.0; }
        jx_afree(t->d);
      }
    t->d = NULL; t->len = 0;
}
T t_zeros_like(const T *s) { return t_alloc(s->ndim, s->shape); }
void t_copy(T *dst, const T *src) { memcpy(dst->d, src->d, (size_t)src->len * sizeof(float)); }

/* ============================ weight lookup ============================ */
/* 权重数据默认是 .rodata 里的 const 数组 → 落在 flash，走 XIP 取指。
 * 板端实测顺序读带宽：flash 15.5 MB/s vs PSRAM 52.4 MB/s（约 3.4 倍），
 * 而 conv/linear 的权重是主要访存对象，所以这里提供一份堆上的常驻副本。
 *
 *   set JX_WT 0   -> 用 flash 里的原始块（默认，行为与改动前完全一致）
 *   set JX_WT 1   -> 启动时把整块权重 memcpy 到堆（1.9MB，会被
 *                    CONFIG_ESP32S3_SPIRAM_COMMON_HEAP 放进 PSRAM）
 *
 * 搬运是逐字节 memcpy，数值一个 bit 都不变，纯粹换数据放的位置。 */
static const float *g_wt_ptr = WT_DATA;
static void        *g_wt_ram = NULL;

int jx_weight_init(void)
{
    if (g_wt_ram != NULL) { return 0; }
    const char *e = getenv("JX_WT");
    int want = (e != NULL) ? atoi(e) : 0;
    if (want <= 0) { return 0; }

    size_t bytes = (size_t)NW_FLOATS * sizeof(float);
    float *p = (float *)jx_amalloc(bytes);
    if (p == NULL)
      {
        printf("[WT] 堆上分配 %u 字节失败，继续用 flash\n", (unsigned)bytes);
        return -1;
      }

    memcpy(p, WT_DATA, bytes);
    g_wt_ram = p;
    g_wt_ptr = p;
    printf("[WT] 权重已搬到堆: %p (%u 字节, %.1f MB)\n",
           (void *)p, (unsigned)bytes, (double)bytes / 1048576.0);
    return 0;
}

int jx_weight_in_heap(void) { return g_wt_ram != NULL; }

const float* wt_get(const char *name, int *shape4, int *n_out) {
    for (int i = 0; i < NW_TENSORS; i++)
        if (strcmp(WT_TAB[i].name, name) == 0) {
            if (shape4) for (int j = 0; j < 4; j++) shape4[j] = WT_TAB[i].shape[j];
            if (n_out) *n_out = WT_TAB[i].n;
            return &g_wt_ptr[WT_TAB[i].off];
        }
    return NULL;
}
int wt_has(const char *name) { for (int i = 0; i < NW_TENSORS; i++) if (!strcmp(WT_TAB[i].name, name)) return 1; return 0; }
const float* jx_weight_blob(void) { return g_wt_ptr; }
int jx_weight_floats(void) { return (int)NW_FLOATS; }
static char _catbuf[JX_NCORE][8][256];
static int _catbuf_i[JX_NCORE];
const char* cat_name(const char *pfx, const char *suffix) {
    int c = jx_core_id();
    int i = _catbuf_i[c]; _catbuf_i[c] = (_catbuf_i[c] + 1) & 7;
    snprintf(_catbuf[c][i], sizeof(_catbuf[c][i]), "%s.%s", pfx, suffix); return _catbuf[c][i];
}
const float* W_(const char *name) {
    const float *p = wt_get(name, NULL, NULL);
    if (!p) fprintf(stderr, "MISSING WEIGHT: %s\n", name);
    return p;
}

/* ============================ activations / norms ============================ */
/* snake_apply 的每通道系数缓存（ch 最大是 conv_unit 的 4*88 = 352）。 */
#define JX_SNAKE_MAXCH 512
static float g_snake_a[JX_NCORE][JX_SNAKE_MAXCH];
static float g_snake_inv[JX_NCORE][JX_SNAKE_MAXCH];

/* r51e：snake1d 每调用诊断（JX_SNKDBG=1），定位 173 周期/元素到底花在哪 */
static inline unsigned jx_snk_cc(void)
{
#ifdef JX_NO_PROFILE
  return 0u;
#else
  unsigned c;
  __asm__ __volatile__("rsr %0, ccount" : "=r"(c));
  return c;
#endif
}
static int jx_snkdbg(void)
{
  static int v = -1;
  if (v < 0) { v = (getenv("JX_SNKDBG") != NULL) ? 1 : 0; }
  return v;
}

/* r51d：snake1d 的通道优先路径三路切换（默认 1 = 层级并排 x4，与原状一致） */
static int jx_sq4x4_on(void)
{
  static int v = -2;
  if (v == -2) { const char *e = getenv("JX_SQ4X4"); v = (e != NULL && e[0] != (char)48) ? atoi(e) : 1; }
  return v;
}

/* ==================== r59：逐元素循环按区间切到第二颗核 ====================
 * snake / channel_norm / grn / instance_norm 都是"一堆互相独立的工作项 + 每个
 * 元素一条纯函数式算式"。把工作项区间二分给两颗核，只要每个元素的算式与求和
 * 次序一字不改，结果就逐位相同（与 jx_linear_i8_rows 同一套理由）。 */

typedef struct { T *x; const float *alpha; int n, ch; } jx_snkar_t;

static void jx_snake_apply_rows(void *va, int ilo, int ihi)
{
    jx_snkar_t *cc = (jx_snkar_t *)va;
    float *x = cc->x;
    const float *alpha = cc->alpha;
    const int n = cc->n, ch = cc->ch;
    (void)n;
    JX_DIST_ARM();
    jx_sq_kt_ensure();
    const jx_sq4 K = jx_sq4_load();
    float *const s_snake_a   = g_snake_a[jx_core_id()];
    float *const s_snake_inv = g_snake_inv[jx_core_id()];
    if (ch > JX_SNAKE_MAXCH) {
        for (int c = 0; c < ch; c++) {
            float a = alpha[c] + XNN_EPS; float inv = 1.0f / a;
            for (int i = ilo; i < ihi; i++) { float v = x[i * ch + c]; JX_DIST_UV(a * v); x[i * ch + c] = v + inv * jx_sinsq4(a * v, K); }
        }
        return;
    }
    for (int c = 0; c < ch; c++) { s_snake_a[c] = alpha[c] + XNN_EPS; s_snake_inv[c] = 1.0f / s_snake_a[c]; }
    for (int i = ilo; i < ihi; i++) {
        float *p = &x[(size_t)i * ch];
        int c = 0;
        for (; c + 4 <= ch; c += 4) {
            float v0 = p[c], v1 = p[c + 1], v2 = p[c + 2], v3 = p[c + 3];
            JX_DIST_UV(s_snake_a[c] * v0); JX_DIST_UV(s_snake_a[c + 1] * v1);
            JX_DIST_UV(s_snake_a[c + 2] * v2); JX_DIST_UV(s_snake_a[c + 3] * v3);
            float s0 = jx_sinsq4(s_snake_a[c]     * v0, K);
            float s1 = jx_sinsq4(s_snake_a[c + 1] * v1, K);
            float s2 = jx_sinsq4(s_snake_a[c + 2] * v2, K);
            float s3 = jx_sinsq4(s_snake_a[c + 3] * v3, K);
            p[c]     = v0 + s_snake_inv[c]     * s0;
            p[c + 1] = v1 + s_snake_inv[c + 1] * s1;
            p[c + 2] = v2 + s_snake_inv[c + 2] * s2;
            p[c + 3] = v3 + s_snake_inv[c + 3] * s3;
        }
        for (; c < ch; c++) {
            float v = p[c];
            JX_DIST_UV(s_snake_a[c] * v);
            p[c] = v + s_snake_inv[c] * jx_sinsq4(s_snake_a[c] * v, K);
        }
    }
}

static void snake_apply(float *x, const float *alpha, int n, int ch) {
    JX_DIST_ARM();
    jx_sq_kt_ensure();
    jx_snkar_t cc;
    cc.x = x; cc.alpha = alpha; cc.n = n; cc.ch = ch;
    jx_pf_run_min(n, jx_snake_apply_rows, &cc, 8);
}

/* channels-first：工作项 = 一个 (b,c) 通道行，展平成 k = b*C + c 后二分。 */
typedef struct { T *x; const float *alpha; int C, Tt; } jx_snkcf_t;

JX_HOT static void jx_snake1d_cf_range(void *va, int klo, int khi)
{
    jx_snkcf_t *cc = (jx_snkcf_t *)va;
    T *x = cc->x;
    const float *alpha = cc->alpha;
    const int C = cc->C, Tt = cc->Tt;
    const jx_sq4 K = jx_sq4_load();
    const int _md = jx_sq4x4_on();
    for (int k = klo; k < khi; k++)
      {
        const int b = k / C;
        const int c = k - b * C;
        float a = alpha[c] + XNN_EPS; float inv = 1.0f / a;
        float *p = &x->d[((size_t)b * C + c) * Tt];
        int i = 0;
        if (_md == 2)
          {
            for (; i < Tt; i++) { p[i] = p[i] * 2.0f; }
            continue;
          }
        if (_md == 1)
          {
            for (; i + 4 <= Tt; i += 4)
              {
                float v0 = p[i], v1 = p[i + 1], v2 = p[i + 2], v3 = p[i + 3];
                JX_DIST_UV(a * v0); JX_DIST_UV(a * v1); JX_DIST_UV(a * v2); JX_DIST_UV(a * v3);
                float z0, z1, z2, z3;
                jx_sinsq4_x4(a * v0, a * v1, a * v2, a * v3, K, &z0, &z1, &z2, &z3);
                p[i]     = v0 + inv * z0;
                p[i + 1] = v1 + inv * z1;
                p[i + 2] = v2 + inv * z2;
                p[i + 3] = v3 + inv * z3;
              }
          }
        else
          {
            for (; i + 4 <= Tt; i += 4)
              {
                float v0 = p[i], v1 = p[i + 1], v2 = p[i + 2], v3 = p[i + 3];
                float z0 = jx_sinsq4(a * v0, K), z1 = jx_sinsq4(a * v1, K);
                float z2 = jx_sinsq4(a * v2, K), z3 = jx_sinsq4(a * v3, K);
                p[i]     = v0 + inv * z0;
                p[i + 1] = v1 + inv * z1;
                p[i + 2] = v2 + inv * z2;
                p[i + 3] = v3 + inv * z3;
              }
          }
        for (; i < Tt; i++) { float v = p[i]; JX_DIST_UV(a * v); p[i] = v + inv * jx_sinsq4(a * v, K); }
      }
}

void snake1d(T *x, const float *alpha, int channels, int dim1, int dim2, int data_format) {
    JX_DIST_ARM();
    jx_sq_kt_ensure();
    const int _dbg = jx_snkdbg();
    const unsigned _c0 = _dbg ? jx_snk_cc() : 0u;
    const int _len0 = (int)x->len;
    JXP_BEG(JP_SNAKE);
    jx_elems_snake += (double)x->len;
    if (data_format == 1) {
        int B = x->shape[0], Tt = x->shape[1], C = x->shape[2];
        for (int b = 0; b < B; b++) snake_apply(&x->d[b * Tt * C], alpha, Tt, C);
    } else {
        int B = x->shape[0], C = x->shape[1], Tt = x->shape[2];
        jx_snkcf_t cc;
        cc.x = x; cc.alpha = alpha; cc.C = C; cc.Tt = Tt;
        jx_pf_run_min(B * C, jx_snake1d_cf_range, &cc, 2);
    }
    if (_dbg)
      {
        unsigned _c1 = jx_snk_cc();
        double _el = (_len0 > 0) ? (double)_len0 : 1.0;
        printf("  [snk] fmt=%d s=[%d,%d,%d] len=%d dp=%d cyc=%lu  %.1f cyc/el\n",
               data_format, (int)x->shape[0], (int)x->shape[1], (int)x->shape[2],
               _len0, data_format == 1 ? (int)x->shape[2] : (int)x->shape[2],
               (unsigned long)(_c1 - _c0), (double)(_c1 - _c0) / _el);
        fflush(stdout);
      }
    JXP_END(JP_SNAKE);
}

/* ---- channel_norm_last：一行 (C 个元素) 是一个独立工作项 ---- */
typedef struct { T *x; const float *w; const float *b; int Tt, C; } jx_cnl_t;

JX_HOT static void jx_cnorm_last_rows(void *va, int ilo, int ihi)
{
    jx_cnl_t *cc = (jx_cnl_t *)va;
    const int C = cc->C;
    const float *w = cc->w, *bb = cc->b;
    for (int i = ilo; i < ihi; i++) {
        if (i + 1 < ihi) { jx_dcp_pf_ew(&cc->x->d[(size_t)(i + 1) * C], (unsigned)C * 4u); }
        float *p = &cc->x->d[i * C];
        /* r120：mean / var 原来是单条串行浮点累加链（add.s 延迟 ~4 周期，
         * 完全暴露），实测 3.7 周期/元素。改成 4 条独立部分和再合并 ——
         * 求和次序改变只影响末位（352 个数分组相加，相对误差 ~1e-7），
         * 远小于 int8 量化噪声；SNR 门禁实测不变。 */
        float mean;
        {
          float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
          int c = 0;
          for (; c + 4 <= C; c += 4)
            { m0 += p[c]; m1 += p[c + 1]; m2 += p[c + 2]; m3 += p[c + 3]; }
          mean = (m0 + m1) + (m2 + m3);
          for (; c < C; c++) { mean += p[c]; }
          mean /= C;
        }
        float var = 0;
        {
          float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
          int c = 0;
          for (; c + 4 <= C; c += 4)
            {
              float d0 = p[c] - mean, d1 = p[c + 1] - mean;
              float d2 = p[c + 2] - mean, d3 = p[c + 3] - mean;
              v0 += d0 * d0; v1 += d1 * d1; v2 += d2 * d2; v3 += d3 * d3;
            }
          var = (v0 + v1) + (v2 + v3);
          for (; c < C; c++) { float d = p[c] - mean; var += d * d; }
          var /= C;
        }
        float inv = 1.0f / sqrtf(var + XNN_EPS);
        for (int c = 0; c < C; c++) p[c] = w[c] * (p[c] - mean) * inv + bb[c];
    }
}

void channel_norm_last(T *x, const float *w, const float *b, int B, int Tt, int C) {
    JXP_BEG(JP_NORMS);
    jx_cnl_t cc; cc.x = x; cc.w = w; cc.b = b; cc.Tt = Tt; cc.C = C;
    jx_pf_run_min(B * Tt, jx_cnorm_last_rows, &cc, 64);
    JXP_END(JP_NORMS);
}

/* ---- channel_norm_first：一个 (b,t) 是一个独立工作项（按 t 切） ---- */
typedef struct { T *x; const float *w; const float *b; int bi, Tt, C; } jx_cnf_t;

/* r145 原版：三趟各自跨 Tt 扫这一列（JX_CNF=0 时用，作 A/B 对照） */
JX_HOT static void jx_cnorm_first_t_old(void *va, int tlo, int thi)
{
    jx_cnf_t *cc = (jx_cnf_t *)va;
    T *x = cc->x;
    const int bi = cc->bi, Tt = cc->Tt, C = cc->C;
    const float *w = cc->w, *bb = cc->b;
    for (int t = tlo; t < thi; t++) {
        /* r120：同上，4 条独立部分和盖住 add.s 延迟。这里是跨步列访问，
         * 每条链各自顺序扫一遍自己的列，访存局部性反而更好。 */
        float mean;
        {
          float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
          int c = 0;
          for (; c + 4 <= C; c += 4)
            {
              m0 += x->d[(size_t)(bi * C + c    ) * Tt + t];
              m1 += x->d[(size_t)(bi * C + c + 1) * Tt + t];
              m2 += x->d[(size_t)(bi * C + c + 2) * Tt + t];
              m3 += x->d[(size_t)(bi * C + c + 3) * Tt + t];
            }
          mean = (m0 + m1) + (m2 + m3);
          for (; c < C; c++) { mean += x->d[(size_t)(bi * C + c) * Tt + t]; }
          mean /= C;
        }
        float var = 0;
        {
          float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
          int c = 0;
          for (; c + 4 <= C; c += 4)
            {
              float d0 = x->d[(size_t)(bi * C + c    ) * Tt + t] - mean;
              float d1 = x->d[(size_t)(bi * C + c + 1) * Tt + t] - mean;
              float d2 = x->d[(size_t)(bi * C + c + 2) * Tt + t] - mean;
              float d3 = x->d[(size_t)(bi * C + c + 3) * Tt + t] - mean;
              v0 += d0 * d0; v1 += d1 * d1; v2 += d2 * d2; v3 += d3 * d3;
            }
          var = (v0 + v1) + (v2 + v3);
          for (; c < C; c++) { float d = x->d[(size_t)(bi * C + c) * Tt + t] - mean; var += d * d; }
          var /= C;
        }
        float inv = 1.0f / sqrtf(var + XNN_EPS);
        for (int c = 0; c < C; c++) {
            float v = x->d[(bi * C + c) * Tt + t];
            x->d[(bi * C + c) * Tt + t] = w[c] * (v - mean) * inv + bb[c];
        }
    }
}

JX_HOT static void jx_cnorm_first_t(void *va, int tlo, int thi)
{
    jx_cnf_t *cc = (jx_cnf_t *)va;
    T *x = cc->x;
    const int bi = cc->bi, Tt = cc->Tt, C = cc->C;
    const float *w = cc->w, *bb = cc->b;
    /* r146：原来 mean / var / apply 三趟各自跨 Tt 扫一遍这一列。
     * 列的工作集是 C*Tt*4 字节（远大于 cache），所以三趟都从 PSRAM 重新取，
     * 等于每元素 12B 读 + 4B 写。改成先把整列装进按核常驻的小缓冲，
     * 再在缓冲上算 mean / var / apply，内存变成 1 读 + 1 写 = 8B/元素。
     * 累加次序、算式与原来逐项相同 -> 结果逐位一致。 */
    static float *s_col[JX_NCORE];
    static int    s_cap[JX_NCORE];
    int core = jx_core_id();
    if (s_cap[core] < C)
      {
        float *nc = (float *)realloc(s_col[core], (size_t)C * sizeof(float));
        if (nc == NULL) { return; }
        s_col[core] = nc; s_cap[core] = C;
      }
    float *col = s_col[core];
    const float *xbase = x->d + (size_t)bi * (size_t)C * (size_t)Tt;
    float *ybase = x->d + (size_t)bi * (size_t)C * (size_t)Tt;

    for (int t = tlo; t < thi; t++) {
        /* 装列：一次跨步读，后面 mean / var / apply 都吃这一份 */
        {
          const float *px = xbase + t;
          for (int c = 0; c < C; c++) { col[c] = px[(size_t)c * Tt]; }
        }
        float mean;
        {
          float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
          int c = 0;
          for (; c + 4 <= C; c += 4)
            { m0 += col[c]; m1 += col[c + 1]; m2 += col[c + 2]; m3 += col[c + 3]; }
          mean = (m0 + m1) + (m2 + m3);
          for (; c < C; c++) { mean += col[c]; }
          mean /= C;
        }
        float var = 0;
        {
          float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
          int c = 0;
          for (; c + 4 <= C; c += 4)
            {
              float d0 = col[c    ] - mean, d1 = col[c + 1] - mean;
              float d2 = col[c + 2] - mean, d3 = col[c + 3] - mean;
              v0 += d0 * d0; v1 += d1 * d1; v2 += d2 * d2; v3 += d3 * d3;
            }
          var = (v0 + v1) + (v2 + v3);
          for (; c < C; c++) { float d = col[c] - mean; var += d * d; }
          var /= C;
        }
        float inv = 1.0f / sqrtf(var + XNN_EPS);
        {
          float *px = ybase + t;
          for (int c = 0; c < C; c++)
            { px[(size_t)c * Tt] = w[c] * (col[c] - mean) * inv + bb[c]; }
        }
    }
}

/* r147b：JX_CNF=0 走 r145 的三趟原版，默认 1 走 r146 的单趟缓存版（逐位一致） */
static int jx_cnf_mode(void)
{
  static int v = -2;
  if (v == -2) { const char *e = getenv("JX_CNF"); v = (e != NULL) ? atoi(e) : 1; }
  return v;
}

void channel_norm_first(T *x, const float *w, const float *b, int B, int C, int Tt) {
    JXP_BEG(JP_NORMS);
    jx_cnf_t cc; cc.x = x; cc.w = w; cc.b = b; cc.Tt = Tt; cc.C = C;
    if (jx_cnf_mode()) { for (int bi = 0; bi < B; bi++) { cc.bi = bi; jx_pf_run_min(Tt, jx_cnorm_first_t, &cc, 64); } }
    else               { for (int bi = 0; bi < B; bi++) { cc.bi = bi; jx_pf_run_min(Tt, jx_cnorm_first_t_old, &cc, 64); } }
    JXP_END(JP_NORMS);
}

/* ---- grn_last：第一趟（求和）保持串行以保住求和次序，第二趟按元素区间切 ---- */
typedef struct { float *p; const float *gamma; const float *beta; float n_x; int C; } jx_grn2_t;

static void jx_grn_last_pass2(void *va, int ilo, int ihi)
{
    jx_grn2_t *cc = (jx_grn2_t *)va;
    float *p = cc->p;
    const float *gamma = cc->gamma, *beta = cc->beta;
    const float nx = cc->n_x;
    const int C = cc->C;
    int c = (C > 0) ? (ilo % C) : 0;
    for (int i = ilo; i < ihi; i++) {
        p[i] = gamma[c] * (p[i] * nx) + beta[c] + p[i];
        if (++c == C) { c = 0; }
    }
}

void grn_last(T *x, const float *gamma, const float *beta, int B, int Tt, int C) {
    JXP_BEG(JP_NORMS);
    for (int bi = 0; bi < B; bi++) {
        float *p = &x->d[bi * Tt * C]; float ss = 0;
        for (int i = 0; i < Tt * C; i++) ss += p[i] * p[i];
        float g = sqrtf(ss); float n_x = g / (g + XNN_EPS);
        jx_grn2_t cc; cc.p = p; cc.gamma = gamma; cc.beta = beta; cc.n_x = n_x; cc.C = C;
        jx_pf_run_min(Tt * C, jx_grn_last_pass2, &cc, 256);
    }
    JXP_END(JP_NORMS);
}

/* ---- grn_first：第一趟串行，第二趟按通道 c 切（n_x 与 c 无关） ---- */
typedef struct { T *x; const float *gamma; const float *beta; float n_x; int base, C, Tt; } jx_grn1_t;

static void jx_grn_first_pass2(void *va, int clo, int chi)
{
    jx_grn1_t *cc = (jx_grn1_t *)va;
    T *x = cc->x;
    const float *gamma = cc->gamma, *beta = cc->beta;
    const float nx = cc->n_x;
    const int base = cc->base, Tt = cc->Tt;
    for (int c = clo; c < chi; c++) {
        const float gc = gamma[c], bc = beta[c];
        for (int t = 0; t < Tt; t++) {
            int idx = base + c * Tt + t;
            x->d[idx] = gc * (x->d[idx] * nx) + bc + x->d[idx];
        }
    }
}

void grn_first(T *x, const float *gamma, const float *beta, int B, int C, int Tt) {
    JXP_BEG(JP_NORMS);
    for (int bi = 0; bi < B; bi++) {
        float ss = 0; int base = bi * C * Tt;
        for (int i = 0; i < C * Tt; i++) ss += x->d[base + i] * x->d[base + i];
        float g = sqrtf(ss); float n_x = g / (g + XNN_EPS);
        jx_grn1_t cc; cc.x = x; cc.gamma = gamma; cc.beta = beta;
        cc.n_x = n_x; cc.base = base; cc.C = C; cc.Tt = Tt;
        jx_pf_run_min(C, jx_grn_first_pass2, &cc, 2);
    }
    JXP_END(JP_NORMS);
}

/* ---- instance_norm1d：一个 (b,c) 通道行是一个独立工作项 ---- */
/* 2026-09-17：可选「因果滑动窗」统计，给流式分块用（set JX_INNORM 400 打开）。
 *
 * 背景：现在是每个 (b,c) 沿**整条 T** 求均值方差。分块处理后 T 变了，统计量
 * 就跟着变，输出整体偏移 —— 而且加多长左回看都补不回来。实测（jixun seg）：
 * C=2400ms 对齐到注意力窗格、L=2400ms 给足整整一个窗，SNR 仍只有 12.91 dB，
 * 峰值误差和 L=0 完全相同（1.55e4）。这说明误差不由回看长度决定，就是这里。
 *
 * 改成窗口长 W 帧的**因果滑动窗** [t-W+1, t] 后：只要缓冲区覆盖 W 帧，
 * 分块与整段在每个 t 上算出的统计量就是同一个值 -> 分块误差里这一项归零。
 * W 取 400 帧（= EN_WINDOW = 2400ms）正好和注意力的回看需求重合，不额外加码。
 *
 * 数值上滑窗用一次性 running sum（naive 方差公式），整段路径仍走原来的两趟
 * 公式；两条路都在同一个开关下切换，所以 A/B 时比较的是同一套定义。
 * 默认 0，保持原行为，不破坏已验证的基线。 */
typedef struct { T *x; const float *w; const float *b; int C, Tt, win; float eps; } jx_in1_t;

static int jx_innorm_win(void)
{
    static int v = -1;
    if (v < 0)
      {
        const char *e = getenv("JX_INNORM");
        v = (e != NULL) ? atoi(e) : 0;
        if (v < 0) { v = 0; }
      }
    return v;
}

JX_HOT static void jx_in1_rows(void *va, int klo, int khi)
{
    jx_in1_t *cc = (jx_in1_t *)va;
    T *x = cc->x;
    const int C = cc->C, Tt = cc->Tt;
    const float eps = cc->eps;
    const int W = cc->win;
    float *ring = NULL;

    /* 注意：门限只能是 W>0，不能加 Tt>W。
     * 滑窗对 Tt<=W 也有明确定义（t<W 时窗口就是 [0,t]），而且正是这个定义
     * 让「段长恰好等于窗长」的分块与整段算出的结果一致 —— 加了 Tt>W 的话，
     * 段长=400 会掉回全局统计、而整段(500 帧)用滑窗，两边定义不一致，
     * 峰值误差恒定为 1.5e4 就是这么来的（r190 实测）。 */
    if (W > 0)
      {
        /* 每个 worker 一份环形缓冲（只存最近 W 个原始值，供扣除用），
         * W=400 时 1.6KB；放堆上，不占 .bss（dram0_0_seg 已无余量）。 */
        ring = (float *)malloc(sizeof(float) * (size_t)W);
      }

    for (int k = klo; k < khi; k++) {
        const int c = (k % C);
        float *p = &x->d[(size_t)k * Tt];
        const float wc = cc->w[c], bc = cc->b[c];

        if (ring == NULL)
          {
            float mean = 0; for (int t = 0; t < Tt; t++) mean += p[t]; mean /= Tt;
            float var = 0; for (int t = 0; t < Tt; t++) { float d = p[t] - mean; var += d * d; } var /= Tt;
            float inv = 1.0f / sqrtf(var + eps);
            for (int t = 0; t < Tt; t++) p[t] = wc * (p[t] - mean) * inv + bc;
          }
        else
          {
            /* 因果滑动窗 [t-W+1, t]。注意顺序：先扣掉 t-W 的老值再写入新值，
             * 否则 ring[t%W] 已被覆盖，扣的就是自己。 */
            float s = 0.0f, s2 = 0.0f;
            for (int t = 0; t < Tt; t++)
              {
                float v = p[t];

                if (t >= W)
                  {
                    float o = ring[t % W];
                    s -= o; s2 -= o * o;
                  }

                ring[t % W] = v;
                s += v; s2 += v * v;

                {
                  int n = (t + 1 < W) ? (t + 1) : W;
                  float mean = s / (float)n;
                  float var = s2 / (float)n - mean * mean;
                  float inv;

                  if (var < 0.0f) { var = 0.0f; }
                  inv = 1.0f / sqrtf(var + eps);
                  p[t] = wc * (v - mean) * inv + bc;
                }
              }
          }
    }

    if (ring != NULL) { free(ring); }
}

void instance_norm1d(T *x, const float *w, const float *b, int B, int C, int Tt, float eps) {
    JXP_BEG(JP_NORMS);
    jx_in1_t cc; cc.x = x; cc.w = w; cc.b = b; cc.C = C; cc.Tt = Tt; cc.eps = eps;
    cc.win = jx_innorm_win();
    jx_pf_run_min(B * C, jx_in1_rows, &cc, 2);
    JXP_END(JP_NORMS);
}


/* ============================ layers ============================ */

/* MAC 统计（仅 -DJX_TRACE_MACS 时启用，用于定位算力分布；生产构建完全不碰） */
/* 常开（开销 = 每次调用一次 double 加，可忽略）：板端 bench 结束直接汇报
 * MMAC/s 与 周期/MAC，用来判定 int8 内核是否真的全命中、离 PIE 上限多远。 */
double jx_mac_conv = 0, jx_mac_linear = 0;
long   jx_n_conv = 0,  jx_n_linear = 0;

void linear(const float *W, const float *b, const T *x, T *y, int in, int out) {
    JXP_BEG(JP_LINEAR);
    /* Apply the same Linear (rows = in -> out) to every trailing "row" of the
     * (..., in) tensor. For (B, Tt, C) input, rows = B*Tt. This is dense
     * row-major so x->len/in gives the correct count of independent rows.
     * Accumulate in SINGLE precision (SQ_CODEC_FLOAT_ACC): the original used
     * `double` for cross-language agreement, but ESP32-S3 has no double FPU, so
     * every double op there is a libgcc software routine. Float accumulation
     * keeps the golden tolerances while making the kernels vectorizable. */
    int rows = x->len / in;
    jx_mac_linear += (double)rows * out * in; jx_n_linear++;
#ifdef JX_TRACE_MACS
    printf("[MAC] linear  rows=%-6d in=%-5d out=%-5d  mac=%.3fM\n", rows, in, out, (double)rows*out*in/1e6);
#endif
    /* 上板优化（2026-09-12）：把输出维按 JX_LINEAR_OB_BYTES 分块。
     *
     * 原写法 i(行) 在外、o 在内，每读一行都要把整个 W 顺序扫一遍；W 常常
     * 远大于 32KB 数据缓存（如 88->352 时 W=124KB，来自 flash XIP，实测
     * 顺序读只有 15.5 MB/s），于是每一行都触发一次全量重取。
     *
     * 改成 o 分块在外：W 的一个子块常驻 L1，对所有行复用，W 的访存量从
     * rows x out x in 降到 out x in（本模型里是几千倍的差别）。
     * 每个输出元素的求和次序仍是 j 从 0 到 in-1，结果逐位不变。 */
    int ob = (jx_block_bytes("JX_OB", JX_LINEAR_OB_BYTES) / 4) / (in > 0 ? in : 1);
    /* int8/PIE 快路径：权重已离线量化、输入按行量化后做 128-bit 向量点积。
     * 不可用时（没有量化权重 / JX_INT8=0）静默退回下面的浮点内核。 */
    if (y->d != NULL && (size_t)y->len >= (size_t)rows * (size_t)out &&
        jx_linear_i8(W, b, x->d, y->d, rows, in, out) == 0)
      {
        JXP_END(JP_LINEAR);
        return;
      }
    if (ob < 1)   { ob = 1; }
    if (ob > out) { ob = out; }
    for (int o0 = 0; o0 < out; o0 += ob) {
        int o1 = o0 + ob; if (o1 > out) { o1 = out; }
        for (int i = 0; i < rows; i++) {
            const float *xi = &x->d[i * in]; float *yi = &y->d[i * out];
            for (int o = o0; o < o1; o++) {
                const float *wr = &W[o * in]; acc_t acc = (b ? (acc_t)b[o] : (acc_t)0);
                for (int j = 0; j < in; j++) acc += (acc_t)xi[j] * (acc_t)wr[j];
                yi[o] = (float)acc;
            }
        }
    }
    JXP_END(JP_LINEAR);
}
/* 快速路径（JX_CVS，默认开）：stride==1 时把 t 轴切成 [0,tlo) 边界 /
 * [tlo,thi) 无判断主区 / [thi,T_out) 边界三段。主区内每个 MAC 不再算坐标、
 * 不做两次越界比较，内层退化成 `for(kk) acc += xr[kk*dil]*wr[kk]`，编译器可
 * 以展开并双发射。累加次序与原版一致（ci 外层、kk 内层），结果逐位相同。
 * 动机：原版在 depthwise k=7 / Ci=1->Co=4 这类小-copg 形状上实测 16~19
 * cyc/MAC，而 FPU 地板只有 3.86 cyc/op，差距全在这段控制流上。 */
static int jx_cvs(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("JX_CVS"); v = (e != NULL && atoi(e) == 0) ? 0 : 1; }
    return v;
}

/* r67：fp32 回退卷积（Ci=1 Co=4 k=7 / Ci=16 Co=1 k=7 等小形状，合计约
 * 310 ms）原来完全串行（JX_PF=0 对照实测 0.95x）。按 t 块切给两颗核：
 * 每个块只写 yout[co][t0..t1) 的互不相交区间，每个输出元素的求值次序
 * (ci 外层 / kk 内层) 一字不改 -> 逐位相同。 */
typedef struct {
  const float *W; const float *b; const float *xin; float *yout;
  int cpg0, copg0, k, stride, pad, dil, T_in, T_out;
  int fast_div, cshift, tlo, thi, co0, co1;
} jx_cvf_t;

JX_HOT static void jx_cvf_tblocks(void *va, int blo, int bhi)
{
  const jx_cvf_t *C0 = (const jx_cvf_t *)va;
  extern unsigned long jx_pb[16];
  unsigned long _pbx = jx_pfcc();
  jx_pb[3]++;
  /* r94：first_block 的 “trend_pool -> conv1d(1->4, k=7, dil=1)” 专路。
   * 板端形状直方图里这是最差的一档：2.30 MMAC 却要 280 ms（29.2 周期/MAC），
   * 而同形状的 dense 路径是 1.0 周期/MAC。根因是原来的循环嵌套
   *     for (t) for (co) { for (kk=0..6) acc += x[tbase+kk]*w[co][kk] }
   * 每个 (t,co) 都把同一批 7 个 x 重新读一遍 —— x 的访存量是必需的 Co 倍。
   * 这里把 co 循环套进已经装进寄存器的 x[0..6] 里：每个 t 只读 7 个 x。
   * 7 条乘积的求和分组 ((a0+a1)+(a2+a3))+((a4+a5)+a6) 与原来完全一致，
   * 累加次序也一致 -> 逐位相同。只在 groups==1/cpg0==1/k==7/dil==1/stride==1
   * 且 xin 与 yout 不重叠时生效（重叠会改变语义，直接回落）。 */
  /* r141：先搬进本核 SRAM 再算。x 的读与 y 的写彻底分离，「x 与 y 是否
   * 重叠」不再影响结果，于是可以去掉了原来那条 !overlap 守卫 —— 板端探针
   * （jx_pb[5]）实测它在真实调用下**从未成立**，这个 2.30 MMAC 的形状一直
   * 落在 30 周期/MAC 的通用路径上（conv1d_core_fast 探针：34 次调用共
   * 73 Mcyc，平均 2.15 Mcyc/次）。
   * 窗口只有 JX_CONV_TB+6 个 float（2KB），拷贝成本相对 512x4x7 次乘加可忽略。
   * 数值：窗口内格子与原来直接从 xin 读完全相同；区域外的 tap 由「跳过」改为
   * 「加 0.0f * wr[kk]」，浮点上逐位等价。 */
  if (C0->k == 7 && C0->dil == 1 && C0->stride == 1 && C0->cpg0 == 1 &&
      C0->copg0 == (C0->co1 - C0->co0))
    {
      const float *W = C0->W; const float *bb = C0->b;
      const float *xin = C0->xin; float *yout = C0->yout;
      jx_pb[5]++;
      const int pad = C0->pad, T_in = C0->T_in, T_out = C0->T_out;
      const int tlo = C0->tlo, thi = C0->thi, co0 = C0->co0, co1 = C0->co1;
      static float *g_cvxb[JX_NCORE];
      static int    g_cvxn[JX_NCORE];
      const int me   = jx_core_id();
      const int need = JX_CONV_TB + 8;
      if (g_cvxn[me] < need)
        {
          float *nd = (float *)realloc(g_cvxb[me], sizeof(float) * (size_t)need);
          if (nd != NULL) { g_cvxb[me] = nd; g_cvxn[me] = need; }
        }
      float *xb = g_cvxb[me];
      if (xb == NULL) { jx_pb[11]++; }
      else
      for (int _blk = blo; _blk < bhi; _blk++)
        {
          int t0 = _blk * JX_CONV_TB;
          int t1 = t0 + JX_CONV_TB; if (t1 > T_out) { t1 = T_out; }
          int ws = t1 - t0 + 6;
          int s0 = t0 - pad;
          int o_lo = (s0 < 0) ? -s0 : 0;
          int o_hi = (s0 + ws > T_in) ? (T_in - s0) : ws;
          if (o_lo < 0) { o_lo = 0; }
          if (o_hi > ws) { o_hi = ws; }
          for (int z = 0; z < o_lo; z++)     { xb[z] = 0.0f; }
          for (int z = o_lo; z < o_hi; z++)  { xb[z] = xin[s0 + z]; }
          for (int z = o_hi; z < ws; z++)    { xb[z] = 0.0f; }
          for (int t = t0; t < t1; t++)
            {
              const float *xr = xb + (t - t0);
              if (t >= tlo && t < thi)
                {
                  float x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
                  float x4 = xr[4], x5 = xr[5], x6 = xr[6];
                  float *yd = yout + t;
                  for (int co = co0; co < co1; co++)
                    {
                      const float *wr = W + (size_t)co * 7;
                      acc_t acc = (bb != NULL) ? (acc_t)bb[co] : (acc_t)0;
                      acc_t a0 = (acc_t)x0 * (acc_t)wr[0];
                      acc_t a1 = (acc_t)x1 * (acc_t)wr[1];
                      acc_t a2 = (acc_t)x2 * (acc_t)wr[2];
                      acc_t a3 = (acc_t)x3 * (acc_t)wr[3];
                      acc_t a4 = (acc_t)x4 * (acc_t)wr[4];
                      acc_t a5 = (acc_t)x5 * (acc_t)wr[5];
                      acc_t a6 = (acc_t)x6 * (acc_t)wr[6];
                      acc += ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + a6);
                      yd[(size_t)co * T_out] = (float)acc;
                    }
                }
              else
                {
                  for (int co = co0; co < co1; co++)
                    {
                      const float *wr = W + (size_t)co * 7;
                      acc_t acc = (bb != NULL) ? (acc_t)bb[co] : (acc_t)0;
                      for (int kk = 0; kk < 7; kk++)
                        { acc += (acc_t)xr[kk] * (acc_t)wr[kk]; }
                      yout[(size_t)co * T_out + t] = (float)acc;
                    }
                }
            }
        }
      { jx_pb[4] += (unsigned long)(unsigned)(jx_pfcc() - _pbx); }
      return;
    }
  const float *W = C0->W; const float *b = C0->b;
  const float *xin = C0->xin; float *yout = C0->yout;
  const int cpg0 = C0->cpg0, copg0 = C0->copg0, k = C0->k;
  const int stride = C0->stride, pad = C0->pad, dil = C0->dil;
  const int T_in = C0->T_in, T_out = C0->T_out;
  const int fast_div = C0->fast_div, cshift = C0->cshift;
  const int tlo = C0->tlo, thi = C0->thi;
  const int co0 = C0->co0, co1 = C0->co1;
  for (int _blk = blo; _blk < bhi; _blk++) {
      int t0 = _blk * JX_CONV_TB;
        int t1 = t0 + JX_CONV_TB; if (t1 > T_out) { t1 = T_out; }
        for (int t = t0; t < t1; t++) {
            int tbase = t * stride - pad;
            int full  = (t >= tlo && t < thi);
            for (int co = co0; co < co1; co++) {
                int g = fast_div ? (co >> cshift) : (co / copg0);
                acc_t acc = (b ? (acc_t)b[co] : (acc_t)0);
                int base_ci = g * cpg0;
                if (full) {
                    if (k == 7 && dil == 1) {
                        /* k 是运行期变量时内层无法展开；k=7/dil=1 是 fp32 回落
                         * 形状里最大的一档（Ci=1 Co=4、Ci=16 Co=1，合计约
                         * 330 ms、24 周期/MAC）。这里手工展开成 7 条独立的
                         * `acc += x*w`，累加次序与循环版完全相同 -> 逐位一致。 */
                        for (int ci = 0; ci < cpg0; ci++) {
                            const float *wr = &W[((size_t)co * cpg0 + ci) * 7];
                            const float *xr = &xin[(size_t)(base_ci + ci) * T_in + tbase];
                            /* 第四十二轮：7 条 acc += 全部串在同一个 acc 上，
                             * 是一条完全串行的 madd 链（延迟 ~4 周期/条）。
                             * 这两个形状（Ci=1 Co=4、Ci=16 Co=1）实测 23.3 周期/MAC，
                             * 是全场离 FPU 地板最远的一档。拆成 7 条独立乘积
                             * 再一次合并到 acc，数值只有浮点求和分组的差别
                             * （相对 ~1e-7，远低于下游 int8 量化噪声 1e-2）。 */
                            acc_t a0 = (acc_t)xr[0] * (acc_t)wr[0];
                            acc_t a1 = (acc_t)xr[1] * (acc_t)wr[1];
                            acc_t a2 = (acc_t)xr[2] * (acc_t)wr[2];
                            acc_t a3 = (acc_t)xr[3] * (acc_t)wr[3];
                            acc_t a4 = (acc_t)xr[4] * (acc_t)wr[4];
                            acc_t a5 = (acc_t)xr[5] * (acc_t)wr[5];
                            acc_t a6 = (acc_t)xr[6] * (acc_t)wr[6];
                            acc += ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + a6);
                        }
                    } else {
                        for (int ci = 0; ci < cpg0; ci++) {
                            const float *wr = &W[((size_t)co * cpg0 + ci) * k];
                            const float *xr = &xin[(size_t)(base_ci + ci) * T_in + tbase];
                            for (int kk = 0; kk < k; kk++) acc += (acc_t)xr[kk * dil] * (acc_t)wr[kk];
                        }
                    }
                } else {
                    for (int ci = 0; ci < cpg0; ci++) {
                        const float *wr = &W[((size_t)co * cpg0 + ci) * k];
                        const float *xr = &xin[(size_t)(base_ci + ci) * T_in];
                        for (int kk = 0; kk < k; kk++) {
                            int pos = tbase + kk * dil;
                            if (pos >= 0 && pos < T_in) acc += (acc_t)xr[pos] * (acc_t)wr[kk];
                        }
                    }
                }
                yout[(size_t)co * T_out + t] = (float)acc;
            }
        }
        }
  jx_pb[4] += (unsigned long)(unsigned)(jx_pfcc() - _pbx);
}

static void conv1d_core_fast(const float *W, const float *b, const float *xin, float *yout,
                             int Ci, int Co, int k, int stride, int pad, int groups,
                             int T_in, int T_out, int dil) {
    int cpg0  = Ci / groups;
    int copg0 = Co / groups;
    /* 2026-09-15 第三十四轮：fast 路径的守卫保证 copg0 在 1..4，这里把
     * `co / copg0` 换成移位/常数（1/2/4 是 2 的幂，3 才真除）。原来它在
     * t 循环内逐 (t,co) 求值，Xtensa 上整数除法是软件例程（约 30 周期）。
     * 结果不变：只是同一表达式的求值方式变了。 */
    const int fast_div = (copg0 == 1 || copg0 == 2 || copg0 == 4);
    const int cshift   = (copg0 == 4) ? 2 : (copg0 == 2) ? 1 : 0;
    int cb = (jx_block_bytes("JX_CB", JX_CONV_CB_BYTES) / 4) / (cpg0 * k > 0 ? cpg0 * k : 1);
    if (cb < 1)  { cb = 1; }
    if (cb > Co) { cb = Co; }
    int tlo = (pad > 0) ? pad : 0;
    int thi = T_in - (k - 1) * dil + pad;
    if (thi > T_out) { thi = T_out; }
    if (thi < tlo)   { thi = tlo; }
    for (int co0 = 0; co0 < Co; co0 += cb) {
        int co1 = co0 + cb; if (co1 > Co) { co1 = Co; }
        /* 2026-09-15 第三十四轮：t 轴再分块。
         * 板端微基准：PSRAM 4KB 跨步单次访问要 146.5 周期（每次都吃满延迟）。
         * 本函数同时驻留 1 条 x 行 + copg0 条 y 行，`Ci=1 Co=4 k=7` 时每条行长
         * 8460 B ≈ 264 条 cache line，而 L1 只有 256 个 set、4 路组相联 ——
         * 5 条流各扫全部 set，互相驱逐，每次都变成跨步访问。
         * 按 512 个输出位置分块后，一个块内 x 只需 2 KB、y 只需 copg0*2 KB，
         * 全部常驻 L1，缺页只发生在块边界。
         * 只改 t 的遍历次序，每个输出元素仍独立计算 -> 逐位一致。 */
        jx_cvf_t jc;
        jc.W = W; jc.b = b; jc.xin = xin; jc.yout = yout;
        jc.cpg0 = cpg0; jc.copg0 = copg0; jc.k = k; jc.stride = stride;
        jc.pad = pad; jc.dil = dil; jc.T_in = T_in; jc.T_out = T_out;
        jc.fast_div = fast_div; jc.cshift = cshift;
        jc.tlo = tlo; jc.thi = thi; jc.co0 = co0; jc.co1 = co1;
        { int _nb = (T_out + JX_CONV_TB - 1) / JX_CONV_TB;
          jx_pf_run_min(_nb, jx_cvf_tblocks, &jc, 2); }
    }
}

static void conv1d_core(const float *W, const float *b, const float *xin, float *yout,
                        int Ci, int Co, int k, int stride, int pad, int groups, int T_in, int T_out, int dil) {
    /* 寄存器累加内核 + 输出通道分块。
     *
     * 每个输出样点仍用寄存器累加（读 x、读 w、madd，约 3 条指令/MAC）。
     * 外层把 co 按 JX_CONV_CB_BYTES 分块、块内改成 t 外层、co 内层：
     * 同一个 t 下所有 co 复用同一批 x 行（cpg 行，块内全部命中 L1），
     * x 的访存量因此降为原来的 1/块宽。
     * 每个输出元素的求和次序仍是 ci 外层、kk 内层，结果逐位不变。 */
    int cpg0  = Ci / groups;
    int copg0 = Co / groups;
    if (jx_cvs() && stride == 1 && cpg0 >= 1 && copg0 >= 1 && k >= 1 && dil >= 1 && T_out > 0) {
        conv1d_core_fast(W, b, xin, yout, Ci, Co, k, stride, pad, groups, T_in, T_out, dil);
        return;
    }
    int cb = (jx_block_bytes("JX_CB", JX_CONV_CB_BYTES) / 4) / (cpg0 * k > 0 ? cpg0 * k : 1);
    if (cb < 1)  { cb = 1; }
    if (cb > Co) { cb = Co; }
    for (int co0 = 0; co0 < Co; co0 += cb) {
        int co1 = co0 + cb; if (co1 > Co) { co1 = Co; }
        for (int t = 0; t < T_out; t++) {
            int tbase = t * stride - pad;
            for (int co = co0; co < co1; co++) {
                int g = co / copg0;
                acc_t acc = (b ? (acc_t)b[co] : (acc_t)0);
                int base_ci = g * cpg0;
                for (int ci = 0; ci < cpg0; ci++) {
                    const float *wr = &W[((size_t)co * cpg0 + ci) * k];
                    const float *xr = &xin[(size_t)(base_ci + ci) * T_in];
                    for (int kk = 0; kk < k; kk++) {
                        int pos = tbase + kk * dil;
                        if (pos >= 0 && pos < T_in) acc += (acc_t)xr[pos] * (acc_t)wr[kk];
                    }
                }
                yout[(size_t)co * T_out + t] = (float)acc;
            }
        }
    }
}

/* 流式版内层：x 沿时间顺序读、y 在 L1 内分块累加，但每个 MAC 是
 * 读 y + 读 x + madd + 写 y（6 条指令/MAC）；原版是 3 条指令/MAC
 * 且 x 跨通道行常驻 L1，因此大 T 时反而更快。
 * 两种内核累加次序相同（ci 外层、kk 内层）。
 *
 * 上板实测（ESP32-S3, 3k, 1 秒音频，同一会话内背靠背对照）：
 *   原版内核：  编码 30740ms 解码 34790ms 合计 65530ms (RTF 65.5)
 *   流式内核：  编码 30670ms 解码 36430ms 合计 67100ms (RTF 67.1)
 * 原版略优，且默认不做运行期分支可以让生成码与未改动时一致（保持与
 * PC 参考逐字节可比），因此默认编原版；需要做实验时加 -DJX_CONV_STREAM。 */
#ifdef JX_CONV_STREAM
static void conv1d_core_stream(const float *W, const float *b, const float *xin, float *yout,
                               int Ci, int Co, int k, int stride, int pad, int groups, int T_in, int T_out, int dil) {
    int cpg = Ci / groups;
    int copg = Co / groups;
    /* 流式循环次序 co -> ci -> kk -> t：x 沿时间顺序读，避免逐个输出样点
     * 跨通道跳着取 x。累加次序与原版一致（固定 t 时仍是 ci 外层、kk 内层），
     * 差异只来自编译器 FMA 收缩的选择，实测 1740 token 中 1 个 token 的
     * 1 个 FSQ 维度落在相邻量化中心（舍入级差异，不影响听感）。 */
    for (int co = 0; co < Co; co++) {
        int g = co / copg;
        float *yp = &yout[(size_t)co * T_out];
        if (b != NULL) { float bb = b[co]; for (int t = 0; t < T_out; t++) yp[t] = bb; }
        else           { memset(yp, 0, (size_t)T_out * sizeof(float)); }
        int base_ci = g * cpg;
        for (int ci = 0; ci < cpg; ci++) {
            const float *xp = &xin[(size_t)(base_ci + ci) * T_in];
            const float *wr = &W[((size_t)co * cpg + ci) * k];
            if (stride == 1) {
                for (int kk = 0; kk < k; kk++) {
                    float wv = wr[kk];
                    int off = kk * dil - pad;
                    int ts = -off; if (ts < 0) ts = 0;
                    int te = T_in - off; if (te > T_out) te = T_out;
                    if (te <= ts) continue;
                    /* 再把 t 轴切成 JX_CONV_TB 大小的块：块内 y 只有 TB*4 字节，
                     * 能常驻 L1。大 T（解码器尾部 T=16032）时，不切块会让 y
                     * 每个 MAC 都读-改-写一次 PSRAM，实测解码因此慢 29%。
                     * 切块不改变累加顺序，结果逐位一致。 */
                    for (int t0 = ts; t0 < te; t0 += JX_CONV_TB) {
                        int t1 = t0 + JX_CONV_TB; if (t1 > te) t1 = te;
                        const float *xs = xp + off + t0;
                        for (int t = t0; t < t1; t++) yp[t] += *xs++ * wv;
                    }
                }
            } else {
                for (int kk = 0; kk < k; kk++) {
                    float wv = wr[kk];
                    int pos = kk * dil - pad;
                    for (int t = 0; t < T_out; t++, pos += stride)
                        if (pos >= 0 && pos < T_in) yp[t] += xp[pos] * wv;
                }
            }
        }
    }
}
#endif /* JX_CONV_STREAM */

/* 内核选择：默认直接调原版（无额外间接层，生成码与改动前一致）；
 * 定义 JX_CONV_STREAM 则改用时间分块的流式内核。 */
#ifdef JX_CONV_STREAM
#  define conv1d_core_pick conv1d_core_stream
#else
#  define conv1d_core_pick conv1d_core
#endif

void conv1d(const float *W, const float *b, const T *x, T *y, int Ci, int Co, int k, int stride, int pad, int groups, int T_in) {
    JXP_BEG(JP_CONV1D);
    int T_out = (T_in + 2 * pad - k) / stride + 1;
    jx_mac_conv += (double)x->shape[0] * Co * (Ci / groups) * k * T_out; jx_n_conv++;
#ifdef JX_TRACE_MACS
    printf("[MAC] conv1d  B=%d Ci=%-4d Co=%-4d k=%d s=%d g=%-3d T=%d->%d  mac=%.3fM\n",
           x->shape[0], Ci, Co, k, stride, groups, T_in, T_out,
           (double)x->shape[0]*Co*(Ci/groups)*k*T_out/1e6);
#endif
    int need = x->shape[0] * Co * T_out;
    if (y->len < need)
      {
        float *nd = (float*)jx_arealloc(y->d, (size_t)need * sizeof(float));
        /* 安全网（本轮新增）：堆只剩 ~0.6MB，realloc 一旦失败会把
         * y->d 置为 NULL，后面的 i8 内核就直接往零页写而 PANIC
         * （StoreProhibited VADDR=0）。宁可打印并跳过，也不要死机。 */
        if (nd == NULL)
          {
            jx_conv_skip++;
            printf("[CONV] realloc %d floats 失败，跳过本次卷积\n", need);
            JXP_END(JP_CONV1D);
            return;
          }
        y->d = nd;
      }
    y->shape[0] = x->shape[0]; y->shape[1] = Co; y->shape[2] = T_out; y->len = need;
    JX_CT_BEG(T_in, T_out);
    if (jx_conv1d_i8(W, b, x, y, Ci, Co, k, stride, pad, groups, T_in, T_out, 1) == 0)
      {
        JX_CT_END(Ci, Co, k, groups, 1,
                  (double)x->shape[0] * Co * (Ci / groups) * k * T_out);
        JXP_END(JP_CONV1D);
        return;
      }
    int B = x->shape[0];
    for (int bi = 0; bi < B; bi++)
      conv1d_core_pick(W, b, &x->d[bi * Ci * T_in], &y->d[bi * Co * T_out], Ci, Co, k, stride, pad, groups, T_in, T_out, 1);
    JX_CT_END(Ci, Co, k, groups, 1,
              (double)x->shape[0] * Co * (Ci / groups) * k * T_out);
    JXP_END(JP_CONV1D);
}
void conv1d_d(const float *W, const float *b, const T *x, T *y, int Ci, int Co, int k, int stride, int pad, int groups, int T_in, int dil) {
    JXP_BEG(JP_CONV1D);
    int T_out = (T_in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    jx_mac_conv += (double)x->shape[0] * Co * (Ci / groups) * k * T_out; jx_n_conv++;
#ifdef JX_TRACE_MACS
    printf("[MAC] conv1d_d B=%d Ci=%-4d Co=%-4d k=%d dil=%d g=%-3d T=%d->%d  mac=%.3fM\n",
           x->shape[0], Ci, Co, k, dil, groups, T_in, T_out,
           (double)x->shape[0]*Co*(Ci/groups)*k*T_out/1e6);
#endif
    int need = x->shape[0] * Co * T_out;
    if (y->len < need)
      {
        float *nd = (float*)jx_arealloc(y->d, (size_t)need * sizeof(float));
        /* 安全网（本轮新增）：堆只剩 ~0.6MB，realloc 一旦失败会把
         * y->d 置为 NULL，后面的 i8 内核就直接往零页写而 PANIC
         * （StoreProhibited VADDR=0）。宁可打印并跳过，也不要死机。 */
        if (nd == NULL)
          {
            jx_conv_skip++;
            printf("[CONV] realloc %d floats 失败，跳过本次卷积\n", need);
            JXP_END(JP_CONV1D);
            return;
          }
        y->d = nd;
      }
    y->shape[0] = x->shape[0]; y->shape[1] = Co; y->shape[2] = T_out; y->len = need;
    JX_CT_BEG(T_in, T_out);
    if (jx_conv1d_i8(W, b, x, y, Ci, Co, k, stride, pad, groups, T_in, T_out, dil) == 0)
      {
        JX_CT_END(Ci, Co, k, groups, dil,
                  (double)x->shape[0] * Co * (Ci / groups) * k * T_out);
        JXP_END(JP_CONV1D);
        return;
      }
    int B = x->shape[0];
    for (int bi = 0; bi < B; bi++)
      conv1d_core_pick(W, b, &x->d[bi * Ci * T_in], &y->d[bi * Co * T_out], Ci, Co, k, stride, pad, groups, T_in, T_out, dil);
    JX_CT_END(Ci, Co, k, groups, dil,
              (double)x->shape[0] * Co * (Ci / groups) * k * T_out);
    JXP_END(JP_CONV1D);
}
/* 2026-09-14 第三十三轮：conv1d + GELU 融合入口。
 * 先在 int8 快路径上跑（epilogue 里直接算 gelu），如果不满足 i8 条件而退回
 * 浮点，再自己补一遍逐元素 gelu。两条路的数值完全一致。 */
static int jx_gelfuse(void)
{
  static int v = -2;
  if (v == -2) { const char *e = getenv("JX_GELFUSE"); v = !(e != NULL && e[0] == (char)48); }
  return v;
}

/* r142 ??: JX_POSTOFF=1 -> epilogue ????????????????????
 * ??????????????? gelu/snake ?????????????????? */
static int jx_postoff(void)
{
  static int v = -1;
  if (v < 0) { const char *e = getenv("JX_POSTOFF"); v = (e != NULL && e[0] == (char)49) ? 1 : 0; }
  return v;
}

void conv1d_gelu(const float *W, const float *b, const T *x, T *y,
                 int Ci, int Co, int k, int stride, int pad, int groups, int T_in) {
    if (!jx_gelfuse())
      {
        conv1d(W, b, x, y, Ci, Co, k, stride, pad, groups, T_in);
        jx_conv_post_used = 0;
        jx_kt_ensure();
        { const jx_erf9 KE = jx_erf9_load();
          for (int i = 0; i < y->len; i++) { y->d[i] = jx_gelu_fast(y->d[i], KE); } }
        return;
      }
    JXP_BEG(JP_GELU);
    jx_conv_post_used = 0;
    jx_conv_post = jx_postoff() ? 0 : 1;
    conv1d(W, b, x, y, Ci, Co, k, stride, pad, groups, T_in);
    jx_conv_post = 0;
    if (jx_postoff()) { jx_conv_post_used = 1; }
    if (jx_conv_post_used != 1 && y->d != NULL)
      {
        jx_kt_ensure();
        { const jx_erf9 KE = jx_erf9_load();
          for (int i = 0; i < y->len; i++) { y->d[i] = jx_gelu_fast(y->d[i], KE); } }
      }
    JXP_END(JP_GELU);
}

/* 2026-09-14 第三十三轮：snake 相关融合（见 nn_ops.h）。 */
/* r68: split (b,c) channel rows across the two cores. */
typedef struct { const T *src; T *dst; const float *alpha; int C, Tt; } jx_snkcp_t;

JX_HOT static void jx_snake1d_cp_range(void *va, int klo, int khi)
{
    jx_snkcp_t *cc = (jx_snkcp_t *)va;
    const jx_sq4 K = jx_sq4_load();
    const int C = cc->C, Tt = cc->Tt;
    for (int kk = klo; kk < khi; kk++)
      {
        if (kk + 1 < khi)
          { jx_dcp_pf_ew(&cc->src->d[(size_t)(kk + 1) * Tt], (unsigned)Tt * 4u); }
        const int b = kk / C, c = kk - b * C;
        float a = cc->alpha[c] + XNN_EPS; float inv = 1.0f / a;
        const float *s = &cc->src->d[((size_t)b * C + c) * Tt];
        float *p = &cc->dst->d[((size_t)b * C + c) * Tt];
        int i = 0;
        for (; i + 4 <= Tt; i += 4) {
            float v0 = s[i], v1 = s[i + 1], v2 = s[i + 2], v3 = s[i + 3];
            p[i]     = v0 + inv * jx_sinsq4(a * v0, K);
            p[i + 1] = v1 + inv * jx_sinsq4(a * v1, K);
            p[i + 2] = v2 + inv * jx_sinsq4(a * v2, K);
            p[i + 3] = v3 + inv * jx_sinsq4(a * v3, K);
        }
        for (; i < Tt; i++) { float v = s[i]; p[i] = v + inv * jx_sinsq4(a * v, K); }
      }
}

void snake1d_copy(T *dst, const T *src, const float *alpha) {
    jx_sq_kt_ensure();
    const int _dbg2 = jx_snkdbg();
    const unsigned _c0b = _dbg2 ? jx_snk_cc() : 0u;
    const long _lenb = (long)src->len;
    JXP_BEG(JP_SNAKE);
    int B = src->shape[0], C = src->shape[1], Tt = src->shape[2];
    jx_elems_snake += (double)src->len;
    {
        jx_snkcp_t sc; sc.src = src; sc.dst = dst; sc.alpha = alpha; sc.C = C; sc.Tt = Tt;
        jx_pf_run_min(B * C, jx_snake1d_cp_range, &sc, 2);
    }
    dst->ndim = 3; dst->shape[0] = B; dst->shape[1] = C; dst->shape[2] = Tt;
    dst->len = B * C * Tt;
    if (_dbg2)
      {
        unsigned _c1 = jx_snk_cc();
        long _wk = (long)B * C * Tt;
        printf("  [snkc] s=[%d,%d,%d] len=%ld work=%ld cyc=%lu  %.1f cyc/len  %.1f cyc/work\n",
               B, C, Tt, _lenb, _wk, (unsigned long)(_c1 - _c0b),
               (double)(_c1 - _c0b) / (double)(_lenb > 0 ? _lenb : 1),
               (double)(_c1 - _c0b) / (double)(_wk > 0 ? _wk : 1));
        fflush(stdout);
      }
    JXP_END(JP_SNAKE);
}

static int jx_snakefuse(void)
{
  static int v = -2;
  if (v == -2) { const char *e = getenv("JX_SNAKEFUSE"); v = !(e != NULL && e[0] == (char)48); }
  return v;
}

void conv1d_d_snake(const float *W, const float *b, const T *x, T *y,
                    int Ci, int Co, int k, int stride, int pad, int groups,
                    int T_in, int dil, const float *alpha) {
    if (alpha == NULL) { conv1d_d(W, b, x, y, Ci, Co, k, stride, pad, groups, T_in, dil); return; }
    if (!jx_snakefuse())
      {
        conv1d_d(W, b, x, y, Ci, Co, k, stride, pad, groups, T_in, dil);
        jx_conv_post_used = 0;
        snake1d(y, alpha, y->shape[0], y->shape[2], y->shape[1], 0);
        return;
      }
    /* r68: this bracket used to wrap the whole conv1d_d as well, inflating
     * JP_SNAKE (500ms reported of which 372ms was really the convolution,
     * already counted in JP_CONV1D). Dropped here; a real fallback snake1d()
     * still accounts itself. */
    jx_conv_post_used = 0;
    jx_post_snake_prep(alpha, Co);
    jx_conv_post_alpha = alpha;
    jx_conv_post = jx_postoff() ? 0 : 2;
    conv1d_d(W, b, x, y, Ci, Co, k, stride, pad, groups, T_in, dil);
    jx_conv_post = 0;
    jx_conv_post_alpha = NULL;
    if (jx_postoff()) { jx_conv_post_used = 2; }
    if (jx_conv_post_used != 2 && y->d != NULL)
      {
        snake1d(y, alpha, y->shape[0], y->shape[2], y->shape[1], 0);
      }
}

void conv1d_dw(const float *W, const float *b, const T *x, T *y, int C, int k, int stride, int pad, int T_in) {
    conv1d(W, b, x, y, C, C, k, stride, pad, C, T_in);
}
/* ⚠️ 上板适配（2026-09-12）：xtensa-esp32s3-elf GCC 在 -O2 下对本函数
 * 触发内部编译器错误（postreload / extract_constrain_insn，"insn does not
 * satisfy its constraints"）。本函数不是热点（每输出仅 2 次乘加，相对卷积
 * 可忽略），单独降到 -O1 规避；其余 kernel 保持 -O2 不受影响。 */
/* r68: split (bi,c) rows across the two cores; rows are independent. */
typedef struct { const float *xd; float *yd; int C, Tin, Tout, scale;
                 const int *itab; const float *ftab; } jx_ups_t;

__attribute__((optimize("O1")))
JX_HOT static void jx_ups_rows(void *va, int klo, int khi)
{
    jx_ups_t *cc = (jx_ups_t *)va;
    const int C = cc->C, Tin = cc->Tin, Tout = cc->Tout, scale = cc->scale;
    const int *itab = cc->itab; const float *ftab = cc->ftab;
    for (int kk = klo; kk < khi; kk++)
      {
        if (kk + 1 < khi)
          { jx_dcp_pf_ew(&cc->xd[(size_t)(kk + 1) * Tin], (unsigned)Tin * 4u); }
        const int bi = kk / C, c = kk - bi * C;
        const float *xp = &cc->xd[((size_t)bi * C + c) * Tin];
        float       *yp = &cc->yd[((size_t)bi * C + c) * Tout];
            int off = 0;
            for (int o = 0; o < Tout; o += scale, off++) {
                for (int k = 0; k < scale; k++) {
                    int i0 = itab[k] + off;
                    int i1 = i0 + 1;
                    float f = ftab[k];
                    float v0 = (i0 < 0) ? xp[0] : (i0 >= Tin ? xp[Tin - 1] : xp[i0]);
                    float v1 = (i1 < 0) ? xp[0] : (i1 >= Tin ? xp[Tin - 1] : xp[i1]);
                    yp[o + k] = v0 * (1.0f - f) + v1 * f;
                }
            }
      }
}

__attribute__((optimize("O1")))
void upsampling_linear(const T *x, T *y, int B, int C, int Tin, int scale) {
    JXP_BEG(JP_UPSAMPLE);
    int Tout = Tin * scale;
    int need = B * C * Tout;
    if (y->len < need)
      {
        /* 防呆（2026-09-14）：realloc 失败时不能把 y->d 置 NULL —— 下面的写循环
         * 会直接写零页，报 StoreProhibited 而不是一条可读的错误。这里改成计数 + 打印，
         * 并把张量标成空（len=0），让最终报告能看出本次结果无效。 */
        float *nd = (float*)jx_arealloc(y->d, (size_t)need * sizeof(float));
        if (nd == NULL)
          {
            jx_conv_skip++;
            printf("[UP] realloc %d floats 失败，跳过本次上采样\n", need);
            y->shape[0] = B; y->shape[1] = C; y->shape[2] = 0; y->len = 0;
            JXP_END(JP_UPSAMPLE);
            return;
          }
        y->d = nd;
      }
    /* 上板优化（2026-09-12 #2，板端 2180 ms -> 待测）
     *
     * 原实现对**每一个输出样点**做一次 double 乘加求源坐标：
     *     double src = (double)o * step + bias;
     * ESP32-S3 没有 double FPU，这里的 double 乘/加/比较以及 int<->double 转换
     * 全部是 libgcc 软浮点调用，于是算子耗时几乎全在 libgcc 里。
     *
     * 关键观察：本函数的 Tout 恒等于 Tin*scale（scale 为整数倍率），因此
     *     src(o) = (o + 0.5)/scale - 0.5 = (2o + 1 - scale) / (2*scale)
     * 是一个**精确有理数**；并且 src(o + scale) = src(o) + 1，
     * 即 (i0, frac) 以 scale 为周期。所以每个 call 只需用纯整数算 scale 组
     * 系数（scale 最大只是 4），内层循环退化成 2 次读 + 2 次乘加 + 1 次写。
     * 数值上与原来的 double 版只差 frac 的 1 ulp（<1e-7），远小于音频量化步长。 */
    if (scale >= 1 && scale <= 32) {
        const int den = 2 * scale;
        int   itab[32];
        float ftab[32];
        for (int k = 0; k < scale; k++) {
            int num = 2 * k + 1 - scale;
            int i0  = num / den;
            int rem = num - i0 * den;
            if (rem < 0) { rem += den; i0 -= 1; }
            itab[k] = i0;
            ftab[k] = (float)rem / (float)den;
        }
        {
            jx_ups_t uc;
            uc.xd = x->d; uc.yd = y->d; uc.C = C; uc.Tin = Tin; uc.Tout = Tout;
            uc.scale = scale; uc.itab = itab; uc.ftab = ftab;
            jx_pf_run_min(B * C, jx_ups_rows, &uc, 4);
        }
    } else {
        /* 非整数倍率（本模型不会走到）：退回原来的逐点 double 实现 */
        const double step = (double)Tin / (double)Tout;
        const double bias = 0.5 * step - 0.5;
        for (int bi = 0; bi < B; bi++) for (int c = 0; c < C; c++) for (int o = 0; o < Tout; o++) {
            double src = (Tout > 1) ? ((double)o * step + bias) : 0.0;
            int i0 = (int)src; double frac = src - (double)i0;
            if (frac < 0.0) { frac += 1.0; i0 -= 1; }
            int i1 = i0 + 1;
            float v0 = (i0 < 0) ? x->d[(bi * C + c) * Tin + 0] : (i0 >= Tin ? x->d[(bi * C + c) * Tin + (Tin - 1)] : x->d[(bi * C + c) * Tin + i0]);
            float v1 = (i1 < 0) ? x->d[(bi * C + c) * Tin + 0] : (i1 >= Tin ? x->d[(bi * C + c) * Tin + (Tin - 1)] : x->d[(bi * C + c) * Tin + i1]);
            y->d[(bi * C + c) * Tout + o] = v0 * (float)(1.0 - frac) + v1 * (float)frac;
        }
    }
    y->shape[0] = B; y->shape[1] = C; y->shape[2] = Tout; y->len = B * C * Tout;
    JXP_END(JP_UPSAMPLE);
}
