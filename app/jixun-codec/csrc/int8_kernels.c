/* ============================================================================
 * int8_kernels.c — int8 量化 + ESP32-S3 PIE 128-bit 向量指令的算子内核。
 *
 * 背景（板端实测，2026-09-12）：
 *   FP32 FPU 上限        58.9 MMAC/s（4.08 周期/MAC，madd.s 不流水）
 *   模型需求             424 MMAC / 音频秒  -> FP32 下限 RTF >= 7.2
 *   PIE int8（纯寄存器） 1219 MMAC/s（20 倍）
 *   PIE int8 + 128b 装载 985 MMAC/s（[8] 微基准，就是本文件的点积形态）
 *   注意：PIE 需要 CPENABLE 打开（默认只开了 CP0=FPU），见 jx_pie_enable()。
 *
 * 数值：int8×int8 -> int32 累加是**精确**的，所以 PIE 内核与可移植 C 内核
 *       逐位一致；主机上验证过的精度，板上必然相同。
 *
 * 量化：权重离线 per-output-channel 对称量化（weights_q8.h，flash 常驻，
 *       不占堆）；激活每行一个 scale（量化与 im2col 融合，无额外往返）。
 * ========================================================================== */

#include "nn_ops.h"
#include "jx_percore.h"
#include "jxprof.h"
#include "jx_dcp.h"        /* DCache 硬件预载 */
#include "jx_amax.h"       /* P0：逐位置最大幅值边车 */

/* ============================================================================
 * r81：量化取整的无分支写法。
 *
 * 反汇编（v194 固件，jx_q_row_auto 的 8 路量化循环）显示，原式
 *     jx_rndf(v)
 * 每个元素要 5 条指令：ole.s(比较) + bt(条件跳转) + l32r(常量池取 -0.5f)
 * + wfr(写进 FP 寄存器) + add.s —— 而且那个条件跳转在真实数据上
 * 正负各半，完全不可预测。整段量化循环实测 ~9 条指令/元素。
 *
 * 下面用"符号位直接拼 ±0.5f"：and/or/wfr/add.s 四步，零分支。
 * 加进去的仍然是 IEEE 的单精度 ±0.5f（位模式与原来一字不差），
 * 所以结果与改动前**逐位相同**（主机 .idx 门禁已验）。
 * ========================================================================== */
static inline int jx_rndf(float v)
{
#if defined(__XTENSA__) && !defined(JX_NO_RNDS)
  /* r94：Xtensa 的 FPU 有一条 `round.s aR, fS, 0` = 取整到最近整数（.5 远离零），
   * 与 (int)(v + copysign(0.5f,v)) 完全同义，但只需 1 条指令：
   * 省掉 2 次跨域搬运（rfr/wfr）+ and/or + add.s + trunc.s 共 5~6 条，
   * 而且不再占用整数寄存器做位拼接。量化路径全场约 7.7M 个元素受益。
   * 数值差异只可能出现在 v 的小数部分恰好为 0.5 的输入上（IEEE 默认 ties-to-even
   * 与 ties-away 的差别），相对 1 LSB，远低于 int8 量化噪声。JX_NO_RNDS 可回退。 */
  int r;
  __asm__ ("round.s %0, %1, 0" : "=r"(r) : "f"(v));
  return r;
#else
  unsigned u;
  float h;
  __builtin_memcpy(&u, &v, 4);
  u = 0x3f000000u | (u & 0x80000000u);   /* ±0.5f 的位模式 */
  __builtin_memcpy(&h, &u, 4);
  return (int)(v + h);
#endif
}
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* NuttX getenv() is not safe for concurrent calls from two cores.  PF runs
 * the same hot kernel on CPU0/CPU1, so serialize all environment lookups in
 * this translation unit with a short cross-core spinlock. */
static const char *jx_getenv_guarded(const char *name);
/* ---- r90：分段并行开关（同一份固件内 A/B）--------------------------------
 * r87_serial 证据：JX_PF=0（全串行）时 [i8mm] epilogue 49.61 cyc/元素，
 * 而 r89 并行下是 58.63 cyc/元素；[dt] 量化单核 54.9 Mcyc vs 双核 85.6 Mcyc。
 * 也就是说"访存受限的段开双核反而更慢"——两个核抢同一条 PSRAM 通路，
 * 再加同步开销。这里给三个最大的并行点各配一个开关，不新增任何逻辑分支
 * 到热循环里，只切换调用方式，保证数值逐位相同。
 *   JX_PFL : jx_linear_i8_rows（I8 matmul+epilogue，最大单项）
 *   JX_PFD : jx_conv1d_dense_tile_tb（dt dense tile）
 *   JX_PFK : jx_conv1d_k1_cf_tb（k1 channels-first）
 * 未设置或非 '0' 开头 = 开并行（与 r89 行为完全一致）。
 * getenv 每次读、不缓存，便于同一次 boot 内扫多组。 ---- */
static int jx_pfx(const char *nm)
{
  const char *e = jx_getenv_guarded(nm);
  return !(e != NULL && e[0] == (char)48);
}

/* ---- 诊断：把 JP_I8MM 拆成 matmul / epilogue 两段（JX_NO_PROFILE 下不编译）
 * 纯整数计数（rsr.ccount），热循环里不放任何 double 运算。 ---- */
unsigned long jx_cyc_mm = 0, jx_cyc_ep = 0;
unsigned long jx_cyc_q  = 0;
double jx_epi_elems = 0.0, jx_q_elems = 0.0;
double jx_fast_elems = 0.0, jx_slow_elems = 0.0;
unsigned long jx_slow_calls = 0, jx_fast_calls = 0;
unsigned long jx_dcol_calls = 0;
/* 2026-09-15 第三十六轮：jx_linear_i8_cf（conv_unit l2 的 channels-first 变体）
 * 之前**一个计数器都没有**，它在 JP_I8MM 桶里那 ~900 ms 只能靠猜。
 *   [0] 激活量化 prologue  [1] dotcol 点积批  [2] epilogue(acc->fp32)  [3] 转置落盘
 * 其中 [2] 是本轮的重点：它每元素只有 4 条 FP 指令（cvt + mul + mul + add）+
 * 4 次访存，是全场唯一能干净地反推「FPU 每指令要几个周期」的地方，
 * 用来和 [1b] 的纯寄存器微基准交叉验证。 */
unsigned long jx_cf_cyc[4];
double jx_cf_elems = 0.0;      /* epilogue 元素数 */
double jx_cf_in_elems = 0.0;   /* 量化 prologue 元素数 */
unsigned long jx_cf_rows = 0;
/* 2026-09-15 第三十六轮：pert（逐 t）路径的四个子段。
 * [k1] accx 分支总共 188.5 Mcyc / 105499 位置 = 1787 周期/位置，但里面
 * 混着 im2col 量化、点积、epilogue 三段完全不同的东西。微基准 [16] 表明
 * 点积本身只要 ~17 周期/输出元素，那么这 1787 里到底谁是大头必须先量出来。
 *   [0] Pass A 求逐位置 max   [1] Pass B 量化写 im2col
 *   [2] 点积批（jx_dotrow_i8）  [3] epilogue（反量化 + 激活 + 落盘） */
unsigned long jx_k1s_cyc[4];

/* ===== 第四十四轮：jx_conv1d_k1_cf 的子阶段周期拆分 =====
 * 目标：回答 "Ci=20 Co=80 k=1 的 680 ms 到底花在哪"。
 * 每个形状一个槽：[0]=PassA求max [1]=逐位置scale [2]=PassB量化+补零
 *                 [3]=点积 [4]=epilogue [5]=整块合计
 * 形状相同的多次调用累加到同一槽；JX_KC=0 可关掉插桩（对照用）。 */
unsigned long jx_kc_blk, jx_kc_pos, jx_kc_calls;
#define JX_KC_NS 24
int           jx_kc_n = 0;
int           jx_kc_sc5[JX_KC_NS][5];
unsigned long jx_kc_scy[JX_KC_NS][6];
unsigned long jx_kc_spos[JX_KC_NS], jx_kc_sblk[JX_KC_NS];
unsigned long jx_kc_sd[JX_KC_NS][3];   /* r44h: o==0 / o==1 / o>=2 dot cyc */
unsigned long jx_kc_sepi[JX_KC_NS][3];  /* r51b: epilogue 按 o 桶 (o==0/o==1/o>=2) */
int           jx_kc_spost[JX_KC_NS];    /* r51b: 该形状的 jx_conv_post */

static int jx_kc_slot(int Ci, int Co, int T_in, int T_out, int TB)
{
  jx_kc_calls++;
  for (int i = 0; i < jx_kc_n; i++)
    if (jx_kc_sc5[i][0] == Ci && jx_kc_sc5[i][1] == Co &&
        jx_kc_sc5[i][2] == T_in && jx_kc_sc5[i][3] == T_out &&
        jx_kc_sc5[i][4] == TB) { return i; }
  if (jx_kc_n >= JX_KC_NS) { return 0; }
  int i = jx_kc_n++;
  jx_kc_sc5[i][0] = Ci; jx_kc_sc5[i][1] = Co; jx_kc_sc5[i][2] = T_in;
  jx_kc_sc5[i][3] = T_out; jx_kc_sc5[i][4] = TB;
  for (int k = 0; k < 6; k++) { jx_kc_scy[i][k] = 0; }
  for (int k = 0; k < 3; k++) { jx_kc_sd[i][k] = 0; }
  for (int k = 0; k < 3; k++) { jx_kc_sepi[i][k] = 0; }
  jx_kc_spost[i] = -1;
  jx_kc_spos[i] = 0; jx_kc_sblk[i] = 0;
  return i;
}

double jx_k1s_qe = 0.0;    /* Pass A/B 各处理的 (ci,kk,j) 元素数 */
double jx_k1s_oe = 0.0;    /* 输出元素数 */
static inline unsigned jx_ccount(void)
{
  unsigned c;
  __asm__ __volatile__("rsr %0, ccount" : "=r"(c));
  return c;
}

#if defined(__has_include)
#  if __has_include("weights_q8.h")
#    include "weights_q8.h"
#    define JX_HAVE_WQ8 1
#  endif
#endif

/* ---------------- PIE 使能 ---------------- */
static inline unsigned jx_cp_get(void)
{
  unsigned v;
  __asm__ __volatile__("rsr %0, cpenable" : "=r"(v));
  return v;
}
static inline void jx_cp_set(unsigned v)
{
  __asm__ __volatile__("wsr %0, cpenable ; rsync" :: "r"(v) : "memory");
}

/* PIE 指令实测需要 CPENABLE 里除 CP0(FPU) 以外的位；出厂只开了 0x01。
 * 默认打开全部 8 位（实测可跑通且长期稳定），可用 JX_CP 覆盖。 */
/* ---- 卷积 epilogue 融合的后处理（2026-09-14 第三十三轮）---------------
 * 动机：first_block 的 conv_1（20->80, k=1）输出 h1（80 x Tt = 170K 个 float /
 * 次，24 次）后面紧跟一个纯逐元素 gelu 循环。这个循环把 4.5M 个元素
 * 「读一遍 + 写一遍」（约 36 MB PSRAM 往返），实测 1660 ms 里绝大部分是它。
 * 而 gelu 只是 conv 输出元素的一个纯函数 —— 直接在 epilogue 里算掉即可，
 * 数值与「先卷积再逐元素 gelu」逐位相同（同一个输入、同一个 gelu_f）。
 *
 * jx_conv_post : 调用方设置的期望后处理（0=无, 1=gelu）
 * jx_conv_post_used : 由 jx_conv1d_i8 在真正走了 int8 快路径时回填，
 *                     调用方据此判断要不要自己做浮点回退版的后处理。 */
int jx_conv_post = 0;
int jx_conv_post_used = 0;
const float *jx_conv_post_alpha = NULL;   /* post==2（snake）时的逐输出通道 alpha */

static inline __attribute__((always_inline)) float jx_post_f(float v)
{
  return (jx_conv_post == 1) ? gelu_f(v) : v;
}
/* 第三十九轮：系数由调用方持有（见 jx_fastmath.h 顶部）。
 * gelu 的 9 个系数在热 epilogue 里被 GCC 每个元素重载一次，
 * 这里改成循环前装载一次。算式与 gelu_f 逐字符相同 -> 逐位一致。 */
static inline __attribute__((always_inline)) float jx_post_f9(float v, jx_erf9 E)
{
  return (jx_conv_post == 1) ? jx_gelu_fast(v, E) : v;
}
/* snake 后处理：v + (1/a) * sin^2(a*v)，a = alpha[c] + eps，
 * 与 snake1d(data_format=0) 的元素内算式逐字符相同。 */
/* snake 后处理的系数表：sa = alpha[c]+eps，si = 1/sa。
 * 它们只跟输出通道有关，提前算好放在内部 SRAM —— 否则 epilogue 每个元素
 * 都要做一次浮点除法（板端实测：dense-tile 的 epi 从 587 周期/位置涨到
 * 2204 周期/位置）。 */
#define JX_POST_N 2048
static float g_post_sa[JX_POST_N];
static float g_post_si[JX_POST_N];
/* 第四十二轮：sin^2 的 4 个系数 + 归约常数在 jx_post_snake_prep 里一次性装好，
 * 热循环里不再读全局 jx_kt（原每元素 7 次 l32r+lsi）。 */
static jx_sq4 g_post_K;

JX_HOT void jx_post_snake_prep(const float *alpha, int n)
{
  jx_sq_kt_ensure();
  g_post_K = jx_sq4_load();
  if (n > JX_POST_N) { n = JX_POST_N; }
  for (int i = 0; i < n; i++)
    {
      float sa = alpha[i] + XNN_EPS;
      g_post_sa[i] = sa;
      g_post_si[i] = 1.0f / sa;
    }
}

static inline __attribute__((always_inline)) float jx_post_snake(float v, int c)
{
  float sa = g_post_sa[c];
  return v + g_post_si[c] * jx_sinsq_k(sa * v, g_post_K);
}

/* 同上：sin² 的 4 个系数由调用方持有。 */
static inline __attribute__((always_inline)) float jx_post_snake_k(float v, int c, jx_sq4 K)
{
  float sa = g_post_sa[c];
  return v + g_post_si[c] * jx_sinsq4(sa * v, K);
}

void jx_pie_enable(void)
{
#if defined(JX_PIE)
  /* 2026-09-13 第十二轮：原来用一个 static done 只做一次。
   * SMP 下 CPENABLE 随任务保存/恢复，而 NSH 每条命令都是新任务，
   * 于是第二条命令（bench）从未打开 PIE 位 -> ee.vmulas 报
   * EXCCAUSE=0x23（协处理器未使能）直接 panic。
   * 改成每次进 int8 内核前 rsr 查一次（几拍），缺位就补上。 */
  static unsigned want = 0;
  unsigned cur;

  if (want == 0)
    {
      const char *e = jx_getenv_guarded("JX_CP");
      want = (e != NULL) ? (unsigned)strtoul(e, NULL, 0) : 0xffu;
      if (want == 0) { want = 0xffu; }
    }

  cur = jx_cp_get();
  if ((cur & want) != want) { jx_cp_set(cur | want); }
#endif
}

#ifdef JX_HAVE_WQ8

/* 诊断用：记录本函数最终走的路径（0=fp32 退回 / 1=dense tile / 2=pre_q / 3=per_t） */
extern int jx_conv_path;

/* ---------------- 开关 ---------------- */
/* JX_INT8: 0=全关，1=只开 linear，2=只开 conv1d，3=全开（默认）。
 * 分组开关是排查数值问题的利器：能立刻定位是哪个内核的锅。 */
int jx_i8_enabled(void)
{
  /* 2026-09-13 第八轮：这里原来是 static 缓存，于是板端 `set JX_INT8 0`
   * 在第一次 bench 之后完全失效（A/B 对照是假的）。改成每次读 getenv： */
  /* 每次调用一次 getenv，整个 bench 只有几百次调用，开销可忽略。
   * 这样可以一块固件扫完所有 int8 开关组合。 */
  const char *e = jx_getenv_guarded("JX_INT8");
  int v = (e != NULL) ? atoi(e) : 3;
  if (v < 0) { v = 0; }
  if (v > 3) { v = 3; }
  return v;
}

/* 融合 epilogue 里 jx_sinsq() 的累计调用次数（板端 profile 用）。
 * 定义放在 nn_ops.c（始终编译），这里只用 jxprof.h 的 extern 声明。 */

/* ---------------- 权重查找 ---------------- */
/* 权重常驻位置：生成头里的 static const 数组落在 .rodata -> flash，读走 XIP，
 * 板端实测只有 13.5 MB/s；PSRAM 顺序读 52.4 MB/s（约 4 倍）。
 * JX_I8WT 默认 1：启动时把整块 int8 权重 memcpy 到堆（PSRAM 公共堆）。
 * 纯搬运，数值一个 bit 都不变；JX_I8WT=0 可退回 flash XIP 做对照。 */
static const signed char *g_wq_blob = JXWQ_DATA;
static void              *g_wq_ram  = NULL;
static void              *g_wq_raw  = NULL;

void jx_wq_init(void)
{
  const char *e;
  int want;
  if (g_wq_ram != NULL) { return; }
  e = jx_getenv_guarded("JX_I8WT");
  want = (e != NULL) ? atoi(e) : 1;
  if (want <= 0) { return; }
  g_wq_raw = malloc((size_t)NQ_BLOB + 64u);
  if (g_wq_raw == NULL) { return; }
  /* r173 BUGFIX：JXWQ_DATA 带 __attribute__((aligned(16)))，但堆拷贝用
   * malloc 只保证 8 字节对齐（PSRAM 上实测落在 ...18）。而 PIE
   * 的 ee.vld.128 要求 16 字节对齐：未对齐时读到的是错位的
   * 权重（整个 conv 结果不一样，且随堆地址漂移 -> “冷启动结果
   * 不确定”）。JX_I8WT=0 走 flash 时对齐正常，所以只有堆路径出错。 */
  g_wq_ram = (void *)(((uintptr_t)g_wq_raw + 15u) & ~(uintptr_t)15u);
  memcpy(g_wq_ram, JXWQ_DATA, (size_t)NQ_BLOB);
  g_wq_blob = (const signed char *)g_wq_ram;
}

/* 分块字节数（运行期可调，方便一次烧录扫多组）：
 *   JX_I8RB  linear 激活子块 / conv 的 x 子块
 *   JX_I8OB  linear 权重子块 / conv 的权重子块
 * 两者之和要放进 32KB 的 L1 D-cache。原来硬编码 32768+32768 = 64KB，
 * 直接把 L1 冲垮 -> 权重每行都被迫回 flash 重读。 */
/* r51h：gelu 实现选择（默认 1 = 旧 LUT，与原状一致） */
static int jx_epg_mode(void)
{
  /* 每次读环境，不缓存：一个 boot 里可以连续 A/B 多个模式 */
  const char *e = jx_getenv_guarded("JX_EPG");
  return (e != NULL) ? atoi(e) : 4;   /* 默认 4 = 旧 LUT 4 路展开（实测 epi 3841->3200） */
}

static int jx_i8_blk(const char *env, int defval)
{
  const char *e = jx_getenv_guarded(env);
  if (e == NULL) { return defval; }
  { int v = atoi(e); return (v < 0) ? 0 : v; }
}

static const JXWQ *jx_wq_find(const float *W)
{
  const float *base = jx_weight_blob();
  long off = (long)(W - base);
  if (off < 0 || off >= jx_weight_floats()) { return NULL; }
  for (int i = 0; i < NQ_TENSORS; i++)
    {
      if (JXWQ_TAB[i].nch >= 2 && JXWQ_TAB[i].off == (int)off)
        {
          return &JXWQ_TAB[i];
        }
    }
  return NULL;
}

/* ---------------- scratch（复用，避免每次 malloc） ---------------- */
static float       *g_sf[JX_NCORE];
static int          g_sf_cap[JX_NCORE];
static signed char *g_sq_raw[JX_NCORE];
static int          g_sq_cap[JX_NCORE];

static float *scratch_f(int n)
{
  int c = jx_core_id();
  if (n > g_sf_cap[c])
    {
      free(g_sf[c]);
      g_sf_cap[c] = n + 64;
      g_sf[c] = (float *)memalign(16u,
                                  sizeof(float) * (size_t)g_sf_cap[c]);
    }
  return g_sf[c];
}

static signed char *scratch_q(int n)
{
  int c = jx_core_id();
  int need = n + 32;
  if (need > g_sq_cap[c])
    {
      free(g_sq_raw[c]);
      g_sq_cap[c] = need;
      g_sq_raw[c] = (signed char *)malloc((size_t)g_sq_cap[c] + 16);
    }
  if (g_sq_raw[c] == NULL) { return NULL; }
  return (signed char *)(((uintptr_t)g_sq_raw[c] + 15u) & ~(uintptr_t)15u);
}

/* 预量化输入用的第三块 scratch（JX_I8AQ=1 时用）。 */
static signed char *g_qx_raw[JX_NCORE];
static int          g_qx_cap[JX_NCORE];
static signed char *scratch_qx(int n)
{
  int c = jx_core_id();
  if (n + 16 > g_qx_cap[c])
    {
      free(g_qx_raw[c]);
      g_qx_cap[c] = n + 64;
      g_qx_raw[c] = (signed char *)malloc((size_t)g_qx_cap[c] + 16);
    }
  if (g_qx_raw[c] == NULL) { return NULL; }
  return (signed char *)(((uintptr_t)g_qx_raw[c] + 15u) & ~(uintptr_t)15u);
}

/* ---------------- conv t 分块用的**静态**缓冲（不占堆） ----------------
 * 为什么不用 scratch_q()：分块需求比单行大，
 * scratch_q 会 free+malloc 换块；**堆余量只有 ~0.6MB**，
 * 一次大 malloc 就会让后面的 t_alloc 失败，表现为 conv 往 NULL
 * 写而挂机（StoreProhibited VADDR=0）。改成静态数组后，
 * 能耗一次分配都不会发生，比旧路径还安全。
 *
 * 容量依据：从主机 MAC 轨迹统计，本模型 conv 的 K16 最大**272**
 *   （ Ci=88 k=3 g=1 ），K16>256 的只占 1.5% MAC；K16=112 占 45%。
 *   这里取 288 作为单行上限，16 行 -> 4608 字节。 */
/* 2026-09-14 第三十五轮：上限 16 -> 64。
 * pert 路径每个输出位置只有 K16 个数据，而它是按“通道行”取 x 的：
 * TN=16 时每条通道行只读 16*4=64 字节（ 2 条 cache line）就跳到下一个通道，
 * PSRAM 的~170 周期延迟完全没被重叠。TN 放到 64 后同一条行连续读
 * 256 字节（8 条 line），访存也能串联。计算结果与 TN 无关（每个输出位置
 * 的 max/scale/点积都是独立算的），所以逐位一致。
 * 缓冲容量取决于 TN*K16 <= JX_I8TN_L1，与 JX_I8TN_N 无关，所以这里反而可以把
 * g_tnq 从 N*K = 4608 字节缩到 L1+64 = 4160 字节/核。 */
#define JX_I8TN_N  512   /* 同时执行的输出位置数上限（r43b：由 L1/K16 与写工作集自适应） */
#define JX_I8TN_K  288   /* 单行 K16 上限（实测模型最大 272） */
#define JX_I8TN_L1 8192  /* 分块缓冲的 L1 占用目标（v153：容纳 K16=96 x TN=64） */

/* 2026-09-12 第四轮：改成 per-core（JX_NCORE 份），双核并行时编码核/解码核
 * 各用自己的这一份，无需加锁。单核构建 JX_NCORE=1，布局与原来完全一样。 */
static signed char g_tnq[JX_NCORE][JX_I8TN_L1 + 64] __attribute__((aligned(16)));
static float       g_tns[JX_NCORE][JX_I8TN_N];
static float       g_tni[JX_NCORE][JX_I8TN_N];
static float       g_tnm[JX_NCORE][JX_I8TN_N];
/* linear 批点积的输出行数上限（oblk 最大 768）与 int32 暂存（per-core） */
#define JX_I8OC_N 1024   /* 每次批处理的输出行数上限（oblk 最大 768） */
/* linear 批点积的 int32 输出暂存（per-core，见 jx_dotcol_i8） */
static int32_t     g_ocacc[JX_NCORE][JX_I8OC_N];
/* P0 边车：epilogue 写 amax 的 **SRAM 暂存**。
 * 直接向堆（PSRAM）里的 amax 做读改写会每个元素多出 ~14 周期：
 * epilogue 每个 t 块要流式写 Co*w*4 字节（Co=80/TB=102 时正好 32KB
 * = 整个 L1），把 amax 的那几条 cache line 每轮都冲掉。
 * 改成先累在 SRAM（无 cache，~1 周期），每个 t 块结束时整块回写一次。
 * 回写仅改存储位置，数值逐位不变。
 * dram0_0_seg 只剩 ~200 字节，周期不重叠的缓存直接复用下面那块
 * per-t 路径的 s_tnm：k1cf 与 jx_conv1d_i8_tb 从不嵌套，且每次调用都会
 * 自己重置（covers 全部 TB <= JX_I8TN_N）。 */
static int32_t     g_ocacc2[JX_NCORE][JX_I8OC_N];

/* r79?k1cf?jx_conv1d_k1_cf_tb???? int32 ????
 * ???r78???? + ????????
 *   1) `jixun mem` ? [19] ?????? PSRAM?0x3d80....??
 *   2) jx_conv1d_k1_cf_tb ???? 0x1160 = 4448 ????? 4096 ????
 *      acc[512] + acc2[512]??? epilogue ????? acc[j] ??**PSRAM ?**?
 *   3) 4448 ?????? Xtensa `movi` ? 12 ???????>2047??GCC ???
 *      ????? "l32r <????>; add.n aN, aN, a1; l32i.n/s32i.n, aN, 0" ??
 *      ???? 2961 ???? 615 ? l32r??????????????
 * ?? .bss?per-core??? SRAM?????? ~350 ???movi ????
 * ?????? PSRAM???????????????? -> ????? */
static int32_t     g_k1acc [JX_NCORE][JX_I8TN_N];
static int32_t     g_k1acc2[JX_NCORE][JX_I8TN_N];

/* ==================== r140 probe ====================
 * 只累加，不改变任何计算路径。jixun mem 末尾打印。
 *   0/1/2   jxu_maxabs 调用数 / 元素数 / 周期
 *   3/4     conv1d_core_fast 调用数 / 周期
 *   5       k=7 寄存器快路径进入次数
 *   6/7     conv1d_core(非 fast) 调用数 / 周期
 *   8/9/10  jx_fbr_range 调用数 / trend_pool 周期 / branch 周期
 *   11/12/13 jx_conv1d_i8 拒绝数 / 接受数 / 周期
 *   14/15   dense_tile 调用数 / jx_fbr_range 总周期
 * ==================================================== */
unsigned long jx_pb[16];

/* r79?????????????? .bss ?????? SRAM ?? PSRAM?? */
void jx_dump_static_addrs(void)
{
  printf("[addr] g_tnq    %p  size %u\n", (void *)g_tnq,   (unsigned)sizeof(g_tnq));
  printf("[addr] g_tns    %p  size %u\n", (void *)g_tns,   (unsigned)sizeof(g_tns));
  printf("[addr] g_ocacc  %p  size %u\n", (void *)g_ocacc, (unsigned)sizeof(g_ocacc));
  printf("[addr] g_k1acc  %p  size %u\n", (void *)g_k1acc, (unsigned)sizeof(g_k1acc));
  printf("[addr] g_wq_blob %p  g_post_sa %p\n", (void *)g_wq_blob, (void *)g_post_sa);
  printf("[pb] maxabs n=%lu elems=%lu cyc=%lu (%.1f cyc/el)\n",
         jx_pb[0], jx_pb[1], jx_pb[2],
         jx_pb[1] ? (double)jx_pb[2] / (double)jx_pb[1] : 0.0);
  printf("[pb] cfast n=%lu cyc=%lu (%.2f cyc/call) k7path=%lu\n",
         jx_pb[3], jx_pb[4],
         jx_pb[3] ? (double)jx_pb[4] / (double)jx_pb[3] : 0.0, jx_pb[5]);
  printf("[pb] cslow n=%lu cyc=%lu\n", jx_pb[6], jx_pb[7]);
  printf("[pb] fbr   n=%lu tp=%lu br=%lu cyc\n", jx_pb[8], jx_pb[9], jx_pb[10]);
  printf("[pb] i8    rej=%lu acc=%lu cyc=%lu\n", jx_pb[11], jx_pb[12], jx_pb[13]);
  printf("[pb] dt    calls=%lu fbr_cyc=%lu\n", jx_pb[14], jx_pb[15]);
}
/* r53：per-o 的 wscale[]/b[] 缓存。这两条流在 epilogue 的每个输出元素上
 * 都要从 PSRAM 读一次（3.55M 元素 x 2 = 7.1M 次），而它们只有 cn 个值、
 * 在整块里不变。缓存到内部 SRAM 后每块只读一遍。数值完全不变。 */
#define JX_I8WSC_N 256   /* r53: per-o wscale/b cache depth (this model out<=88) */
static float       g_wsc[JX_NCORE][JX_I8WSC_N];
static float       g_bb [JX_NCORE][JX_I8WSC_N];

/* r43c 诊断：JX_K1EP=1 时把 k1s epilogue 的输出落到内部 SRAM（结果无效，
 * 只为把「反量化+激活」与「PSRAM float 写」两笔成本分开量）。 */
static float  g_ep_scratch[1024];
static const char *jx_getenv_guarded(const char *name)
{
  volatile int *lock = (volatile int *)(void *)g_ep_scratch;
  const char *v;

  while (__sync_lock_test_and_set((int *)lock, 1) != 0) { }
  v = getenv(name);
  __sync_lock_release((int *)lock);
  return v;
}
static int jx_gx4(void)
{
  static int v = -1;
  if (v < 0) { const char *e = jx_getenv_guarded("JX_GX4"); v = (e == NULL || atoi(e) != 0); }
  return v;
}
float *jx_ep_dst = NULL;
static int jx_ep_on(void)
{
  static int v = -1;
  if (v < 0) { const char *e = jx_getenv_guarded("JX_K1EP"); v = (e != NULL && atoi(e) != 0); }
  return v;
}
static inline float *jx_ep_pick(float *real)
{
  return (jx_ep_dst != NULL) ? jx_ep_dst : real;
}

/* depthwise conv 直接写 channels-last 输出：y[B][T][C]。
 * conv_unit 的下一层 pw_conv1 GEMM 正好是 [T][C] 输入，因此省掉原来
 * c0 [C][T] -> c1 [T][C] 的整块转置。时间小块暂存复用 g_ocacc2（该阶段
 * 尚未进入线性 GEMM，不会与之冲突）。 */
/* r67：depthwise conv（conv_unit 的 dw_conv，约 310 ms）原来完全串行。
 * 按 t 块切给两颗核：块与块只碰 yout 的不相交时间区间，每通道的求和
 * 次序与算式一字不改 -> 逐位相同。tile 用 per-core 的 g_ocacc2。 */
typedef struct {
  const float *W; const float *b; const float *xin; float *yout;
  int C, k, stride, pad, T_in, T_out, TB;
} jx_dwtc_t;

JX_HOT static void jx_dwtc_blocks(void *va, int blo, int bhi)
{
  const jx_dwtc_t *cc = (const jx_dwtc_t *)va;
  const float *W = cc->W; const float *b = cc->b;
  const float *xin = cc->xin; float *yout = cc->yout;
  const int C = cc->C, k = cc->k, stride = cc->stride, pad = cc->pad;
  const int T_in = cc->T_in, T_out = cc->T_out, TB = cc->TB;
  float *tile = (float *)g_ocacc2[jx_core_id()];
  for (int _blk = blo; _blk < bhi; _blk++)
    {
      int t0 = _blk * TB;
      int t1 = t0 + TB; if (t1 > T_out) { t1 = T_out; }
      int w = t1 - t0;
            /* r118：k=7/stride=1 的“内部区”快路。
             *
             * 原写法每个 tap 都要 `pos>=0 && pos<T_in` 两次比较，而这在一个
             * 整段里只有表头表尾共 6 个位置真正需要它。更关键的是每个输出
             * 元素的 7 条 `acc += x*w` 全部串在同一个 acc 上，是一条 7 级
             * 的 madd 依赖链（FPU 单条延迟约 4 周期），而相邻 j 的两个输出
             * 互相独立却没有任何交错。
             *
             * 两条一起改：
             *   1) 内部区的 7 个 tap 全部展开、系数提前装进寄存器；
             *   2) j 方向 4 路展开，4 条独立累加链交错，盖住 madd 延迟。
             * 每个 acc 的加法次序仍然是 bb -> kk 递增，与慢路逐项相同
             * （浮点加法次序不变 -> 逐位一致）。 */
            const int fast_k7 = (k == 7 && stride == 1);
            int jlo = 0, jhi = w;
            if (fast_k7)
              {
                if (pad > t0) { jlo = pad - t0; }
                { int lim = T_in - 1 - 6 + pad - t0 + 1;
                  if (jhi > lim) { jhi = lim; } }
                if (jhi < 0) { jhi = 0; }
                if (jlo > jhi) { jlo = jhi; }
              }
            for (int c = 0; c < C; c++)
              {
                const float *xr = xin + (size_t)c * T_in;
                const float *wr = W + (size_t)c * k;
                float bb = (b != NULL) ? b[c] : 0.0f;
                int j = 0;
                if (fast_k7)
                  {
                    const float w0 = wr[0], w1 = wr[1], w2 = wr[2], w3 = wr[3];
                    const float w4 = wr[4], w5 = wr[5], w6 = wr[6];
                    for (; j < jlo; j++)
                      {
                        int pos0 = (t0 + j) - pad;
                        float acc = bb;
                        for (int kk = 0; kk < k; kk++)
                          {
                            int pos = pos0 + kk;
                            if (pos >= 0 && pos < T_in) { acc += xr[pos] * wr[kk]; }
                          }
                        tile[(size_t)j * C + c] = acc;
                      }
                    for (; j + 4 <= jhi; j += 4)
                      {
                        const float *xp = xr + ((size_t)(t0 + j) - (size_t)pad);
                        float a0 = bb, a1 = bb, a2 = bb, a3 = bb;
                        a0 += xp[0] * w0; a1 += xp[1] * w0; a2 += xp[2] * w0; a3 += xp[3] * w0;
                        a0 += xp[1] * w1; a1 += xp[2] * w1; a2 += xp[3] * w1; a3 += xp[4] * w1;
                        a0 += xp[2] * w2; a1 += xp[3] * w2; a2 += xp[4] * w2; a3 += xp[5] * w2;
                        a0 += xp[3] * w3; a1 += xp[4] * w3; a2 += xp[5] * w3; a3 += xp[6] * w3;
                        a0 += xp[4] * w4; a1 += xp[5] * w4; a2 += xp[6] * w4; a3 += xp[7] * w4;
                        a0 += xp[5] * w5; a1 += xp[6] * w5; a2 += xp[7] * w5; a3 += xp[8] * w5;
                        a0 += xp[6] * w6; a1 += xp[7] * w6; a2 += xp[8] * w6; a3 += xp[9] * w6;
                        tile[(size_t)(j    ) * C + c] = a0;
                        tile[(size_t)(j + 1) * C + c] = a1;
                        tile[(size_t)(j + 2) * C + c] = a2;
                        tile[(size_t)(j + 3) * C + c] = a3;
                      }
                    for (; j < jhi; j++)
                      {
                        const float *xp = xr + ((size_t)(t0 + j) - (size_t)pad);
                        float acc = bb;
                        acc += xp[0] * w0;
                        acc += xp[1] * w1;
                        acc += xp[2] * w2;
                        acc += xp[3] * w3;
                        acc += xp[4] * w4;
                        acc += xp[5] * w5;
                        acc += xp[6] * w6;
                        tile[(size_t)j * C + c] = acc;
                      }
                  }
                for (; j < w; j++)
                  {
                    int pos0 = (t0 + j) * stride - pad;
                    float acc = bb;
                    for (int kk = 0; kk < k; kk++)
                      {
                        int pos = pos0 + kk;
                        if (pos >= 0 && pos < T_in) { acc += xr[pos] * wr[kk]; }
                      }
                    tile[(size_t)j * C + c] = acc;
                  }
              }
            for (int j = 0; j < w; j++)
              {
                memcpy(&yout[(size_t)(t0 + j) * C], &tile[(size_t)j * C],
                       (size_t)C * sizeof(float));
              }
          }
}

void jx_conv1d_dw_tc(const float *W, const float *b, const T *x, T *y,
                     int C, int k, int stride, int pad, int T_in)
{
    JXP_BEG(JP_CONV1D);
    int B = x->shape[0];
    int T_out = (T_in + 2 * pad - k) / stride + 1;
    int need = B * T_out * C;
    if (y->len < need)
      {
        float *nd = (float*)jx_arealloc(y->d, (size_t)need * sizeof(float));
        if (nd == NULL)
          {
            jx_conv_skip++;
            printf("[CONV] dw_tc realloc %d floats 失败，跳过本次卷积\n", need);
            JXP_END(JP_CONV1D);
            return;
          }
        y->d = nd;
      }
    y->shape[0] = B; y->shape[1] = T_out; y->shape[2] = C; y->len = need;

    int TB = JX_I8OC_N / C;
    if (TB < 1) { TB = 1; }
    if (TB > T_out) { TB = T_out; }
    for (int bi = 0; bi < B; bi++)
      {
        jx_dwtc_t cc;
        cc.W = W; cc.b = b;
        cc.xin = x->d + (size_t)bi * C * T_in;
        cc.yout = y->d + (size_t)bi * T_out * C;
        cc.C = C; cc.k = k; cc.stride = stride; cc.pad = pad;
        cc.T_in = T_in; cc.T_out = T_out; cc.TB = TB;
        { int _nb = (T_out + TB - 1) / TB;
          jx_pf_run_min(_nb, jx_dwtc_blocks, &cc, 2); }
      }
    JXP_END(JP_CONV1D);
}

/* aq==3（k=1 顺序预量化）的 per-core 缓冲（PSRAM 堆，一次分配长期复用）。
 * q 布局为 [t][cpad]，cpad = align16(cpg)；分配时整块清零，
 * 之后每次只写 [0..cpg) 部分，所以 padding 字节永远是 0。 */
static signed char *g_aq3_q[JX_NCORE];
static int          g_aq3_qcap[JX_NCORE];
static int          g_aq3_cpad[JX_NCORE];
static float       *g_aq3_m[JX_NCORE];
static int          g_aq3_mcap[JX_NCORE];

/* 行级内部 SRAM 暂存（2KB/核）。
 * 定义提前到这里，jx_q_row（行量化）与 jx_q_row_grn（GRN 衍射）
 * 互斥使用：同一时刻只会有一个在跑。 */
#define JX_GRN_MAX 512  /* 行缓存上限（本模型 in 最大 352） */
static float g_grnrow[JX_NCORE][JX_GRN_MAX];
#define JX_NORM_MAX 128
static float *g_normrow[JX_NCORE];
static int    g_normcap[JX_NCORE];

/* JX_QSR：行量化是否用行级 SRAM 暂存（默认 1）。 */
static int jx_q_sr(void)
{
  static int v = -2;
  if (v == -2) { const char *e = jx_getenv_guarded("JX_QSR"); v = !(e != NULL && e[0] == (char)48); }
  return v;
}

static __attribute__((noinline)) void jx_norm_row(const float *src, float *dst,
                                                  int n, const float *w,
                                                  const float *b)
{
  float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
  int c = 0;
  for (; c + 4 <= n; c += 4)
    { m0 += src[c]; m1 += src[c + 1]; m2 += src[c + 2]; m3 += src[c + 3]; }
  float mean = (m0 + m1) + (m2 + m3);
  for (; c < n; c++) { mean += src[c]; }
  mean /= (float)n;
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  c = 0;
  for (; c + 4 <= n; c += 4)
    {
      float d0 = src[c    ] - mean, d1 = src[c + 1] - mean;
      float d2 = src[c + 2] - mean, d3 = src[c + 3] - mean;
      v0 += d0 * d0; v1 += d1 * d1; v2 += d2 * d2; v3 += d3 * d3;
    }
  float var = (v0 + v1) + (v2 + v3);
  for (; c < n; c++) { float d = src[c] - mean; var += d * d; }
  var /= (float)n;
  float inv = 1.0f / sqrtf(var + XNN_EPS);
  for (c = 0; c < n; c++) dst[c] = w[c] * (src[c] - mean) * inv + b[c];
}

/* ---------------- 行量化：float[n] -> int8[n16]，返回 scale ---------------- */
/* 2026-09-14 第十七轮：单趟化实验**已回退**。
 * 方案：用每核 4KB 的 int32 暂存 g_ocacc2 存 float 位模式，第二趟从
 * 内部 SRAM 读回，省掉对 x 的第二趟 PSRAM 读。
 * 板端 A/B（其余完全相同的两版固件）：
 *     两趟（原始）  量化 prologue 220.2 Mcyc / 4.146M 元素 = 53.11 cyc/元素
 *     单趟（实验）  量化 prologue 229.0 Mcyc / 4.146M 元素 = 55.24 cyc/元素
 * 反而慢 9 Mcyc（+4%）。说明这个 prologue 的瓶颈**不是**第二趟 PSRAM 读
 * ——行块只有 rblk*in 字节，第二趟基本命中 L1；而每元素多出的
 * 「暂存写 + 暂存读 + 指针判空」反而净亏。已回退为原来的两趟实现。 */
static float jx_q_row(const float *x, signed char *q, int n)
{
  /* 2026-09-14 第十八轮：把第二趟的输入从 PSRAM 挪到内部 SRAM。
   *
   * 两趟版的第一趟读 x 求 max，第二趟**再读一遍** x 做量化。第二趟本来
   * 指望命中 L1，但行块是 rblk*in*4 字节（in=352、JX_I8RB=12288 时
   * rblk=34 -> 47.9KB），已经超过 32KB 的 L1，所以有相当一部分仍要回
   * PSRAM —— 也就是同一条数据被读了两趟 PSRAM。
   *
   * 改法：第一趟读 x 时顺手把值存进行级 SRAM 暂存（g_grnrow，2KB/核），
   * 第二趟改从 SRAM 读。PSRAM 读从两趟降为一趟。
   * （第十七轮试过的"按行块暂存"失败，是因为 rblk 级需要 45KB、SRAM 装不下；
   *   行级只需要 in*4 <= 2KB，装得下。）
   *
   * 逐位一致：max 的比较次序、量化的算式与舍入全部不变，只是第二趟的
   * 数据来源从 PSRAM 换成同日期的 SRAM 副本。
   * JX_QSR=0 可退回原来的两趟实现做 A/B。 */
  float mx = 0.0f;
  int n16 = (n + 15) & ~15;
  float *row = (jx_q_sr() && n <= JX_GRN_MAX) ? g_grnrow[jx_core_id()] : NULL;
  if (row != NULL)
    {
      for (int i = 0; i < n; i++)
        {
          float xv = x[i];
          row[i] = xv;
          float a = fabsf(xv);
          if (a > mx) { mx = a; }
        }
    }
  else
    {
      for (int i = 0; i < n; i++)
        {
          float a = fabsf(x[i]);
          if (a > mx) { mx = a; }
        }
    }
  if (mx <= 1e-30f) { memset(q, 0, (size_t)n16); return 1e-30f; }
  {
    float inv = 127.0f / mx;
    const float *xs = (row != NULL) ? row : x;
    for (int i = 0; i < n; i++)
      {
        float v = xs[i] * inv;
        int iv = jx_rndf(v);
        if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
        q[i] = (signed char)iv;
      }
  }
  for (int i = n; i < n16; i++) { q[i] = 0; }
  return mx / 127.0f;
}

/* ---------------- 行内展开的行量化（2026-09-14 第十八轮：攻 FPU 依赖链）----
 * [15] 探针否决了"行间并行"（4 行 0.86x，反而更慢）。同一轮 [1a] 给出真因：
 *     FPU 1 链 9.01 周期/MAC，2 链 6.01，4 链 4.79，8 链 3.79（上限）
 * —— Xtensa 的 FP 运算延迟很高，单链完全被延迟吃满。
 * 行量化的两趟都是**长依赖链**：
 *   趟1 fabsf -> compare/select            链长 ~5 周期/元素
 *   趟2 mul -> add .5 -> cvt -> clamp -> 存 链长 ~16 周期/元素
 * 把一行内部切成 U 条独立的链（i, i+1, ... i+U-1 各一条），延迟就被隐藏：
 *   趟1 8 路 -> ~0.7 周期/元素；趟2 8 路 -> ~2 周期/元素
 * 逐位一致：max 是结合的（浮点 max 无舍入），趟 2 每个元素独立，
 * 同一元素内的算式次序与 jx_q_row 完全相同。
 */
static int jx_q_unroll(void)
{
  static int v = -2;
  if (v == -2)
    {
      const char *e = jx_getenv_guarded("JX_QU");
      /* 2026-09-14 第三十三轮：默认从 0（逐元素）改成 8（8 路展开）。
       * 旧的 A/B 结论是"8 路只换 3.5%"，但那次展开的只是 GRN 行量化**没有**
       * 覆盖到的普通行量化；GRN 版展开后（0/8 对照）总耗时 -710 ms，
       * 说明这条路径确实是延迟受限，普通版同样该展开。JX_QU=0 可回退。 */
      v = (e != NULL) ? atoi(e) : 8;
      if (v != 4 && v != 8) { v = 0; }
    }
  return v;
}

/* 第四十二轮：钳位是死代码。inv = 127/mx 且 mx 就是同一批数据的 |v| 上界，
 * 故 |v*inv| <= 127*(1+eps)，(int)(v +- 0.5) 本身就落在 [-127,127]。去掉
 * 省 2 条 cmp + 2 条 select/元素（全场约 9M 元素）。指纹已校验。 */
#define JX_QCLAMP(w) do { (void)0; } while (0)

/* 2026-09-15 第四十三轮：行 max 从浮点域搬到整数域。
 *
 * 反汇编证据（r43c 固件，jx_q_row_grn 内层循环体 0x..d5 起）：
 *   8 路展开的 8 个 running max 有 5 个被 GCC 溢出到栈槽
 *   （ssi f*, a1, 24/28/32/36/44），内层每轮 lsi + ssi 各若干次 ——
 *   每个元素凭空多出两次栈访问；而每次更新是 olt.s + bf 分支。
 *   根因是只有 16 个浮点寄存器：8 个 max + 8 个待处理值 + 系数装不下。
 *
 * IEEE-754 去掉符号位之后，位模式的**无符号比较**与 |x| 的大小比较同序
 * （有限数、非 NaN）。于是 max|x| 整段可以放到整数域做：
 *   - 累加器用 16 个 a 寄存器（本函数只占几个指针），不再溢出；
 *   - 整数比较 1 周期，替掉 4 周期延迟的 olt.s + bf。
 * 最后一次性把选中的位模式还原成 float。
 *
 * 数值逐位一致：fabsf + 浮点比较 与 掩码 + 整数比较，选出的是同一个
 * float 位模式；后续算式、次序、舍入全部不变。 */
typedef unsigned jx_u32a __attribute__((may_alias));

static inline unsigned jx_ubits(float v)
{
  unsigned u;
  __builtin_memcpy(&u, &v, sizeof(u));
  return u;
}

static inline float jx_u2f(unsigned u)
{
  float f;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}

static inline unsigned jx_uabs(float v)
{
  return jx_ubits(v) & 0x7fffffffu;
}

/* ============================================================================
 * r86：PIE 128 位「逐元素 |x| 最大值」扫描。
 *
 * 动机（r86 反汇编证据）：dt 的 mx 扫描内层循环里，8 个 running max 与 8 个
 * 待处理值抢 16 个通用寄存器，GCC 把一半溢出到栈槽 —— 每元素凭空多出
 * 1 条 s32i + 1 条 l32i，而且 8 条 PSRAM load 串行等待。实测该桶
 * 83.9 Mcyc / 817K 元素 = 102 周期/元素，是全场单价最高的一处。
 *
 * 手法：4 个 float 一次装载（1 条 ee.vld.128.ip），用 ee.andq 在 128 位里
 * 一次清掉 4 个符号位，再用 ee.vmax.s32 做 4 路运行最大 —— 每 4 个元素
 * 只要 3 条指令，且不占通用寄存器。
 *
 * 数值逐位一致：max 与次序无关；去掉符号位后的位模式在有限非 NaN 值上
 * 与 |x| 同序（最高位恒 0，有符号/无符号比较同序），与 jx_uabs+maxu 等价。
 * 首尾不满足 16 字节对齐/不足 4 个的部分仍走标量，拼接顺序不影响 max。
 * ========================================================================== */
JX_HOT static unsigned jxu_maxabs(const float *p, int n)
{
  unsigned m = 0u;
  int i = 0;
  unsigned long _pb0 = jx_ccount();
  jx_pb[0]++;
  if (n <= 0) { jx_pb[2] += (unsigned long)(unsigned)(jx_ccount() - _pb0); return 0u; }
  jx_pb[1] += (unsigned long)n;
  {
    unsigned long ap = (unsigned long)(const void *)p;
    int head = (int)((ap & 15ul) >> 2);          /* 对齐前的元素个数（0~3） */
    if (head > n) { head = n; }
    for (; i < head; i++) { unsigned v = jx_uabs(p[i]); if (v > m) { m = v; } }
  }
#if defined(JX_PIE)
  {
    int ng = (n - i) & ~3;
    if (ng >= 4)
      {
        const float *qp = p + i;
        unsigned lanes[4] __attribute__((aligned(16)));
        static const unsigned jxu_mk = 0x7fffffffu;
        void *mkp = (void *)(&jxu_mk);
        void *lbp = (void *)(lanes);
        int k = ng >> 2;
        __asm__ __volatile__(
            "ee.vldbc.32.ip q7, %[MK], 0\n\t"
            "ee.zero.q q6\n\t"
            "1:\n\t"
            "ee.vld.128.ip q0, %[Q], 16\n\t"
            "ee.andq q1, q0, q7\n\t"
            "ee.vmax.s32 q6, q6, q1\n\t"
            "addi %[K], %[K], -1\n\t"
            "bnez %[K], 1b\n\t"
            "ee.vst.128.ip q6, %[LB], 16\n\t"
            : [Q] "+r"(qp), [K] "+r"(k), [MK] "+r"(mkp), [LB] "+r"(lbp)
            :
            : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
        for (int j = 0; j < 4; j++) { if (lanes[j] > m) { m = lanes[j]; } }
        i += ng;
      }
  }
#endif
  for (; i < n; i++) { unsigned v = jx_uabs(p[i]); if (v > m) { m = v; } }
  jx_pb[2] += (unsigned long)(unsigned)(jx_ccount() - _pb0);
  return m;
}

/* r86：dt 的 max 扫描换 PIE 版的开关（板上 A/B 用，默认 1）。 */
static int jx_dtmax(void)
{
  static int v = -2;
  if (v == -2) { v = jx_i8_blk("JX_DTMAX", 1); }
  return v;
}

/* U=4 */
static float jx_q_row_u4(const float *x, signed char *q, int n)
{
  int n16 = (n + 15) & ~15;
  int n4 = n & ~3;
  int i = 0;
  {
    /* r97：整段 max 扫描换成 PIE 128 位（见 jxu_maxabs 的说明）。 */
    float mx = jx_u2f(jxu_maxabs(x, n));
    if (mx <= 1e-30f) { memset(q, 0, (size_t)n16); return 1e-30f; }
    {
      float inv = 127.0f / mx;
      i = 0;
      for (; i < n4; i += 4)
        {
          float v0 = x[i    ] * inv, v1 = x[i + 1] * inv;
          float v2 = x[i + 2] * inv, v3 = x[i + 3] * inv;
          int w0 = jx_rndf(v0);
          int w1 = jx_rndf(v1);
          int w2 = jx_rndf(v2);
          int w3 = jx_rndf(v3);
          JX_QCLAMP(w0); JX_QCLAMP(w1); JX_QCLAMP(w2); JX_QCLAMP(w3);
          q[i] = (signed char)w0; q[i + 1] = (signed char)w1;
          q[i + 2] = (signed char)w2; q[i + 3] = (signed char)w3;
        }
      for (; i < n; i++)
        {
          float v = x[i] * inv;
          int w = jx_rndf(v);
          JX_QCLAMP(w); q[i] = (signed char)w;
        }
    }
    for (i = n; i < n16; i++) { q[i] = 0; }
    return mx / 127.0f;
  }
}

/* U=8 */
static float jx_q_row_u8(const float *x, signed char *q, int n)
{
  int n16 = (n + 15) & ~15;
  int n8 = n & ~7;
  int i = 0;
  {
    /* r97：整段 max 扫描换成 PIE 128 位（见 jxu_maxabs 的说明）。 */
    float mx = jx_u2f(jxu_maxabs(x, n));
    if (mx <= 1e-30f) { memset(q, 0, (size_t)n16); return 1e-30f; }
    {
      float inv = 127.0f / mx;
      i = 0;
      for (; i < n8; i += 8)
        {
          float v0 = x[i    ] * inv, v1 = x[i + 1] * inv;
          float v2 = x[i + 2] * inv, v3 = x[i + 3] * inv;
          float v4 = x[i + 4] * inv, v5 = x[i + 5] * inv;
          float v6 = x[i + 6] * inv, v7 = x[i + 7] * inv;
          int w0 = jx_rndf(v0);
          int w1 = jx_rndf(v1);
          int w2 = jx_rndf(v2);
          int w3 = jx_rndf(v3);
          int w4 = jx_rndf(v4);
          int w5 = jx_rndf(v5);
          int w6 = jx_rndf(v6);
          int w7 = jx_rndf(v7);
          JX_QCLAMP(w0); JX_QCLAMP(w1); JX_QCLAMP(w2); JX_QCLAMP(w3);
          JX_QCLAMP(w4); JX_QCLAMP(w5); JX_QCLAMP(w6); JX_QCLAMP(w7);
          q[i]     = (signed char)w0; q[i + 1] = (signed char)w1;
          q[i + 2] = (signed char)w2; q[i + 3] = (signed char)w3;
          q[i + 4] = (signed char)w4; q[i + 5] = (signed char)w5;
          q[i + 6] = (signed char)w6; q[i + 7] = (signed char)w7;
        }
      for (; i < n; i++)
        {
          float v = x[i] * inv;
          int w = jx_rndf(v);
          JX_QCLAMP(w); q[i] = (signed char)w;
        }
    }
    for (i = n; i < n16; i++) { q[i] = 0; }
    return mx / 127.0f;
  }
}

/* 按 JX_QU 分派（0/1 = 原逐元素实现） */
static float jx_q_row_auto(const float *x, signed char *q, int n)
{
  int u = jx_q_unroll();
  if (u == 8) { return jx_q_row_u8(x, q, n); }
  if (u == 4) { return jx_q_row_u4(x, q, n); }
  return jx_q_row(x, q, n);
}

/* ---------------- 多路并行的行量化（2026-09-14 第十八轮：攻访存受限）-------
 * 动机：板端 [11] 探针显示 PSRAM 顺序单流是**延迟受限**而不是带宽受限——
 *     1 路 39.0 周期/次 (24.7 MB/s)
 *     2 路 25.0 周期/次 (38.1 MB/s)
 *     4 路 20.5 周期/次 (46.6 MB/s)  <- 饱和
 *     8 路 172 周期/次 (5.6 MB/s)    <- 地址低位 [12:6] 相同，全撞同一 cache set
 * 行量化的第一趟（求 max）与第二趟（量化写 q）都是一行一条顺序流，天然
 * 卡在单流延迟上。把 PW 行**同时**处理，PW 个独立 miss 就能重叠。
 *
 * 逐位一致：每行的 max 与量化算式与 jx_q_row 完全相同（同一行内仍是
 * 低地址到高地址、同一比较/取整顺序），只是行与行交错执行。
 *
 * cache set：D-cache 64KB / 8 ways / 64B 行 -> 128 个 set，索引 = 地址 [12:6]。
 * 本模型的行字节步长 in*4，in 属于 {16,20,28,40,56,64,80,81,88,352}，
 * in*4 都不是 4096 的倍数，PW 行的 set 索引天然错开，不会重演 8 路塌陷。
 */
static int jx_q_par(void)
{
  static int v = -2;
  if (v == -2)
    {
      const char *e = jx_getenv_guarded("JX_QPAR");
      v = (e != NULL) ? atoi(e) : 0;
      if (v != 2 && v != 4) { v = 0; }
    }
  return v;
}

/* PW=2：两行交错。sout 接收两行的 scale（= max/127）。 */
static void jx_q_row2(const float *x, signed char *q, int n, int in16, float *sout)
{
  const float *x0 = x, *x1 = x + n;
  signed char *q0 = q, *q1 = q + in16;
  int n16 = (n + 15) & ~15;
  float m0 = 0.0f, m1 = 0.0f;
  for (int i = 0; i < n; i++)
    {
      float a0 = fabsf(x0[i]); if (a0 > m0) { m0 = a0; }
      float a1 = fabsf(x1[i]); if (a1 > m1) { m1 = a1; }
    }
  if (m0 > 1e-30f && m1 > 1e-30f)
    {
      float i0 = 127.0f / m0, i1 = 127.0f / m1;
      for (int i = 0; i < n; i++)
        {
          float v0 = x0[i] * i0;
          float v1 = x1[i] * i1;
          int w0 = jx_rndf(v0);
          int w1 = jx_rndf(v1);
          if (w0 > 127) { w0 = 127; } else if (w0 < -127) { w0 = -127; }
          if (w1 > 127) { w1 = 127; } else if (w1 < -127) { w1 = -127; }
          q0[i] = (signed char)w0;
          q1[i] = (signed char)w1;
        }
    }
  else
    {
      for (int i = 0; i < n; i++)
        {
          float v0 = (m0 > 1e-30f) ? x0[i] * (127.0f / m0) : 0.0f;
          float v1 = (m1 > 1e-30f) ? x1[i] * (127.0f / m1) : 0.0f;
          int w0 = jx_rndf(v0);
          int w1 = jx_rndf(v1);
          if (w0 > 127) { w0 = 127; } else if (w0 < -127) { w0 = -127; }
          if (w1 > 127) { w1 = 127; } else if (w1 < -127) { w1 = -127; }
          q0[i] = (signed char)w0;
          q1[i] = (signed char)w1;
        }
    }
  for (int i = n; i < n16; i++) { q0[i] = 0; q1[i] = 0; }
  sout[0] = (m0 > 1e-30f) ? m0 / 127.0f : 1e-30f;
  sout[1] = (m1 > 1e-30f) ? m1 / 127.0f : 1e-30f;
}

/* PW=4：四行交错。第一趟是本轮访存优化的主战场（4 路 miss 重叠）。 */
static void jx_q_row4(const float *x, signed char *q, int n, int in16, float *sout)
{
  const float *x0 = x, *x1 = x + n, *x2 = x + 2 * n, *x3 = x + 3 * n;
  signed char *q0 = q, *q1 = q + in16, *q2 = q + 2 * in16, *q3 = q + 3 * in16;
  int n16 = (n + 15) & ~15;
  float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
  for (int i = 0; i < n; i++)
    {
      float a0 = fabsf(x0[i]); if (a0 > m0) { m0 = a0; }
      float a1 = fabsf(x1[i]); if (a1 > m1) { m1 = a1; }
      float a2 = fabsf(x2[i]); if (a2 > m2) { m2 = a2; }
      float a3 = fabsf(x3[i]); if (a3 > m3) { m3 = a3; }
    }
  if (m0 > 1e-30f && m1 > 1e-30f && m2 > 1e-30f && m3 > 1e-30f)
    {
      float i0 = 127.0f / m0, i1 = 127.0f / m1;
      float i2 = 127.0f / m2, i3 = 127.0f / m3;
      for (int i = 0; i < n; i++)
        {
          float v0 = x0[i] * i0, v1 = x1[i] * i1;
          float v2 = x2[i] * i2, v3 = x3[i] * i3;
          int w0 = jx_rndf(v0);
          int w1 = jx_rndf(v1);
          int w2 = jx_rndf(v2);
          int w3 = jx_rndf(v3);
          if (w0 > 127) { w0 = 127; } else if (w0 < -127) { w0 = -127; }
          if (w1 > 127) { w1 = 127; } else if (w1 < -127) { w1 = -127; }
          if (w2 > 127) { w2 = 127; } else if (w2 < -127) { w2 = -127; }
          if (w3 > 127) { w3 = 127; } else if (w3 < -127) { w3 = -127; }
          q0[i] = (signed char)w0; q1[i] = (signed char)w1;
          q2[i] = (signed char)w2; q3[i] = (signed char)w3;
        }
    }
  else
    {
      for (int i = 0; i < n; i++)
        {
          float v0 = (m0 > 1e-30f) ? x0[i] * (127.0f / m0) : 0.0f;
          float v1 = (m1 > 1e-30f) ? x1[i] * (127.0f / m1) : 0.0f;
          float v2 = (m2 > 1e-30f) ? x2[i] * (127.0f / m2) : 0.0f;
          float v3 = (m3 > 1e-30f) ? x3[i] * (127.0f / m3) : 0.0f;
          int w0 = jx_rndf(v0);
          int w1 = jx_rndf(v1);
          int w2 = jx_rndf(v2);
          int w3 = jx_rndf(v3);
          if (w0 > 127) { w0 = 127; } else if (w0 < -127) { w0 = -127; }
          if (w1 > 127) { w1 = 127; } else if (w1 < -127) { w1 = -127; }
          if (w2 > 127) { w2 = 127; } else if (w2 < -127) { w2 = -127; }
          if (w3 > 127) { w3 = 127; } else if (w3 < -127) { w3 = -127; }
          q0[i] = (signed char)w0; q1[i] = (signed char)w1;
          q2[i] = (signed char)w2; q3[i] = (signed char)w3;
        }
    }
  for (int i = n; i < n16; i++) { q0[i] = 0; q1[i] = 0; q2[i] = 0; q3[i] = 0; }
  sout[0] = (m0 > 1e-30f) ? m0 / 127.0f : 1e-30f;
  sout[1] = (m1 > 1e-30f) ? m1 / 127.0f : 1e-30f;
  sout[2] = (m2 > 1e-30f) ? m2 / 127.0f : 1e-30f;
  sout[3] = (m3 > 1e-30f) ? m3 / 127.0f : 1e-30f;
}

/* ---------------- GRN 仿射 + 行量化（conv_unit 融合用） ----------------
 * y = (gamma[c]*(x*nx) + beta[c]) + x，表达式与 grn_last 逐项一致；
 * 之后按行 max 量化，数值与「先 grn_last 落盘再 jx_q_row」逐位相同。 */
#define JX_SN_MAX 384   /* snake 系数缓存上限（本模型 dim4 最大 352） */
static float g_sna[JX_NCORE][JX_SN_MAX];
static float g_sni[JX_NCORE][JX_SN_MAX];
/* r49：l1 epilogue 写「待量化的整行」用的行级 SRAM 暂存（与 g_sna 同尺寸）。
 * 与 g_grnrow 不是同一用途：这里存的是 snake 之后、GRN 之前的行。 */
static float g_l1row[JX_NCORE][JX_SN_MAX];
/* r138：双行 epilogue 的第二条行缓冲。dram0_0_seg 已无余量，只能 malloc
 * （JX_NCORE*JX_SN_MAX*4 = 3KB，一次性）。取不到就自动退回单行路径。 */
static float *g_l1row2 = NULL;
static int jx_l1r2_on(void)
{
  static int v = -2;
  /* r147c：板端同固件 A/B（各 3 次，重复性 ±10 ms）实测"双行块"反而更慢：
   * 关掉 3670/3680/3670 -> 开 3590/3590/3590（encode -50，decode -30）。
   * 原因与 r54 的 PassA 4 路 ci 交错同型：两条行的中间量+12 条系数加载
   * 让活跃寄存器超了，GCC 溢出把交错收益吃光。默认改为 0（单行路径），
   * JX_L1R2=1 仍可编回旧行为。 */
  if (v == -2) { const char *e = jx_getenv_guarded("JX_L1R2"); v = (e != NULL && e[0] != (char)48); }
  return v;
}
static float *jx_l1row2_get(void)
{
  if (g_l1row2 == NULL)
    {
      float *p = (float *)malloc((size_t)JX_NCORE * (size_t)JX_SN_MAX * sizeof(float));
      if (p == NULL) { return NULL; }
      g_l1row2 = p;
    }
  return g_l1row2;
}

/* r85：GRN 的 gamma/beta 也搬进内部 SRAM。
 * 动机（r84 板端实测）：PSRAM 顺序读只有 ~85 MB/s、顺序写 ~49 MB/s，而
 * 这两条逐通道参数数组在本循环里**每个元素都要读一次**。工作集是几 MB 的
 * 流式张量，64KB 的 D-cache 被冲得一干二净，这两条小数组几乎每次都要吃
 * 一次 cache line 填充。swc/sbb/sna/sni 早已是同样做法
 * （g_wsc/g_bb/g_sna/g_sni），这里把最后两条补齐：内容一字不改地拷进来，
 * 读法完全等同，数值逐位一致。 */
static float g_gg[JX_NCORE][JX_SN_MAX];
static float g_gb[JX_NCORE][JX_SN_MAX];

/* 2026-09-14 第三十三轮：GRN prologue 的 8 路展开（隐藏 FPU 依赖链）。
 *
 * 板端 [i8mm] 探针：量化 prologue 703.3 Mcyc / 12.42M 元素 = 56.62 cyc/元素，
 * 而这段每元素只有约 12 条浮点指令 —— LX7 是**单发射 in-order**，两趟都是
 * 长依赖链：
 *     趟1  读 x -> 乘 nx -> 乘 gamma -> 加 beta -> 加 x -> fabsf -> max
 *     趟2  读 row -> 乘 inv -> 加 .5 -> cvt -> clamp -> 存 int8
 * 一次只喂一条链时流水线全在等延迟（[1a] 实测单链 9.01 周期/操作）。
 * 相邻元素完全独立，展开 8 路后 8 条链交错执行。
 *
 * 逐位一致：趟1 的 max 是浮点 max（无舍入、与比较次序无关），趟2 每元素
 * 独立且元素内算式次序与展开前逐字符相同。JX_GRN8=0 退回旧实现做 A/B。 */
static int jx_grn8(void)
{
  static int v = -2;
  if (v == -2) { const char *e = jx_getenv_guarded("JX_GRN8"); v = !(e != NULL && e[0] == (char)48); }
  return v;
}

static float jx_q_row_grn(const float *x, signed char *q, int n,
                          const float *gamma, const float *beta, float nx)
{
  float mx = 0.0f;
  int n16 = (n + 15) & ~15;
  float *row = (n <= JX_GRN_MAX) ? g_grnrow[jx_core_id()] : NULL;
  if (row != NULL && jx_grn8())
    {
      int n8 = n & ~7;
      int i = 0;
      for (; i < n8; i += 8)
        {
          float v0 = gamma[i    ] * (x[i    ] * nx) + beta[i    ] + x[i    ];
          float v1 = gamma[i + 1] * (x[i + 1] * nx) + beta[i + 1] + x[i + 1];
          float v2 = gamma[i + 2] * (x[i + 2] * nx) + beta[i + 2] + x[i + 2];
          float v3 = gamma[i + 3] * (x[i + 3] * nx) + beta[i + 3] + x[i + 3];
          float v4 = gamma[i + 4] * (x[i + 4] * nx) + beta[i + 4] + x[i + 4];
          float v5 = gamma[i + 5] * (x[i + 5] * nx) + beta[i + 5] + x[i + 5];
          float v6 = gamma[i + 6] * (x[i + 6] * nx) + beta[i + 6] + x[i + 6];
          float v7 = gamma[i + 7] * (x[i + 7] * nx) + beta[i + 7] + x[i + 7];
          row[i    ] = v0; row[i + 1] = v1; row[i + 2] = v2; row[i + 3] = v3;
          row[i + 4] = v4; row[i + 5] = v5; row[i + 6] = v6; row[i + 7] = v7;
        }
      for (; i < n; i++)
        {
          float v = gamma[i] * (x[i] * nx) + beta[i] + x[i];
          row[i] = v;
        }
      /* r97：max 不再逐元素累加，改由 PIE 128 位第二遍扫 row[0..n)。
       * row[] 的写入值与次序一字未改，max 与次序无关 -> 逐位一致。 */
      mx = jx_u2f(jxu_maxabs(row, n));
      if (mx <= 1e-30f) { memset(q, 0, (size_t)n16); return 1e-30f; }
      {
        float inv = 127.0f / mx;
        i = 0;
        for (; i < n8; i += 8)
          {
            float v0 = row[i    ] * inv, v1 = row[i + 1] * inv;
            float v2 = row[i + 2] * inv, v3 = row[i + 3] * inv;
            float v4 = row[i + 4] * inv, v5 = row[i + 5] * inv;
            float v6 = row[i + 6] * inv, v7 = row[i + 7] * inv;
            int w0 = jx_rndf(v0);
            int w1 = jx_rndf(v1);
            int w2 = jx_rndf(v2);
            int w3 = jx_rndf(v3);
            int w4 = jx_rndf(v4);
            int w5 = jx_rndf(v5);
            int w6 = jx_rndf(v6);
            int w7 = jx_rndf(v7);
            JX_QCLAMP(w0); JX_QCLAMP(w1); JX_QCLAMP(w2); JX_QCLAMP(w3);
            JX_QCLAMP(w4); JX_QCLAMP(w5); JX_QCLAMP(w6); JX_QCLAMP(w7);
            q[i    ] = (signed char)w0; q[i + 1] = (signed char)w1;
            q[i + 2] = (signed char)w2; q[i + 3] = (signed char)w3;
            q[i + 4] = (signed char)w4; q[i + 5] = (signed char)w5;
            q[i + 6] = (signed char)w6; q[i + 7] = (signed char)w7;
          }
        for (; i < n; i++)
          {
            float v = row[i] * inv;
            int iv = jx_rndf(v);
            if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
            q[i] = (signed char)iv;
          }
      }
      for (int k = n; k < n16; k++) { q[k] = 0; }
      return mx / 127.0f;
    }
  for (int i2 = 0; i2 < n; i2++)
    {
      float v = gamma[i2] * (x[i2] * nx) + beta[i2] + x[i2];
      float av = fabsf(v);
      if (row != NULL) { row[i2] = v; }
      if (av > mx) { mx = av; }
    }
  if (mx <= 1e-30f) { memset(q, 0, (size_t)n16); return 1e-30f; }
  {
    float inv = 127.0f / mx;
    for (int i2 = 0; i2 < n; i2++)
      {
        float y = (row != NULL) ? row[i2] : (gamma[i2] * (x[i2] * nx) + beta[i2] + x[i2]);
        float v = y * inv;
        int iv = jx_rndf(v);
        if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
        q[i2] = (signed char)iv;
      }
  }
  for (int i2 = n; i2 < n16; i2++) { q[i2] = 0; }
  return mx / 127.0f;
}

/* r73：jx_q_row_grn 的"趟 2 单独版"。
 *
 * 动机：r49 把 l2 的 GRN+量化折进 l1 的 epilogue 之后，jx_q_row_grn 变成
 * 每行两次全行扫描：趟1 读 row -> 乘 nx -> 乘 gamma -> 加 beta -> 加 row
 * -> fabs -> max（还要把中间结果写进 g_grnrow）；趟2 再读一遍做量化。
 * 实测这一段摊到 3.48M 元素上是 ~43 周期/元素（[i8mm] epilogue 总共
 * 72.36 = snake 29.5 + 本函数 43）。
 *
 * 而 l1 的 epilogue 本来就在逐元素算 snake 输出、并把它写进 g_l1row —
 * 只要顺手把 GRN 仿射做掉、并把 |u| 的**逐行 max** 一起统计出来
 * （浮点位整数比较，和 jx_q_row_grn 趟1 的 jx_uabs + max 完全同源），
 * 趟1 就整趟消失，g_grnrow 的写/读往返也一起消失。
 *
 * 数值：mx 必须**逐位**等于 jx_q_row_grn 趟1 会算出的那一个。
 *   - max 与次序无关，4 路部分量再合并与 8 路部分量再合并结果相同；
 *   - 每元素算式 (gamma*(x*nx) + beta) + x 与展开/循环次序无关；
 * 本函数其余部分（inv、取整、钳位、补零、返回值）与 jx_q_row_grn 的趟2
 * 逐字符相同 -> 结果逐位一致。 */
static float jx_q_row_quant(const float *row, signed char *q, int n, float mx)
{
  int n16 = (n + 15) & ~15;
  if (mx <= 1e-30f) { memset(q, 0, (size_t)n16); return 1e-30f; }
  {
    float inv = 127.0f / mx;
    int n8 = n & ~7;
    int i = 0;
    for (; i < n8; i += 8)
      {
        float v0 = row[i    ] * inv, v1 = row[i + 1] * inv;
        float v2 = row[i + 2] * inv, v3 = row[i + 3] * inv;
        float v4 = row[i + 4] * inv, v5 = row[i + 5] * inv;
        float v6 = row[i + 6] * inv, v7 = row[i + 7] * inv;
        int w0 = jx_rndf(v0);
        int w1 = jx_rndf(v1);
        int w2 = jx_rndf(v2);
        int w3 = jx_rndf(v3);
        int w4 = jx_rndf(v4);
        int w5 = jx_rndf(v5);
        int w6 = jx_rndf(v6);
        int w7 = jx_rndf(v7);
        JX_QCLAMP(w0); JX_QCLAMP(w1); JX_QCLAMP(w2); JX_QCLAMP(w3);
        JX_QCLAMP(w4); JX_QCLAMP(w5); JX_QCLAMP(w6); JX_QCLAMP(w7);
        q[i    ] = (signed char)w0; q[i + 1] = (signed char)w1;
        q[i + 2] = (signed char)w2; q[i + 3] = (signed char)w3;
        q[i + 4] = (signed char)w4; q[i + 5] = (signed char)w5;
        q[i + 6] = (signed char)w6; q[i + 7] = (signed char)w7;
      }
    for (; i < n; i++)
      {
        float v = row[i] * inv;
        int iv = jx_rndf(v);
        if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
        q[i] = (signed char)iv;
      }
  }
  for (int k = n; k < n16; k++) { q[k] = 0; }
  return mx / 127.0f;
}

/* ---------------- int8 点积：n16 = 向量个数（16 字节/向量） ----------------
 * PIE：一条 ee.vmulas.s8.accx 做 16 个 int8 MAC，累加进 40 位 accx。
 * 这里 4 路展开（4 次装载 + 4 条 MAC / 轮），实测 985 MMAC/s 形态。 */
JX_HOT static int32_t jx_dot_i8(const signed char *a, const signed char *b, int n16)
{
  int32_t acc = 0;
  if (n16 <= 0) { return 0; }

#if defined(JX_PIE)
  /* 2026-09-12 修正（本轮最大单点收益，来源：主机 MAC 轨迹 + 板端周期核算）
   *
   * 原实现只把 n16>=4 的那一段交给 PIE，n16<4 的"尾巴"走标量 C 循环。
   * 但本模型的点积绝大多数都很短：
   *     linear  in=28   -> n16 = 2      （全是尾巴）
   *     linear  in=16   -> n16 = 1      （全是尾巴）
   *     conv1d  Ci=20 k=1 -> n16 = 2    （全是尾巴）
   *     conv1d  Ci=16 k=1 -> n16 = 1    （全是尾巴）
   *     depthwise k=7     -> n16 = 1    （全是尾巴）
   * Xtensa LX7 没有 8 位乘法，标量 int8 乘加要经过 取字节+符号扩展+32 位乘+加，
   * 实测约 8 周期/MAC；于是这些点积 100% 走标量，成为 conv/linear 的唯一瓶颈
   * （板端核算：linear 8.4 周期/MAC、conv 21.5 周期/MAC，与纯 PIE 内核的
   *  0.08 周期/MAC 差两个数量级，与访存无关）。
   *
   * 改法：整段点积放进同一次 accx 会话——先 4 路展开，再用 1 路补齐余数，
   * 只在最后读一次 accx。结果与标量版逐位一致（int8×int8->int32 精确）。 */
  {
    int c4  = n16 >> 2;
    int rem = n16 & 3;
    const signed char *pa = a;
    const signed char *pb = b;
    int32_t r0 = 0;
    __asm__ __volatile__(
        "ee.zero.accx\n\t"
        "beqz %3, 2f\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %1, 16\n\t"
        "ee.vld.128.ip q1, %2, 16\n\t"
        "ee.vld.128.ip q2, %1, 16\n\t"
        "ee.vld.128.ip q3, %2, 16\n\t"
        "ee.vld.128.ip q4, %1, 16\n\t"
        "ee.vld.128.ip q5, %2, 16\n\t"
        "ee.vld.128.ip q6, %1, 16\n\t"
        "ee.vld.128.ip q7, %2, 16\n\t"
        "ee.vmulas.s8.accx q0, q1\n\t"
        "ee.vmulas.s8.accx q2, q3\n\t"
        "ee.vmulas.s8.accx q4, q5\n\t"
        "ee.vmulas.s8.accx q6, q7\n\t"
        "addi %3, %3, -1\n\t"
        "bnez %3, 1b\n\t"
        "2:\n\t"
        "beqz %4, 4f\n\t"
        "3:\n\t"
        "ee.vld.128.ip q0, %1, 16\n\t"
        "ee.vld.128.ip q1, %2, 16\n\t"
        "ee.vmulas.s8.accx q0, q1\n\t"
        "addi %4, %4, -1\n\t"
        "bnez %4, 3b\n\t"
        "4:\n\t"
        "rur.accx_0 %0\n\t"
        : "=r"(r0), "+r"(pa), "+r"(pb), "+r"(c4), "+r"(rem)
        :
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    acc = r0;
  }
#else
  /* 可移植参考实现（主机 / 无 PIE 目标），与 PIE 版逐位一致 */
  {
    const signed char *pa = a;
    const signed char *pb = b;
    for (int v = 0; v < n16; v++, pa += 16, pb += 16)
      {
        int32_t s = 0;
        for (int j = 0; j < 16; j++) { s += (int32_t)pa[j] * (int32_t)pb[j]; }
        acc += s;
      }
  }
#endif
  return acc;
}

/* ---- 注意力专用（2026-09-13 第七轮） ----
 * 行量化公开包装：把 n 个 float 量化成 int8（返回 scale = max|.|/127）。
 * jx_q_row 本来是 static，注意力侧要在 transformer.c 里逐行量化 q/k。 */
float jx_qrow_i8(const float *x, signed char *q, int n)
{
  return jx_q_row(x, q, n);
}

/* QKᵀ 批量点积：q16 是**恰好一个 16 字节向量**（本模型 dh=16），
 * K 是 nk 个 key 行紧密排列（每行 16 字节），out 收 nk 个 int32。
 * 每个 key 一条 ee.vmulas.s8.accx（16 个 MAC）+ 一次 rur 读回，
 * 实测 7 条指令/key（对标量版是 dh 次乘加 + 一次函数调用）。
 * int8×int8->int32 精确，与标量实现逐位一致。 */
void jx_qk_i8(const signed char *q16, const signed char *K, int nk, int32_t *out)
{
#if defined(JX_PIE)
  signed char *q = (signed char *)q16;
  signed char *k = (signed char *)K;
  int32_t     *o = out;
  int          c = nk;
  int32_t      t;
  if (nk <= 0) { return; }
  __asm__ __volatile__(
      "ee.vld.128.ip q4, %[Q], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q0, %[K], 16\n\t"
      "ee.zero.accx\n\t"
      "ee.vmulas.s8.accx q0, q4\n\t"
      "rur.accx_0 %[T]\n\t"
      "s32i %[T], %[O], 0\n\t"
      "addi %[O], %[O], 4\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
        : [Q] "+r"(q), [K] "+r"(k), [O] "+r"(o), [C] "+r"(c), [T] "=&r"(t)
        :
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
#else
  for (int i = 0; i < nk; i++) { out[i] = jx_dot_i8(q16, K + (size_t)i * 16u, 1); }
#endif
}

/* ---- 注意力 int8 开关（两个，分开控制） ----
 * QKᵀ 的 int8：**默认关**。原因（2026-09-13 实测）：q/k 按行 int8 量化后，
 * 点积的相对误差不是 1/127，而是 (Σ|q_d·k_d| / |Σ q_d·k_d|) × 1/127 ——
 * dh=16 的短点积在有相消时 L1/|和| 能到 10~30，实测重建 SNR 从 14.2 dB
 * 掉到 3.56 dB。要用必须换 int16（精度 1/32767）或共享 scale 的重排。
 * 需要时置 JX_ATTN_I8QK=1 打开（仅用于对照实验）。 */
int jx_attn_qk_i8_enabled(void)
{
  static int on = -1;
  if (on < 0)
    {
      const char *e = jx_getenv_guarded("JX_ATTN_I8QK");
      if (jx_i8_enabled() && e != NULL && e[0] != '0')
        { jx_pie_enable(); on = 1; }
      else { on = 0; }
    }
  return on;
}

/* attn·V 的 int8：**默认开**（JX_ATTN_I8=0 关闭）。
 * 与 QKᵀ 不同，这里是"权重非负的加权和"（softmax 权重 >= 0），没有相消，
 * 相对误差就是量化噪声本身（~1/127 + 1/127）；代价也低：v 转置成
 * [d][a] 并按列量化（列 scale），w 按行量化（w<=1，scale 固定 1/127），
 * 内层每 16 个 key 一条 ee.vmulas.s8.accx。 */
int jx_attn_sv_i8_enabled(void)
{
  static int on = -1;
  if (on < 0)
    {
      const char *e = jx_getenv_guarded("JX_ATTN_I8");
      if (jx_i8_enabled() && !(e != NULL && e[0] == '0'))
        { jx_pie_enable(); on = 1; }
      else { on = 0; }
    }
  return on;
}

/* attn·V 的 int8 内积：W 是量化后的 softmax 权重块（每 16 个 key 一字节组），
 * V 是**转置后**的 v 行（同一组 key，int8）。返回 int32 和。
 * 每 16 个 key：2 条 128-bit 装载 + 1 条 MAC。 */
int32_t jx_sv_i8(const signed char *W, const signed char *V, int nblk)
{
#if defined(JX_PIE)
  signed char *w = (signed char *)W;
  signed char *v = (signed char *)V;
  int          c = nblk;
  int32_t      t;
  if (nblk <= 0) { return 0; }
  __asm__ __volatile__(
      "ee.zero.accx\n\t"
      "1:\n\t"
      "ee.vld.128.ip q0, %[W], 16\n\t"
      "ee.vld.128.ip q1, %[V], 16\n\t"
      "ee.vmulas.s8.accx q1, q0\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      "rur.accx_0 %[T]\n\t"
        : [W] "+r"(w), [V] "+r"(v), [C] "+r"(c), [T] "=&r"(t)
        :
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
  return t;
#else
  int32_t acc = 0;
  for (int i = 0; i < nblk; i++)
    {
      for (int j = 0; j < 16; j++) { acc += (int32_t)W[i * 16 + j] * (int32_t)V[i * 16 + j]; }
    }
  return acc;
#endif
}

/* ---- 注意力 QKᵀ 的 int16 路径（2026-09-15 第三十六轮）----
 * int8 的 QK 精度不足（SNR 14.2 -> 3.56 dB），原因见 jx_qk_i8 上方注释：
 * dh=16 的短点积有相消，per-row int8 的相对误差被 L1/|sum| 放大。
 * 这里改用 13 位有效量化（±8191）：精度是 int8 的 65 倍，同时保证
 * 16 项点积 <= 16*8191^2 = 1.07e9 < 2^31，可安全地一次 zero +
 * 两条 ee.vmulas.s16.accx + 一次 rur.accx_0 取回，不溢出 int32。
 * 与 int8 一样逐位可解释：int16 x int16 -> int40 精确累加。 */
#define JX_I16_QMAX 8191

float jx_qrow_i16(const float *x, signed short *q, int n)
{
  int i;
  float mx = 0.0f;
  for (i = 0; i < n; i++)
    {
      float a = fabsf(x[i]);
      if (a > mx) { mx = a; }
    }
  if (mx <= 1e-30f)
    {
      for (i = 0; i < n; i++) { q[i] = 0; }
      return 1e-30f;
    }
  {
    float inv = (float)JX_I16_QMAX / mx;
    for (i = 0; i < n; i++)
      {
        float v = x[i] * inv;
        int w = jx_rndf(v);
        if (w > JX_I16_QMAX) { w = JX_I16_QMAX; }
        else if (w < -JX_I16_QMAX) { w = -JX_I16_QMAX; }
        q[i] = (signed short)w;
      }
  }
  return mx / (float)JX_I16_QMAX;
}

/* QKᵀ 批量点积（int16）：q16 是 dh=16 个 int16（32 字节，2 个 128 位向量），
 * K 是 nk 个 key 行紧密排列（每行 dh=16 个 int16），out 收 nk 个 int32。
 * 每个 key：2 条 128-bit 装载 + 2 条 ee.vmulas.s16.accx + 一次 rur。 */
void jx_qk_i16(const signed short *q16, const signed short *K, int nk, int32_t *out)
{
#if defined(JX_PIE)
  signed short *q = (signed short *)q16;
  signed short *k = (signed short *)K;
  int32_t     *o = out;
  int          c = nk;
  int32_t      t;
  if (nk <= 0) { return; }
  __asm__ __volatile__(
      "ee.vld.128.ip q0, %[Q], 16\n\t"
      "ee.vld.128.ip q1, %[Q], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q4, %[K], 16\n\t"
      "ee.vld.128.ip q5, %[K], 16\n\t"
      "ee.zero.accx\n\t"
      "ee.vmulas.s16.accx q0, q4\n\t"
      "ee.vmulas.s16.accx q1, q5\n\t"
      "rur.accx_0 %[T]\n\t"
      "s32i %[T], %[O], 0\n\t"
      "addi %[O], %[O], 4\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
        : [Q] "+r"(q), [K] "+r"(k), [O] "+r"(o), [C] "+r"(c), [T] "=&r"(t)
        :
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
#else
  for (int i = 0; i < nk; i++)
    {
      int32_t s = 0;
      for (int j = 0; j < 16; j++)
        {
          s += (int32_t)q16[j] * (int32_t)K[(size_t)i * 16u + j];
        }
      out[i] = s;
    }
#endif
}

/* QKᵀ int16：默认开（相对 fp32 的精度余量足够），JX_ATTN_I8QK16=0 关闭。 */
int jx_attn_qk_i16_enabled(void)
{
  static int on = -1;
  if (on < 0)
    {
      const char *e = jx_getenv_guarded("JX_ATTN_I8QK16");
      if (jx_i8_enabled() && !(e != NULL && e[0] == '0'))
        { jx_pie_enable(); on = 1; }
      else { on = 0; }
    }
  return on;
}

/* 微基准探针：板端 `jixun codec mem` 用来量短点积的真实往返开销。 */
/* ---------------- 行点积批：权重常驻 PIE 寄存器 ----------------
 * 2026-09-12 第五轮追加（本文件里性能最关键的一段）。
 *
 * 问题（板端实测 + 指令核算）：原先 Pass C 是
 *     for (co) for (j) jx_dot_i8(act[j], w[co], nrow)
 * 每个 112-MAC 的点积要花 14 次 128-bit 装载（7 次激活 + 7 次权重），
 * 而真正干活的只有 7 条 ee.vmulas —— 三分之二的协处理器操作都在搬数据。
 * 权重行在 j 方向上被完全重复装载了 nj 次。
 *
 * 改法：把权重行一次性装进 q0..q(N-1)，然后对 nj 个激活行依次点积。
 * ESP32-S3 的 PIE 只有 8 个 128 位寄存器（q0..q7，q8 以上汇编器直接报
 * register number out of range），所以 N 最多 4（权重 4 个 + 激活 4 个）。
 * N>4 时由 jx_dotrow_i8() 拆成两半，两半的 int32 部分和在整数里相加
 * —— 整数加法精确，结果与原来逐位一致。
 *
 * 每个点积的协处理器操作从 2N+2 降到 N+2（权重装载摊到 nj 行上）：
 *   N=7、nj=16 时，每通道 21*16 = 336 次 -> 7 + 16*(7+7+1) = 247 次，约 -26%。
 *
 * A：激活行起点，每行 astride 字节。asm 每行装 N 个向量后用
 *    pad = astride - N*16 补回行距，所以 astride 必须是 16 的倍数。
 * B：权重行起点（连续 N 个向量，无需 padding）。 */
JX_HOT static void jx_dotmj_i8(const signed char *A, const signed char *B,
                        int N, int nj, int pad, int32_t *out)
{
#if defined(JX_PIE)
  signed char *a = (signed char *)A;              /* asm 里会被推进 */
  signed char *b = (signed char *)B;
  int32_t     *o = out;
  int          cnt = nj;
  int32_t      t;
  if (nj <= 0) { return; }
  switch (N)
    {
    case 1:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 2:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 3:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 4:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.vld.128.ip q3, %[B], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "ee.vld.128.ip q7, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "ee.vmulas.s8.accx q3, q7\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
      default: break;
    }
#else
  /* 主机 / 无 PIE 目标：等价的可移植实现 */
  for (int j = 0; j < nj; j++)
    {
      int32_t s = 0;
      const signed char *ar = A + (size_t)j * (size_t)(pad + N * 16);
      for (int n = 0; n < N; n++)
        for (int e = 0; e < 16; e++)
          s += (int32_t)ar[n * 16 + e] * (int32_t)B[n * 16 + e];
      out[j] = s;
    }
#endif
}

/* 2026-09-15 v134：fused-load-MAC 版本。
 * 把「vld + vmulas」合并成一条 ee.vmulas.s8.accx.ld.ip：
 * 加载 16 字节激活的同时直接乘加，少一条发射指令，也让协处理器
 * 自己把装载和 MAC 排进同一条流水。
 * 数值与 jx_dotmj_i8 完全相同：同一组 int8 输入、同一累加次序。
 * 默认关，JX_DOTF=1 打开，方便 A/B。 */
static int jx_dotf(void)
{
  static int v = -2;
  if (v == -2)
    {
      const char *e = jx_getenv_guarded("JX_DOTF");
      v = (e != NULL && e[0] == '1');
    }
  return v;
}

JX_HOT static void jx_dotmj_i8_ld(const signed char *A, const signed char *B,
                           int N, int nj, int pad, int32_t *out)
{
#if defined(JX_PIE)
  signed char *a = (signed char *)A;
  signed char *b = (signed char *)B;
  int32_t     *o = out;
  int          cnt = nj;
  int32_t      t;
  if (nj <= 0) { return; }
  switch (N)
    {
    case 1:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx.ld.ip q4, %[A], 16, q4, q0\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 2:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx.ld.ip q4, %[A], 16, q4, q0\n\t"
        "ee.vmulas.s8.accx.ld.ip q5, %[A], 16, q5, q1\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 3:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx.ld.ip q4, %[A], 16, q4, q0\n\t"
        "ee.vmulas.s8.accx.ld.ip q5, %[A], 16, q5, q1\n\t"
        "ee.vmulas.s8.accx.ld.ip q6, %[A], 16, q6, q2\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 4:
      __asm__ __volatile__(
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.vld.128.ip q3, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx.ld.ip q4, %[A], 16, q4, q0\n\t"
        "ee.vmulas.s8.accx.ld.ip q5, %[A], 16, q5, q1\n\t"
        "ee.vmulas.s8.accx.ld.ip q6, %[A], 16, q6, q2\n\t"
        "ee.vmulas.s8.accx.ld.ip q7, %[A], 16, q7, q3\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[A], %[A], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
      default: break;
    }
#else
  for (int j = 0; j < nj; j++)
    {
      int32_t ss = 0;
      const signed char *ar = A + (size_t)j * (size_t)(pad + N * 16);
      for (int n = 0; n < N; n++)
        for (int e = 0; e < 16; e++)
          ss += (int32_t)ar[n * 16 + e] * (int32_t)B[n * 16 + e];
      out[j] = ss;
    }
#endif
}

/* 对一行权重、nj 个激活行做点积，覆盖任意 nrow。
 *   nrow <= 4 : 一次装完
 *   5..8      : 分两半（见上面注释）
 *   >8        : 退回逐个点积（本模型里只占少量 MAC） */
static void jx_dotrow_i8(const signed char *A, int astride, const signed char *B,
                         int nrow, int nj, int32_t *out)
{
  if (nj <= 0) { return; }
  if (nrow <= 4)
    {
      if (jx_dotf()) { jx_dotmj_i8_ld(A, B, nrow, nj, astride - nrow * 16, out); }
      else          { jx_dotmj_i8(A, B, nrow, nj, astride - nrow * 16, out); }
      return;
    }
  if (nrow <= 8)
    {
      int32_t tmp[JX_I8TN_N];
      jx_dotmj_i8(A, B, 4, nj, astride - 64, out);
      jx_dotmj_i8(A + 64, B + 64, nrow - 4, nj, astride - (nrow - 4) * 16, tmp);
      for (int j = 0; j < nj; j++) { out[j] += tmp[j]; }
      return;
    }
  for (int j = 0; j < nj; j++)
    {
      out[j] = jx_dot_i8(A + (size_t)j * (size_t)astride, B, nrow);
    }
}

/* ---------------- 列点积批：一行激活常驻 PIE 寄存器 ----------------
 * 2026-09-13 第五轮追加。
 *
 * 问题：linear 的内层是
 *     for (o) for (iv) jx_dot_i8(act_row, w[o], nvec)
 * 每个输出元素都走一次函数调用 + 一次 accx 归零 + 一次 rur 读回，再乘一遍
 * scale。而在本模型里 in 很小（16/28/64 -> nvec = 1/2/4），一次点积只有
 * 1~4 条 ee.vmulas，调用与收尾开销远比 MAC 本身大。板端实测 linear
 * 3.21 周期/MAC，而同形态的纯 PIE 微基准是 0.24 周期/MAC。
 *
 * 改法：激活行（in16 <= 64 字节）一次性装进 q4..q7，然后对 nout 个权重行
 * 循环，每轮只装权重 + vmulas + 读回 + 存 int32。调用开销从"每个输出元素
 * 一次"摊成"每行激活一次"，浮点收尾也整体挪到循环外。
 *
 * 结果逐位一致：int8 x int8 -> int32 是精确累加，每个输出元素的点积次序
 * 与 jx_dot_i8 完全相同（低地址在前），只是不再每次都重新装激活。
 * nvec > 4 或 nvec < 1 时由调用方退回 jx_dot_i8。 */

JX_HOT static void __attribute__((noinline))
jx_dotcol_i8(const signed char *A, const signed char *B, int nvec,
                         int stride, int nout, int32_t *out)
{
#if defined(JX_PIE)
  signed char *a = (signed char *)A;
  signed char *b = (signed char *)B;
  int32_t     *o = out;
  int          cnt = nout;
  int          pad = stride - nvec * 16;
  int32_t      t;
  if (nout <= 0) { return; }
  switch (nvec)
    {
    case 1:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 2:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 3:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 4:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "ee.vld.128.ip q7, %[A], 16\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.vld.128.ip q3, %[B], 16\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "ee.vmulas.s8.accx q3, q7\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
      default: break;
    }
#else
  for (int i = 0; i < nout; i++)
    {
      out[i] = jx_dot_i8(A, B + (size_t)i * (size_t)stride, nvec);
    }
#endif
}

/* r147：jx_dotcol_i8 的**延迟读回**版（JX_K1DP=1 启用，默认 1）。
 * 原版每轮都是 zero.accx -> vmulas... -> rur.accx_0 -> s32i **首尾相接**：
 * rur 在 vmulas 结果刚出流水线时就去读，accx 的读出延迟完全暴露在关键路径上，
 * 实测约 2.5 CPI（9~11 条指令/输出却要 22 周期）。这里把"取下一行权重"
 * 插到 vmulas 与 rur 之间：vmulas 是在**发射时**读 q 寄存器的，之后再覆写
 * 同号 q 寄存器是安全的，于是 rur 之前多了 4~6 条指令的间隔，把 accx
 * 读出延迟藏进去。
 * 累加次序、每条输出元素的算式与 jx_dotcol_i8 完全相同 -> 逐位一致。 */
static int jx_k1dp(void)
{
  static int v = -2;
  /* r148：板端同固件 A/B 实测与 r145 原版完全同速（3670/3670），没有任何收益，
   * 因此默认关闭，保留代码备查（JX_K1DP=1 可开）。 */
  if (v == -2) { const char *e = jx_getenv_guarded("JX_K1DP"); v = (e != NULL && e[0] != (char)48); }
  return v;
}

JX_HOT static void __attribute__((noinline))
jx_dotcol_i8_p(const signed char *A, const signed char *B, int nvec,
               int stride, int nout, int32_t *out)
{
#if defined(JX_PIE)
  signed char *a = (signed char *)A;
  signed char *b = (signed char *)B;
  int32_t     *o = out;
  int          cnt = nout;
  int          pad = stride - nvec * 16;
  int32_t      t;
  if (nout <= 0) { return; }
  switch (nvec)
    {
    case 1:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "addi %[C], %[C], -1\n\t"
        "beqz %[C], 2f\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "add %[B], %[B], %[P]\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 2:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "addi %[C], %[C], -1\n\t"
        "beqz %[C], 2f\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "add %[B], %[B], %[P]\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 3:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "addi %[C], %[C], -1\n\t"
        "beqz %[C], 2f\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "add %[B], %[B], %[P]\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    case 4:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "ee.vld.128.ip q7, %[A], 16\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.vld.128.ip q3, %[B], 16\n\t"
        "1:\n\t"
        "ee.zero.accx\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "ee.vmulas.s8.accx q3, q7\n\t"
        "addi %[C], %[C], -1\n\t"
        "beqz %[C], 2f\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.vld.128.ip q3, %[B], 16\n\t"
        "add %[B], %[B], %[P]\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
        "addi %[O], %[O], 4\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "rur.accx_0 %[T]\n\t"
        "s32i %[T], %[O], 0\n\t"
          : [A] "+r"(a), [B] "+r"(b), [O] "+r"(o), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      break;
    default: break;
    }
#else
  for (int i = 0; i < nout; i++)
    {
      out[i] = jx_dot_i8(A, B + (size_t)i * (size_t)stride, nvec);
    }
#endif
}

/* ---------------- QACC 16 通道并行点积（JX_QACC A/B 开关） ----------------
 * QACC 在 8-bit 模式下是 16 个 20-bit 并行累加器。配合
 * ee.vsmulas.s8.qacc.ld.incp，一条指令同时完成 16 个输出通道的乘加，
 * 并自动装入下一组 16 个权重。板端微基准约 0.17 cyc/MAC。
 *
 * 约束：每个 lane 只有 20-bit，因此调用方必须按 <=2 个 16 字节块分批，
 * 然后把 int32 部分和相加；单次 nchunks=2 最坏累加 516128，仍小于 2^19。 */
static int jx_qacc_want(void)
{
  static int v = -2;
  if (v == -2)
    {
      const char *e = jx_getenv_guarded("JX_QACC");
      v = (e != NULL && e[0] == '1');
    }
  return v;
}

static inline int32_t jx_qacc_sxt20(uint32_t v)
{
  return (int32_t)((v ^ 0x80000u) - 0x80000u);
}

static void jx_qacc_unpack10(const uint32_t *qw, int32_t *out)
{
  /* r50：全展开 + 常量移位。原实现用运行期 bit/wi/sh，Xtensa 没有单指令
   * 变长移位，GCC 只能生成移位循环，16 条 lane 要 200+ 周期；板端实测
   * 读出本体 13.6 周期/输出，比 accx 路径的 7.5 还贵，QACC 因此整体亏。
   * 这里把同一份 20-bit 位域公式手工展开成常量移位：
   *   lane i 取 qw[wi]>>sh（跨字时再 | qw[wi+1]<<(32-sh)）
   * 再用 (int32_t)(v<<12)>>12 做 20-bit 符号扩展 —— 左移天然丢掉高位，
   * 所以不需要 & 0xfffff 掩码。数值与原实现逐位相同。 */
  uint32_t a0 = qw[0], a1 = qw[1], a2 = qw[2], a3 = qw[3], a4 = qw[4];
  uint32_t a5 = qw[5], a6 = qw[6], a7 = qw[7], a8 = qw[8], a9 = qw[9];
#define JX_Q20(v) ((int32_t)((uint32_t)(v) << 12) >> 12)
  out[0]  = JX_Q20(a0);
  out[1]  = JX_Q20((a0 >> 20) | (a1 << 12));
  out[2]  = JX_Q20(a1 >> 8);
  out[3]  = JX_Q20((a1 >> 28) | (a2 << 4));
  out[4]  = JX_Q20((a2 >> 16) | (a3 << 16));
  out[5]  = JX_Q20(a3 >> 4);
  out[6]  = JX_Q20((a3 >> 24) | (a4 << 8));
  out[7]  = JX_Q20(a4 >> 12);
  out[8]  = JX_Q20(a5);
  out[9]  = JX_Q20((a5 >> 20) | (a6 << 12));
  out[10] = JX_Q20(a6 >> 8);
  out[11] = JX_Q20((a6 >> 28) | (a7 << 4));
  out[12] = JX_Q20((a7 >> 16) | (a8 << 16));
  out[13] = JX_Q20(a8 >> 4);
  out[14] = JX_Q20((a8 >> 24) | (a9 << 8));
  out[15] = JX_Q20(a9 >> 12);
#undef JX_Q20
}

static void jx_qacc_dot16(const signed char *wT, const signed char *x,
                          int nchunks, int32_t out[16])
{
#if defined(JX_PIE)
  signed char *wp = (signed char *)wT;
  signed char *xp = (signed char *)x;
  int cnt = nchunks;
  uint32_t qw[10];
  __asm__ __volatile__(
      "ee.zero.qacc\n\t"
      "ee.vld.128.ip q0, %[W], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 0\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 1\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 2\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 3\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 4\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 5\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 6\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 7\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 8\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 9\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 10\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 11\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 12\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 13\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 14\n\t"
      "ee.vsmulas.s8.qacc.ld.incp q0, %[W], q0, q1, 15\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      "rur.qacc_l_0 %[Q0]\n\t"
      "rur.qacc_l_1 %[Q1]\n\t"
      "rur.qacc_l_2 %[Q2]\n\t"
      "rur.qacc_l_3 %[Q3]\n\t"
      "rur.qacc_l_4 %[Q4]\n\t"
      "rur.qacc_h_0 %[Q5]\n\t"
      "rur.qacc_h_1 %[Q6]\n\t"
      "rur.qacc_h_2 %[Q7]\n\t"
      "rur.qacc_h_3 %[Q8]\n\t"
      "rur.qacc_h_4 %[Q9]\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [C] "+r"(cnt),
        [Q0] "=&r"(qw[0]), [Q1] "=&r"(qw[1]), [Q2] "=&r"(qw[2]),
        [Q3] "=&r"(qw[3]), [Q4] "=&r"(qw[4]), [Q5] "=&r"(qw[5]),
        [Q6] "=&r"(qw[6]), [Q7] "=&r"(qw[7]), [Q8] "=&r"(qw[8]),
        [Q9] "=&r"(qw[9])
      :
      : "f0", "f1", "memory");
  jx_qacc_unpack10(qw, out);
#else
  (void)wT; (void)x; (void)nchunks;
  for (int o = 0; o < 16; o++) { out[o] = 0; }
#endif
}

static inline int64_t jx_qacc_sxt40(uint64_t v)
{
  return (int64_t)((v ^ (1ull << 39)) - (1ull << 39));
}

static void jx_qacc_unpack10_s16(const uint32_t *qw, int64_t *out)
{
  for (int i = 0; i < 8; i++)
    {
      int bit = i * 40;
      int wi = bit >> 5;
      int sh = bit & 31;
      uint64_t pair = qw[wi];
      if (wi + 1 < 10)
        {
          pair |= (uint64_t)qw[wi + 1] << 32;
        }
      uint64_t v = (pair >> sh) & ((1ull << 40) - 1u);
      out[i] = jx_qacc_sxt40(v);
    }
}

/* s16 QACC：8 个 40-bit lane，整段 K 累加也不易溢出。wT 布局为
 * [k][8 个输出通道]，x 为 [k]（每 8 个 int16 一组）。每个 chunk 是 8 个
 * 输入位置，LD.INCP 的地址会按 16 字节推进；首组权重在循环外预装，
 * 避免“初始 vld + 8 次 LD.INCP”把第二 chunk 的起始权重多跳一组。 */
static void jx_qacc_dot8_s16(const int16_t *wT, const int16_t *x,
                             int nchunks, int64_t out[8])
{
#if defined(JX_PIE)
  int16_t *wp = (int16_t *)wT;
  int16_t *xp = (int16_t *)x;
  int cnt = nchunks;
  uint32_t qw[10];
  __asm__ __volatile__(
      "ee.zero.qacc\n\t"
      "ee.vld.128.ip q0, %[W], 16\n\t"
      "1:\n\t"
      "ee.vld.128.ip q1, %[X], 16\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 0\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 1\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 2\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 3\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 4\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 5\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q2, %[W], q0, q1, 6\n\t"
      "ee.vsmulas.s16.qacc.ld.incp q0, %[W], q2, q1, 7\n\t"
      "addi %[C], %[C], -1\n\t"
      "bnez %[C], 1b\n\t"
      "rur.qacc_l_0 %[Q0]\n\t"
      "rur.qacc_l_1 %[Q1]\n\t"
      "rur.qacc_l_2 %[Q2]\n\t"
      "rur.qacc_l_3 %[Q3]\n\t"
      "rur.qacc_l_4 %[Q4]\n\t"
      "rur.qacc_h_0 %[Q5]\n\t"
      "rur.qacc_h_1 %[Q6]\n\t"
      "rur.qacc_h_2 %[Q7]\n\t"
      "rur.qacc_h_3 %[Q8]\n\t"
      "rur.qacc_h_4 %[Q9]\n\t"
      : [W] "+r"(wp), [X] "+r"(xp), [C] "+r"(cnt),
        [Q0] "=&r"(qw[0]), [Q1] "=&r"(qw[1]), [Q2] "=&r"(qw[2]),
        [Q3] "=&r"(qw[3]), [Q4] "=&r"(qw[4]), [Q5] "=&r"(qw[5]),
        [Q6] "=&r"(qw[6]), [Q7] "=&r"(qw[7]), [Q8] "=&r"(qw[8]),
        [Q9] "=&r"(qw[9])
      :
      : "f0", "f1", "f2", "memory");
  jx_qacc_unpack10_s16(qw, out);
#else
  (void)wT; (void)x; (void)nchunks;
  for (int o = 0; o < 8; o++) { out[o] = 0; }
#endif
}

int32_t jx_dot_i8_probe(const signed char *a, const signed char *b, int n16)
{
  return jx_dot_i8(a, b, n16);
}

/* ---------------- 第三十六轮：dot 阶段「结构」对照探针（jx mem [16]） ------
 * 背景：k=1 卷积加起来只占 185.5 MMAC 里的一小半，却要 ~1250 ms；而 PIE 的
 * 纯寄存器 MAC 上限是 0.08 周期/MAC、linear 路径实测 0.13 周期/MAC。
 * 差距全部落在「每个输出元素一次 accx 会话 + 一次 rur 读回」这个结构上。
 * 下面 4 个探针把同一份权重/激活（都放 .bss，L1 常驻）按不同结构跑，
 * 直接量出「周期 / 输出元素」，用来判定瓶颈到底是 MAC 吞吐、rur 往返，
 * 还是循环/调用开销。全部只做计时，不参与数值正确性。 */

void jx_dotcol_i8_probe(const signed char *A, const signed char *B, int nvec,
                        int stride, int nout, int32_t *out)
{
  jx_dotcol_i8(A, B, nvec, stride, nout, out);
}

/* dotcol 的「无读回」版：整批输出累加进同一个 accx，只在最后读一次。
 * 数值上没有意义（所有输出被加到一起），纯粹用来量「去掉每输出的
 * rur.accx_0 往返」之后还能跑到多少。 */
void jx_dotcol_i8_norb(const signed char *A, const signed char *B, int nvec,
                       int stride, int nout, int32_t *out)
{
#if defined(JX_PIE)
  signed char *a = (signed char *)A;
  signed char *b = (signed char *)B;
  int          cnt = nout;
  int          pad = stride - nvec * 16;
  int32_t      t;
  if (nout <= 0) { return; }
  switch (nvec)
    {
    case 1:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
        "rur.accx_0 %[T]\n\t"
          : [A] "+r"(a), [B] "+r"(b), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      out[0] = t;
      break;
    case 2:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
        "rur.accx_0 %[T]\n\t"
          : [A] "+r"(a), [B] "+r"(b), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      out[0] = t;
      break;
    case 3:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
        "rur.accx_0 %[T]\n\t"
          : [A] "+r"(a), [B] "+r"(b), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      out[0] = t;
      break;
    case 4:
      __asm__ __volatile__(
        "ee.vld.128.ip q4, %[A], 16\n\t"
        "ee.vld.128.ip q5, %[A], 16\n\t"
        "ee.vld.128.ip q6, %[A], 16\n\t"
        "ee.vld.128.ip q7, %[A], 16\n\t"
        "ee.zero.accx\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %[B], 16\n\t"
        "ee.vld.128.ip q1, %[B], 16\n\t"
        "ee.vld.128.ip q2, %[B], 16\n\t"
        "ee.vld.128.ip q3, %[B], 16\n\t"
        "ee.vmulas.s8.accx q0, q4\n\t"
        "ee.vmulas.s8.accx q1, q5\n\t"
        "ee.vmulas.s8.accx q2, q6\n\t"
        "ee.vmulas.s8.accx q3, q7\n\t"
        "add %[B], %[B], %[P]\n\t"
        "addi %[C], %[C], -1\n\t"
        "bnez %[C], 1b\n\t"
        "rur.accx_0 %[T]\n\t"
          : [A] "+r"(a), [B] "+r"(b), [C] "+r"(cnt), [T] "=&r"(t)
          : [P] "r"(pad)
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
      out[0] = t;
      break;
      default: break;
    }
#else
  out[0] = jx_dot_i8(A, B, nvec);
#endif
}

/* 一行权重 × nj 个激活位置（权重常驻寄存器，逐位置换激活） */
void jx_dotmj_i8_probe(const signed char *A, const signed char *B,
                       int N, int nj, int pad, int32_t *out)
{
  jx_dotmj_i8(A, B, N, nj, pad, out);
}

/* x 必须是**恰好一个** 16 字节向量时才是安全的；探针里只看时间。 */
void jx_qacc_dot16_probe(const signed char *wT, const signed char *x,
                         int nchunks, int32_t out[16])
{
  jx_qacc_dot16(wT, x, nchunks, out);
}

const signed char *jx_wq_blob_ptr(void)
{
  return g_wq_blob;
}

/* ---------------- linear: y[rows][out] = x[rows][in] * W[out][in]^T + b ---- */
int jx_linear_i8(const float *W, const float *b, const float *x, float *y,
                 int rows, int in, int out)
{
  return jx_linear_i8_ex(W, b, x, y, rows, in, out, NULL);
}

/* r56：把 l1 的行循环抽成**可重入**函数，供双核按行切分调用。
 * 每一行的计算只依赖本行的输入/权重（只读共享）与本核的 scratch，
 * 行与行之间没有任何依赖，所以按行切分与串行结果逐位相同。 */
typedef struct {
  const float *x; float *y; const float *b;
  const float *wscale; const signed char *wdata; const JXWQ *wq;
  const jx_linear_fuse_t *fuse;
  int rows, in, in16, out, rblk, oblk, in16q, qfuse;
} jx_l1ctx_t;

JX_HOT static void jx_linear_i8_rows(void *va, int rlo, int rhi)
{
  jx_l1ctx_t *cc = (jx_l1ctx_t *)va;
  const float *x      = cc->x;
  float       *y      = cc->y;
  const float *b      = cc->b;
  const float *wscale = cc->wscale;
  const signed char *wdata = cc->wdata;
  const JXWQ  *wq     = cc->wq;
  const jx_linear_fuse_t *fuse = cc->fuse;
  const int in = cc->in, in16 = cc->in16, out = cc->out;
  const int rblk = cc->rblk, oblk = cc->oblk, in16q = cc->in16q;
  const int qfuse = cc->qfuse;
  signed char *qb = scratch_q(rblk * in16);
  if (qb == NULL) { return; }
  /* r85：GRN 参数一次性搬进 SRAM（in <= JX_SN_MAX 时）。 */
  const float *gg = (fuse != NULL) ? fuse->grn_gamma : NULL;
  const float *gbp = (fuse != NULL) ? fuse->grn_beta : NULL;
  if (gg != NULL && gbp != NULL && in <= JX_SN_MAX)
    {
      int core = jx_core_id();
      for (int i = 0; i < in; i++) { g_gg[core][i] = gg[i]; g_gb[core][i] = gbp[i]; }
      gg = g_gg[core]; gbp = g_gb[core];
    }

    for (int r0 = rlo; r0 < rhi; r0 += rblk)
      {
        int r1 = r0 + rblk; if (r1 > rhi) { r1 = rhi; }
        float *sx = scratch_f(r1 - r0);
        if (sx == NULL) { return; }
        JXP_BEG(JP_I8Q);
#ifndef JX_NO_PROFILE
        unsigned _jqq0 = jx_ccount();
#endif
        {
        int r = r0;
        if (fuse == NULL || fuse->grn_gamma == NULL)
          {
            /* 2026-09-14 第十八轮：PW 行同时量化，PW 路 miss 重叠（JX_QPAR）
             * PW=0/1 保持原来的逐行调用，结果逐位相同。 */
            int par = (fuse != NULL && fuse->in_norm_w != NULL) ? 0 : jx_q_par();
            if (par == 4)
              {
                for (; r + 4 <= r1; r += 4)
                  {
                    jx_q_row4(x + (size_t)r * in, qb + (size_t)(r - r0) * in16, in, in16,
                              sx + (r - r0));
                  }
              }
            else if (par == 2)
              {
                for (; r + 2 <= r1; r += 2)
                  {
                    jx_q_row2(x + (size_t)r * in, qb + (size_t)(r - r0) * in16, in, in16,
                              sx + (r - r0));
                  }
              }
          }
        for (; r < r1; r++)
          {
            const float *xr = x + (size_t)r * in;
            if (fuse != NULL && fuse->in_norm_w != NULL)
              {
                int core = jx_core_id();
                if (g_normrow[core] == NULL || g_normcap[core] < in)
                  {
                    float *nn = (float *)realloc(g_normrow[core],
                                                sizeof(float) * (size_t)in);
                    if (nn == NULL) { return; }
                    g_normrow[core] = nn; g_normcap[core] = in;
                  }
                jx_norm_row(xr, g_normrow[core], in,
                            fuse->in_norm_w, fuse->in_norm_b);
                xr = g_normrow[core];
              }
            if (fuse != NULL && fuse->grn_gamma != NULL)
              {
                sx[r - r0] = jx_q_row_grn(xr, qb + (size_t)(r - r0) * in16, in,
                                          gg, gbp, fuse->grn_nx);
              }
            else
              {
                sx[r - r0] = jx_q_row_auto(xr, qb + (size_t)(r - r0) * in16, in);
              }
          }
        }
#ifndef JX_NO_PROFILE
        jx_cyc_q += (unsigned long)(unsigned)(jx_ccount() - _jqq0);
        jx_q_elems += (double)(r1 - r0) * (double)in;
#endif
        JXP_END(JP_I8Q);
    float *l1rowp = qfuse ? g_l1row[jx_core_id()] : NULL;
        /* r138：复用同一对行缓冲做 4 通道双行块（JX_L1R2，默认开）。 */
        float *l1rowb = NULL;
        if (qfuse && jx_l1r2_on()) { l1rowb = jx_l1row2_get(); if (l1rowb != NULL) { l1rowb += (size_t)jx_core_id() * JX_SN_MAX; } }
        for (int o0 = 0; o0 < out; o0 += oblk)
          {
            int o1 = o0 + oblk; if (o1 > out) { o1 = out; }
            if (o1 < out)
              {
                int nb = (out - o1 < oblk) ? (out - o1) : oblk;
                jx_dcp_pf_i8(wdata + (size_t)o1 * wq->kstride,
                          (unsigned)nb * (unsigned)wq->kstride);
              }
            int nv = in16 >> 4;
            int32_t *accb = g_ocacc[jx_core_id()];
            int32_t *accb2 = g_ocacc2[jx_core_id()];
            int cn = o1 - o0;
            /* r53：o 块入口把 wscale[]/b[] 读进 SRAM（原来内层 r 循环每行重读） */
            int cok = (cn <= JX_I8WSC_N);
            float *swc = g_wsc[jx_core_id()];
            float *sbb = g_bb [jx_core_id()];
            if (cok) { for (int i = 0; i < cn; i++) { swc[i] = wscale[o0 + i]; sbb[i] = (b != NULL) ? b[o0 + i] : 0.0f; } }
            if (nv >= 1 && nv <= 32 && cn <= JX_I8OC_N && cok)
              {
                /* 2026-09-13 第五轮：激活常驻 PIE 寄存器，批量算完再统一收尾。
                 * 见 jx_dotcol_i8() 的注释。每个元素的算式与旧路径逐位相同。
                 * nv>4 时 PIE 的 8 个寄存器装不下，拆成两半分别累加：
                 * int8×int8->int32 是精确累加，两半 int32 相加与单个 accx
                 * 一次累加完全等价（与求和次序无关）。 */
                const signed char *wblk = wdata + (size_t)o0 * wq->kstride;
                int kst = wq->kstride;
                if (fuse != NULL && fuse->snake_alpha != NULL)
                  {
                    /* 每个 o 块重建一次 snake 系数缓存（a 与 1/a，与 snake1d 的
                     * 预计算逐位一致），行循环里就不再做除法。 */
                    int core = jx_core_id();
                    for (int i = 0; i < cn; i++)
                      {
                        float av = fuse->snake_alpha[o0 + i] + XNN_EPS;
                        g_sna[core][i] = av;
                        g_sni[core][i] = 1.0f / av;
                      }
                    if (qfuse)
                      {
                        /* r85：GRN gamma/beta 同一块里搬进 SRAM（每个 o 块一次），
                         * 行循环内就不再逐元素去 PSRAM 取。 */
                        const float *qgs = fuse->q_gamma;
                        const float *qbs = fuse->q_beta;
                        for (int i = 0; i < cn; i++)
                          { g_gg[core][i] = qgs[o0 + i] + 1.0f; g_gb[core][i] = qbs[o0 + i]; }   /* r94: qgp = 1+gamma */
                      }
                  }
#ifndef JX_NO_PROFILE
                jx_fast_calls++;  jx_fast_elems += (double)(r1 - r0) * (double)cn;
#endif
                int _rpair = r0;
                /* ================= r138：双行块 =================
                 * 动机（r137 PFSTAT + r138 反汇编）：qfuse epilogue 每 4 个元素
                 * 96 条指令，其中 **32 条是 swc/sbb/sna/sni/qg/qb 的加载** ——
                 * 这 6 条数组都只按输出通道 i 索引，与行 r 无关，却在
                 * 「行外层」的循环里每行重读一遍。实测 40.52 周期/元素
                 * （IPC 0.59），是全场最大的单一开销桶（139.5 Mcyc）。
                 *
                 * 这里把行循环改成「两条**相邻**行成对、按 2 个通道展开」：
                 * 同一组 12 条参数加载同时喂 4 个元素（2 行 x 2 通道），
                 * 加载数 8 -> 4.5 条/元素，指令数 24 -> 18 条/元素；
                 * 同时 8 条独立链交错，盖住 FPU 4.13 周期的延迟。
                 *
                 * 逐位一致：每个元素自己的算式、运算次序与单行版逐字符相同
                 * （swap 只发生在**不同行之间**，行内 i 仍递增）；ssr 的读改写
                 * 仍是「先读本行槽、末尾写回本行槽」，槽号仍用 (r - r0)。
                 * 关闭（JX_L1R2=0）或拿不到第二块行缓冲时行为与以前完全相同。 */
                if (l1rowb != NULL && cok && nv <= 4 && nv >= 1 && cn <= JX_I8OC_N)
                  {
                    int core = jx_core_id();
                    const float *sna = g_sna[core];
                    const float *sni = g_sni[core];
                    const float * const _qg2 = g_gg[core];
                    const float * const _qb2 = g_gb[core];
                    int r = r0;
                    for (; r + 1 < r1; r += 2)
                      {
                        const signed char *qr0 = qb + (size_t)(r - r0) * in16;
                        const signed char *qr1 = qr0 + in16;
                        float s0 = sx[r - r0], s1 = sx[r - r0 + 1];
                        {
                          unsigned _jxp0 = jx_ccount();
                          jx_dotcol_i8(qr0, wblk, nv, kst, cn, accb);
                          jx_dotcol_i8(qr1, wblk, nv, kst, cn, accb2);
                          jx_cyc_mm += (unsigned long)(unsigned)(jx_ccount() - _jxp0);
                        }
                        /* r157（2026-09-17 双核竞态修复）：ss_rows 必须用**全局行号 r**。
                         * 原来用的是「本 rblk 块内的局部行号 r - r0」，而消费者（blocks.c
                         * 的 `for (r < rows) ss += ssr[r]`）用的是全局行号。双核切分后 CPU1 的 r0
                         * 从 mid 开始，两核于是往同一批槽位 [0, rows-mid) 里相加 —— 整机结果
                         * 每次上电都不一样（JX_PFL=0 把这条路径串行后则逐位稳定）。
                         * 改成全局 r 后两核写互不相交的行，且与单核语义一致。 */
                        float *ssr = fuse->ss_rows + r;
                        float ss0 = ssr[0], ss1 = ssr[1];
                        jx_elems_sinsq += (double)cn * 2.0;
                        {
                          float * __restrict yow0 = l1rowp;
                          float * __restrict yow1 = l1rowb;
                          int i = 0;
                          for (; i + 2 <= cn; i += 2)
                            {
                              float w0 = swc[i], w1 = swc[i + 1];
                              float b0 = sbb[i], b1 = sbb[i + 1];
                              float a0 = sna[i], a1 = sna[i + 1];
                              float n0 = sni[i], n1 = sni[i + 1];
                              float g0 = _qg2[i], g1 = _qg2[i + 1];
                              float h0 = _qb2[i], h1 = _qb2[i + 1];
                              float v0 = (float)accb[i]     * (s0 * w0) + b0;
                              float v1 = (float)accb[i + 1] * (s0 * w1) + b1;
                              float z0 = jx_sinsq_lut(a0 * v0);
                              float z1 = jx_sinsq_lut(a1 * v1);
                              v0 = v0 + n0 * z0;
                              v1 = v1 + n1 * z1;
                              yow0[i] = g0 * v0 + h0;
                              yow0[i + 1] = g1 * v1 + h1;
                              ss0 += v0 * v0;
                              ss0 += v1 * v1;
                              float u0 = (float)accb2[i]     * (s1 * w0) + b0;
                              float u1 = (float)accb2[i + 1] * (s1 * w1) + b1;
                              float y0 = jx_sinsq_lut(a0 * u0);
                              float y1 = jx_sinsq_lut(a1 * u1);
                              u0 = u0 + n0 * y0;
                              u1 = u1 + n1 * y1;
                              yow1[i] = g0 * u0 + h0;
                              yow1[i + 1] = g1 * u1 + h1;
                              ss1 += u0 * u0;
                              ss1 += u1 * u1;
                            }
                          for (; i < cn; i++)
                            {
                              float v = (float)accb[i] * (s0 * swc[i]) + sbb[i];
                              v = v + sni[i] * jx_sinsq_lut(sna[i] * v);
                              yow0[i] = _qg2[i] * v + _qb2[i];
                              ss0 += v * v;
                              float u = (float)accb2[i] * (s1 * swc[i]) + sbb[i];
                              u = u + sni[i] * jx_sinsq_lut(sna[i] * u);
                              yow1[i] = _qg2[i] * u + _qb2[i];
                              ss1 += u * u;
                            }
                        }
                        ssr[0] = ss0; ssr[1] = ss1;
                        {
                          unsigned _mm0 = jxu_maxabs(l1rowp, cn);
                          unsigned _mm1 = jxu_maxabs(l1rowb, cn);
                          fuse->out_sc[r] = jx_q_row_quant(l1rowp,
                              fuse->out_q + (size_t)r * (size_t)in16q, out, jx_u2f(_mm0));
                          fuse->out_sc[r + 1] = jx_q_row_quant(l1rowb,
                              fuse->out_q + (size_t)(r + 1) * (size_t)in16q, out, jx_u2f(_mm1));
                        }
#ifndef JX_NO_PROFILE
                        jx_dcol_calls += 2;
                        jx_epi_elems += (double)cn * 2.0;
#endif
                      }
                    _rpair = r;
                  }
                for (int r = _rpair; r < r1; r++)
                  {
                    const signed char *qr = qb + (size_t)(r - r0) * in16;
                    float *yo = y + (size_t)r * out;
                    /* r119：restrict —— 输出是 float*，与 swc/sbb/sna/sni/qg/qb
                     * 同类型，别名假设会强迫每个元素重取一遍。 */
                    float * __restrict const yow = qfuse ? l1rowp : yo;
                    float s = sx[r - r0];
#ifndef JX_NO_PROFILE
                    unsigned _jxc0 = jx_ccount();
#endif
                    /* 2026-09-13 第六轮：nv 不限 8。l2（in=dim4=352 -> nv=22）
                     * 原来掉进标量兜底，现在按 4 个向量一块批处理，块间用
                     * int32 部分和相加——int8xint8->int32 精确，结果逐位一致。 */
                    {
                      int c0 = (nv <= 4) ? nv : 4;
                      jx_dotcol_i8(qr, wblk, c0, kst, cn, accb);
                      while (c0 < nv)
                        {
                          int nc = nv - c0; if (nc > 4) { nc = 4; }
                          jx_dotcol_i8(qr + (size_t)c0 * 16u, wblk + (size_t)c0 * 16u,
                                       nc, kst, cn, accb2);
                          for (int i = 0; i < cn; i++) { accb[i] += accb2[i]; }
                          c0 += nc;
                        }
                    }
#ifndef JX_NO_PROFILE
                    unsigned _jxc1 = jx_ccount();
                    jx_cyc_mm += (unsigned long)(unsigned)(_jxc1 - _jxc0);
#endif
                    /* r73：qfuse 时逐行 max 在这里顺手统计（见 jx_q_row_quant） */
                    unsigned _mm = 0u;
                    if (fuse != NULL && fuse->snake_alpha != NULL)
                      {
                        int core = jx_core_id();
                        /* r157（2026-09-17 双核竞态修复）：ss_rows 必须用**全局行号 r**。
                         * 原来用的是「本 rblk 块内的局部行号 r - r0」，而消费者（blocks.c
                         * 的 `for (r < rows) ss += ssr[r]`）用的是全局行号。双核切分后 CPU1 的 r0
                         * 从 mid 开始，两核于是往同一批槽位 [0, rows-mid) 里相加 —— 整机结果
                         * 每次上电都不一样（JX_PFL=0 把这条路径串行后则逐位稳定）。
                         * 改成全局 r 后两核写互不相交的行，且与单核语义一致。 */
                        float *ssr = fuse->ss_rows + r;
                        const float *sna = g_sna[core];
                        const float *sni = g_sni[core];
                        /* 行首装载、行尾写回：循环内保持寄存器累加，且逐元素
                         * 累加次序与 grn_last 的 i 序完全一致（逐位相同）。 */
                        float ssr_acc = *ssr;
                        jx_elems_sinsq += (double)cn;
                        /* 2026-09-13 第九轮：4 路展开。
                         * 每个元素内部是一条 ~8 级串行的 Horner 链（延迟约
                         * 30+ 周期），一次只喂一条链时 FPU 流水线全在等延迟；
                         * 相邻元素彼此独立，展开后 4 条链交错执行。
                         * ssr 的累加次序仍按 i 递增（v0,v1,v2,v3），
                         * 与展开前逐位相同。展开段与尾段写到 yo 的位置、
                         * 每个元素的算式也完全不变。 */
                        /* 第三十九轮：4 个 sin² 系数在行循环里装载一次。 */
                        const jx_sq4 KS = jx_sq4_load();
                        int i = 0;
                        /* 2026-09-14 第三十三轮：4 路 -> 8 路。
                         * jx_sinsq 是一条 ~8 级、延迟 30+ 周期的 Horner 链，
                         * 4 条链仍喂不满单发射流水线；epilogue 实测 49.60
                         * 周期/元素（每元素约 22 条指令），还有一半是空泡。
                         * ssr 的累加次序仍严格按 i 递增（v0..v7），逐位一致。 */
                        /* 2026-09-14 第三十三轮：这一支试过 8 路展开，板端 A/B
                         * 反而把 epilogue 从 49.60 拉到 51.73 周期/元素 ——
                         * 每元素约 22 条指令，8 路让 8 个 v 同时活跃，
                         * 16 个 FP 寄存器装不下，出现溢出。保持 4 路。 */
                        if (qfuse)
                          {
                            /* r73：GRN 仿射（nx 恒为 1.0f，见 jx_linear_i8_ex 的调用）
                             * + |u| 逐行 max 一起做掉，l1rowp 存**GRN 之后**的行。
                             * 算式与 jx_q_row_grn 趟1 逐字符相同 -> 逐位一致。 */
                            const int _qcore = jx_core_id();
                            const float * const _qg = g_gg[_qcore];
                            const float * const _qb = g_gb[_qcore];
                            for (; i + 4 <= cn; i += 4)
                              {
                                int o = o0 + i;
                                float v0 = (float)accb[i]     * (s * swc[o - o0])     + sbb[o - o0];
                                float v1 = (float)accb[i + 1] * (s * swc[o + 1 - o0]) + sbb[o + 1 - o0];
                                float v2 = (float)accb[i + 2] * (s * swc[o + 2 - o0]) + sbb[o + 2 - o0];
                                float v3 = (float)accb[i + 3] * (s * swc[o + 3 - o0]) + sbb[o + 3 - o0];
                                float z0, z1, z2, z3;
                                jx_sinsq4_x4(sna[i]     * v0, sna[i + 1] * v1,
                                             sna[i + 2] * v2, sna[i + 3] * v3, KS,
                                             &z0, &z1, &z2, &z3);
                                v0 = v0 + sni[i]     * z0;
                                v1 = v1 + sni[i + 1] * z1;
                                v2 = v2 + sni[i + 2] * z2;
                                v3 = v3 + sni[i + 3] * z3;
                                float u0 = _qg[i]     * v0 + _qb[i];
                                float u1 = _qg[i + 1] * v1 + _qb[i + 1];
                                float u2 = _qg[i + 2] * v2 + _qb[i + 2];
                                float u3 = _qg[i + 3] * v3 + _qb[i + 3];
                                yow[o]     = u0; ssr_acc += v0 * v0;
                                yow[o + 1] = u1; ssr_acc += v1 * v1;
                                yow[o + 2] = u2; ssr_acc += v2 * v2;
                                yow[o + 3] = u3; ssr_acc += v3 * v3;
                              }
                            for (; i < cn; i++)
                              {
                                int o = o0 + i;
                                float v = (float)accb[i] * (s * swc[o - o0]) + sbb[o - o0];
                                v = v + sni[i] * jx_sinsq4(sna[i] * v, KS);
                                yow[o] = _qg[i] * v + _qb[i];    /* r94: qgp folded */
                                ssr_acc += v * v;
                              }
                          }
                        else
                          {
                        for (; i + 4 <= cn; i += 4)
                          {
                            int o = o0 + i;
                            float v0 = (float)accb[i]     * (s * swc[o - o0])     + sbb[o - o0];
                            float v1 = (float)accb[i + 1] * (s * swc[o + 1 - o0]) + sbb[o + 1 - o0];
                            float v2 = (float)accb[i + 2] * (s * swc[o + 2 - o0]) + sbb[o + 2 - o0];
                            float v3 = (float)accb[i + 3] * (s * swc[o + 3 - o0]) + sbb[o + 3 - o0];
                            /* r43f：层级并排的 4 路交错（算式逐项相同） */
                            float z0, z1, z2, z3;
                            jx_sinsq4_x4(sna[i]     * v0, sna[i + 1] * v1,
                                         sna[i + 2] * v2, sna[i + 3] * v3, KS,
                                         &z0, &z1, &z2, &z3);
                            v0 = v0 + sni[i]     * z0;
                            v1 = v1 + sni[i + 1] * z1;
                            v2 = v2 + sni[i + 2] * z2;
                            v3 = v3 + sni[i + 3] * z3;
                            yow[o]     = v0; ssr_acc += v0 * v0;
                            yow[o + 1] = v1; ssr_acc += v1 * v1;
                            yow[o + 2] = v2; ssr_acc += v2 * v2;
                            yow[o + 3] = v3; ssr_acc += v3 * v3;
                          }
                        for (; i < cn; i++)
                          {
                            int o = o0 + i;
                            float v = (float)accb[i] * (s * swc[o - o0]) + sbb[o - o0];
                            v = v + sni[i] * jx_sinsq4(sna[i] * v, KS);
                            yow[o] = v;
                            ssr_acc += v * v;
                          }
                          }
                        *ssr = ssr_acc;
                      }
                    else
                      {
                        /* 2026-09-14 第三十三轮：这一支原来完全没展开（逐元素）。
                         * l2（in=dim4, out=C, 无 snake）走的就是这里，每元素
                         * 只有 6 条指令却要 20+ 周期（cvt->mul->add->store 串行）。
                         * 8 路展开后 8 条独立链交错。逐元素算式不变。 */
                        int i = 0;
                        for (; i + 8 <= cn; i += 8)
                          {
                            int o = o0 + i;
                            yo[o]     = (float)accb[i]     * (s * swc[o - o0])     + sbb[o - o0];
                            yo[o + 1] = (float)accb[i + 1] * (s * swc[o + 1 - o0]) + sbb[o + 1 - o0];
                            yo[o + 2] = (float)accb[i + 2] * (s * swc[o + 2 - o0]) + sbb[o + 2 - o0];
                            yo[o + 3] = (float)accb[i + 3] * (s * swc[o + 3 - o0]) + sbb[o + 3 - o0];
                            yo[o + 4] = (float)accb[i + 4] * (s * swc[o + 4 - o0]) + sbb[o + 4 - o0];
                            yo[o + 5] = (float)accb[i + 5] * (s * swc[o + 5 - o0]) + sbb[o + 5 - o0];
                            yo[o + 6] = (float)accb[i + 6] * (s * swc[o + 6 - o0]) + sbb[o + 6 - o0];
                            yo[o + 7] = (float)accb[i + 7] * (s * swc[o + 7 - o0]) + sbb[o + 7 - o0];
                          }
                        for (; i < cn; i++)
                          {
                            int o = o0 + i;
                            yo[o] = (float)accb[i] * (s * swc[o - o0]) + sbb[o - o0];
                          }
                      }
                    if (qfuse)
                      {
                        /* r97：对刚写出的 l1rowp[0..cn) 求 max —— 与原来逐元素
                         * jx_uabs 累加等价（qfuse 保证 oblk==out、o0==0），
                         * 但走 PIE 128 位，省掉每元素 5 条栈往返指令。 */
                        _mm = jxu_maxabs(l1rowp, cn);
                      }
                    if (qfuse)
                      {
                        /* r49：这一行 out 个通道已全部算完（oblk == out），就地
                         * 做 GRN 仿射 + 逐行 max + 量化；算式与 l2 prologue 的
                         * jx_q_row_grn(..., nx=1.0f) 完全同一份代码，逐位一致。 */
                        fuse->out_sc[r] = jx_q_row_quant(l1rowp,
                            fuse->out_q + (size_t)r * (size_t)in16q, out,
                            jx_u2f(_mm));
                      }
#ifndef JX_NO_PROFILE
                    jx_cyc_ep += (unsigned long)(unsigned)(jx_ccount() - _jxc1);
                    jx_dcol_calls++;
                    jx_epi_elems += (double)cn;
#endif
                  }
              }
            else
              {
                int _ob0 = o0;  (void)_ob0;
#ifndef JX_NO_PROFILE
                jx_slow_calls++;  jx_slow_elems += (double)(r1 - r0) * (double)(o1 - o0);
                { static int _ns = 0; if (_ns < 24) { _ns++; printf("[SLOW] rows=%d in=%d out=%d nv=%d cn=%d oblk=%d o0=%d cok=%d\n", (int)cc->rows, in, out, nv, cn, oblk, o0, cok); } }
#endif
                /* 第四十二轮：按 o 块预算 a 与 1/a（与快路径同一份缓存），

                 * jx_sq4 提到行循环外，省掉每元素一次浮点除法 + 7 次全局读。 */

                const jx_sq4 KSE = jx_sq4_load();
                if (fuse != NULL && fuse->snake_alpha != NULL)
                  {
                    int core = jx_core_id();
                    for (int i = 0; i < cn; i++)
                      {
                        float av = fuse->snake_alpha[o0 + i] + XNN_EPS;
                        g_sna[core][i] = av;
                        g_sni[core][i] = 1.0f / av;
                      }
                  }
                for (int r = r0; r < r1; r++)
                  {
                    const signed char *qr = qb + (size_t)(r - r0) * in16;
                    float *yo = y + (size_t)r * out;
                    float s = sx[r - r0];
                    jx_elems_sinsq += (double)(o1 - o0);
                    jx_epi_elems += (double)(o1 - o0);
                    for (int o = o0; o < o1; o++)
                      {
                        int32_t acc = jx_dot_i8(qr, wdata + (size_t)o * wq->kstride, in16 >> 4);
                        float v = (float)acc * (s * wscale[o]) + (b != NULL ? b[o] : 0.0f);
                        if (fuse != NULL && fuse->snake_alpha != NULL)
                          {
                            int _oi = o - o0;
                            float av = g_sna[jx_core_id()][_oi];
                            v = v + g_sni[jx_core_id()][_oi] * jx_sinsq_k(av * v, KSE);
                            fuse->ss_rows[r] += v * v;   /* r157: 全局行号，两核互不相交 */
                          }
                        yo[o] = v;
                      }
                  }
              }
          }
      }
}

int jx_linear_i8_ex(const float *W, const float *b, const float *x, float *y,
                    int rows, int in, int out, const jx_linear_fuse_t *fuse)
{
  const int _tr = (jx_getenv_guarded("JX_I8TRACE") != NULL);
  double _t0 = 0.0;
  const JXWQ *wq;
  const signed char *wdata;
  const float *wscale;
  int in16, rblk, oblk, ret = -1;

  if (!jx_i8_enabled() || in <= 0 || out <= 0 || rows <= 0) { return -1; }
  if (fuse != NULL && fuse->in_norm_w != NULL && in > JX_NORM_MAX) { return -1; }
  jx_sq_kt_ensure();
  if (fuse != NULL && fuse->snake_alpha != NULL && out > JX_SN_MAX) { return -1; }
  if (fuse != NULL && fuse->snake_alpha != NULL && fuse->ss_rows == NULL) { return -1; }
  if ((jx_i8_enabled() & 1) == 0) { return -1; }
  jx_pie_enable();
  jx_wq_init();
  wq = jx_wq_find(W);
  if (wq == NULL || wq->nch != out || wq->chlen != in) { return -1; }
  if (_tr) { _t0 = jx_prof_now(); }

  wdata  = g_wq_blob + wq->qoff;
  wscale = JXWQ_SCALES + wq->sc;
  in16   = (in + 15) & ~15;

  /* 分块：权重子块 + 量化后的 x 子块都留在 L1（32KB）内 */
  rblk = jx_i8_blk("JX_I8RB", 16384) / in16; if (rblk < 1) { rblk = 1; }
  if (rblk > rows) { rblk = rows; }
  oblk = jx_i8_blk("JX_I8OB", 16384) / wq->kstride; if (oblk < 1) { oblk = 1; }
  if (oblk > out) { oblk = out; }
  { static int _n1=0; if (jx_getenv_guarded("JX_LINFO") && _n1<40) { _n1++; printf("[l1] rows=%d in=%d out=%d rblk=%d oblk=%d kstr=%d\n", rows,in,out,rblk,oblk,(int)wq->kstride); } }
  /* r49：把 l2 的 GRN 仿射 + 逐行 max + 量化折进 l1 的 epilogue。
   * 条件：snake 融合已开、out_q/out_sc/out_qdone/q_gamma/q_beta 齐备、
   * oblk == out（epilogue 一次拿到整行）、out 是 16 的倍数（l2 的 in16 == out，
   * 量化缓冲的行步长才能对上）。任一不满足就完全走原路径。 */
  int qfuse = (fuse != NULL && fuse->snake_alpha != NULL &&
               fuse->out_q != NULL && fuse->out_sc != NULL &&
               fuse->out_qdone != NULL && fuse->q_gamma != NULL &&
               fuse->q_beta != NULL && oblk == out && (out & 15) == 0);
  const int in16q = (out + 15) & ~15;
  if (fuse != NULL && fuse->out_qdone != NULL) { *fuse->out_qdone = 0; }

    ret = 0;
    JXP_BEG(JP_I8MM);
    {
      jx_l1ctx_t cc;
      cc.x = x; cc.y = y; cc.b = b;
      cc.wscale = wscale; cc.wdata = wdata; cc.wq = wq; cc.fuse = fuse;
      cc.rows = rows; cc.in = in; cc.in16 = in16; cc.out = out;
      cc.rblk = rblk; cc.oblk = oblk; cc.in16q = in16q; cc.qfuse = qfuse;
      if (jx_pfx("JX_PFL")) { jx_pf_run(rows, jx_linear_i8_rows, &cc); }
      else { jx_linear_i8_rows(&cc, 0, rows); }
    }
    if (qfuse) { *fuse->out_qdone = 1; }
    JXP_END(JP_I8MM);
  if (_tr) printf("[I8EX] rows=%-6d in=%-5d out=%-5d snake=%d grn=%d %8.2f ms\n",
                  rows, in, out, fuse != NULL && fuse->snake_alpha != NULL,
                  fuse != NULL && fuse->grn_gamma != NULL, jx_prof_now() - _t0);
  return ret;
}

/* ---------------- linear 的 channels-first 输出变体（conv_unit l2 用）-------
 * 数学与 jx_linear_i8_ex 完全一致：输入 x[rows][in]，权重 W[out][in]。
 * 区别只在于输出直接写 y[out][rows]（channels-first），省掉 conv_unit 里
 * p2 [T][C] 的整块转置与临时张量往返。 */
/* r56：cf 变体的行循环抽成**可重入**函数，供双核按行切分调用。
 * 与 jx_linear_i8_rows 同理：每行的量化 scale / im2col / 点积 / epilogue
 * 只依赖本行输入与本核 scratch，行间零依赖 -> 切分后逐位相同。
 * 用到的 acc/acc2/tile 都是本函数栈上的临时量，天然 per-core。 */
typedef struct {
  const float *x; float *y; const float *b;
  const float *wscale; const signed char *wdata; const JXWQ *wq;
  const jx_linear_fuse_t *fuse;
  signed char *qb;
  int rows, in, in16, out, rblk, oblk, qin;
} jx_l2ctx_t;

JX_HOT static void jx_linear_i8_cf_rows(void *va, int rlo, int rhi)
{
  jx_l2ctx_t *cc = (jx_l2ctx_t *)va;
  const float *x      = cc->x;
  float       *y      = cc->y;
  const float *b      = cc->b;
  const float *wscale = cc->wscale;
  const signed char *wdata = cc->wdata;
  const JXWQ  *wq     = cc->wq;
  const jx_linear_fuse_t *fuse = cc->fuse;
  const int rows = cc->rows, in = cc->in, in16 = cc->in16;
  const int out = cc->out, rblk = cc->rblk, qin = cc->qin;
  const int nv = in16 >> 4;
  /* r157（2026-09-17 双核竞态修复）：分块量化缓冲必须**在本函数内**按当前核取。
   * 原来读的是 cc->qb —— 那是调用方（CPU0）用 jx_core_id() 取好后塞进 ctx 的，
   * worker（CPU1）拿到的是 CPU0 的那一份，两个核于是往同一块 SRAM 里写各自的
   * im2col 量化结果。症状：同一块板 / 同一份固件 / 同一份输入，每次上电
   * `jixun bench 1` 的 idx/rec 指纹都不一样（JX_PF=0 单核则逐位稳定）。
   * 改成本核现场取：两核各写各的，结果与串行逐位相同。 */
  signed char *qb     = qin ? NULL : scratch_q(rblk * in16);
  if (!qin && qb == NULL) { return; }
  /* r122：acc/acc2 从栈上的 VLA 换成 per-core 静态 SRAM。
   * 并行 worker 的 96KB 栈落在 PSRAM（见 jx_pf_ready 的 r76 注释），原来
   * `int32_t acc[out]; int32_t acc2[out]; float tile[1024];` 全是 PSRAM 上的
   * VLA —— 每个输出元素都要吃一次 PSRAM 往返。g_ocacc/g_ocacc2 本来就是
   * 这两条累加器的 per-core 静态版，直接复用。 */
  int32_t *acc  = g_ocacc [jx_core_id()];
  int32_t *acc2 = g_ocacc2[jx_core_id()];
  for (int r0 = rlo; r0 < rhi; r0 += rblk)
      {
        int r1 = r0 + rblk; if (r1 > rhi) { r1 = rhi; }
        int nr = r1 - r0;
        float *sx = qin ? ((float *)fuse->in_sc + r0) : scratch_f(nr);
        if (sx == NULL) { break; }
        const signed char *qbb = qin ? ((const signed char *)fuse->in_q +
                                        (size_t)r0 * (size_t)in16) : qb;
        if (!qin)
          {
        JXP_BEG(JP_I8Q);
#ifndef JX_NO_PROFILE
        unsigned _cfq0 = jx_ccount();
#endif
        for (int r = r0; r < r1; r++)
          {
            if (fuse != NULL && fuse->grn_gamma != NULL)
              {
                sx[r - r0] = jx_q_row_grn(x + (size_t)r * in,
                                          qb + (size_t)(r - r0) * in16, in,
                                          fuse->grn_gamma, fuse->grn_beta, fuse->grn_nx);
              }
            else
              {
                sx[r - r0] = jx_q_row_auto(x + (size_t)r * in,
                                           qb + (size_t)(r - r0) * in16, in);
              }
          }
#ifndef JX_NO_PROFILE
        jx_cf_cyc[0] += (unsigned long)(unsigned)(jx_ccount() - _cfq0);
        jx_cf_in_elems += (double)nr * (double)in;
        jx_cf_rows += (unsigned long)nr;
#endif
        JXP_END(JP_I8Q);
          }
        /* r122：循环次序改成「输出通道块外层 / 行内层」。
         *
         * 原写法（行外层、一次点积扫完全部 out）在 l2 上实测 243038 周期/行：
         * out=88、kstride=352 时每行要把 88*352 = 31KB 权重全部摸一遍，而
         * 它和刚被量化流冲过的 L1 争不过 —— 每一行都从 PSRAM 重读一次
         * （34KB @ ~40MB/s ≈ 186K 周期，与实测吻合）。
         * 改成 o 块外层后，块内权重（ob*kstride，默认目标 16KB）对整段 nr
         * 行只从 PSRAM 读一次，块间再做一次 DCP 预载。
         *
         * 同时去掉 tile 中转与转置：直接写 y[(o0+o)*rows + r0 + j] ——
         * 固定 o 时 j 连续（原来转置也是同一个写序），省掉 [cf] 里那
         * 28.9 Mcyc 的转置，也不再需要 4KB 的栈上 tile。
         *
         * 每个输出元素的算式、点积的分块累加次序（低地址在前、acc/acc2
         * 两半相加）一字未改 -> 逐位一致；写入地址与转置版完全相同。 */
        for (int o0 = 0; o0 < out; o0 += cc->oblk)
          {
            int o1 = o0 + cc->oblk; if (o1 > out) { o1 = out; }
            const int cn = o1 - o0;
            const signed char *wblk = wdata + (size_t)o0 * wq->kstride;
            const float *ws = wscale + o0;
            const float *bs = (b != NULL) ? (b + o0) : NULL;
            if (o1 < out)
              {
                int nb = (out - o1 < cc->oblk) ? (out - o1) : cc->oblk;
                jx_dcp_pf_i8(wblk + (size_t)cn * wq->kstride,
                             (unsigned)nb * (unsigned)wq->kstride);
              }
            for (int j = 0; j < nr; j++)
              {
                const signed char *qr = qbb + (size_t)j * in16;
                int c0 = (nv <= 4) ? nv : 4;
#ifndef JX_NO_PROFILE
                unsigned _cfd0 = jx_ccount();
#endif
                jx_dotcol_i8(qr, wblk, c0, wq->kstride, cn, acc);
                while (c0 < nv)
                  {
                    int nc = nv - c0; if (nc > 4) { nc = 4; }
                    jx_dotcol_i8(qr + (size_t)c0 * 16u, wblk + (size_t)c0 * 16u,
                                 nc, wq->kstride, cn, acc2);
                    for (int o = 0; o < cn; o++) { acc[o] += acc2[o]; }
                    c0 += nc;
                  }
#ifndef JX_NO_PROFILE
                unsigned _cfd1 = jx_ccount();
                jx_cf_cyc[1] += (unsigned long)(unsigned)(_cfd1 - _cfd0);
#endif
                const float s = sx[j];
                float * __restrict ybase = y + (size_t)o0 * rows + r0 + j;
                for (int o = 0; o < cn; o++)
                  {
                    ybase[(size_t)o * rows] = (float)acc[o] * (s * ws[o])
                                            + ((bs != NULL) ? bs[o] : 0.0f);
                  }
#ifndef JX_NO_PROFILE
                jx_cf_cyc[2] += (unsigned long)(unsigned)(jx_ccount() - _cfd1);
                jx_cf_elems += (double)cn;
#endif
              }
          }
      }
}

int jx_linear_i8_cf(const float *W, const float *b, const float *x, float *y,
                    int rows, int in, int out, const jx_linear_fuse_t *fuse)
{
  const int _tr = (jx_getenv_guarded("JX_I8TRACE") != NULL);
  double _t0 = 0.0;
  const JXWQ *wq;
  const signed char *wdata;
  const float *wscale;
  int in16, rblk, ret = -1;

  if (!jx_i8_enabled() || in <= 0 || out <= 0 || rows <= 0) { return -1; }
  if (fuse != NULL && fuse->snake_alpha != NULL) { return -1; }
  if ((jx_i8_enabled() & 1) == 0) { return -1; }
  jx_pie_enable();
  jx_wq_init();
  wq = jx_wq_find(W);
  if (wq == NULL || wq->nch != out || wq->chlen != in) { return -1; }
  if (_tr) { _t0 = jx_prof_now(); }

  wdata  = g_wq_blob + wq->qoff;
  wscale = JXWQ_SCALES + wq->sc;
  in16   = (in + 15) & ~15;

  rblk = jx_i8_blk("JX_I8RB", 16384) / in16; if (rblk < 1) { rblk = 1; }
  int rmax = jx_i8_blk("JX_I8R2", 4096) / out; if (rmax < 1) { rmax = 1; }
  if (rblk > rmax) { rblk = rmax; }
  if (rblk > rows) { rblk = rows; }
  /* r122：输出通道分块（r122 前是"每行扫全部 out"）。默认目标 16KB/块，
   * 让块内权重在 L1 里对整段 rblk 行复用。JX_I8OB2 可在板端扫。 */
  int oblk = jx_i8_blk("JX_I8OB2", 16384) / (int)wq->kstride; if (oblk < 1) { oblk = 1; }
  if (oblk > JX_I8OC_N) { oblk = JX_I8OC_N; }
  if (oblk > out) { oblk = out; }
  { static int _n2=0; if (jx_getenv_guarded("JX_LINFO") && _n2<40) { _n2++; printf("[l2] rows=%d in=%d out=%d rblk=%d rmax=%d oblk=%d kstr=%d\n", rows,in,out,rblk,rmax,oblk,(int)wq->kstride); } }

  {
    const int qin = (fuse != NULL && fuse->in_q != NULL && fuse->in_sc != NULL);
    signed char *qb = qin ? NULL : scratch_q(rblk * in16);
    if (qb == NULL && !qin) { return -1; }
    ret = 0;
    JXP_BEG(JP_I8MM);
    {
      jx_l2ctx_t cc;
      cc.x = x; cc.y = y; cc.b = b;
      cc.wscale = wscale; cc.wdata = wdata; cc.wq = wq; cc.fuse = fuse;
      cc.qb = qb;
      cc.rows = rows; cc.in = in; cc.in16 = in16; cc.out = out;
      cc.rblk = rblk; cc.oblk = oblk; cc.qin = qin;
      jx_pf_run(rows, jx_linear_i8_cf_rows, &cc);
    }
    JXP_END(JP_I8MM);
  }
  if (_tr) printf("[I8CF] rows=%-6d in=%-5d out=%-5d grn=%d %8.2f ms\n",
                  rows, in, out, fuse != NULL && fuse->grn_gamma != NULL,
                  jx_prof_now() - _t0);
  return ret;
}

/* ---------------- 密集卷积：时间分块 + 逐块预量化 + 批量点积 ----------------
 * 2026-09-13 第六轮（冲击 1s 级）。
 *
 * 目标：decoder 尾部 legacy_unit 的密集 16->16 k=7 膨胀卷积（T=16032，86 MMAC）。
 * 老路径每个输出位置都要对 112 个 tap 重新求 max 再量化（约 224 次浮点操作），
 * 实测 9.3 周期/MAC，是全场离 PIE 地板（0.43）最远的一档。
 *
 * 改法：把时间轴切成块，一块内
 *   1) 先求这一块的 scale（感受野 = 块宽 + (k-1)*dil），只扫一遍；
 *   2) 把块内输入量化成 int8（也只扫一遍），放在 L1 大小的静态块里；
 *   3) 每个输出位置只做 im2col 收集（int8 字节拷贝，无浮点）+ 批量 PIE 点积
 *      （激活常驻 q4..q7，复用 jx_dotcol_i8）。
 * 量化开销从"每个输出位置一次"摊成"每块一次"。
 *
 * 数值：scale 从"逐位置"改成"逐块"（块内 WT 个输出位置共用一个 scale，
 * WT 默认 256 ≈ 16ms 音频，动态范围有限）。这不是逐位一致改动，
 * 必须用 SNR 回归确认（host 对照，见 _REALTIME_ROUND6_20260913.md）。 */
/* r61b：容量**必须**盖住 WT=256 时所有层的需要，否则 WT 会被缩小、
 * 每块的量化 scale 跟着变 -> 不再逐位一致。
 * r60 用的 16384 就是这样被主机 SNR 回归抓到的（.idx 从第 82 帧起漂移）。
 * 需求 = max over layers of  cpg*(WT+(k-1)*dil+2*stride+2)：
 *   本模型 dense-tile 路径里最大的是 Ci=64,k=4,dil=1,stride=1
 *   -> 64*(256+3+2+2) = 16832。取 17408（17KB）留 576 字节余量。
 * 每核一份：2*17408 = 34816 字节（r57 单份 32768）。 */
#define JX_DTQ_BYTES 8192   /* r61c：dense tile 已改成按处理子块**流式**量化，
                             * 缓冲只需容纳一个子块的窗口（实测最大 ~3KB）。
                             * 注意：这个值**不再影响数值**（TB 只改分块，
                             * 每块的 mx/is 仍在整个 WT 窗口上求），只要不为 0 就行。 */
static signed char g_dtq[JX_NCORE][JX_DTQ_BYTES] __attribute__((aligned(16)));

/* QACC 转置权重暂存：[k][16 个输出通道]。只用于 JX_QACC=1 且形状能放下的
 * dense-tile 路径；K16 * copg 超过该容量时自动回退原 accx 路径。 */
#define JX_QW_BYTES 2048   /* r61b：QACC 只有 JX_QACC=1 才走，已被实测证伪、
                            * 默认永不命中；8192 纯属占 DRAM。缩到 2048 腾出 6KB。 */
static signed char g_qwT[JX_QW_BYTES] __attribute__((aligned(16)));

/* 2026-09-14 第十六轮：dense tile 内部周期拆分（rsr.ccount，整数计数）。
 * 0=逐块量化 1=im2col 收集 2=dotcol 3=epilogue */
unsigned long jx_dt_cyc[4];
unsigned long jx_dt_pos;
unsigned long jx_dt_blk;
unsigned long jx_dt_calls;
/* 2026-09-15 第三十五轮：pert 路径 TN 分块里 QACC 与 accx 两条点积分支的
 * 周期计数（只看点积+epilogue 段，不含量化 prologue）。 */
unsigned long jx_k1q_cyc;
unsigned long jx_k1n_cyc;
unsigned long jx_k1q_pos;
unsigned long jx_k1n_pos;

/* r60：dense tile 的 t 块循环抽成**可重入**函数，供双核按块区间切分。
 * 每个块只碰自己的 g_dtq[core] 与 per-core 的 sq/tile，输出写到 yout 的
 * 不相交时间区间；块间零依赖 -> 切分后逐位相同。 */
typedef struct {
  const float *b; const float *xin; float *yout;
  const signed char *wgc; const float *wsc; const float *galph;
  int cpg, copg, k, stride, pad, dil, T_in, T_out, K, K16, kstride, gbase;
  int WT, cob, quse, inl;
  signed char *sq; float *tile; signed char *dq;
} jx_dtctx_t;

JX_HOT static void jx_conv1d_dense_tile_tb(void *va, int blo, int bhi)
{
  jx_dtctx_t *cc = (jx_dtctx_t *)va;
  const float *b = cc->b; const float *xin = cc->xin; float *yout = cc->yout;
  const signed char *wgc = cc->wgc; const float *wsc = cc->wsc;
  const int cpg = cc->cpg, copg = cc->copg, k = cc->k, stride = cc->stride;
  const int pad = cc->pad, dil = cc->dil, T_in = cc->T_in, T_out = cc->T_out;
  const int K = cc->K, K16 = cc->K16, kstride = cc->kstride, gbase = cc->gbase;
  const int WT = cc->WT, cob = cc->cob, quse = cc->quse;
  /* r157（2026-09-17 双核竞态修复）：sq/tile/dq 原来取自 cc->*，而那是调用方
   * （CPU0）用 jx_core_id() 取好塞进来的，worker（CPU1）跟着用 CPU0 那一份 —— 两核同时写
   * 同一块量化/暂存缓冲，整机结果每次上电都不同。
   * 改成本核现场取（g_tnq / g_ocacc2 / g_dtq 本就是 per-core 静态）。 */
  const int _jc_dt = jx_core_id();
  signed char *sq = g_tnq[_jc_dt];
  float *tile = (float *)g_ocacc2[_jc_dt];
  signed char *dq = g_dtq[_jc_dt];
  const int inl = cc->inl;
  const int nv = K16 >> 4;
  int32_t accb[copg];
  int32_t accb2[copg];
  (void)cob; (void)gbase;
  for (int _tb = blo; _tb < bhi; _tb++)
    {
      int t0 = _tb * WT;
      unsigned long _dt_t0 = jx_ccount();
      int t1 = t0 + WT; if (t1 > T_out) { t1 = T_out; }
      /* 本块输出 [t0,t1) 的感受野 [A,B]（含端点，输入坐标） */
      int A  = t0 * stride - pad;
      int Bp = (t1 - 1) * stride - pad + (k - 1) * dil;
      int a0 = A < 0 ? 0 : A;
      int b0 = Bp > T_in - 1 ? T_in - 1 : Bp;
      int W  = b0 - a0 + 1;
      if (W <= 0) { W = 1; }
      /* 内部块所有 tap 都在有效区内：免掉逐 tap 边界判断（gather 的主开销） */
      int nocheck = (t0 * stride - pad >= 0) &&
                    ((t1 - 1) * stride - pad + (k - 1) * dil <= T_in - 1);
      /* r74：逐块 scale。原写法虽然 8 路展开，但 8 个比较**都更新同一个
       * float mx** —— 整段是一条串行 fabsf+cmp+select 依赖链，实测
       * [dt] 量化桶 79.3 Mcyc / ~0.83M 元素 = **~99 周期/元素**，是全场
       * 最离谱的一处。改成 8 个独立累加器 + **整数（浮点位）**比较，与
       * jx_q_row_u8 / k1cf PassA 同一手法。
       *
       * 逐位一致：max 与次序无关，8 个独立部分量再串行合并 == 一条链扫完；
       * |x| 的位模式（去掉符号位）在有限非 NaN 值上与浮点数值同序，
       * 且 jx_uabs 对 +0/-0 给出同一码 -> mx 逐位相同。 */
      unsigned m0 = 0u, m1 = 0u, m2 = 0u, m3 = 0u;
      unsigned m4 = 0u, m5 = 0u, m6 = 0u, m7 = 0u;
      if (jx_dtmax() == 1)
        {
          /* r86：PIE 128 位扫描（见 jxu_maxabs） */
          int W_ = b0 - a0 + 1;
          for (int c = 0; c < cpg; c++)
            {
              unsigned v = jxu_maxabs(xin + (size_t)c * T_in + a0, W_);
              if (v > m0) { m0 = v; }
            }
        }
      else
      for (int c = 0; c < cpg; c++)
        {
          const float *xr = xin + (size_t)c * T_in;
          int j = a0;
          for (; j + 8 <= b0; j += 8)
            {
              unsigned v0 = jx_uabs(xr[j    ]); if (v0 > m0) { m0 = v0; }
              unsigned v1 = jx_uabs(xr[j + 1]); if (v1 > m1) { m1 = v1; }
              unsigned v2 = jx_uabs(xr[j + 2]); if (v2 > m2) { m2 = v2; }
              unsigned v3 = jx_uabs(xr[j + 3]); if (v3 > m3) { m3 = v3; }
              unsigned v4 = jx_uabs(xr[j + 4]); if (v4 > m4) { m4 = v4; }
              unsigned v5 = jx_uabs(xr[j + 5]); if (v5 > m5) { m5 = v5; }
              unsigned v6 = jx_uabs(xr[j + 6]); if (v6 > m6) { m6 = v6; }
              unsigned v7 = jx_uabs(xr[j + 7]); if (v7 > m7) { m7 = v7; }
            }
          for (; j <= b0; j++)
            { unsigned vv = jx_uabs(xr[j]); if (vv > m0) { m0 = vv; } }
        }
      float mx;
      {
        unsigned mm = m0;
        if (m1 > mm) { mm = m1; }
        if (m2 > mm) { mm = m2; }
        if (m3 > mm) { mm = m3; }
        if (m4 > mm) { mm = m4; }
        if (m5 > mm) { mm = m5; }
        if (m6 > mm) { mm = m6; }
        if (m7 > mm) { mm = m7; }
        mx = jx_u2f(mm);
      }

      { static int _dz = 0; if (jx_getenv_guarded("JX_DT_TRACE") != NULL && _dz < 6) { _dz++; unsigned _mu; float _tf = mx; memcpy(&_mu, &_tf, 4); printf("[DTZ] post=%d alpha=%p W=%d a0=%d b0=%d mx=%08x t0=%d t1=%d\n", (int)jx_conv_post, (void *)jx_conv_post_alpha, (int)W, a0, b0, _mu, t0, t1); } }
      float s  = (mx > 1e-30f) ? (mx / 127.0f) : 1e-30f;
      float is = (mx > 1e-30f) ? (127.0f / mx) : 0.0f;
      /* 2026-09-14 第十七轮：布局从 [c*W + j] 改成**通道交错** [j*cpg + c]。
       *
       * 板端 [dt] 探针：每个输出位置的"收集"要花 998 周期，而它只是把
       * cpg*k 个字节拼进 sq。按老布局，dil==1 时每个通道一次 memcpy(k)
       * —— 88 次小 memcpy 的循环就是 998 周期的主因。
       * 交错布局后 k 个 tap 在通道方向上连续，dil==1 时退化成
       * **一次连续的 memcpy(k*cpg)**（616 字节），同一份数据、同一顺序，
       * 拼出来的 sq 与原来逐字节相同。
       *
       * 2026-09-15：量化填充改成 c 外层、j 内层 8 路展开。读 x 变成每个
       * 通道内连续流，g_dtq 的写是 cpg 字节等距的 SRAM 写（同一 cache
       * line 内），元素公式与舍入顺序不变。 */
      /* r66 计时修正：_dt_t0 原来只在 _tb 入口取一次，而累加在 ts 子块
       * 循环里，多子块时会把前一个子块的 gather/dot/epi 重复计入 [0]，
       * 实测把量化桶放大 3 倍。这里在 mx 扫描之后先把这段记掉并重置。 */
      jx_dt_cyc[0] += (unsigned long)(unsigned)(jx_ccount() - _dt_t0);
      _dt_t0 = jx_ccount();
      jx_dt_blk++;
      /* 每个输出位置：im2col 收集 + 批量点积 */
      /* r70：子块宽度从 JX_I8OC_N/copg 解耦。原写法 copg=16 时 TB=64，
       * 一个 WT=256 的块被拆成 4 个量化子块，每个子块把自己的窗口
       * （TB + (k-1)*dil 个样点）重新量化一遍 -> 量化工作量放大约 1.5 倍。
       * 这里直接取 TB = WT（受 dq 容量约束时下面的 while 会自动二分缩小）。
       * 每个输出位置的 is = 127/mx 仍在**整个 WT 窗口**上求，TB 只影响分块，
       * 结果逐位相同。 */
      /* r70 诊断：JX_DTTB>0 时覆盖子块宽度（0/未设 = 旧行为 JX_I8OC_N/copg）。 */
      int _dtb = jx_i8_blk("JX_DTTB", 0);
      int TB = (_dtb > 0) ? _dtb : (JX_I8OC_N / copg);
      if (TB < 1) { TB = 1; }
      if (TB > WT) { TB = WT; }
      /* r61c：dq 只放**一个处理子块**的窗口，所以先把 TB 压到
       *   cpg*((TB-1)*stride + (k-1)*dil + 1) <= JX_DTQ_BYTES
       * 为止（至少 1）。每个输出位置的 int8 激活只由本 WT 块的
       * is = 127/mx 决定（mx 仍在整个块窗口 [a0,b0] 上求），与 TB 无关，
       * 所以 TB 变小只多做一点重叠区的量化，结果逐位相同。
       * 旧版门禁是 cpg*(WT+(k-1)*dil+2*stride+2)：stride>1 时它比真实
       * 需求 (WT-1)*stride+(k-1)*dil+1 **小得多** —— r60 的 16384 正是
       * 在 stride=4/6 的层上写爆了 g_dtq（主机 SNR 从第 82 帧起漂移）。 */
      while (TB > 1 &&
             (size_t)cpg * (size_t)((TB - 1) * stride + (k - 1) * dil + 1) >
             (size_t)JX_DTQ_BYTES)
        { TB >>= 1; }
      for (int ts = t0; ts < t1; ts += TB)
        {
          int te = ts + TB; if (te > t1) { te = t1; }
          /* 本子块真正用到的输入窗口 [aS,bS]（含端点） */
          int As = ts * stride - pad;
          int Bs = (te - 1) * stride - pad + (k - 1) * dil;
          int aS = As < 0 ? 0 : As;
          int bS = Bs > T_in - 1 ? T_in - 1 : Bs;
          int WS = bS - aS + 1; if (WS <= 0) { WS = 1; }
          int nchk = (As >= 0) && (Bs <= T_in - 1);
          /* r70：4 通道打包写。原写法逐通道处理、每个元素一条 8 位 store，
           * 且同一次展开里的 8 条 store 相隔 cpg 字节（cpg=16 时跨越 128 字节）。
           * 同一 j 上连续 4 个通道的 4 个字节落在 [j*cpg + c4 .. +3]，连续且
           * 4 字节对齐（cpg 是 4 的倍数时），于是合成一条 32 位 store。
           * 写入的字节值与逐条 8 位 store 逐字节相同 -> 结果逐位一致。 */
          int c4 = 0;
          if ((cpg & 3) == 0)
            {
              for (; c4 + 4 <= cpg; c4 += 4)
                {
                  const float *x0 = xin + (size_t)(c4    ) * T_in + aS;
                  const float *x1 = xin + (size_t)(c4 + 1) * T_in + aS;
                  const float *x2 = xin + (size_t)(c4 + 2) * T_in + aS;
                  const float *x3 = xin + (size_t)(c4 + 3) * T_in + aS;
                  for (int j = 0; j < WS; j++)
                    {
                      float a0 = x0[j] * is, a1 = x1[j] * is;
                      float a2 = x2[j] * is, a3 = x3[j] * is;
                      int i0 = jx_rndf(a0);
                      int i1 = jx_rndf(a1);
                      int i2 = jx_rndf(a2);
                      int i3 = jx_rndf(a3);
                      if (i0 > 127) { i0 = 127; } else if (i0 < -127) { i0 = -127; }
                      if (i1 > 127) { i1 = 127; } else if (i1 < -127) { i1 = -127; }
                      if (i2 > 127) { i2 = 127; } else if (i2 < -127) { i2 = -127; }
                      if (i3 > 127) { i3 = 127; } else if (i3 < -127) { i3 = -127; }
                      unsigned pk = ((unsigned)i0 & 0xffu) | (((unsigned)i1 & 0xffu) << 8)
                                  | (((unsigned)i2 & 0xffu) << 16) | (((unsigned)i3 & 0xffu) << 24);
                      *(jx_u32a *)(void *)(dq + (size_t)j * cpg + c4) = (jx_u32a)pk;
                    }
                }
            }
          for (; c4 < cpg; c4++)
            {
              const float *xr = xin + (size_t)c4 * T_in + aS;
              int j = 0;
              for (; j + 8 <= WS; j += 8)
                {
                  float v0 = xr[j] * is, v1 = xr[j + 1] * is;
                  float v2 = xr[j + 2] * is, v3 = xr[j + 3] * is;
                  float v4 = xr[j + 4] * is, v5 = xr[j + 5] * is;
                  float v6 = xr[j + 6] * is, v7 = xr[j + 7] * is;
                  int w0 = jx_rndf(v0);
                  int w1 = jx_rndf(v1);
                  int w2 = jx_rndf(v2);
                  int w3 = jx_rndf(v3);
                  int w4 = jx_rndf(v4);
                  int w5 = jx_rndf(v5);
                  int w6 = jx_rndf(v6);
                  int w7 = jx_rndf(v7);
                  if (w0 > 127) { w0 = 127; } else if (w0 < -127) { w0 = -127; }
                  if (w1 > 127) { w1 = 127; } else if (w1 < -127) { w1 = -127; }
                  if (w2 > 127) { w2 = 127; } else if (w2 < -127) { w2 = -127; }
                  if (w3 > 127) { w3 = 127; } else if (w3 < -127) { w3 = -127; }
                  if (w4 > 127) { w4 = 127; } else if (w4 < -127) { w4 = -127; }
                  if (w5 > 127) { w5 = 127; } else if (w5 < -127) { w5 = -127; }
                  if (w6 > 127) { w6 = 127; } else if (w6 < -127) { w6 = -127; }
                  if (w7 > 127) { w7 = 127; } else if (w7 < -127) { w7 = -127; }
                  dq[(size_t)(j + 0) * cpg + c4] = (signed char)w0;
                  dq[(size_t)(j + 1) * cpg + c4] = (signed char)w1;
                  dq[(size_t)(j + 2) * cpg + c4] = (signed char)w2;
                  dq[(size_t)(j + 3) * cpg + c4] = (signed char)w3;
                  dq[(size_t)(j + 4) * cpg + c4] = (signed char)w4;
                  dq[(size_t)(j + 5) * cpg + c4] = (signed char)w5;
                  dq[(size_t)(j + 6) * cpg + c4] = (signed char)w6;
                  dq[(size_t)(j + 7) * cpg + c4] = (signed char)w7;
                }
              for (; j < WS; j++)
                {
                  float v = xr[j] * is;
                  int iv = jx_rndf(v);
                  if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
                  dq[(size_t)j * cpg + c4] = (signed char)iv;
                }
            }
          { unsigned long _t = jx_ccount();
            jx_dt_cyc[0] += (unsigned long)(unsigned)(_t - _dt_t0);
            _dt_t0 = _t; }
          for (int t = ts; t < te; t++)
            {
              int base = t * stride - pad;
              int off  = base - aS;
              unsigned long _c1 = jx_ccount(), _c2;
              signed char *d = sq;
              /* 2026-09-14 第二十五轮【修 bug】：im2col 行的字节顺序必须是
               * [kk][c]（tap 外层、通道内层），才能与 int8 权重行对上。
               *
               * 第十七轮把 g_dtq 改成"通道交错" [j*cpg + c] 后，这里用一次
               * memcpy(d, dq + off*cpg, k*cpg) 直接搬，拼出来的却是
               * [c][kk] 的**转置**（原始连续块 = 时间外层、通道内层）。于是
               * 每个输出通道的 K 维被整体错排，int8 卷积输出全是错的 ——
               * 这就是"播放全是噪音"的根因（板端 A/B：encoder_out 最大误差
               * 5.12 -> 修复后 0.083）。
               *
               * 统一约定：激活的 im2col 行与权重的每一行都按 [kk][c] 排布
               * （gen_wq8.py 生成时对 3-D 卷积权重做同样的 (ci,k)->(k,ci)
               * 交错）。这样 dil==1 时整块连续，仍然是一次 memcpy。 */
              if (nchk)
                {
                  if (dil == 1 && !inl)
                    {
                      /* dil=1：k 个 tap 在交错布局里连续，且顺序恰为 [kk][c] */
                      memcpy(d, dq + (size_t)off * cpg, (size_t)k * (size_t)cpg);
                    }
                  else
                    {
                      /* dil>1：按 [kk][c] 取 k 组、每组 cpg 个连续字节。
                       * cpg 是 16 字节对齐，d 也是 16 字节对齐，直接 memcpy
                       * 让编译器用 128-bit 装载；字节顺序与逐字节拷贝完全一致。 */
                      for (int kk = 0; kk < k; kk++)
                        {
                          const signed char *qr = dq + (size_t)(off + kk * dil) * cpg;
                          memcpy(d + (size_t)kk * cpg, qr, (size_t)cpg);
                        }
                    }
                }
              else
                {
                  for (int kk = 0; kk < k; kk++)
                    {
                      int pos = off + kk * dil;
                      if (pos >= 0 && pos < WS)
                        {
                          memcpy(d + (size_t)kk * cpg, dq + (size_t)pos * cpg, (size_t)cpg);
                        }
                      else
                        {
                          memset(d + (size_t)kk * cpg, 0, (size_t)cpg);
                        }
                    }
                }
              _c2 = jx_ccount();
              jx_dt_cyc[1] += (unsigned long)(unsigned)(_c2 - _c1);
              for (int z = K; z < K16; z++) { sq[z] = 0; }
              if (quse)
                {
                  for (int o0 = 0; o0 < copg; o0 += 16)
                    {
                      const signed char *wt = g_qwT + (size_t)(o0 >> 4) * (size_t)K16 * 16u;
                      int32_t tmp[16];
                      int nn = copg - o0; if (nn > 16) { nn = 16; }
                      memset(accb + o0, 0, (size_t)nn * sizeof(int32_t));
                      for (int c0 = 0; c0 < nv; c0 += 2)
                        {
                          int nc = nv - c0;
                          if (nc > 2) { nc = 2; }
                          jx_qacc_dot16(wt + (size_t)c0 * 256u,
                                        sq + (size_t)c0 * 16u, nc, tmp);
                          for (int i = 0; i < nn; i++)
                            {
                              accb[o0 + i] += tmp[i];
                            }
                        }
                    }
                }
              else if (nv <= 4)
                {
                  jx_dotcol_i8(sq, wgc, nv, kstride, copg, accb);
                }
              else
                {
                  jx_dotcol_i8(sq, wgc, 4, kstride, copg, accb);
                  jx_dotcol_i8(sq + 64, wgc + 64, nv - 4, kstride, copg, accb2);
                  for (int i = 0; i < copg; i++) { accb[i] += accb2[i]; }
                }
              { unsigned long _c3 = jx_ccount();
                jx_dt_cyc[2] += (unsigned long)(unsigned)(_c3 - _c2);
                /* r120：restrict，解除与 accb/wsc/b 的别名假设 */
                float * __restrict trow = tile + (size_t)(t - ts) * copg;
                /* 第三十九轮：系数在分支外装载一次。 */
                const jx_erf9 KE = jx_erf9_load();
                const jx_sq4  KS = jx_sq4_load();
                if (jx_conv_post == 1)
                  {
                    int co = 0;
                    for (; co + 4 <= copg; co += 4)
                      {
                        trow[co] =
                            jx_gelu_fast((float)accb[co] * (s * wsc[co]) + (b != NULL ? b[co] : 0.0f), KE);
                        trow[co + 1] =
                            jx_gelu_fast((float)accb[co + 1] * (s * wsc[co + 1]) + (b != NULL ? b[co + 1] : 0.0f), KE);
                        trow[co + 2] =
                            jx_gelu_fast((float)accb[co + 2] * (s * wsc[co + 2]) + (b != NULL ? b[co + 2] : 0.0f), KE);
                        trow[co + 3] =
                            jx_gelu_fast((float)accb[co + 3] * (s * wsc[co + 3]) + (b != NULL ? b[co + 3] : 0.0f), KE);
                      }
                    for (; co < copg; co++)
                      {
                        trow[co] =
                            jx_gelu_fast((float)accb[co] * (s * wsc[co]) + (b != NULL ? b[co] : 0.0f), KE);
                      }
                  }
                else if (jx_conv_post == 2)
                  {
                    int co = 0;
                    for (; co + 4 <= copg; co += 4)
                      {
                        trow[co] =
                            jx_post_snake_k((float)accb[co] * (s * wsc[co]) + (b != NULL ? b[co] : 0.0f),
                                            gbase + co, KS);
                        trow[co + 1] =
                            jx_post_snake_k((float)accb[co + 1] * (s * wsc[co + 1]) + (b != NULL ? b[co + 1] : 0.0f),
                                            gbase + co + 1, KS);
                        trow[co + 2] =
                            jx_post_snake_k((float)accb[co + 2] * (s * wsc[co + 2]) + (b != NULL ? b[co + 2] : 0.0f),
                                            gbase + co + 2, KS);
                        trow[co + 3] =
                            jx_post_snake_k((float)accb[co + 3] * (s * wsc[co + 3]) + (b != NULL ? b[co + 3] : 0.0f),
                                            gbase + co + 3, KS);
                      }
                    for (; co < copg; co++)
                      {
                        trow[co] =
                            jx_post_snake_k((float)accb[co] * (s * wsc[co]) + (b != NULL ? b[co] : 0.0f),
                                            gbase + co, KS);
                      }
                  }
                else
                  {
                    int co = 0;
                    for (; co + 4 <= copg; co += 4)
                      {
                        trow[co] =
                            (float)accb[co] * (s * wsc[co]) + (b != NULL ? b[co] : 0.0f);
                        trow[co + 1] =
                            (float)accb[co + 1] * (s * wsc[co + 1]) + (b != NULL ? b[co + 1] : 0.0f);
                        trow[co + 2] =
                            (float)accb[co + 2] * (s * wsc[co + 2]) + (b != NULL ? b[co + 2] : 0.0f);
                        trow[co + 3] =
                            (float)accb[co + 3] * (s * wsc[co + 3]) + (b != NULL ? b[co + 3] : 0.0f);
                      }
                    for (; co < copg; co++)
                      {
                        trow[co] =
                            (float)accb[co] * (s * wsc[co]) + (b != NULL ? b[co] : 0.0f);
                      }
                  }
                jx_dt_cyc[3] += (unsigned long)(unsigned)(jx_ccount() - _c3);
              }
              jx_dt_pos++;
            }
          /* r70 计时修正：上面那个 t 循环必须在**本轮**记完步，否则下一子块的
           * 量化累加会把这一轮的 gather+dot+epi 整段算进"量化"桶（实测把量化
           * 桶放大 3 倍，[dt] 的分项完全不可读）。这里补一次取样。 */
          _dt_t0 = jx_ccount();
          { static int _dz2 = 0; if (jx_getenv_guarded("JX_DT_TRACE") != NULL && _dz2 < 6) { _dz2++; unsigned _u0, _u1, _u2, _u3; float _f0 = tile[0], _f1 = tile[1], _f2 = tile[2], _f3 = tile[3]; memcpy(&_u0, &_f0, 4); memcpy(&_u1, &_f1, 4); memcpy(&_u2, &_f2, 4); memcpy(&_u3, &_f3, 4); printf("[DTY] ts=%d te=%d y0=%08x %08x %08x %08x\n", ts, te, _u0, _u1, _u2, _u3); } }
          for (int co = 0; co < copg; co++)
            {
              float *dst = yout + (size_t)co * T_out + ts;
              int j = 0;
              for (; j < te - ts; j++)
                {
                  dst[j] = tile[(size_t)j * copg + co];
                }
            }
        }
    }
}

static int jx_conv1d_dense_tile(const float *b, const float *xin, float *yout,
                                int cpg, int copg, int k, int stride, int pad,
                                int dil, int T_in, int T_out, int K, int K16,
                                int kstride, const signed char *wgc,
                                const float *wsc, const float *galph, int gbase)
{
  int WT = jx_i8_blk("JX_DTW", 256);
  if (WT < 16) { WT = 16; }
  int inl = jx_i8_blk("JX_DTC", 0);
  jx_dt_calls++;
  int nv = K16 >> 4;
  if (nv < 1 || nv > 8 || copg > JX_I8OC_N) { return -1; }
  if (jx_conv_post == 2 && galph == NULL) { return -1; }
  /* r61d：**不要再按缓冲容量缩 WT！**
   * WT 决定每个块的量化 scale（mx 在整个块窗口 [a0,b0] 上求），缩 WT 就改数值。
   * 旧循环用的容量公式 cpg*(WT+(k-1)*dil+2*stride+2) 在 stride>1 时**远小于**
   * 真实需求 cpg*((WT-1)*stride+(k-1)*dil+1)：stride=4 的层在 WT=256 时要 32768
   * 字节，公式只报 8512 —— 这才是 r60 写爆 g_dtq、主机 SNR 从第 82 帧漂移的根因。
   * 现在 dq 只放「一个处理子块」的窗口（见 TB 循环里的流式量化），所以这里
   * 只要保证**最小窗口**（TB=1，(k-1)*dil+1 个位置）放得下即可。 */
  if ((size_t)cpg * (size_t)((k - 1) * dil + 1) > (size_t)JX_DTQ_BYTES)
    {
      return -1;
    }

  int32_t accb[copg];
  int32_t accb2[copg];
  float *tile = (float *)g_ocacc2[jx_core_id()];
  /* 2026-09-14 第十六轮：sq 从 scratch_q()（malloc -> PSRAM）改成复用
   * per-core 静态缓冲 g_tnq（内部 SRAM）。
   *
   * 板端 [dt] 探针把 dense tile 每个输出位置拆成
   *     量化 1003 + 收集 1000 + 点积 838 + epilogue 609 = 3455 周期
   * 其中"收集"只是把 112 字节 im2col 拷进 sq —— 112 字节花掉 1000 周期
   * （8.9 周期/字节），正是 PSRAM 32 字节行写分配+回写的代价。
   * sq 每行只有 K16+16 <= 144 字节，本该待在内部 SRAM。
   * g_tnq 现在是 JX_I8TN_L1+64 = 4160 字节/核（第三十五轮缩小），pert 路径
   * 受 TN*K16 <= JX_I8TN_L1 约束，每行最多用 K16 <= 288 字节；
   * 而 dense tile 与 pert 路径在 jx_conv1d_i8 里互斥，故可安全复用。
   * 数值完全不变（只是缓冲位置不同）。 */
  signed char *sq = g_tnq[jx_core_id()];
  if (sq == NULL) { return -1; }

  /* 2026-09-15 第三十五轮 —— QACC 已实测证伪，门禁退回原样，**不要再放宽**。
   *
   * 本轮把门禁放宽到 nv>=1 / copg>=16（16 条 lane 本来就是 16 个输出通道）
   * 并在板端用新加的 [k1] 计数器做了决定性测量：
   *     accx 分支 : 1786 周期/输出位置（历史路径）
   *     QACC 分支 : 3081 周期/输出位置（1.72x **更慢**）
   * 而且整机指纹从 idx=422c4900301a699f 变成 idx=2d3944d7b8c6a78c，
   * 说明 jx_qacc_dot16 的 nchunks=2 语义与自检（只覆盖 nchunks=1）不符，
   * **结果不正确**。
   *
   * 结论：这颗芯片上 16 条 20-bit lane 的读出代价（10 条 rur + 320 bit
   * 位域拆包）高于它省下的 accx 往返，QACC 对这套形状是负收益。
   * 真要用必须重写读出（避免 C 里拆 20-bit 位域）并通过逐位校验。
   *
   * 门禁恢复为 nv==2 && copg>=32 && copg%16==0（即原来的窄门禁）。 */
  int cob = (copg + 15) & ~15;
  int quse = jx_qacc_want() && nv == 2 && copg >= 32 && ((copg & 15) == 0) &&
             (size_t)K16 * (size_t)copg <= (size_t)JX_QW_BYTES;
  if (quse)
    {
      memset(g_qwT, 0, (size_t)K16 * (size_t)copg);
      for (int o = 0; o < copg; o++)
        {
          int block = o >> 4;
          int oo = o & 15;
          signed char *wt = g_qwT + (size_t)block * (size_t)K16 * 16u + oo;
          const signed char *src = wgc + (size_t)o * (size_t)kstride;
          for (int kk = 0; kk < K; kk++)
            {
              wt[(size_t)kk * 16u] = src[kk];
            }
        }
    }

  { int _nblk = (T_out + WT - 1) / WT;
    jx_dtctx_t cc;
    cc.b = b; cc.xin = xin; cc.yout = yout;
    cc.wgc = wgc; cc.wsc = wsc; cc.galph = galph;
    cc.cpg = cpg; cc.copg = copg; cc.k = k; cc.stride = stride; cc.pad = pad;
    cc.dil = dil; cc.T_in = T_in; cc.T_out = T_out; cc.K = K; cc.K16 = K16;
    cc.kstride = kstride; cc.gbase = gbase;
    cc.WT = WT; cc.cob = cob; cc.quse = quse; cc.inl = inl;
    cc.sq = sq; cc.tile = tile; cc.dq = g_dtq[jx_core_id()];
    if (jx_pfx("JX_PFD")) { jx_pf_run_min(_nblk, jx_conv1d_dense_tile_tb, &cc, 2); }
    else { jx_conv1d_dense_tile_tb(&cc, 0, _nblk); }
  }
  return 0;
}

/* ---------------- channels-first k=1 卷积：分块量化 + 批量点积 ---------- */
/* r44i 实验：点积换成 jx_dotcol_i8（权重行常驻 q4/q5，激活矩阵流式）。
 * JX_K1DOT=1 开。与 jx_dotmj_i8 数值逐位一致（同一组 int8 输入、同一累加次序）。 */
static int jx_k1dot(void)
{
  static int v = -2;
  if (v == -2) { const char *e = jx_getenv_guarded("JX_K1DOT"); v = (e != NULL && e[0] == '1'); }
  return v;
}
/* r57k：k1cf 的 t 块循环抽成**可重入**函数，供双核按块区间切分。
 * 每个块只碰 yout[o][t0..t1) 与自己那一份 qb/sc/s_tni（全是 per-core 缓冲），
 * 块与块之间零依赖 -> 切分后逐位相同。 */
typedef struct {
  const float *b; const float *xin; float *yout;
  const signed char *wgc; const float *wsc;
  int Ci, Co, T_in, T_out, kstride, gbase;
  int cpad, nv, TB, kcsi;
  signed char *qb; float *sc; float *s_tni;
  /* P0 边车：输入侧（可直接免掉 Pass A）与输出侧（epilogue 顺手写） */
  const unsigned *amx; int amc; unsigned *am;
} jx_k1ctx_t;

/* P0 边车：生产者 epilogue 里把 |v| 折进本块的 SRAM 暂存。
 * 位运算与 Pass A 的 jx_uabs 完全同源 -> 逐位一致。 */
#define JX_AMU(_amb, _t, _v) do { if ((_amb) != NULL) { \
    unsigned _au = jx_uabs(_v); unsigned *_ap = (_amb) + (size_t)(_t); \
    if (_au > *_ap) { *_ap = _au; } } } while (0)

JX_HOT static void jx_conv1d_k1_cf_tb(void *va, int tblo, int tbhi)
{
  jx_k1ctx_t *cc = (jx_k1ctx_t *)va;
  const float *b = cc->b; const float *xin = cc->xin; float *yout = cc->yout;
  const signed char *wgc = cc->wgc; const float *wsc = cc->wsc;
  const int Ci = cc->Ci, Co = cc->Co, T_in = cc->T_in, T_out = cc->T_out;
  const int kstride = cc->kstride, gbase = cc->gbase;
  const int cpad = cc->cpad, nv = cc->nv, TB = cc->TB;
  const int _kcsi = cc->kcsi;
  /* r157（2026-09-17 双核竞态修复）：同 jx_conv1d_dense_tile_tb —— qb/sc/s_tni 必须在**本函数内**
   * 按当前核取。cc->* 是 CPU0 取好的那一份，两核共用会互相覆盖量化缓冲。 */
  const int _jc_k1 = jx_core_id();
  signed char *qb = g_tnq[_jc_k1]; float *sc = g_tns[_jc_k1];
  float *s_tni = g_tni[_jc_k1];
  /* r144 probe: epilogue / dot cost isolation (default 0 = bit-identical).
   *   JX_K1EPV 0=normal 1=dst[j]=bb 2=full math -> SRAM 3=const -> SRAM
   *   JX_K1DOV 1=skip dot (acc=0) */
  const int _epv = jx_i8_blk("JX_K1EPV", 0);
  const int _dov = jx_i8_blk("JX_K1DOV", 0);
  const int _pas = jx_i8_blk("JX_K1PAS", 0);
  /* P0 边车：_amx = 输入张量的逐位置 max（生产者送来），
   * _am = 本算子作为生产者的边车缓冲（无则 NULL）。 */
  const unsigned * const _amx = cc->amx;
  const int _ci0 = (cc->amc > 0) ? cc->amc : 0;
  unsigned * const _am = cc->am;
  unsigned * const _ams = (_am != NULL) ? (unsigned *)(void *)g_tnm[jx_core_id()] : NULL;
  for (int _tb = tblo; _tb < tbhi; _tb++)
    {
      int t0 = _tb * TB;
      int t1 = t0 + TB; if (t1 > T_out) { t1 = T_out; }
      int w = t1 - t0;
      unsigned _kc0 = jx_ccount();
      unsigned _kcA = _kc0;
      jx_kc_spos[_kcsi] += (unsigned long)w;
      jx_kc_sblk[_kcsi] += 1ul;
      if (_ams != NULL) { for (int _j = 0; _j < w; _j++) { _ams[_j] = 0u; } }

      JXP_BEG(JP_K1Q);
      /* 2026-09-15 第三十六轮：把这里的"逐位置 gather + jx_q_row_auto"换成
       * **块内两趟连续读**。原来每个位置 j 都要 xin[ci*T_in + t] 逐个 ci 取，
       * 一个元素一条 cache line，TB 个位置就是 Ci*TB 次 PSRAM 缺失；实测
       * JX_K1CF 开启后 im2col 桶 4547 周期/位置（对照 pert 路径 1524），
       * 整机 6850 -> 7380 ms，白白亏 530 ms。
       *
       * 现在：Pass A 逐通道连续读一遍求**逐位置** max，Pass B 再连续读一遍
       * 量化写进转置缓冲 qb[j*cpad + ci]。访存从"每元素一条新行"变成
       * "每条通道行连续 TB*4 字节"，而且每行只过两遍且都在 L1。
       *
       * 逐位一致：max 的累加顺序仍是 ci 递增（与原 gather 后的 i 递增同序），
       * 量化算式 (int)(v + (v>=0?0.5:-0.5)) + 钳位 + 补零、
       * scale = mx/127.0f、inv = 127.0f/mx 全部与 jx_q_row 相同，
       * 所以开关前后指纹必须完全一致（v184 已实测 K1CF 与基线指纹相同）。 */
      /* r43e：PassA 的逐位置 max 同样搬整数域（累加器天然在内存里，
       * 但 per-element 的 fabsf + olt.s + bf 换成 and + 整数比较）。 */
      /* r52b：位置分块（JX_K1JB，默认 32）。Pass A / Pass B 原本是两趟全宽扫描，
       * 工作集 Ci*w*4 字节（Ci=81,w=170 时 55KB）远超 L1，第二趟等于又从 PSRAM
       * 读了一遍（实测 A 371 / B 405 周期/位置，两者几乎相等 = 零复用）。
       * 分块后每块只需 Ci*JB*4 字节（81*32*4 = 10KB），第二趟基本全落 L1。
       * 逐位置独立：max 的 ci 递增顺序、量化算式、scale 一字不改 -> 逐位一致。 */
      int _JB = jx_i8_blk("JX_K1JB", 32); if (_JB < 1) { _JB = 1; }
      /* r120：JX_K1PIE / JX_K1PA 原来在 jb 子块循环里**逐块** jx_getenv_guarded()。
       * getenv 要线性扫描环境表并做字符串比较，而一次 k1cf 调用的子块数是
       * w/_JB（TB=512 时 16 个），乘上 550 个块就是上万次。提到循环外只改
       * 取值位置，取值结果完全相同。 */
      int _pie_hoist = 0;
#if defined(JX_PIE)
      _pie_hoist = ((T_in & 3) == 0) ? jx_i8_blk("JX_K1PIE", 1) : 0;
#endif
      if (_ci0 != 0) { _pie_hoist = 0; }   /* P0：前 _ci0 条通道已由边车覆盖 */
      const int _PA_hoist = (_ci0 == 0) ? jx_i8_blk("JX_K1PA", 0) : 0;
      unsigned long _cyA = 0, _cyS = 0, _cyB = 0;
      for (int jb = 0; jb < w; jb += _JB)
        {
          int jw = w - jb; if (jw > _JB) { jw = _JB; }
          jx_u32a *scb = (jx_u32a *)(void *)(sc + jb);
          float *stn = s_tni + jb;
          unsigned _ta = jx_ccount();
          if (_amx != NULL)
            { const unsigned *_ax = _amx + (size_t)t0 + (size_t)jb;
              for (int j = 0; j < jw; j++) { scb[j] = _ax[j]; } }
          else
            { for (int j = 0; j < jw; j++) { scb[j] = 0u; } }
      /* r145 JX_K1PAS=1: PassA 改成"逐通道整块顺序扫"。
       * 原 PIE 路径每 16 个位置只抓一条 64B 行，然后跳到 16KB 外
       * 的下一条通道 —— 实测 36.5 周期/(ci,pos)（PSRAM 跨步 146 周期/
       * 行的全额罚）；而顺序流读实测可达 85 MB/s ≈ 11.3 周期/元素。
       * 新写法：ci 外层、j 内层，每行连续读 jw*4 字节；每个 j 的 max
       * 累加器放在 SRAM 的 scb（j 方向 8 路展开，8 个独立链盖住延迟）。
       * max 与次序无关，结果逐位相同。 */
      if (_pas)
        {
          for (int ci = 0; ci < Ci; ci++)
            {
              const float *xr = xin + (size_t)ci * T_in + t0 + jb;
              int j = 0;
              for (; j + 8 <= jw; j += 8)
                {
                  unsigned a0 = jx_uabs(xr[j    ]), a1 = jx_uabs(xr[j + 1]);
                  unsigned a2 = jx_uabs(xr[j + 2]), a3 = jx_uabs(xr[j + 3]);
                  unsigned a4 = jx_uabs(xr[j + 4]), a5 = jx_uabs(xr[j + 5]);
                  unsigned a6 = jx_uabs(xr[j + 6]), a7 = jx_uabs(xr[j + 7]);
                  if (a0 > scb[j    ]) { scb[j    ] = a0; }
                  if (a1 > scb[j + 1]) { scb[j + 1] = a1; }
                  if (a2 > scb[j + 2]) { scb[j + 2] = a2; }
                  if (a3 > scb[j + 3]) { scb[j + 3] = a3; }
                  if (a4 > scb[j + 4]) { scb[j + 4] = a4; }
                  if (a5 > scb[j + 5]) { scb[j + 5] = a5; }
                  if (a6 > scb[j + 6]) { scb[j + 6] = a6; }
                  if (a7 > scb[j + 7]) { scb[j + 7] = a7; }
                }
              for (; j < jw; j++)
                { unsigned a = jx_uabs(xr[j]); if (a > scb[j]) { scb[j] = a; } }
            }
        }
      else
      /* 第四十二轮：8 路展开。原写法每个元素一条 load->fabs->cmp->select->store
       * 的 read-modify-write 链，实测 17.85 周期/(ci,j)。展开后 8 个不同的 sc[j]
       * 互相无依赖，延迟被盖住。max 与求和次序无关，结果逐位一致。 */
      /* r54：PassA 的 4 路 ci 交错**已回退**。板端实测（同固件 A/B）：
       *   Ci=20/80 A_max 376.9 -> 524.3，Ci=81/16 1216 -> 1858（更差）；
       *   同轮 PassB 反而 427 -> 313 / 1466 -> 1070（更好，保留）。
       * 原因：PassA 的 8 个 b 值要跨 4 个通道块保持活跃，16 个通用寄存器
       * 装不下 -> 溢出；而跨步读本身已被 8 路 j 展开重叠得差不多
       * （实测已到 ~51 MB/s，PSRAM 上限 84 MB/s）。PassB 每元素独立、
       * 无活跃值压力，交错收益是真的。 */
      /* r68 JX_K1PA=1: hoist the 8 running maxima into registers and keep
       * them across the whole ci loop. The old form did a load+store to scb
       * for every (ci,j); this does them once per 8 j. max is order-free,
       * so the result is bit-identical. */
      /* r98：PIE 128 位 PassA。
       * 原标量版把 8 个 running max 和 8 个待处理值同时压在 16 个 AR 寄存器上，
       * GCC 只能把一半溢出到栈槽 —— 每个元素凭空多出 1 条 s32i + 1 条 l32i，
       * 还制造 store->load 依赖。板端 [kc] 显示 Ci=20 时 A_max 高达 40 周期/元素。
       * 改成 4 个 128 位向量累加器（q1..q4）+ 1 条 andq + 1 条 vmax.s32，
       * 每元素 0.75 条指令且完全不占 AR 寄存器。max 是次序无关的归约，
       * 结果与标量版逐位一致；表头/表尾不足 16 个位置的部分仍走标量。
       * 仅在 T_in 是 4 的倍数（行间保持 16 字节对齐）时启用。 */
      { /* 原 PassA 路径 */
      const int _pie = _pie_hoist;
      if (_pie)
        {
          const float *row0 = xin + t0 + jb;
          unsigned long _ap = (unsigned long)(const void *)row0;
          int head = (int)((((16ul - (_ap & 15ul)) & 15ul)) >> 2);
          if (head > jw) { head = jw; }
          int _j = 0;
          for (; _j < head; _j++)
            {
              unsigned b = 0u;
              for (int ci = 0; ci < Ci; ci++)
                { unsigned a = jx_uabs(row0[(size_t)ci * T_in + _j]); if (a > b) { b = a; } }
              scb[_j] = b;
            }
          {
            static const unsigned _jmk = 0x7fffffffu;
            const unsigned long _jrs = ((unsigned long)T_in - 16ul) * 4ul;
            for (; _j + 16 <= jw; _j += 16)
              {
                unsigned _jl[16] __attribute__((aligned(16)));
                const float *_jp = row0 + _j;
                int _jc = Ci;
                void *_jmkp = (void *)(&_jmk);
                void *_jlp = (void *)(_jl);
                __asm__ __volatile__(
                  "ee.vldbc.32.ip q7, %[MK], 0\n\t"
                  "ee.vld.128.ip q1, %[P], 16\n\t"
                  "ee.vld.128.ip q2, %[P], 16\n\t"
                  "ee.vld.128.ip q3, %[P], 16\n\t"
                  "ee.vld.128.ip q4, %[P], 16\n\t"
                  "ee.andq q1, q1, q7\n\t"
                  "ee.andq q2, q2, q7\n\t"
                  "ee.andq q3, q3, q7\n\t"
                  "ee.andq q4, q4, q7\n\t"
                  "addi %[C], %[C], -1\n\t"
                  "beqz %[C], 2f\n\t"
                  /* r171 BUGFIX: 4 条 vld 已把 P 推进 16 个 float，而每行只需 T_in 个。
                   * 原实现把 (T_in-16) 的补偿放在循环**尾部**，于是第 1 次迭代从
                   * 第 0 行的 [s+16,s+32) 读起 —— 每 16 条 lane 的 max 全部错位，
                   * 既漏掉第 Ci-1 行，又读出行外数据（末块直接越界，取值随堆内容变化，
                   * 这就是"冷启动结果不确定"的根因）。补偿必须发生在读之前。 */
                  "1:\n\t"
                  "add %[P], %[P], %[RS]\n\t"
                  "ee.vld.128.ip q5, %[P], 16\n\t"
                  "ee.vld.128.ip q6, %[P], 16\n\t"
                  "ee.andq q5, q5, q7\n\t"
                  "ee.andq q6, q6, q7\n\t"
                  "ee.vmax.s32 q1, q1, q5\n\t"
                  "ee.vmax.s32 q2, q2, q6\n\t"
                  "ee.vld.128.ip q5, %[P], 16\n\t"
                  "ee.vld.128.ip q6, %[P], 16\n\t"
                  "ee.andq q5, q5, q7\n\t"
                  "ee.andq q6, q6, q7\n\t"
                  "ee.vmax.s32 q3, q3, q5\n\t"
                  "ee.vmax.s32 q4, q4, q6\n\t"
                  "addi %[C], %[C], -1\n\t"
                  "bnez %[C], 1b\n\t"
                  "2:\n\t"
                  "ee.vst.128.ip q1, %[L], 16\n\t"
                  "ee.vst.128.ip q2, %[L], 16\n\t"
                  "ee.vst.128.ip q3, %[L], 16\n\t"
                  "ee.vst.128.ip q4, %[L], 16\n\t"
                    : [P] "+r"(_jp), [C] "+r"(_jc), [MK] "+r"(_jmkp), [L] "+r"(_jlp)
                    : [RS] "r"(_jrs)
                    : "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
                for (int _x = 0; _x < 16; _x++) { scb[_j + _x] = _jl[_x]; }
              }
          }
          for (; _j < jw; _j++)
            {
              unsigned b = 0u;
              for (int ci = 0; ci < Ci; ci++)
                { unsigned a = jx_uabs(row0[(size_t)ci * T_in + _j]); if (a > b) { b = a; } }
              scb[_j] = b;
            }
        }
      else
        {
      const int _PA = _PA_hoist;
      if (_PA == 1)
        {
          int j = 0;
          for (; j + 8 <= jw; j += 8)
            {
              unsigned b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, b7 = 0;
              for (int ci = 0; ci < Ci; ci++)
                {
                  const float *xr = xin + (size_t)ci * T_in + t0 + jb + j;
                  unsigned a0 = jx_uabs(xr[0]), a1 = jx_uabs(xr[1]);
                  unsigned a2 = jx_uabs(xr[2]), a3 = jx_uabs(xr[3]);
                  unsigned a4 = jx_uabs(xr[4]), a5 = jx_uabs(xr[5]);
                  unsigned a6 = jx_uabs(xr[6]), a7 = jx_uabs(xr[7]);
                  if (a0 > b0) { b0 = a0; } if (a1 > b1) { b1 = a1; }
                  if (a2 > b2) { b2 = a2; } if (a3 > b3) { b3 = a3; }
                  if (a4 > b4) { b4 = a4; } if (a5 > b5) { b5 = a5; }
                  if (a6 > b6) { b6 = a6; } if (a7 > b7) { b7 = a7; }
                }
              scb[j    ] = b0; scb[j + 1] = b1; scb[j + 2] = b2; scb[j + 3] = b3;
              scb[j + 4] = b4; scb[j + 5] = b5; scb[j + 6] = b6; scb[j + 7] = b7;
            }
          for (; j < jw; j++)
            {
              unsigned b = 0;
              for (int ci = 0; ci < Ci; ci++)
                {
                  unsigned a = jx_uabs(xin[(size_t)ci * T_in + t0 + jb + j]);
                  if (a > b) { b = a; }
                }
              scb[j] = b;
            }
        }
      for (int ci = _ci0; _PA != 1 && ci < Ci; ci++)
        {
          const float *xr = xin + (size_t)ci * T_in + t0 + jb;
          int j = 0;
          for (; j + 8 <= jw; j += 8)
            {
              unsigned b0 = scb[j    ], b1 = scb[j + 1], b2 = scb[j + 2], b3 = scb[j + 3];
              unsigned b4 = scb[j + 4], b5 = scb[j + 5], b6 = scb[j + 6], b7 = scb[j + 7];
              unsigned a0 = jx_uabs(xr[j    ]), a1 = jx_uabs(xr[j + 1]);
              unsigned a2 = jx_uabs(xr[j + 2]), a3 = jx_uabs(xr[j + 3]);
              unsigned a4 = jx_uabs(xr[j + 4]), a5 = jx_uabs(xr[j + 5]);
              unsigned a6 = jx_uabs(xr[j + 6]), a7 = jx_uabs(xr[j + 7]);
              if (a0 > b0) { b0 = a0; } if (a1 > b1) { b1 = a1; }
              if (a2 > b2) { b2 = a2; } if (a3 > b3) { b3 = a3; }
              if (a4 > b4) { b4 = a4; } if (a5 > b5) { b5 = a5; }
              if (a6 > b6) { b6 = a6; } if (a7 > b7) { b7 = a7; }
              scb[j    ] = b0; scb[j + 1] = b1; scb[j + 2] = b2; scb[j + 3] = b3;
              scb[j + 4] = b4; scb[j + 5] = b5; scb[j + 6] = b6; scb[j + 7] = b7;
            }
          for (; j < jw; j++)
            {
              unsigned a = jx_uabs(xr[j]);
              if (a > scb[j]) { scb[j] = a; }
            }
        }
        }
      }
          { unsigned _t = jx_ccount(); _cyA += (unsigned long)(unsigned)(_t - _ta); _ta = _t; }
          for (int j = 0; j < jw; j++)
            {
              float mx = jx_u2f(scb[j]);
              if (mx > 1e-30f) { stn[j] = 127.0f / mx; scb[j] = jx_ubits(mx / 127.0f); }
              else             { stn[j] = 0.0f;       scb[j] = jx_ubits(1e-30f); }
            }
          { unsigned _t = jx_ccount(); _cyS += (unsigned long)(unsigned)(_t - _ta); _ta = _t; }
      /* r54：PassB 同样 4 路 ci 交错（4 条输入流同时在飞），而且同一个 j 上
       * 4 个通道的写出 qb[j*cpad + ci .. ci+3] 正好是连续的 4 个字节。
       * 每个输出字节各自独立 -> 逐位一致。 */
      {
        int ci4 = 0;
        for (; ci4 + 4 <= Ci; ci4 += 4)
          {
            const float *xr0 = xin + (size_t)(ci4    ) * T_in + t0 + jb;
            const float *xr1 = xin + (size_t)(ci4 + 1) * T_in + t0 + jb;
            const float *xr2 = xin + (size_t)(ci4 + 2) * T_in + t0 + jb;
            const float *xr3 = xin + (size_t)(ci4 + 3) * T_in + t0 + jb;
            signed char *dq = qb + ci4 + (size_t)jb * cpad;
            for (int j = 0; j < jw; j++)
              {
                float s  = stn[j];
                float v0 = xr0[j] * s;
                float v1 = xr1[j] * s;
                float v2 = xr2[j] * s;
                float v3 = xr3[j] * s;
                /* r70：4 个相邻通道在这个位置上正好连续 4 字节（cpad 是 16 的
                 * 倍数、ci4 是 4 的倍数 -> 4 字节对齐），合成一条 32 位 store。
                 * 写入字节与逐条 8 位 store 完全相同 -> 逐位一致。 */
                int q0 = jx_rndf(v0);
                int q1 = jx_rndf(v1);
                int q2 = jx_rndf(v2);
                int q3 = jx_rndf(v3);
                unsigned pk = ((unsigned)q0 & 0xffu) | (((unsigned)q1 & 0xffu) << 8)
                            | (((unsigned)q2 & 0xffu) << 16) | (((unsigned)q3 & 0xffu) << 24);
                *(jx_u32a *)(void *)(dq + (size_t)j * cpad) = (jx_u32a)pk;
              }
          }
        for (; ci4 < Ci; ci4++)
          {
            const float *xr = xin + (size_t)ci4 * T_in + t0 + jb;
            signed char *dq = qb + ci4 + (size_t)jb * cpad;
            for (int j = 0; j < jw; j++)
              {
                float v = xr[j] * stn[j];
                dq[(size_t)j * cpad] = (signed char)jx_rndf(v);
              }
          }
      }
          { unsigned _t = jx_ccount(); _cyB += (unsigned long)(unsigned)(_t - _ta); }
        }
      jx_kc_scy[_kcsi][0] += _cyA;
      jx_kc_scy[_kcsi][1] += _cyS;
      jx_kc_scy[_kcsi][2] += _cyB;
      _kcA = jx_ccount();
      for (int j = 0; j < w; j++)
        {
          signed char *qr = qb + (size_t)j * cpad;
          if (sc[j] == 1e-30f) { memset(qr, 0, (size_t)cpad); }
          else { for (int z = Ci; z < cpad; z++) { qr[z] = 0; } }
        }
      { unsigned _t = jx_ccount(); jx_kc_scy[_kcsi][2] += (unsigned long)(unsigned)(_t - _kcA); _kcA = _t; }
      JXP_END(JP_K1Q);
      JXP_BEG(JP_K1MM);
      /* r79???? .bss ? per-core ??????????? 4KB ????
       * ????? PSRAM??? g_k1acc ??????? */
      int32_t *acc  = g_k1acc [jx_core_id()];
      int32_t *acc2 = g_k1acc2[jx_core_id()];
      /* 第三十九轮：epilogue 的系数与后处理分支在循环外定下来。 */
      const int    post = jx_conv_post;
      const int    _dbg_gm = jx_epg_mode();
      const jx_sq4 KS = jx_sq4_load();
      const jx_erf9 KE = jx_erf9_load();
      /* r119：gelu 的 off/scale/hi 与 LUT 基址在 o 循环外装载一次。
       * 详见 jx_fastmath.h 的 jx_gelu_kt 注释：原来每个元素都要重取。 */
      const jx_glut4 KL = jx_glut4_load();
      const float * const jxLUT = jx_glut;
      for (int o = 0; o < Co; o++)
        {
          const signed char *wr = wgc + (size_t)o * kstride;
          if (_dov) { for (int j = 0; j < w; j++) { acc[j] = 0; } }
          else if (nv <= 4)
            {
              if (jx_k1dot())
                {
                  if (jx_k1dp()) { jx_dotcol_i8_p(wr, qb, nv, cpad, w, acc); }
                  else           { jx_dotcol_i8(wr, qb, nv, cpad, w, acc); }
                }
              else            { jx_dotmj_i8(qb, wr, nv, w, cpad - nv * 16, acc); }
            }
          else
            {
              if (jx_k1dot())
                {
                  if (jx_k1dp())
                    {
                      jx_dotcol_i8_p(wr, qb, 4, cpad, w, acc);
                      jx_dotcol_i8_p(wr + 64, qb + 64, nv - 4, cpad, w, acc2);
                    }
                  else
                    {
                      jx_dotcol_i8(wr, qb, 4, cpad, w, acc);
                      jx_dotcol_i8(wr + 64, qb + 64, nv - 4, cpad, w, acc2);
                    }
                }
              else
                {
                  jx_dotmj_i8(qb, wr, 4, w, cpad - 64, acc);
                  jx_dotmj_i8(qb + 64, wr + 64, nv - 4, w, cpad - (nv - 4) * 16, acc2);
                }
              for (int j = 0; j < w; j++) { acc[j] += acc2[j]; }
            }
          { unsigned _t = jx_ccount(); jx_kc_scy[_kcsi][3] += (unsigned long)(unsigned)(_t - _kcA); _kcA = _t; }
          unsigned _ep0 = _kcA;
          float ws = wsc[o];
          float bb = (b != NULL) ? b[o] : 0.0f;
          /* r43a：去掉 tile 中转，反量化结果直接按 channels-first 连续写
           * yout[o][t0..t0+w)。算式与 k1s 的 epilogue 逐位相同，但 TB 不再
           * 受 4KB tile 限制（改由 qb 容量 JX_I8TN_L1/cpad 决定）。 */
          /* r119：__restrict —— dst 是 float*、jx_kt/jx_glut 也是 float[]，
           * 严格别名规则下 GCC 必须假设每次 dst[j]= 都可能改到它们，于是
           * 每个元素重取 3 条 l32r + 3 条 lsi，且 4 条 lane 无法交错。 */
          float * __restrict dst = jx_ep_pick(yout + (size_t)o * T_out + t0);   /* r99: JX_K1EP=1 -> SRAM 探针 */
          /* r51h：gelu 实现四路 A/B（JX_EPG 0=恒等 1=旧LUT 2=分支消除 3=Horner 4=旧LUT四路展开，默认 4） */
          const int _gm = _dbg_gm;
          if (_epv == 1)
            { for (int j = 0; j < w; j++) { dst[j] = bb; } }
          else if (_epv == 2)
            { float *s2 = s_tni;
              for (int j = 0; j < w; j++) { s2[j] = (float)acc[j] * (sc[j] * ws) + bb; } }
          else if (_epv == 3)
            { float *s2 = s_tni;
              for (int j = 0; j < w; j++) { s2[j] = bb; } }
          else if (post == 1 && _gm == 0)
            {
              for (int j = 0; j < w; j++)
                { float _v = (float)acc[j] * (sc[j] * ws) + bb;
                  dst[j] = _v; JX_AMU(_ams, j, _v); }
            }
          else if (post == 1 && _gm == 2)
            {
              for (int j = 0; j < w; j++)
                { float _v = jx_gelu_br((float)acc[j] * (sc[j] * ws) + bb, KE);
                  dst[j] = _v; JX_AMU(_ams, j, _v); }
            }
          else if (post == 1 && _gm == 4)
            {
              /* r51j：旧 LUT，但 4 路展开交给 GCC 交错（单个元素是 ~15 级串行链）。
               * r71 试过手工按层级交错（jx_gelu4_x4），板端实测更慢
               * （gelu 290 -> 430 ms：4 条链同时活跃需要 ~28 个浮点寄存器，
               * 16 个装不下 -> 溢出）。保持逐个算完的写法。 */
              int j = 0;
              for (; j + 4 <= w; j += 4)
                {
                  float u0 = (float)acc[j    ] * (sc[j    ] * ws) + bb;
                  float u1 = (float)acc[j + 1] * (sc[j + 1] * ws) + bb;
                  float u2 = (float)acc[j + 2] * (sc[j + 2] * ws) + bb;
                  float u3 = (float)acc[j + 3] * (sc[j + 3] * ws) + bb;
                  /* r100b 已回退：相位交错的 4 路 gelu（jx_gelu4_fast）上板实测
                   * 更慢（3790 -> 3800，epi o2+ 4533 -> 4660）——16 个 f 寄存器
                   * 装不下 4 条 lane 的相位中间量，溢出把交错收益吃光。 */
                  float r0, r1, r2, r3;
                  /* r146：改成相位交错的 jx_gelu_kt4。
                   * 原来 4 次 jx_gelu_kt 各自是一条 ~9 级串行 FP 链
                   * （实测 41.7 周期/元素 = 链长 x FPU 4.13 延迟），
                   * 4 条 lane 在指令流里首尾相接、延迟全暴露。
                   * 算式与 jx_gelu_kt 逐项相同 -> 逐位一致。 */
                  r0 = jx_gelu_kt(u0, KL, jxLUT);
                  r1 = jx_gelu_kt(u1, KL, jxLUT);
                  r2 = jx_gelu_kt(u2, KL, jxLUT);
                  r3 = jx_gelu_kt(u3, KL, jxLUT);
                  dst[j    ] = r0;
                  dst[j + 1] = r1;
                  dst[j + 2] = r2;
                  dst[j + 3] = r3;
                  JX_AMU(_ams, j    , r0); JX_AMU(_ams, j + 1, r1);
                  JX_AMU(_ams, j + 2, r2); JX_AMU(_ams, j + 3, r3);
                }
              for (; j < w; j++)
                { float _v = jx_gelu_kt((float)acc[j] * (sc[j] * ws) + bb, KL, jxLUT);
                  dst[j] = _v; JX_AMU(_ams, j, _v); }
            }
          else if (post == 1 && _gm == 3)
            {
              for (int j = 0; j < w; j++)
                { float _v = jx_gelu9((float)acc[j] * (sc[j] * ws) + bb, KE);
                  dst[j] = _v; JX_AMU(_ams, j, _v); }
            }
          else if (post == 1)
            {
              for (int j = 0; j < w; j++)
                { float _v = jx_gelu_kt((float)acc[j] * (sc[j] * ws) + bb, KL, jxLUT);
                  dst[j] = _v; JX_AMU(_ams, j, _v); }
            }
          else if (post == 2)
            {
              /* r97：4 路展开。原来逐元素，一条 "cvt -> mul -> sin² LUT ->
               * mul -> add" 的串行链（实测 3828 周期/位置 / 80 通道 = 48
               * 周期/元素），4 条独立链交错后可盖住大部分延迟。
               * 每个元素各自独立、算式逐字符未变 -> 逐位一致。 */
              int j = 0;
              for (; j + 8 <= w; j += 8)
                {
                  float u0 = (float)acc[j    ] * (sc[j    ] * ws) + bb;
                  float u1 = (float)acc[j + 1] * (sc[j + 1] * ws) + bb;
                  float u2 = (float)acc[j + 2] * (sc[j + 2] * ws) + bb;
                  float u3 = (float)acc[j + 3] * (sc[j + 3] * ws) + bb;
                  float u4 = (float)acc[j + 4] * (sc[j + 4] * ws) + bb;
                  float u5 = (float)acc[j + 5] * (sc[j + 5] * ws) + bb;
                  float u6 = (float)acc[j + 6] * (sc[j + 6] * ws) + bb;
                  float u7 = (float)acc[j + 7] * (sc[j + 7] * ws) + bb;
                  dst[j    ] = jx_post_snake_k(u0, gbase + o, KS);
                  dst[j + 1] = jx_post_snake_k(u1, gbase + o, KS);
                  dst[j + 2] = jx_post_snake_k(u2, gbase + o, KS);
                  dst[j + 3] = jx_post_snake_k(u3, gbase + o, KS);
                  dst[j + 4] = jx_post_snake_k(u4, gbase + o, KS);
                  dst[j + 5] = jx_post_snake_k(u5, gbase + o, KS);
                  dst[j + 6] = jx_post_snake_k(u6, gbase + o, KS);
                  dst[j + 7] = jx_post_snake_k(u7, gbase + o, KS);
                  JX_AMU(_ams, j    , dst[j    ]); JX_AMU(_ams, j + 1, dst[j + 1]);
                  JX_AMU(_ams, j + 2, dst[j + 2]); JX_AMU(_ams, j + 3, dst[j + 3]);
                  JX_AMU(_ams, j + 4, dst[j + 4]); JX_AMU(_ams, j + 5, dst[j + 5]);
                  JX_AMU(_ams, j + 6, dst[j + 6]); JX_AMU(_ams, j + 7, dst[j + 7]);
                }
              for (; j < w; j++)
                { float _v = jx_post_snake_k((float)acc[j] * (sc[j] * ws) + bb,
                                             gbase + o, KS);
                  dst[j] = _v; JX_AMU(_ams, j, _v); }
            }
          else
            {
              int j = 0;
              for (; j + 8 <= w; j += 8)
                {
                  dst[j    ] = (float)acc[j    ] * (sc[j    ] * ws) + bb;
                  dst[j + 1] = (float)acc[j + 1] * (sc[j + 1] * ws) + bb;
                  dst[j + 2] = (float)acc[j + 2] * (sc[j + 2] * ws) + bb;
                  dst[j + 3] = (float)acc[j + 3] * (sc[j + 3] * ws) + bb;
                  dst[j + 4] = (float)acc[j + 4] * (sc[j + 4] * ws) + bb;
                  dst[j + 5] = (float)acc[j + 5] * (sc[j + 5] * ws) + bb;
                  dst[j + 6] = (float)acc[j + 6] * (sc[j + 6] * ws) + bb;
                  dst[j + 7] = (float)acc[j + 7] * (sc[j + 7] * ws) + bb;
                  JX_AMU(_ams, j    , dst[j    ]); JX_AMU(_ams, j + 1, dst[j + 1]);
                  JX_AMU(_ams, j + 2, dst[j + 2]); JX_AMU(_ams, j + 3, dst[j + 3]);
                  JX_AMU(_ams, j + 4, dst[j + 4]); JX_AMU(_ams, j + 5, dst[j + 5]);
                  JX_AMU(_ams, j + 6, dst[j + 6]); JX_AMU(_ams, j + 7, dst[j + 7]);
                }
              for (; j < w; j++)
                { float _v = (float)acc[j] * (sc[j] * ws) + bb;
                  dst[j] = _v; JX_AMU(_ams, j, _v); }
            }
{ unsigned _t = jx_ccount(); jx_kc_scy[_kcsi][4] += (unsigned long)(unsigned)(_t - _kcA); _kcA = _t;
            int _ob = (o == 0) ? 0 : ((o == 1) ? 1 : 2);
            jx_kc_sepi[_kcsi][_ob] += (unsigned long)(unsigned)(_t - _ep0); }
        }
      if (_ams != NULL)   /* P0：本 t 块的逐位置 max 整块回写 */
        { unsigned *_ad = _am + (size_t)t0;
          for (int _j = 0; _j < w; _j++) { _ad[_j] = _ams[_j]; } }
      { unsigned _t = jx_ccount(); jx_kc_scy[_kcsi][5] += (unsigned long)(unsigned)(_t - _kc0); }
      JXP_END(JP_K1MM);
    }
}

static int jx_conv1d_k1_cf(const float *b, const float *xin, float *yout,
                           int Ci, int Co, int T_in, int T_out, int kstride,
                           const signed char *wgc, const float *wsc, int gbase)
{
  const int cpad = (Ci + 15) & ~15;
  const int nv = cpad >> 4;
  if (nv < 1 || nv > 32 || Co > JX_I8OC_N) { return -1; }
  if (jx_conv_post == 2 && jx_conv_post_alpha == NULL) { return -1; }
  int jc = jx_core_id();
  signed char *qb = g_tnq[jc];
  float *sc = g_tns[jc];
  /* 第三十六轮：sf（逐位置 gather 用的暂存）随 gather 一起去掉，
   * 改用逐位置 1/scale 暂存 s_tni（容量 JX_I8TN_N，与 TB 上限一致）。 */
  float *s_tni = g_tni[jc];

  /* r43b：分块尺寸要同时满足两个约束
   *   (a) 量化缓冲 TB*cpad <= JX_I8TN_L1（qb 容量）；
   *   (b) 输出写工作集 Co*TB*4 <= 8KB —— 否则每个 t 块都要重踩 4x 缓存行，
   *       实测 Co=80 时 TB=256 反而比 k1s 的 64 慢（700 vs 610 ms）。
   * 两条一起取，既拿到大块的点积复用，又不让写出缓存。 */
  int TB = jx_i8_blk("JX_K1TB", 0);
  /* r44f：覆盖值也必须服从 qb 容量（TB*cpad <= JX_I8TN_L1），
   * 否则 JX_K1TB 一大就把 g_tnq 写爆（Ci=81 时 256*96=24KB > 8KB）。 */
  { int cap = JX_I8TN_L1 / cpad; if (cap < 1) { cap = 1; }
    if (TB <= 0 || TB > cap) { TB = cap; } }
  if (TB > JX_I8TN_N) { TB = JX_I8TN_N; }
  { /* r51a：输出写工作集上限可调（JX_K1WC，0=不限），用于扫描 TB 与散点写的取舍 */
    int _wc = jx_i8_blk("JX_K1WC", 8192);
    if (_wc > 0) { int wcap = _wc / Co; if (wcap < 1) { wcap = 1; } if (TB > wcap) { TB = wcap; } } }
  if (TB > T_out) { TB = T_out; }
  int _kcsi = jx_kc_slot(Ci, Co, T_in, T_out, TB);
  jx_kc_spost[_kcsi] = jx_conv_post;
  int _nblk = (T_out + TB - 1) / TB;
  {
    jx_k1ctx_t cc;
    cc.b = b; cc.xin = xin; cc.yout = yout;
    cc.wgc = wgc; cc.wsc = wsc;
    cc.Ci = Ci; cc.Co = Co; cc.T_in = T_in; cc.T_out = T_out;
    cc.kstride = kstride; cc.gbase = gbase;
    cc.cpad = cpad; cc.nv = nv; cc.TB = TB; cc.kcsi = _kcsi;
    cc.qb = qb; cc.sc = sc; cc.s_tni = s_tni;
    /* P0：输入侧先探边车（必须在 arm 之前，否则会把自己要读的那片清掉） */
    {
      int _amc = 0;
      const unsigned *_amx = jx_amax_probe(xin, Ci, T_in, &_amc);
      cc.amx = _amx; cc.amc = _amc; cc.am = NULL;
      if (jx_i8_blk("JX_K1EPV", 0) == 0) { jx_amax_arm(yout, Co, T_out); cc.am = jx_amax_cur; }
      else { jx_amax_kill(); }
      if (jx_pfx("JX_PFK")) { jx_pf_run_min(_nblk, jx_conv1d_k1_cf_tb, &cc, 2); }
      else { jx_conv1d_k1_cf_tb(&cc, 0, _nblk); }
      if (cc.am != NULL) { jx_amax_pub(); }
    }
  }
  return 0;
}

/* ---------------- conv1d: im2col(融合量化) + int8 点积 ----------------
 * groups==1（普通卷积）与 groups==Ci==Co（depthwise）都走这一条路径。 */

/* r80：k1 per-t 路径的 t 块循环抽成可重入函数，供双核按块区间切分。
 * 数值逐位不变：每个块只写自己的输出区间，块间无共享写状态；
 * s_tnq/s_tns/s_tni/s_tnm/s_tnacc 全部是 per-core 私有。 */
typedef struct {
  int T_out, TN, K, K16, cpg, copg, k, dil, pad, stride, T_in, g, nrow, quse_k1, kstride;
  const float *b;
  const signed char *wgc;
  const float *wsc;
  const float *xin;
  float *yout;
} jx_c1i8_ctx_t;

JX_HOT static void jx_conv1d_i8_tb(void *va, int blo, int bhi)
{
  jx_c1i8_ctx_t *C = (jx_c1i8_ctx_t *)va;
  const int T_out = C->T_out, TN = C->TN, K = C->K, K16 = C->K16;
  const int cpg = C->cpg, copg = C->copg, k = C->k, dil = C->dil;
  const int pad = C->pad, stride = C->stride, T_in = C->T_in, g = C->g;
  const int nrow = C->nrow, quse_k1 = C->quse_k1, kstride = C->kstride;
  const float *b = C->b;
  const signed char *wgc = C->wgc;
  const float *wsc = C->wsc;
  const float *xin = C->xin;
  float *yout = C->yout;
  const int _jc = jx_core_id();
  signed char *s_tnq = g_tnq[_jc];
  float *s_tns = g_tns[_jc];
  float *s_tni = g_tni[_jc];
  float *s_tnm = g_tnm[_jc];
  int32_t s_tnacc[JX_I8TN_N];
  (void)K;
for (int t0 = blo * TN; t0 < T_out && t0 < bhi * TN; t0 += TN)
                {
                  int tn = t0 + TN; if (tn > T_out) { tn = T_out; }
                  int nj = tn - t0;
                  /* 2026-09-12 #8：im2col 改成"通道外层 + 两遍"。
                   *
                   * 原来是 j 外层、ci 内层，一个 j 要踩 cpg 条相隔
                   * T_in*4 字节的缓存行。本模型最热的两个卷积 cpg=20 / 81，
                   * 而 32KB L1（32B/行）的 set 数只有 128：步长 Tt*4 对应的
                   * set 步进恰好是小数（例：Tt=1152 -> 4608B -> 144 行 -> set 步进 16），
                   * 于是 81 条行全落在同一小殤 8 个 set 里，8-way 被撞爆，
                   * 每个元素都变成一次 PSRAM 伪命中（≈170 周期）。
                   * 这就是 conv 长期卡在 9~最近 12 周期/MAC、而 linear（行内连续）
                   * 只有 2.8 周期/MAC 的根因。
                   *
                   * 现在改成：对每个 ci 把 xr[t0..t0+nj) 连续扫完（工作集只有 1~2 条行），
                   * 第一遍求每行的 max，第二遍直接量化写进 s_tnq。
                   * 每行的 scale、舍入、截断公式与 jx_q_row 完全相同，
                   * 所以结果仍然逐位一致（max 与求和顺序无关）。 */
                 JXP_BEG(JP_K1Q);
                  { unsigned long _s0 = jx_ccount();
                  for (int j = 0; j < nj; j++) { s_tnm[j] = 0.0f; }
                  for (int ci = 0; ci < cpg; ci++)
                    {
                      const float *xr = xin + (size_t)(g * cpg + ci) * T_in;
                      if (stride == 1)
                        {
                          /* 2026-09-12 第五轮：stride==1 时把 j 循环拆成只走有效区。
                           * 原写法每个元素都要算一次 pos = (t0+j)*stride + dpos，
                           * 再做两次越界判断；而在这一层 stride/dpos 都是常量，
                           * 有效区内根本不需要判断。越界位置原本按 0 参与
                           * （|0| = 0）不影响 max（max 从 0 起），所以直接跳过即可，
                           * 结果逐位不变。边界区最多 k-1 个元素，代价可忽。 */
                          for (int kk = 0; kk < k; kk++)
                            {
                              int base = t0 + kk * dil - pad;
                              const float *xs = xr + base;
                              int jlo = (base < 0) ? -base : 0;
                              int jhi = (base + nj > T_in) ? (T_in - base) : nj;
                              if (jlo > nj) { jlo = nj; }
                              if (jhi > nj) { jhi = nj; }
                              if (jhi < jlo) { jhi = jlo; }
                              for (int j = jlo; j < jhi; j++)
                                {
                                  float a = fabsf(xs[j]);
                                  if (a > s_tnm[j]) { s_tnm[j] = a; }
                                }
                            }
                          continue;
                        }
                      for (int kk = 0; kk < k; kk++)
                        {
                          int dpos = kk * dil - pad;
                          for (int j = 0; j < nj; j++)
                            {
                              int pos = (t0 + j) * stride + dpos;
                              if (pos >= 0 && pos < T_in)
                                {
                                  float a = fabsf(xr[pos]);
                                  if (a > s_tnm[j]) { s_tnm[j] = a; }
                                }
                            }
                        }
                    }
                  jx_k1s_cyc[0] += (unsigned long)(unsigned)(jx_ccount() - _s0);
                  jx_k1s_qe    += (double)cpg * (double)k * (double)nj; }
                  { unsigned long _s1 = jx_ccount();
                  for (int j = 0; j < nj; j++)
                    {
                      s_tni[j] = (s_tnm[j] > 1e-30f) ? (127.0f / s_tnm[j]) : 0.0f;
                      s_tns[j] = (s_tnm[j] > 1e-30f) ? (s_tnm[j] / 127.0f) : 1e-30f;
                      for (int t = K; t < K16; t++) { s_tnq[(size_t)j * K16 + t] = 0; }
                    }
                  for (int ci = 0; ci < cpg; ci++)
                    {
                      const float *xr = xin + (size_t)(g * cpg + ci) * T_in;
                      if (stride == 1)
                        {
                          /* 同 Pass A：有效区内去掉 pos 计算与越界判断。
                           * 越界位置原来写的是 quantize(0.0f) = 0，这里显式补 0。 */
                          for (int kk = 0; kk < k; kk++)
                            {
                              int base = t0 + kk * dil - pad;
                              const float *xs = xr + base;
                              int ci0  = kk * cpg + ci;   /* [kk][c]，与权重行同序 */
                              int jlo = (base < 0) ? -base : 0;
                              int jhi = (base + nj > T_in) ? (T_in - base) : nj;
                              signed char *dq = s_tnq + ci0;
                              if (jlo > nj) { jlo = nj; }
                              if (jhi > nj) { jhi = nj; }
                              if (jhi < jlo) { jhi = jlo; }
                              for (int j = 0; j < jlo; j++) { dq[(size_t)j * K16] = 0; }
                              /* 第三十六轮：Pass B 实测 33.5 周期/(ci,j)（101 Mcyc，
                               * 3.03M 元素对），同样的病：运行期长度的 j 循环不展开，
                               * 每元素一条 load→mul→cvt→clamp→store 串行链。
                               * 4 路展开，逐元素算式不变。 */
                              int j = jlo;
                              for (; j + 4 <= jhi; j += 4)
                                {
                                  float q0 = xs[j]     * s_tni[j];
                                  float q1 = xs[j + 1] * s_tni[j + 1];
                                  float q2 = xs[j + 2] * s_tni[j + 2];
                                  float q3 = xs[j + 3] * s_tni[j + 3];
                                  int i0 = jx_rndf(q0);
                                  int i1 = jx_rndf(q1);
                                  int i2 = jx_rndf(q2);
                                  int i3 = jx_rndf(q3);
                                  if (i0 > 127) { i0 = 127; } else if (i0 < -127) { i0 = -127; }
                                  if (i1 > 127) { i1 = 127; } else if (i1 < -127) { i1 = -127; }
                                  if (i2 > 127) { i2 = 127; } else if (i2 < -127) { i2 = -127; }
                                  if (i3 > 127) { i3 = 127; } else if (i3 < -127) { i3 = -127; }
                                  dq[(size_t)j * K16]       = (signed char)i0;
                                  dq[(size_t)(j + 1) * K16] = (signed char)i1;
                                  dq[(size_t)(j + 2) * K16] = (signed char)i2;
                                  dq[(size_t)(j + 3) * K16] = (signed char)i3;
                                }
                              for (; j < jhi; j++)
                                {
                                  float v = xs[j];
                                  float q = v * s_tni[j];
                                  int iv = jx_rndf(q);
                                  if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
                                  dq[(size_t)j * K16] = (signed char)iv;
                                }
                              for (int j = jhi; j < nj; j++) { dq[(size_t)j * K16] = 0; }
                            }
                          continue;
                        }
                      for (int kk = 0; kk < k; kk++)
                        {
                          int dpos = kk * dil - pad;
                          int ci0  = kk * cpg + ci;   /* [kk][c]，与权重行同序 */
                          for (int j = 0; j < nj; j++)
                            {
                              int pos = (t0 + j) * stride + dpos;
                              float v = (pos >= 0 && pos < T_in) ? xr[pos] : 0.0f;
                              float q = v * s_tni[j];
                              int iv = jx_rndf(q);
                              if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
                              s_tnq[(size_t)j * K16 + ci0] = (signed char)iv;
                            }
                        }
                    }
                  jx_k1s_cyc[1] += (unsigned long)(unsigned)(jx_ccount() - _s1); }
                  JXP_END(JP_K1Q);
                  JXP_BEG(JP_K1MM);
                  { unsigned long _k1c = jx_ccount();
                  if (quse_k1)
                    {
                      /* 2026-09-15 第三十五轮：输出秩序改成「先按 16 通道一组
                       * 算完 nj 个位置，再按通道连续写出」。
                       *
                       * 原来写成 `for j { for o0 { yout[oc*T_out + t0 + j] } }`，
                       * 每个输出位置都要往 80 个相隔 T_out*4 字节的地址各散写一次，
                       * PSRAM 上每次都是新 cache line（写分配 + 回写），把 QACC
                       * 省下来的 accx 往返又全部吃掉 —— 板端实测 dot 桶 730 -> 840 ms，
                       * 整机零收益。改成按通道连续写 nj 个 float 后，与 accx 路径的
                       * `yo[j] = ...` 完全同构。
                       *
                       * 数值不变：每个元素的点积与 epilogue 算式逐项相同，只是
                       * 中间结果先落在 tile 里。tile 复用 g_ocacc（该分支不用它）：
                       * 16 x JX_I8TN_N 恰好是 JX_I8OC_N 个 int32。 */
                      float *tile = (float *)g_ocacc[jx_core_id()];
                      int32_t qout[16], qtmp[16];
                      for (int o0 = 0; o0 < copg; o0 += 16)
                        {
                          int nn = copg - o0; if (nn > 16) { nn = 16; }
                          const signed char *wt = g_qwT + (size_t)(o0 >> 4) * (size_t)K16 * 16u;
                          for (int j = 0; j < nj; j++)
                            {
                              const signed char *xrow = s_tnq + (size_t)j * K16;
                              /* 20-bit lane：单次调用最多 2 个 16 字节块，
                               * 超出部分按 2 块一批降到 int32 累加。 */
                              if (nrow <= 2)
                                {
                                  jx_qacc_dot16(wt, xrow, nrow, qout);
                                }
                              else
                                {
                                  memset(qout, 0, sizeof(qout));
                                  for (int c0 = 0; c0 < nrow; c0 += 2)
                                    {
                                      int nc = nrow - c0;
                                      if (nc > 2) { nc = 2; }
                                      jx_qacc_dot16(wt + (size_t)c0 * 256u,
                                                    xrow + (size_t)c0 * 16u, nc, qtmp);
                                      for (int i = 0; i < 16; i++)
                                        {
                                          qout[i] += qtmp[i];
                                        }
                                    }
                                }
                              float *tr = tile + (size_t)j * 16u;
                              for (int oo = 0; oo < nn; oo++)
                                {
                                  int co = o0 + oo;
                                  int oc = g * copg + co;
                                  float v = (float)qout[oo] * (s_tns[j] * wsc[co]) +
                                            ((b != NULL) ? b[oc] : 0.0f);
                                  if (jx_conv_post == 1)
                                    {
                                      tr[oo] = gelu_f(v);
                                    }
                                  else if (jx_conv_post == 2)
                                    {
                                      tr[oo] = jx_post_snake(v, oc);
                                    }
                                  else
                                    {
                                      tr[oo] = v;
                                    }
                                }
                            }
                          for (int oo = 0; oo < nn; oo++)
                            {
                              int oc = g * copg + o0 + oo;
                              float *yo = &yout[(size_t)oc * T_out + t0];
                              for (int j = 0; j < nj; j++)
                                {
                                  yo[j] = tile[(size_t)j * 16u + oo];
                                }
                          }
                        }
                      jx_k1q_cyc += (unsigned long)(unsigned)(jx_ccount() - _k1c);
                      jx_k1q_pos += (unsigned long)nj;
                    }
                  else
                    {
                      unsigned long _s2 = jx_ccount();
                      for (int co = 0; co < copg; co++)
                        {
                          const signed char *wp = wgc + (size_t)co * kstride;
                          int   oc   = g * copg + co;
                          float ws   = wsc[co];
                          float bb   = (b != NULL) ? b[oc] : 0.0f;
                          float *yo  = jx_ep_pick(&yout[(size_t)oc * T_out + t0]);
                          /* 2026-09-12 第五轮：改成"权重行常驻 PIE 寄存器"的
                           * 点积批。旧写法每个 (co,j) 都把权重行从内存重新装一遍，
                           * 而权重在 j 方向上完全重复。现在每个 co 只装一次。
                           * 每个输出元素的点积求和次序不变，结果逐位一致。 */
                          jx_dotrow_i8(s_tnq, K16, wp, nrow, nj, s_tnacc);
                          jx_k1s_cyc[2] += (unsigned long)(unsigned)(jx_ccount() - _s2);
                          /* 分支外提：jx_post_f 在这个循环里逐元素调用会让 GCC 生成
                           * 真实函数调用（模块内 static inline 未内联），板端实测
                           * 光这一项就多花 1.7 s。这里把 gelu/无 gelu 拆成两条紧循环，
                           * 数值与「先算卷积再逐元素 gelu」逐位相同。 */
                          if (jx_conv_post == 1)
                            {
                              /* 2026-09-15 第三十六轮：4 路展开。
                               * 新计数器 [k1s] 实测这段 epilogue 占 63.4 周期/输出
                               * 元素（3.06M 元素 = 194 Mcyc，全场最贵），而同样
                               * 4 条 FP 指令的 l2 epilogue（写 L1 tile）只要
                               * 22.2 周期/元素。差别不在 FPU：这里的 j 循环是
                               * 运行期长度、GCC 不展开，每个元素一条
                               * cvt→mul→mul→add→store 的串行依赖链（FPU 延迟
                               * 4.13 周期/条）完全暴露，再加上 gelu/snake 的
                               * 长 Horner 链，延迟乘 3~4 倍。
                               * 4 条互相独立的链并行后延迟被盖住，逐元素算式
                               * 一字未改，结果逐位一致。 */
                              /* 第三十九轮：9 个 gelu 系数在这里装载一次，
                               * 不再每个元素 l32r+wfr 重载。 */
                              const jx_erf9 KE = jx_erf9_load();
                              const int gx4 = jx_gx4();
                              int j = 0;
                              for (; j + 4 <= nj; j += 4)
                                {
                                  float u0 = (float)s_tnacc[j]     * (s_tns[j]     * ws) + bb;
                                  float u1 = (float)s_tnacc[j + 1] * (s_tns[j + 1] * ws) + bb;
                                  float u2 = (float)s_tnacc[j + 2] * (s_tns[j + 2] * ws) + bb;
                                  float u3 = (float)s_tnacc[j + 3] * (s_tns[j + 3] * ws) + bb;
                                  if (gx4)
                                    {
                                      yo[j]     = jx_gelu_fast(u0, KE);
                                      yo[j + 1] = jx_gelu_fast(u1, KE);
                                      yo[j + 2] = jx_gelu_fast(u2, KE);
                                      yo[j + 3] = jx_gelu_fast(u3, KE);
                                    }
                                  else
                                    {
                                      yo[j]     = jx_gelu_fast(u0, KE);
                                      yo[j + 1] = jx_gelu_fast(u1, KE);
                                      yo[j + 2] = jx_gelu_fast(u2, KE);
                                      yo[j + 3] = jx_gelu_fast(u3, KE);
                                    }
                                }
                              for (; j < nj; j++)
                                {
                                  yo[j] = jx_gelu_fast((float)s_tnacc[j] * (s_tns[j] * ws) + bb, KE);
                                }
                            }
                          else if (jx_conv_post == 2)
                            {
                              /* snake 折进 epilogue（legacy_unit）：原来 conv1d_d 之后
                               * 还要把整个 [B,C,T] 读一遍再写一遍，现在直接在这里算掉，
                               * 数值与 snake1d 逐位相同。 */
                              /* 第三十九轮：4 个 sin² 系数在这里装载一次。 */
                              const jx_sq4 KS = jx_sq4_load();
                              int j = 0;
                              for (; j + 4 <= nj; j += 4)
                                {
                                  yo[j]     = jx_post_snake_k((float)s_tnacc[j]     * (s_tns[j]     * ws) + bb, oc, KS);
                                  yo[j + 1] = jx_post_snake_k((float)s_tnacc[j + 1] * (s_tns[j + 1] * ws) + bb, oc, KS);
                                  yo[j + 2] = jx_post_snake_k((float)s_tnacc[j + 2] * (s_tns[j + 2] * ws) + bb, oc, KS);
                                  yo[j + 3] = jx_post_snake_k((float)s_tnacc[j + 3] * (s_tns[j + 3] * ws) + bb, oc, KS);
                                }
                              for (; j < nj; j++)
                                {
                                  yo[j] = jx_post_snake_k((float)s_tnacc[j] * (s_tns[j] * ws) + bb, oc, KS);
                                }
                            }
                          else
                            {
                              int j = 0;
                              for (; j + 4 <= nj; j += 4)
                                {
                                  yo[j]     = (float)s_tnacc[j]     * (s_tns[j]     * ws) + bb;
                                  yo[j + 1] = (float)s_tnacc[j + 1] * (s_tns[j + 1] * ws) + bb;
                                  yo[j + 2] = (float)s_tnacc[j + 2] * (s_tns[j + 2] * ws) + bb;
                                  yo[j + 3] = (float)s_tnacc[j + 3] * (s_tns[j + 3] * ws) + bb;
                                }
                              for (; j < nj; j++)
                                {
                                  yo[j] = (float)s_tnacc[j] * (s_tns[j] * ws) + bb;
                                }
                            }
                          jx_k1s_cyc[3] += (unsigned long)(unsigned)(jx_ccount() - _s2);
                          _s2 = jx_ccount();
                        }
                      jx_k1s_oe += (double)copg * (double)nj;
                      jx_k1n_cyc += (unsigned long)(unsigned)(jx_ccount() - _k1c);
                      jx_k1n_pos += (unsigned long)nj;
                    }
                  }
                  JXP_END(JP_K1MM);
                }

}

int jx_conv1d_i8(const float *W, const float *b, const T *x, T *y,
                 int Ci, int Co, int k, int stride, int pad, int groups,
                 int T_in, int T_out, int dil)
{
  jx_sq_kt_ensure();
  const JXWQ *wq;
  const signed char *wdata;
  const float *wscale;
  int cpg, copg, K, K16, ret = -1;
  float *sf;
  signed char *sq;
  int B = x->shape[0];

  /* per-core 分块缓冲：局部同名变量遮蔽文件作用域的 per-core 数组，
   * 函数体内所有 s_tnq/s_tns/s_tni/s_tnm 的用法保持不变。 */
  int            jc    = jx_core_id();
  signed char   *s_tnq = g_tnq[jc];
  float         *s_tns = g_tns[jc];
  float         *s_tni = g_tni[jc];
  float         *s_tnm = g_tnm[jc];

  if (!jx_i8_enabled() || B <= 0) { return -1; }
  /* Small-T shapes are dominated by int8 quantization/setup overhead.
   * Let them fall back to the direct FP32 kernel. */
  {
    int small_t = jx_i8_blk("JX_I8SMALLT", 0);
    if (small_t > 0 && T_out < small_t && k <= 7) { return -1; }
  }
  if (jx_ep_dst == NULL && jx_ep_on()) { jx_ep_dst = g_ep_scratch; }
  jx_conv_path = 0;
  if ((jx_i8_enabled() & 2) == 0) { return -1; }
  jx_pie_enable();
  jx_wq_init();
  wq = jx_wq_find(W);
  if (wq == NULL) { return -1; }
  if (groups <= 0 || Ci % groups != 0 || Co % groups != 0) { return -1; }
  cpg  = Ci / groups;
  copg = Co / groups;
  K    = cpg * k;
  K16  = (K + 15) & ~15;
  if (wq->nch != Co || wq->chlen != K) { return -1; }

  /* ---------------- 2026-09-12 #9：单向量深度卷积退回 FP32 ----------------
   * 板端 A/B（同一固件，只改 JX_INT8）：
   *     CU.dw_conv+in-permute   int8 2410 ms   vs   全 FP32 730 ms   ——快 3.3 倍
   *     FB.trendpool+branch     int8  710 ms   vs   全 FP32 620 ms
   * 原因：int8 路径每个输出位置都要先把 K 个活跃值量化
   * （求 max + 乘 scale + 舍入 + 截断，两遍访存）；而 K16<=16 的深度卷积
   * 一次量化只服务 **1 个输出×K 个 MAC**（copg=1），量化开销根本摊不薄；
   * 而 PIE 对只有 1 个向量的点积也只能做 16 个 MAC（36 周期往返）。
   * FP32 的 conv1d_core 直接 madd.s，省掉整个量化环节。
   *
   * 反例（不能切）：Ci=16 Co=16 k=1 g=1（K16=16 但 copg=16）、以及所有
   * K16>=32 的密集卷积 —— 它们量化一次能服务十几个输出，
   * int8 依然快 8~20 倍（FB.conv_1 640 vs 4850，FB.conv_2 680 vs 13850）。
   *
   * JX_I8DW=1 可强制这类形状继续走 int8（对照用）。 */
  if (jx_i8_blk("JX_I8DW", 0) == 0 && K16 <= 16 && copg <= 8) { return -1; }

  wdata  = g_wq_blob + wq->qoff;
  wscale = JXWQ_SCALES + wq->sc;

  /* 2026-09-15 结构性优化：channels-first 的 k=1 卷积专门走分块内核。
   * 输入 x[C][T] 直接按时间块量化成内部 q[T][Cpad]，点积后把输出小块先写
   * 内部 SRAM，再按输出通道连续 flush 到 y[C][T]。省掉原来逐位置路径里的
   * 跨通道 gather、im2col 拷贝，以及 channels-first 输出的散点直写。 */
if (jx_i8_blk("JX_K1CF", 1) != 0 &&
      B == 1 && k == 1 && stride == 1 && pad == 0 && dil == 1 && groups == 1)
    {
      int ok = 1;
      for (int bi = 0; bi < B && ok; bi++)
        {
          ok = (jx_conv1d_k1_cf(b, x->d + (size_t)bi * Ci * T_in,
                                 y->d + (size_t)bi * Co * T_out,
                                 Ci, Co, T_in, T_out, wq->kstride,
                                 wdata, wscale, 0) == 0);
        }
      if (ok) { jx_conv_path = 4; jx_conv_post_used = jx_conv_post; return 0; }
    }

  /* 2026-09-13 第六轮：密集大 T 卷积走"分块预量化 + 批量点积"快路径。
   * 只接 k>=3 的密集卷积（copg>=2）；depthwise 在上面已被退回 FP32。 */
  /* 2026-09-13 第八轮：k>=3 的限制去掉（改成 k>=1）。
   * 板端按形状直方图显示，1x1 卷积（first_block conv_1 20->80 / conv_2 81->16、
   * decoder 末端 16->16）全部落在"逐 t im2col"慢路径上：
   *     Ci=81 Co=16 k=1  410 ms  4.49 cyc/MAC
   *     Ci=20 Co=80 k=1  350 ms  3.10 cyc/MAC
   *     Ci=16 Co=16 k=1  370 ms  7.21 cyc/MAC   （合计 1130 ms）
   * 而分块快路径（一块内一次量化 + 权常常驻 PIE 寄存器）对 k=1 完全适用：
   * im2col 退化成"按 t 连续搬 1 个字节"，感受野就是块本身。 */
  /* 2026-09-13 第八轮：k<3 的密集卷积也接进分块快路径，但**限制 cpg**。
   *
   * 分块快路径的量化是"块内共用一个 scale"（max 取遍 cpg x WT 个值）。
   * 主机实测（libri0，8bit，SNR vs fp32）：
   *     JX_K1MAX=0（全部走慢路径） 14.18 dB
   *     =4 / =16 / =20 / =28       14.16 / 14.11 / 14.41 / 14.39 dB
   *     =100（含 Ci=81）           11.91 dB      <-- 掉 2.3 dB
   * 也就是说 cpg=81 的 1x1 卷积（first_block conv_2）块内动态范围太宽，
   * 单一 scale 撑不住。cpg<=32 时反而略优于慢路径（1x1 的慢路径逐位置
   * 量化在主机上偶有舍入方向不同）。
   * JX_DTW 扫描也印证这一点：256->11.91 / 64->12.99 / 16->13.60 dB，
   * 块越小越好，但永远追不上逐位置量化。 */
  if (k >= 1 && copg >= 2 && K16 <= 128 && T_out >= 32 &&
      (k >= 3 || cpg <= jx_i8_blk("JX_K1MAX", 0)))
    {
      int ok = 1;
      for (int bi = 0; bi < B && ok; bi++)
        {
          for (int g = 0; g < groups && ok; g++)
            {
              ok = (jx_conv1d_dense_tile(
                        b, x->d + (size_t)bi * Ci * T_in + (size_t)g * cpg * T_in,
                        y->d + (size_t)bi * Co * T_out + (size_t)g * copg * T_out,
                        cpg, copg, k, stride, pad, dil, T_in, T_out, K, K16,
                        wq->kstride,
                        wdata + (size_t)g * copg * wq->kstride,
                        wscale + (size_t)g * copg,
                        (jx_conv_post_alpha != NULL) ? (jx_conv_post_alpha + (size_t)g * copg) : NULL,
                        g * copg) == 0);
            }
        }
      if (ok) { jx_conv_path = 1; jx_conv_post_used = jx_conv_post; return 0; }
    }

  sf = scratch_f(K);
  sq = scratch_q(K16);
  if (sf == NULL || sq == NULL) { return -1; }

  /* JX_I8AQ —— 激活量化的粒度（决定量化开销能不能从 t 循环里提出来）：
   *   0 = 逐 t 两遍量化（老路径）：每个输出位置都重新求 max 再量化。
   *   1 = 整个输入张量预量化一次（张量级 scale）。
   *   2 = 默认：只在 depthwise（cpg==1）时预量化 —— 此时 group 内只有一个
   *       输入通道，得到的天然就是"逐通道 scale"，精度不降反升；
   *       其余情况仍走逐 t 路径保精度。
   * 背景：绝大多数卷积是 depthwise，每输出通道只做 1 个 16 字节点积，
   *       而逐 t 路径的量化开销是点积的几十倍（实测 17.6 周期/MAC）。
   * 主机实测（libri0，8bit 音频 SNR vs fp32）：aq=0 -> 13.4 dB，
   *       aq=1 -> 9.4 dB（掉 4 dB，弃用），aq=2 -> 见回归结果。 */
  int aq = jx_i8_blk("JX_I8AQ", 2);
  jx_conv_path = 3;
  int pre_q = (aq == 1) || (aq == 2 && cpg == 1);
  if (pre_q) { jx_conv_path = 2; }
  signed char *qx = NULL;
  if (pre_q)
    {
      /* 缓冲只需容纳"本 group"的 cpg 个输入通道：
       *   aq==2 时 cpg 恒为 1 -> 只要 T_in 字节（很小）；
       *   aq==1 才需要 cpg*T_in，为此设上限，超了就退回逐 t 路径。
       * 这一点很关键：模型堆峰值 5.5MB / 总量 6.4MB，一次大 malloc 就会让
       * 后面的 t_alloc 返回 NULL，表现为 conv 里对 y->d 写空指针而炸机。 */
      size_t qx_need = (aq == 1) ? ((size_t)Ci * (size_t)T_in)
                                 : ((size_t)T_in + 64u);
      size_t qx_cap  = (size_t)jx_i8_blk("JX_I8QX", 16384);
      if (qx_need > qx_cap) { pre_q = 0; }
      else
        {
          qx = scratch_qx((int)qx_need);
          if (qx == NULL) { pre_q = 0; }
        }
    }

  ret = 0;
  for (int bi = 0; bi < B; bi++)
    {
      const float *xin  = &x->d[(size_t)bi * Ci * T_in];
      float       *yout = &y->d[(size_t)bi * Co * T_out];
      for (int g = 0; g < groups; g++)
        {
          const signed char *wgc = wdata + (size_t)g * copg * wq->kstride;
          const float *wsc = wscale + (size_t)g * copg;
          /* 2026-09-14 第十八轮：k=1 的“逐位置 scale + 预转置”路径（JX_I8AQ=3）。
           *
           * 背景：pert 路径在 k=1 时的量化 prologue 对每个输出位置
           * 重新跨身读一遍 cpg 行（步长 T_in*4），而且 dot 前还要
           * 把这 cpg 个字节 im2col 收集到 sq。两项合计占了 k=1
           * 卷积的大部分时间（61 MMAC 却要 1.3 s）。
           *
           * k=1 时逐位置的 max 恰好就是“跨通道 max”，与 pert 路径的
           * s_tnm[j] 语义完全相同（求和顺序无关的 max），所以结果逐位一致。
           * 做法：
           *   1) 每行顺序扫一遍求逐位置 max（访存从跨步变顺序）
           *   2) 每行再顺序扫一遍量化，直接写成 [t][cpad] 布局
           *   3) dot 阶段直接取 qx3 + pos*cpad，完全不用 im2col 收集
           * 逐位一致的前提：扩展字节为 0（与原 im2col 填 0 一致）。 */
          if (aq == 3 && k == 1 && stride == 1 && dil == 1 && pad == 0 && cpg >= 2)
            {
              const int cpad  = (cpg + 15) & ~15;
              const int core  = jx_core_id();
              const size_t nq = (size_t)T_in * (size_t)cpad;
              if (g_aq3_cpad[core] != cpad || (size_t)g_aq3_qcap[core] < nq)
                {
                  free(g_aq3_q[core]); g_aq3_q[core] = NULL;
                  g_aq3_qcap[core] = 0; g_aq3_cpad[core] = 0;
                  { void *_r = malloc(nq + 32u);
        g_aq3_q[core] = (_r != NULL) ? (signed char *)(((uintptr_t)_r + 15u) & ~(uintptr_t)15u) : NULL; }
                  if (g_aq3_q[core] != NULL)
                    {
                      memset(g_aq3_q[core], 0, nq);
                      g_aq3_qcap[core] = (int)nq;
                      g_aq3_cpad[core] = cpad;
                    }
                }
              if (g_aq3_mcap[core] < T_in)
                {
                  free(g_aq3_m[core]); g_aq3_m[core] = NULL;
                  g_aq3_m[core] = (float *)malloc(sizeof(float) * (size_t)T_in);
                  g_aq3_mcap[core] = (g_aq3_m[core] != NULL) ? T_in : 0;
                }
              signed char *qx3 = g_aq3_q[core];
              float       *ma  = g_aq3_m[core];
              if (qx3 != NULL && ma != NULL)
                {
                  const float *xgo = xin + (size_t)(g * cpg) * T_in;
                  for (int t = 0; t < T_in; t++) { ma[t] = 0.0f; }
                  for (int ci = 0; ci < cpg; ci++)
                    {
                      const float *xr = xgo + (size_t)ci * T_in;
                      for (int t = 0; t < T_in; t++)
                        {
                          float a = fabsf(xr[t]);
                          if (a > ma[t]) { ma[t] = a; }
                        }
                    }
                  for (int t = 0; t < T_in; t++)
                    {
                      ma[t] = (ma[t] > 1e-30f) ? (127.0f / ma[t]) : 0.0f;
                    }
                  for (int ci = 0; ci < cpg; ci++)
                    {
                      const float *xr = xgo + (size_t)ci * T_in;
                      for (int t = 0; t < T_in; t++)
                        {
                          float v = xr[t] * ma[t];
                          int iv = jx_rndf(v);
                          if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
                          qx3[(size_t)t * cpad + ci] = (signed char)iv;
                        }
                    }
                  for (int t = 0; t < T_out; t++)
                    {
                      const signed char *sq3 = qx3 + (size_t)t * cpad;
                      float s3 = (ma[t] > 0.0f) ? (1.0f / ma[t]) : 1e-30f;
                      for (int co = 0; co < copg; co++)
                        {
                          int32_t acc = jx_dot_i8(sq3, wgc + (size_t)co * wq->kstride,
                                                  K16 >> 4);
                          int oc = g * copg + co;
                          yout[(size_t)oc * T_out + t] =
                              (jx_conv_post == 2)
                                ? jx_post_snake((float)acc * (s3 * wsc[co]) + (b != NULL ? b[oc] : 0.0f), oc)
                                : jx_post_f((float)acc * (s3 * wsc[co]) + (b != NULL ? b[oc] : 0.0f));
                        }
                    }
                  jx_mac_conv += (double)T_out * copg * (double)K;
                  continue;
                }
            }
          if (pre_q)
            {
              const float *xgo = xin + (size_t)(g * cpg) * T_in;
              size_t nin = (size_t)cpg * (size_t)T_in;
              size_t i;
              float mx = 0.0f;
              float s, is;
              for (i = 0; i < nin; i++)
                {
                  float a = fabsf(xgo[i]);
                  if (a > mx) { mx = a; }
                }
              if (mx > 1e-30f) { s = mx / 127.0f; is = 127.0f / mx; }
              else             { s = 1e-30f;      is = 0.0f; }
              for (i = 0; i < nin; i++)
                {
                  float v = xgo[i] * is;
                  int iv = jx_rndf(v);
                  if (iv > 127) { iv = 127; } else if (iv < -127) { iv = -127; }
                  qx[i] = (signed char)iv;
                }
              for (int t = 0; t < T_out; t++)
                {
                  int base = t * stride - pad;
                  int ci;
                  for (ci = 0; ci < cpg; ci++)
                    {
                      const signed char *xr = qx + (size_t)ci * T_in;
                      int kk;
                      for (kk = 0; kk < k; kk++)
                        {
                          int pos = base + kk * dil;
                          sq[(size_t)kk * cpg + ci] =
                              (pos >= 0 && pos < T_in) ? xr[pos] : (signed char)0;
                        }
                    }
                  for (ci = K; ci < K16; ci++) { sq[ci] = 0; }
                  for (int co = 0; co < copg; co++)
                    {
                      int32_t acc = jx_dot_i8(sq, wgc + (size_t)co * wq->kstride, K16 >> 4);
                      int oc = g * copg + co;
                      yout[(size_t)oc * T_out + t] =
                          (jx_conv_post == 2)
                            ? jx_post_snake((float)acc * (s * wsc[co]) + (b != NULL ? b[oc] : 0.0f), oc)
                            : jx_post_f((float)acc * (s * wsc[co]) + (b != NULL ? b[oc] : 0.0f));
                    }
                }
              continue;
            }
          /* 2026-09-12 #6：输出位置分块（t 分块）。
           *
           * 板端子探针实测：first_block 里两个 1x1 卷积（20->80、81->16）
           * 占 conv1d 总时间的 62%（2270 + 3360 ms），而它们的 MAC 只有
           * 3.3M、im2col 元素只有 23K/93K —— 按算子模型只需 5 ms/块，实测
           * 142 / 210 ms。对照：linear 做完全相同的数学（同样的 jx_q_row +
           * jx_dot_i8、同样的分块），只有 2.85 周期/MAC，而这两个卷积是
           * 18~23 周期/MAC。唯一的结构差别是写回：
           *     linear : yo[o]                    —— 连续
           *     conv   : yout[oc*T_out + t]       —— 相邻元素相隔 T_out*4 字节
           * 于是把 t 分块：一次先把 TN 个输出位置的行全部量化好，再按 co
           * 连续写回这 TN 个输出（同时让同一份权重复用 TN 次）。
           * 每个输出元素的点积与 scale 与原来逐位相同。 */
          /* 2026-09-12 #6：输出位置分块（t 分块）。
           *
           * 板端子探针实测：first_block 里两个 1x1 卷积（20->80、81->16）
           * 占 conv1d 总时间的 62%（2270 + 3360 ms），而它们的 MAC 只有
           * 3.3M、im2col 元素只有 23K/93K —— 按算子模型只需 5 ms/块，实测
           * 142 / 210 ms。对照：linear 做完全相同的数学（同样的 jx_q_row +
           * jx_dot_i8、同样的分块），只有 2.85 周期/MAC，而这两个卷积是
           * 18~23 周期/MAC。结构差别有两个：
           *     写回 : linear yo[o] 连续 / conv yout[oc*T_out+t] 相邻隔 T_out*4 字节
           *     读取 : im2col 逐 t 拼行，每个 t 都要重走 cpg 个相隔 T_in*4
           *              字节的缓存行（T_in~3000 -> 每次都是缺页）
           * 于是把 t 分块：一次把 TN 个输出位置的行全部量化好，再按 co
           * 连续写回这 TN 个输出；同时相邻 t 共享同一条 im2col 缓存行
           * （缺页数 ~TN 倍下降），同一份权重也复用 TN 次。
           * 每个输出元素的点积与 scale 与原来逐位相同。 */
int TN = jx_i8_blk("JX_I8TN", 0);
          /* r43b：默认不再固定 64，按「量化缓冲 + 输出写工作集」自适应，
           * JX_I8TN=<n> 可覆盖（0/不设 = 自适应）。同 k1cf 的两条约束。 */
          if (TN <= 0) { TN = JX_I8TN_L1 / K16; if (TN < 1) { TN = 1; } }
          if (TN > JX_I8TN_N) { TN = JX_I8TN_N; }
          if (TN < 1) { TN = 1; }
          if (K16 > JX_I8TN_K) { TN = 1; }            /* 超出静态缓冲 -> 走逐 t 路径 */
          while (TN > 1 && TN * K16 > JX_I8TN_L1) { TN >>= 1; }
          { int _wc = jx_i8_blk("JX_I8WCAP", 8192);
            if (_wc > 0) { int wcap = _wc / copg; if (wcap < 1) { wcap = 1; }
                           if (TN > wcap) { TN = wcap; } } }
          if (TN > T_out) { TN = T_out; }
          if (TN > 1)
            {

              const int nrow = K16 >> 4;
              /* 2026-09-15 第三十五轮：QACC 已实测证伪（1.72x 更慢 + nchunks=2
               * 数值不正确），门禁退回原样，见 jx_conv1d_dense_tile 处注释。 */
              int cobk = (copg + 15) & ~15;
              int quse_k1 = jx_qacc_want() && nrow == 2 &&
                            copg >= 32 && ((copg & 15) == 0) &&
                            (size_t)K16 * (size_t)copg <= (size_t)JX_QW_BYTES;
              if (quse_k1)
                {
                  memset(g_qwT, 0, (size_t)K16 * (size_t)copg);
                  for (int o = 0; o < copg; o++)
                    {
                      int block = o >> 4;
                      int oo = o & 15;
                      signed char *wt = g_qwT + (size_t)block * (size_t)K16 * 16u + oo;
                      const signed char *src = wgc + (size_t)o * (size_t)wq->kstride;
                      for (int kk = 0; kk < K; kk++)
                        {
                          wt[(size_t)kk * 16u] = src[kk];
                        }
                    }
                }
                            (void)cobk;
              {
                jx_c1i8_ctx_t cc;
                cc.T_out = T_out; cc.TN = TN; cc.K = K; cc.K16 = K16;
                cc.cpg = cpg; cc.copg = copg; cc.k = k; cc.dil = dil;
                cc.pad = pad; cc.stride = stride; cc.T_in = T_in; cc.g = g;
                cc.nrow = nrow; cc.quse_k1 = quse_k1; cc.kstride = (int)wq->kstride;
                cc.b = b; cc.wgc = wgc; cc.wsc = wsc; cc.xin = xin; cc.yout = yout;
                jx_pf_run_min((T_out + TN - 1) / TN, jx_conv1d_i8_tb, &cc, 2);
              }
              continue;
            }
          for (int t = 0; t < T_out; t++)
            {
              int base = t * stride - pad;
              for (int kk = 0; kk < k; kk++)
                {
                  for (int ci = 0; ci < cpg; ci++)
                    {
                      const float *xr = xin + (size_t)(g * cpg + ci) * T_in;
                      int pos = base + kk * dil;
                      sf[(size_t)kk * cpg + ci] =
                          (pos >= 0 && pos < T_in) ? xr[pos] : 0.0f;
                    }
                }
              {
                float sx = jx_q_row(sf, sq, K);
                for (int co = 0; co < copg; co++)
                  {
                    int32_t acc = jx_dot_i8(sq, wgc + (size_t)co * wq->kstride, K16 >> 4);
                    int oc = g * copg + co;
                    yout[(size_t)oc * T_out + t] =
                        (jx_conv_post == 2)
                          ? jx_post_snake((float)acc * (sx * wsc[co]) + (b != NULL ? b[oc] : 0.0f), oc)
                          : jx_post_f((float)acc * (sx * wsc[co]) + (b != NULL ? b[oc] : 0.0f));
                  }
              }
            }
        }
    }
  if (ret == 0) { jx_conv_post_used = jx_conv_post; }
  return ret;
}

#else  /* !JX_HAVE_WQ8 —— 没有量化权重时全部退化为"不支持"，走原来的浮点路径 */

int  jx_i8_enabled(void) { return 0; }
void jx_wq_init(void) { }
int  jx_linear_i8(const float *W, const float *b, const float *x, float *y,
                  int rows, int in, int out)
{
  (void)W; (void)b; (void)x; (void)y; (void)rows; (void)in; (void)out;
  return -1;
}
int  jx_conv1d_i8(const float *W, const float *b, const T *x, T *y,
                  int Ci, int Co, int k, int stride, int pad, int groups,
                  int T_in, int T_out, int dil)
{
  (void)W; (void)b; (void)x; (void)y; (void)Ci; (void)Co; (void)k; (void)stride;
  (void)pad; (void)groups; (void)T_in; (void)T_out; (void)dil;
  return -1;
}

#endif /* JX_HAVE_WQ8 */
