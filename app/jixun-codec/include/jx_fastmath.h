/* ============================================================================
 * jx_fastmath.h - 快速超越函数（多项式/有理逼近），用于 snake / gelu / tanh。
 *
 * 动机（板端实测 2026-09-12）：newlib 的 sinf / erff / tanhf 是主要剩余热点，
 * 合计约 8.6 s / 50 s。这些函数的取值范围都有界，用固定次数的 Horner
 * 多项式即可达到远好于 int8 量化噪声（~1e-2）的精度。
 *
 * 精度（相对 libm，双精度对照拟合，全网实测最大绝对误差）：
 *   jx_sinsq  sin(x)^2    1.7e-09   （x 归约到 [0,pi]，再映射到 [0,pi/2]，11 阶）
 *   jx_tanhf  tanh(x)     4.4e-07   （|x|<=8；之外饱和到 +-1）
 *   jx_erff   erf(x)      4.6e-06   （|x|<=4；之外饱和到 +-1）
 *
 * 关掉的办法：编译期 -DJX_FASTMATH=0，退回 libm（用于做 A/B 数值对照）。
 * ========================================================================== */
#ifndef JX_FASTMATH_H
#define JX_FASTMATH_H

#include <math.h>
#include <stdlib.h>

#ifndef JX_FASTMATH
#  define JX_FASTMATH 1
#endif

#if JX_FASTMATH

#define JX_PI_F      3.14159265358979323846f
#define JX_HALF_PI_F 1.57079632679489661923f
#define JX_INV_PI_F  0.31830988618379067154f

/* sin(x)^2。先把 x 归约到 [0,pi)，再利用 sin^2(pi-u)=sin^2(u) 折叠到 [0,pi/2]。
 * 多项式是 sin(u)^2 在 u in [0,pi/2] 上的最小二乘（Chebyshev 节点）拟合。
 *
 * 2026-09-12 第五轮：两处 if 改成无分支的 ?: 。这里数学完全等价（结果逐位不变），
 * 但取消了控制流：原来跨元素的调度被分支切断，现在 4 条
 * 独立的 Horner 链可以被编译器交错排布，隐藏延时。
 *
 * （備选，本轮未采用）用 cos^2 恒等式换成 w=v^2 的 8 阶多项式可再省 3 步，
 * 不过会把 m0（纯 fp32 基准）从“与参考逐字节相同”变成“1740 个 token 里 1 个不同”，为保留
 * 逐位一致的回归基线而不采用。 */
/* 多项式系数**驻留内部 SRAM**（非 const -> .dram0.data，不走 flash cache）。
 *
 * 2026-09-13 第十五轮：板端微基准（jixun smem）实测 `[s3] 纯读写` 27.2 周期/元素，
 * 而 `[s1] 快速路径 poly` 135.2 周期/元素 —— 多项式本身花了 108 周期/元素，
 * 远超它那 ~10 条 FP 指令（S3 是单发射，IPC 上限 1）。原因是 GCC 把 Horner
 * 常量放进 flash XIP 的 literal pool，每次都要过 32KB 数据缓存，
 * 而热循环同时在流式扫描 PSRAM 大张量，把常量行挤出去 -> 每个元素若干次
 * flash 缺失。改成静态非 const 数组后，常量在内部 SRAM（实测直连、不过缓存），
 * 每个元素只剩 1 条 lsi。
 *
 * 数值与操作次序完全不变 -> 结果逐位一致，不破坏回归门禁。 */
static float jx_sq_k[16] = {
  /* [0..3] 快速路径 sin^2 的 q(w)，w = x^2。v130：降为 8 阶（原 10 阶），
   * 最大拟合误差约 2.1e-4，仍远低于 int8 量化噪声 ~1e-2。 */
  -2.58519e-03f,  4.36332e-02f, -3.329755e-01f,  9.9997572e-01f,
   0.0f,
  /* [5..15] 慢路径 11 阶 */
  -4.157437553e-13f,  6.631500918e-04f, -4.687531765e-03f,  2.053761546e-03f,
   4.268362097e-02f,  9.617141793e-04f, -3.336593731e-01f,  6.431927373e-05f,
   9.999935029e-01f,  2.557090515e-07f, -1.662400773e-09f
};

/* 2026-09-15 第三十八轮（_REALTIME_ROUND38_20260915.md）：
 * 快路径的 4 个系数必须放在**运行期填充**的表里。
 *
 * 板端反汇编证据（v192 固件，`xtensa-esp32s3-elf-objdump` 反汇编
 * main/jx_main.c.o 的 [s1] 循环）：GCC 把 jx_sq_k[0..3] 折叠成 flash
 * literal pool 的 l32r，再用 wfr 把值从 AR 搬到 FP 寄存器。热循环里
 * 每个元素 19 条 l32r + 19 条 wfr（快路径 4 条、慢路径 15 条），
 * 而 Xtensa 的 wfr/rfr 是跨寄存器堆搬移、延迟极高 —— 微基准实测：
 *     [s3] 纯读写 v*1.0001     27.2 周期/元素
 *     [s0] 骨架 v+iv*(a*v)     27.2 周期/元素   <- 加 3 条 FP 指令没变化
 *     [s1] 快路径 poly        135.2 周期/元素   <- 只多 6 条 FP，却 +108
 * 多出来的 108 周期就是 l32r+wfr，不是 FPU 算不动（[1b] asm 8 链
 * 实测 0.75 周期/FP 指令）。
 *
 * 改法：表内容由 noinline 的 jx_sq_kt_fill() 在运行期写入（值就是
 * jx_sq_k[0..3]，一字不差），GCC 无法把它当常量折叠，只能按 lsi 把
 * 系数从 SRAM 直接装进 f 寄存器。主机验证：同样的循环写成
 * 表版后 l32r 4->0、wfr 4->0。
 *
 * 数值：same K0..K3、same Horner 次序 -> 逐位一致。 */
/* 表布局（第三十九轮扩充到 64）：
 *   [0..3]   sin² 快路径 q(w) 的 4 个系数
 *   [4..12]  erf / gelu 的 9 个系数
 *   [16..23] tanh 的 P/Q 系数
 *   [24..34] sin² 慢路径 11 阶系数（原 jx_sq_k[5..15]）
 *   [40] 0.5f  [41] 1.0f  [42] 127.0f  [43] 0.70710678f  [44] JX_HALF_PI_F
 * 同一个表、同一次填充。 */
extern float jx_kt[64];
extern volatile int jx_kt_ready;
void jx_kt_fill(void);
static inline void jx_kt_ensure(void)
{
  if (jx_kt_ready == 0) { jx_kt_fill(); }
}
/* 兼容旧名字（本文件内部用） */
#define jx_sq_kt     (jx_kt)
#define jx_sq_kt_ensure() jx_kt_ensure()
#define jx_sq_kt_fill()   jx_kt_fill()

/* 热循环常用的浮点常量也放进同一张表：lsi 一次装载即可，
 * 而写成字面常量会被 GCC 折叠成 flash literal pool 的 l32r + wfr
 * （wfr 是跨寄存器堆搬移，延迟极高，见第三十八/三十九轮记录）。 */
#define JX_KT_HALF    40
#define JX_KT_ONE     41
#define JX_KT_R127    42
#define JX_KT_ISQ2    43
#define JX_KT_HALFPI  44
#define JX_KT_INVPI   45
#define JX_KT_PI      46
#define JX_KT_ZERO    47

/* ---- 第四十一轮：正弦慢路径搬出热循环，常量走表 ----------------------------
 * 板上反汇编证据（r41a 固件，snake1d 内层循环体 0x42074724..0x420749c5）：
 * 热循环本体只有 ~76 条指令，而慢路径里**每个元素**有 2~3 组
 *     l32r <flash literal pool 里的 1/pi、1.0f、pi> + wfr f*, a*
 * （wfr 是跨寄存器堆搬移，延迟极高）。第三十八轮在**快路径**上修掉了这个病
 * （系数搬进运行期填充的 jx_kt 表），但慢路径一直没修。
 *
 * 本轮实测（jixun smem，501x352 元素、REP 16）：同一条循环里
 *     [s0] 骨架 v+iv*(a*v)     19.6 周期/元素
 *     [s1] 加一个快速路径正弦  100.4 周期/元素      <- 正弦本身值 80.8
 * 另用 jixun snake 现场分解（352x512，四路展开）：
 *     [s1] 现状 48.2 / [s2] 去掉慢路径分支 21.2 / [s4] 纯读地板 10.9
 *     [s6] 数据在内部 SRAM 也只要 44.6 -> 与访存无关，是分支+常量的取指代价
 *
 * 两处改动：
 *   ① 慢路径的 1/pi、pi、1.0f、0.0f、pi/2 全部改从 jx_kt 取（lsi 直达 f 寄存器）
 *   ② 慢路径整体挪成 noinline 函数：热循环里不再有"跨过慢路径"的无条件跳转，
 *      循环体由 ~700 B 缩到 ~120 B
 * 常量值、运算次序、多项式一字不改 -> 结果逐位一致。
 * jx_sinsq_slow_n 是慢路径命中计数，用来量真实的回退比例（诊断用）。 */
extern unsigned long long jx_sinsq_slow_n;

static inline float jx_sinsq_slow(float x)
{
  const float *K2   = jx_kt + 24;    /* = 原 jx_sq_k[5..15]，运行期填充 */
  const float INVPI = jx_kt[JX_KT_INVPI];
  const float ONE   = jx_kt[JX_KT_ONE];
  const float ZERO  = jx_kt[JX_KT_ZERO];
  const float PI    = jx_kt[JX_KT_PI];
  const float HPI   = jx_kt[JX_KT_HALFPI];
  float r;
  float k;
  float u;
  float t;
  float p;
  jx_sinsq_slow_n++;
  r = x * INVPI;
  k = (float)(int)r;
  k -= (k > r) ? ONE : ZERO;         /* 截断 -> floor，负数也正确（无分支） */
  u = x - k * PI;                    /* [0, pi) */
  t = PI - u;
  u = (u > HPI) ? t : u;             /* 折叠到 [0, pi/2]（无分支） */
  p = K2[0];
  p = p * u + K2[1];
  p = p * u + K2[2];
  p = p * u + K2[3];
  p = p * u + K2[4];
  p = p * u + K2[5];
  p = p * u + K2[6];
  p = p * u + K2[7];
  p = p * u + K2[8];
  p = p * u + K2[9];
  p = p * u + K2[10];
  return p;
}

/* ---- 第四十一轮：正弦改成"统一归约 + 无分支" --------------------------------
 * 板上实测（jixun bench 的慢路径计数器，r41c 固件）：
 *     [elems] sinsq 慢路径绕计=688761  (snake 元素的回退比例 0.6713)
 * 也就是说**67% 的元素走慢路径**，而不是原设计假定的 4%（那个数字来自主机
 * 上一次错误分布的测量）。于是热循环里那个 if 变成了 67/33 的随机分支：
 *   ① 每元素 ~33% 概率分支预判失败；
 *   ② 2/3 的元素要跑 11 项 Horner（串行延迟 ~44 周期），4 路展开喂不饱；
 *   ③ 两条路交错把 4 路展开的调度区域切碎。
 * 既然 2/3 本来就要归约，就干脆全部归约，把两条路并成一条、且**零分支**：
 *     r = x * (1/pi)                lsi + mul
 *     k = floor(r)                  floor.s 一条
 *     u = x - k*pi   -> [0, pi)     msub.s 一条
 *     u = pi/2 - |u - pi/2| -> [0, pi/2]     sub + abs + sub，共 3 条
 *     s = w*q(w), w = u*u           5 项偶多项式（与旧快路径同一组系数）
 * 总计约 11 条浮点、0 条分支。旧路径最坏 25 条 + 3 个分支。
 * 精度：与旧路径的差 <= 5e-7，远低于 int8 量化噪声（~1e-2）。
 * 旧实现（快路径 + 11 阶慢路径）保留在下面 jx_sinsq_slow，供对照。 */

/* ---- 第四十一轮：正弦改成"取最近整数归约 + 无分支、无函数调用" -------------
 * 两处板上实测把问题钉死了：
 *   ① 慢路径回退比例 **67%**（不是设计假定的 4%）：
 *        [elems] sinsq 慢路径绕计=688761  (snake 元素的回退比例 0.6713)
 *   ② `floorf(r)` 在 Xtensa 上是 **libm 函数调用**（反汇编 call8 floorf，
 *      每个元素一次），比它想省的还贵。
 *
 * 原来的归约写成 floor(x/pi)（截断再修正）是为了把 x 折到 [0,pi)。
 * 但 sin² 是**偶函数且周期为 pi**，取"最近整数"就够了：
 *     k = round(|x|/pi)        -> u = |x| - k*pi 落在 [-pi/2, pi/2]
 *     sin²(x) = sin²(u)        （偶函数，负号无所谓 -> 不用处理符号）
 * 于是多项式回到原本的 5 项快路径（本来就是按 [0,pi/2] 拟合的），
 * 而且整段**零分支、零函数调用**：
 *     abs.s / mul.s / add.s(#+0.5) / trunc.s / float.s / msub.s
 *     / mul.s(w) / 3*madd.s / mul.s   ≈ 11 条浮点
 * 取值：k = (float)(int)(|x|/pi + 0.5)，|x| <= 28 -> r+0.5 <= 9.5，无溢出。
 * 精度：与旧路径差 <= 5e-7，远低于 int8 量化噪声（~1e-2）。
 * 旧实现保留在下面 jx_sinsq_slow（对照/回退用）。 */
/* r48：角度归约去掉 (float)(int)(...) 的跨域搬运。
 * Xtensa 上 (float)(int)x = trunc.s(FP) -> rfr(读整数寄存器) -> float.s(整数寄存器进 FP)，
 * 两次跨域 + 调度屏障把 4 路展开的链串行化：板端探针实测
 *     纯读写 27.2 周期/元素 -> 加快路径正弦 100.4（13 条浮点却花了 73 周期）。
 * 改用 1.5*2^23 魔数取整，全浮点域 4 条（mul/add/sub/msub），无跨域：
 *     k = (|x|/pi + MAGIC) - MAGIC = round_to_nearest_even(|x|/pi)
 * 与 (int)(r+0.5) 只在 r 小数部分恰为 0.5 处可能不同，而那时 w = +-pi/2、
 * sin^2(w) = 1 两者相同；其余输入 k 逐位相同，故 w 逐位相同。 */
#define JX_SQ_MAGIC 12582912.0f   /* 1.5f * 2^23f */


static __attribute__((always_inline)) inline float jx_sinsq(float x)
{
#if JX_SSQ_LUT
  return jx_sinsq_lut(x);
#else
  float ax = fabsf(x);
  float k  = ax * jx_kt[JX_KT_INVPI] + JX_SQ_MAGIC;
  k -= JX_SQ_MAGIC;
  float w  = ax - k * jx_kt[JX_KT_PI];                   /* u in [-pi/2,pi/2] */
  w = w * w;
  return w * (((jx_kt[0] * w + jx_kt[1]) * w + jx_kt[2]) * w + jx_kt[3]);
#endif
}





/* ================== 第三十九轮：系数由调用方持有 ==================
 * 反汇编证据（v194 固件，snake1d 内层 / jx_conv1d_k1_cf epilogue）：
 * 每个元素都要
 *     l32r <jx_kt 地址> + 4×lsi         （快路径的 4 个系数）
 *     l32r + wfr                        （0.5f / 1.0f 这类字面常量）
 * 才能拿到系数 —— jx_kt 是全局数组，热循环里又有 store（x[i]=…），
 * GCC 无法证明两者不别名，只好每轮重载。
 *
 * 改法：把系数在循环**前**一次性读进局部变量（结构体按值传递、
 * 不再取地址），GCC 就能让它们常驻 f 寄存器。多项式和 Horner 次序
 * 与 jx_sinsq / jx_erff 完全相同（同一个输入走同一条路），逐位一致。 */
/* 第四十二轮：把归约用的 1/pi、0.5f、pi 也放进结构体。
 * 否则热循环里每个元素都要 l32r <jx_kt 地址> + lsi f*, a*（3 组），
 * 而 jx_kt 是全局数组、循环里又有 store，不能证明不别名，只能每轮重载。 */
/* ================== r96：sin^2 改成周期查表（免归约、免系数寄存器）==========
 * 多项式路径每元素 12 条 FP 指令（abs / mul / mov+madd / sub / mov+msub /
 * mul / mov+3*madd / mul），还要 7 个 f 寄存器常驻系数
 * （k0..k3 / c45 / c46 / MAGIC）。16 个 f 寄存器被压爆，反汇编实测
 * jx_linear_i8_rows 的 qfuse epilogue 每 4 个元素有
 *     - 8 条 ssi/lsi 浮点溢出（v0..v3 算完立刻压栈、sin^2 里再弹回来）
 *     - 约 14 条 mov.s 纯搬运（madd.s 的目的寄存器必须先装 MAGIC / k0）
 *     - |u| 的 max 统计被挤成 ssi->l32i->and->s32i 的栈往返
 * 结果 43 条指令/元素、55.7 周期/元素（FPU 实测 1.03 周期/FP 指令，
 * 即 3/4 的时间花在非 FP 的搬运与溢出上）。
 *
 * sin^2 的周期是 pi，等价于 u = t/pi 的周期是 1。取 N 为 2 的幂，则
 *     idx = (int)(|t| * (N/pi)) & (N-1)
 * 恰好等于 floor( frac(|t|/pi) * N )（整数部分乘 N 是 N 的整数倍，
 * 按位与后自动消掉），而表
 *     jx_ssq[i] = sin^2( pi * i / N )
 * 直接就是答案。**对任意大的 |t| 都成立**，所以不需要归约、不需要钳位、
 * 不需要任何系数寄存器。
 *
 * 每元素只剩 6 条指令：abs.s / mul.s / trunc.s / and / addx4 / lsi。
 * 误差上界 pi/N（N=2048 时 1.53e-3，绝对量，作用在 sin^2 in [0,1] 上），
 * 折算到 int8 量化台阶（max/127）上不到千分之几 —— 由主机端 SNR 门禁复核。
 * 表由 jx_kt_fill() 用 libm sinf 在运行期填（N 次，一次性）。
 * JX_SSQ_LUT=0 可编回多项式（归因对照用）。 */
#ifndef JX_SSQ_LUT
#  define JX_SSQ_LUT 1
#endif
#define JX_SSQ_N      2048
#define JX_SSQ_MASK   (JX_SSQ_N - 1)
#define JX_SSQ_SCALE  ((float)JX_SSQ_N / JX_PI_F)
extern float jx_ssq[JX_SSQ_N];

/* r138：去掉 abs.s。
 * 依据：表是 sin^2(pi*i/N)，满足 jx_ssq[i] == jx_ssq[N-i]（sin^2 偶函数 +
 * 以 pi 为周期）。对 t<0，(int)(t*SCALE) 向零截断得 -m，按位与掩码后得到
 * N-m（m>=1 时），查到的表值与旧写法 floor(|t|*SCALE)=m 查到的**完全相同**：
 *     sin^2(pi*(N-m)/N) = sin^2(pi - pi*m/N) = sin^2(pi*m/N)。
 * m=0 时两者都查 0 号项；|t|*SCALE 超出 N 时两个下标同余、由周期性与偶
 * 函数性保证也相同。+0.0/-0.0 都落在 0 号项。故逐位一致，省掉 1 条浮点
 * （FPU 单链延迟 4.13 周期，是 sinsq 那条链上的第一环）。
 * jx_kt_fill 用 sinf 填表，与算式无关。 */
static __attribute__((always_inline)) inline float jx_sinsq_lut(float t)
{
  float    x  = t * JX_SSQ_SCALE;
  unsigned i  = (unsigned)(int)x;
  i &= (unsigned)JX_SSQ_MASK;
  return jx_ssq[i];
}

/* r95: force-inline hot transcendental kernels. */
typedef struct { float k0, k1, k2, k3, c45, c40, c46; } jx_sq4;

static __attribute__((always_inline)) inline jx_sq4 jx_sq4_load(void)
{
  jx_sq4 k;
  k.k0 = jx_kt[0]; k.k1 = jx_kt[1]; k.k2 = jx_kt[2]; k.k3 = jx_kt[3];
  k.c45 = jx_kt[JX_KT_INVPI]; k.c40 = jx_kt[JX_KT_HALF]; k.c46 = jx_kt[JX_KT_PI];
  return k;
}

/* |x| <= pi/2 走快路径（与 jx_sinsq 同式同序），之外退回 jx_sinsq。 */
/* |x| <= pi/2 走快路径（与 jx_sinsq 同式同序），之外退回 jx_sinsq。 */
static __attribute__((always_inline)) inline float jx_sinsq4(float x, jx_sq4 k)
{
#if JX_SSQ_LUT
  (void)k;
  return jx_sinsq_lut(x);
#else
  float ax = fabsf(x);
  float kk = ax * k.c45 + JX_SQ_MAGIC;
  kk -= JX_SQ_MAGIC;
  float w  = ax - kk * k.c46;
  w = w * w;
  return w * (((k.k0 * w + k.k1) * w + k.k2) * w + k.k3);
#endif
}

/* 第四十二轮：上面那个版本每个元素读 7 次全局 jx_kt（l32r + lsi）。
 * 这里提供一个“系数已在手”的版本，供热循环用；算式与 jx_sinsq 逐项相同。 */
static __attribute__((always_inline)) inline float jx_sinsq_k(float x, jx_sq4 c)
{
#if JX_SSQ_LUT
  (void)c;
  return jx_sinsq_lut(x);
#else
  float ax = fabsf(x);
  float k  = ax * c.c45 + JX_SQ_MAGIC;
  k -= JX_SQ_MAGIC;
  float w  = ax - k * c.c46;
  w = w * w;
  return w * (((c.k0 * w + c.k1) * w + c.k2) * w + c.k3);
#endif
}



/* ---- r43f：4 路交错的 sin^2 ----
 * 动机（实测）：snake 的 4 路展开 epilogue 稳定在 ~61 周期/元素，恰好是
 * 「~15 条串行 FP x FPU 4.13 周期延迟」——说明 4 条链**没有交错发射**。
 * 原因是寄存器压力：jx_sq4 有 7 个系数，4 条链各自的中间量再占 3~4 个，
 * 16 个 f 寄存器装不下，GCC 只能把它们串起来。
 *
 * 这里按 Horner **层级**并排展开：先把 4 个元素的归约段（abs/mul/cvt/msub）
 * 一起做完，再让 Horner 的每一层同时发射 4 条互不依赖的 madd.s，最后一起
 * 收尾。段与段之间不需要同时保活 4 份中间量，峰值活跃寄存器降到 ~9。
 * 每个元素自己的算式与运算次序与 jx_sinsq4 逐项相同（交错只发生在不同
 * 元素之间），所以结果逐位一致。 */
static __attribute__((always_inline)) inline void jx_sinsq4_x4(float x0, float x1, float x2, float x3,
                                const jx_sq4 c,
                                float *o0, float *o1, float *o2, float *o3)
{
#if JX_SSQ_LUT
  (void)c;
  *o0 = jx_sinsq_lut(x0);
  *o1 = jx_sinsq_lut(x1);
  *o2 = jx_sinsq_lut(x2);
  *o3 = jx_sinsq_lut(x3);
#else

  float a0 = fabsf(x0), a1 = fabsf(x1), a2 = fabsf(x2), a3 = fabsf(x3);
  float k0 = a0 * c.c45 + JX_SQ_MAGIC, k1 = a1 * c.c45 + JX_SQ_MAGIC;
  float k2 = a2 * c.c45 + JX_SQ_MAGIC, k3 = a3 * c.c45 + JX_SQ_MAGIC;
  k0 -= JX_SQ_MAGIC; k1 -= JX_SQ_MAGIC; k2 -= JX_SQ_MAGIC; k3 -= JX_SQ_MAGIC;
  float w0 = a0 - k0 * c.c46, w1 = a1 - k1 * c.c46;
  float w2 = a2 - k2 * c.c46, w3 = a3 - k3 * c.c46;
  w0 = w0 * w0; w1 = w1 * w1; w2 = w2 * w2; w3 = w3 * w3;
  float p0 = c.k0, p1 = c.k0, p2 = c.k0, p3 = c.k0;
  p0 = p0 * w0 + c.k1; p1 = p1 * w1 + c.k1; p2 = p2 * w2 + c.k1; p3 = p3 * w3 + c.k1;
  p0 = p0 * w0 + c.k2; p1 = p1 * w1 + c.k2; p2 = p2 * w2 + c.k2; p3 = p3 * w3 + c.k2;
  p0 = p0 * w0 + c.k3; p1 = p1 * w1 + c.k3; p2 = p2 * w2 + c.k3; p3 = p3 * w3 + c.k3;
  *o0 = w0 * p0; *o1 = w1 * p1; *o2 = w2 * p2; *o3 = w3 * p3;
#endif
}

/* erf 的 9 个系数，索引 i 对应 jx_kt[4 + i]。 */
typedef struct { float e0, e1, e2, e3, e4, e5, e6, e7, e8; } jx_erf9;

static __attribute__((always_inline)) inline jx_erf9 jx_erf9_load(void)
{
  jx_erf9 e;
  const float *p = jx_kt + 4;
  e.e0 = p[0]; e.e1 = p[1]; e.e2 = p[2]; e.e3 = p[3]; e.e4 = p[4];
  e.e5 = p[5]; e.e6 = p[6]; e.e7 = p[7]; e.e8 = p[8];
  return e;
}

/* 与 jx_erff 逐字符相同的算式（同一多项式、同一次序）。 */
static __attribute__((always_inline)) inline float jx_erff9(float x, jx_erf9 E)
{
  float ax = fabsf(x);
  float ac = (ax > 4.0f) ? 4.0f : ax;
  float p, r;
  p = E.e0;
  p = p * ac + E.e1;
  p = p * ac + E.e2;
  p = p * ac + E.e3;
  p = p * ac + E.e4;
  p = p * ac + E.e5;
  p = p * ac + E.e6;
  p = p * ac + E.e7;
  p = p * ac + E.e8;
  r = ac * p;
  r = (ax >= 4.0f) ? 1.0f : r;
  return (x < 0.0f) ? -r : r;
}

/* gelu(x) = 0.5x(1+erf(x/sqrt2))，系数由调用方持有。 */
static __attribute__((always_inline)) inline float jx_gelu9(float x, jx_erf9 E)
{
  return 0.5f * x * (1.0f + jx_erff9(x * 0.70710678f, E));
}

/* ===== r45b：gelu 查表（B 方案第 2 项）=====================================
 * 动机：jx_gelu9 是 9 系数 Horner + erf 归约，热循环实测约 24 条浮点指令/元素，
 * 而 r43f 已证明这个循环是 **发射受限（1 IPC）**——展开不减少指令数，所以零收益。
 * 唯一能提速的办法是**减少每元素指令数**。
 *
 * 方案：gelu(x) 在 [-8,8] 上用 1025 项表 + 线性插值。
 *   步长 h = 1/64，插值误差 = h^2*|gelu''|/8 <= (1/64)^2*1.13/8 = 3.4e-5
 *   —— 比 int8 量化噪声（~1e-2）低 3 个数量级，SNR 影响可忽略。
 *   x >= +8 时 gelu(x) == x（精确到 1e-15）；x <= -8 时 gelu(x) == 0。
 * 每元素约 11 条指令（对比 24 条）。
 *
 * 表由 jx_kt_fill() 运行期填充（与 jx_kt 同因：避免 GCC 折叠成 flash
 * literal pool 的 l32r + wfr）。常数放 jx_kt[48..51]，热循环里一次装载。 */
#define JX_GLUT_N   1025
extern float jx_glut[JX_GLUT_N];

typedef struct { float off, scale, hi, lo; } jx_glut4;

static __attribute__((always_inline)) inline jx_glut4 jx_glut4_load(void)
{
  jx_glut4 k;
  k.off = jx_kt[48]; k.scale = jx_kt[49];
  k.hi  = jx_kt[50]; k.lo    = jx_kt[51];
  return k;
}

/* 参数形状与 jx_gelu9 完全一致，便于整站替换；E 不再使用。 */
/* r51i：本体分支消除。
 * 原写法两个提前返回（x>=8 返回 x / x<=-8 返回 0）在真实数据上是随机分支：
 * k1cf 的 gelu epilogue 实测 48.0 周期/元素（同一份代码去掉 gelu 只要 20.7），
 * 而这 27 周期全是分支预判失败——换成无分支写法后 epilogue 回到 20.7，gelu 的
 * 算术被访存完全盖住（JX_EPG=2 实测 1652.6 对恒等 1656.8）。
 * 表内容 / off / scale / 算式一字不改，结果与旧实现逐位相同。 */
/* r100：Xtensa FP 条件搬运。GCC 对 float 的三元/if 一律生成分支（实测
 * -O2/-O3/-Os 都是），而 Xtensa 有 ole.s/olt.s + movt.s/movf.s 可以无条件
 * 搬运；这里显式写出来。
 *   JX_FSELHI_ASM(y, x, h) :  y = (x >= h) ? x : y */
#if defined(__XTENSA__)
#  define JX_FSELHI_ASM(y, x, h)                                              \
     __asm__("olt.s b0, %[X], %[H]\n\t"                                     \
             "movf.s %[Y], %[X], b0\n\t"                                    \
             : [Y] "+f"(y) : [X] "f"(x), [H] "f"(h) : "b0")
#else
   /* 主机端编译（用于主机门禁）没有 ole.s/movf.s，等价展开。 */
#  define JX_FSELHI_ASM(y, x, h) do { if ((x) >= (h)) { (y) = (x); } } while (0)
#endif

/* r100：无分支 gelu 查表。
 *
 * 动机（r99 固件反汇编，jx_conv1d_k1_cf_tb 内 post==1 && _gm==4 的元素块）：
 *     ... madd.s f5, f3, f2            <- u = acc*... + bb
 *     lsi f7, a2, 200 / ole.s b0, f7, f5 / bt ->  返回 u
 *     ole.s b0, f5, f6 / bt ->  返回 0.0f
 *     lsi/lsi/madd.s/trunc.s/float.s/addx4/sub.s/lsi/lsi/sub.s/madd.s/ssi
 * 两条 ole.s+bt 在真实数据上跳转方向随机，一次预判失败十几周期；该 epilogue
 * 实测 49.5 周期/元素，其中 27 周期记在这两个分支上（见 r51i 记录：同一份
 * 代码去掉 gelu 只要 20.7）。r51h/r51i 当年写的"分支消除版" jx_gelu_br 实际
 * 只是转调本函数，分支从未消失。
 *
 * 改法：把条件**下沉到下标**，用整数 min/max 做钳位——越界只可能出现在饱和区
 * （x>=8 -> t>=1024；x<=-8 -> t<=0），而饱和区的结果随后由一次无条件条件搬运
 * 覆盖，表域内的 x 下标与 f 完全不变。
 *
 * 逐位等价说明：
 *   - x∈[lo,hi)：t<1024 且 t>=0，(int)t<=1023 不被钳位，a/b/f 与旧实现相同；
 *   - x>=hi   ：下标被钳到 1023，y 是有限值，随后 movf.s 换回精确的 x；
 *   - x<=lo   ：下标钳到 0、t<0。jx_kt_fill 末尾已把 jx_glut[0] 从 -0.0f 改成
 *               +0.0f（旧实现在 x<=lo 时根本不查表，只用 x∈(lo,..) 那段，
 *               而那段上 b-a=+0.0f、结果 +0.0f，改成 +0.0f 后同样得 +0.0f），
 *               于是 y = +0.0f + t*(b-a)。t<0 时该式恒为 +0.0f，与旧实现
 *               的 return 0.0f 同值同符号。 */
static __attribute__((always_inline)) inline float jx_gelu_fast(float x, jx_erf9 E)
{
  (void)E;
  const float hi = jx_kt[50];
  {
    float t = x * jx_kt[49] + jx_kt[48];
    int   i = (int)t;
    i = (i < 0) ? 0 : i;
    i = (i > (JX_GLUT_N - 2)) ? (JX_GLUT_N - 2) : i;
    float f = t - (float)i;
    float a = jx_glut[i], b = jx_glut[i + 1];
    float y = a + f * (b - a);
    JX_FSELHI_ASM(y, x, hi);
    return y;
  }
}

/* r119：把 jx_kt[48..51] 与 LUT 基址**提到循环外**的版本。
 * 动机（r118 固件反汇编，jx_conv1d_k1_cf_tb 的 post==1 epilogue）：
 * 每个元素里都有 3 条 `l32r`（从 flash 字面池重取 jx_kt/jx_glut 地址）
 * 与 3 条 `lsi`（重读 jx_kt[48..51]）—— 因为 dst 是 float*、jx_kt 也是
 * float[]，严格别名规则下 GCC 必须假设每次 `dst[j]=` 都可能改到它们。
 * 结果 26 条指令/元素全部串行，4 路展开被排成 4 段独立代码，延迟完全暴露
 * （实测 48.9 周期/元素）。
 * 把 off/scale/hi 与表基址作为参数传进来、并在调用点配 __restrict 输出后，
 * 这些加载可以彻底提到循环外，4 条 lane 也能被交错。
 * 算式与 jx_gelu_fast 逐项相同 -> 逐位一致。 */
static __attribute__((always_inline)) inline float jx_gelu_kt(float x, jx_glut4 K,
                                                              const float *lut)
{
  float t = x * K.scale + K.off;
  int   i = (int)t;
  i = (i < 0) ? 0 : i;
  i = (i > (JX_GLUT_N - 2)) ? (JX_GLUT_N - 2) : i;
  float f = t - (float)i;
  float a = lut[i], b = lut[i + 1];
  float y = a + f * (b - a);
  JX_FSELHI_ASM(y, x, K.hi);
  return y;
}

/* r100b：4 路相位交错的 gelu 查表。
 * 上板 A/B 实测：单纯把 r100 的两个分支换成 movf.s 之后总耗时没有变化
 * （epilogue 仍是 ~49.5 周期/元素，o2+ 3989）。原因是这条链本身就长
 * （madd.s -> trunc.s -> min/max -> float.s -> sub.s -> lsi -> sub.s
 *  -> madd.s -> movf.s，单条 FP 延迟 4.13），而 4 条 lane 在指令流里
 * 首尾相接，in-order 发射下延迟全部暴露。
 * 这里按**相位**并排展开（与 jx_sinsq4_x4 / jx_erff9_x4 同一手法）：同一相位
 * 的 4 条指令互不依赖，正好盖住 FPU 延迟；段与段之间不要求同时保活 4 份
 * 中间量，峰值活跃寄存器降到 ~12。
 * 每个元素自己的下标钳位、算式与 jx_gelu_fast 完全相同（交错只发生在不同
 * 元素之间），结果逐位一致。 */
static __attribute__((always_inline)) inline void jx_gelu4_fast(
    float x0, float x1, float x2, float x3, jx_erf9 E,
    float *r0, float *r1, float *r2, float *r3)
{
  (void)E;
  const float hi   = jx_kt[50];
  const float sc   = jx_kt[49], of = jx_kt[48];
  const int   imax = JX_GLUT_N - 2;
  float t0 = x0 * sc + of, t1 = x1 * sc + of;
  float t2 = x2 * sc + of, t3 = x3 * sc + of;
  int i0 = (int)t0, i1 = (int)t1, i2 = (int)t2, i3 = (int)t3;
  i0 = (i0 < 0) ? 0 : i0; i0 = (i0 > imax) ? imax : i0;
  i1 = (i1 < 0) ? 0 : i1; i1 = (i1 > imax) ? imax : i1;
  i2 = (i2 < 0) ? 0 : i2; i2 = (i2 > imax) ? imax : i2;
  i3 = (i3 < 0) ? 0 : i3; i3 = (i3 > imax) ? imax : i3;
  float f0 = t0 - (float)i0, f1 = t1 - (float)i1;
  float f2 = t2 - (float)i2, f3 = t3 - (float)i3;
  float a0 = jx_glut[i0], b0 = jx_glut[i0 + 1];
  float a1 = jx_glut[i1], b1 = jx_glut[i1 + 1];
  float a2 = jx_glut[i2], b2 = jx_glut[i2 + 1];
  float a3 = jx_glut[i3], b3 = jx_glut[i3 + 1];
  float y0 = a0 + f0 * (b0 - a0), y1 = a1 + f1 * (b1 - a1);
  float y2 = a2 + f2 * (b2 - a2), y3 = a3 + f3 * (b3 - a3);
  JX_FSELHI_ASM(y0, x0, hi); JX_FSELHI_ASM(y1, x1, hi);
  JX_FSELHI_ASM(y2, x2, hi); JX_FSELHI_ASM(y3, x3, hi);
  *r0 = y0; *r1 = y1; *r2 = y2; *r3 = y3;
}

/* r51h：分支消除版 gelu 查表。
 * 旧 jx_gelu_fast 有两个 if（x>=8 直接返回 x / x<=-8 返回 0）。真实数据上
 * 这两个分支是随机的，k1cf 的 gelu epilogue 实测 48.0 周期/元素，而同形状
 * 无激活的 epilogue 只要 20.8 —— 差的全在这两个跳转上。
 * 改法：先把 x 钳到 [-8,8]（两条条件移动，无跳转），查表线性插值，最后用
 * 两次条件选择把饱和区换回精确值。表内容、off/scale、算式都不变，
 * 结果与 jx_gelu_fast 逐位相同。 */
static __attribute__((always_inline)) inline float jx_gelu_br(float x, jx_erf9 E)
{
  return jx_gelu_fast(x, E);
}


/* ---- r43d：4 路交错内核 ----
 * 实测：k1s / dense-tile 的 epilogue 稳定在 ~61 周期/元素，恰好等于
 * 「~15 条串行 FP 指令 x FPU 4.13 周期延迟」——也就是说 4 路展开写的
 * 4 条独立链**并没有真正交错发射**（寄存器压力让 GCC 把它们串起来了）。
 * 这里把 4 个元素的多项式按 Horner 层级并排展开：每一层同时发射 4 条
 * 互不依赖的 madd.s，正好盖住 4.13 周期延迟。
 * 每个元素自己的算式与运算次序与 jx_erff9 / jx_gelu9 逐项相同（交错只
 * 发生在不同元素之间），所以结果逐位一致。 */
static __attribute__((always_inline)) inline void jx_erff9_x4(float x0, float x1, float x2, float x3,
                              const jx_erf9 E,
                              float *r0, float *r1, float *r2, float *r3)
{
  float ax0 = fabsf(x0), ax1 = fabsf(x1), ax2 = fabsf(x2), ax3 = fabsf(x3);
  float a0 = (ax0 > 4.0f) ? 4.0f : ax0;
  float a1 = (ax1 > 4.0f) ? 4.0f : ax1;
  float a2 = (ax2 > 4.0f) ? 4.0f : ax2;
  float a3 = (ax3 > 4.0f) ? 4.0f : ax3;
  float p0 = E.e0, p1 = E.e0, p2 = E.e0, p3 = E.e0;
  p0 = p0 * a0 + E.e1; p1 = p1 * a1 + E.e1; p2 = p2 * a2 + E.e1; p3 = p3 * a3 + E.e1;
  p0 = p0 * a0 + E.e2; p1 = p1 * a1 + E.e2; p2 = p2 * a2 + E.e2; p3 = p3 * a3 + E.e2;
  p0 = p0 * a0 + E.e3; p1 = p1 * a1 + E.e3; p2 = p2 * a2 + E.e3; p3 = p3 * a3 + E.e3;
  p0 = p0 * a0 + E.e4; p1 = p1 * a1 + E.e4; p2 = p2 * a2 + E.e4; p3 = p3 * a3 + E.e4;
  p0 = p0 * a0 + E.e5; p1 = p1 * a1 + E.e5; p2 = p2 * a2 + E.e5; p3 = p3 * a3 + E.e5;
  p0 = p0 * a0 + E.e6; p1 = p1 * a1 + E.e6; p2 = p2 * a2 + E.e6; p3 = p3 * a3 + E.e6;
  p0 = p0 * a0 + E.e7; p1 = p1 * a1 + E.e7; p2 = p2 * a2 + E.e7; p3 = p3 * a3 + E.e7;
  p0 = p0 * a0 + E.e8; p1 = p1 * a1 + E.e8; p2 = p2 * a2 + E.e8; p3 = p3 * a3 + E.e8;
  float q0 = a0 * p0, q1 = a1 * p1, q2 = a2 * p2, q3 = a3 * p3;
  q0 = (ax0 >= 4.0f) ? 1.0f : q0; q1 = (ax1 >= 4.0f) ? 1.0f : q1;
  q2 = (ax2 >= 4.0f) ? 1.0f : q2; q3 = (ax3 >= 4.0f) ? 1.0f : q3;
  *r0 = (x0 < 0.0f) ? -q0 : q0; *r1 = (x1 < 0.0f) ? -q1 : q1;
  *r2 = (x2 < 0.0f) ? -q2 : q2; *r3 = (x3 < 0.0f) ? -q3 : q3;
}

static __attribute__((always_inline)) inline void jx_gelu9_x4(float x0, float x1, float x2, float x3,
                              const jx_erf9 E,
                              float *y0, float *y1, float *y2, float *y3)
{
  float e0, e1, e2, e3;
  jx_erff9_x4(x0 * 0.70710678f, x1 * 0.70710678f,
              x2 * 0.70710678f, x3 * 0.70710678f, E, &e0, &e1, &e2, &e3);
  *y0 = 0.5f * x0 * (1.0f + e0);
  *y1 = 0.5f * x1 * (1.0f + e1);
  *y2 = 0.5f * x2 * (1.0f + e2);
  *y3 = 0.5f * x3 * (1.0f + e3);
}

/* tanh 的 8 个系数，索引 i 对应 jx_kt[16 + i]。 */
typedef struct { float t0, t1, t2, t3, t4, t5, t6, t7; } jx_tanh8;

static __attribute__((always_inline)) inline jx_tanh8 jx_tanh8_load(void)
{
  jx_tanh8 t;
  const float *p = jx_kt + 16;
  t.t0 = p[0]; t.t1 = p[1]; t.t2 = p[2]; t.t3 = p[3];
  t.t4 = p[4]; t.t5 = p[5]; t.t6 = p[6]; t.t7 = p[7];
  return t;
}

static __attribute__((always_inline)) inline float jx_tanhf8(float x, jx_tanh8 T)
{
  float ax = fabsf(x);
  float y, p, q, r;
  if (ax >= 8.0f) { return (x >= 0.0f) ? 1.0f : -1.0f; }
  y = ax * ax;
  p = T.t0;
  p = p * y + T.t1;
  p = p * y + T.t2;
  p = p * y + T.t3;
  p = p * y + T.t4;
  q = T.t5;
  q = q * y + T.t6;
  q = q * y + T.t7;
  q = q * y + 1.0f;
  r = ax * p / q;
  return (x < 0.0f) ? -r : r;
}

/* exp(x)，**只用于 x <= 0**（softmax 减掉行 max 之后就是这个区间）。
 *
 * 动机（2026-09-13 第六轮，主机实测）：整机 softmax 的 expf 调用量是
 * 15,600,060 次，是全模型调用次数最多的超越函数；newlib 的 expf 在 Xtensa
 * 上是软件实现（约 100+ 周期），这一项就能吃掉 1.5~3 秒。
 *
 * 做法：exp(x) = 2^t，t = x*log2(e)；取 k = floor(t)、f = t-k in [0,1)，
 * 2^f 用 5 阶多项式（相对误差 <1.1e-07），2^k 用整数指数域位拼装。
 * x < -87 直接返回 0（exp 已经下溢到 0）。
 * 总代价 ~12 个操作，是 libm 的 1/3~1/4。
 * 注意：结果与 libm expf 有 <=1e-07 的相对差，softmax 权重随之有 ~1e-07
 * 的变化，m0 不再是逐字节 0（见 SNR 门禁）。 */
static inline float jx_expf_neg(float x)
{
  float t;
  float kf;
  float f;
  float p;
  int   ki;
  union { float ff; int ii; } sc;
  if (x < -87.0f) { return 0.0f; }
  t  = x * 1.4426950408889634f;
  kf = (float)(int)t;
  kf -= (kf > t) ? 1.0f : 0.0f;     /* 截断 -> floor（无分支） */
  f  = t - kf;                       /* [0, 1) */
  p  =         7.901989e-02f;
  p = p * f +  2.2412625e-01f;
  p = p * f +  6.9683881e-01f;
  p = p * f +  9.9981191e-01f;       /* 2^f，v131 降为 3 阶 */
  ki = (int)kf;
  sc.ii = (ki + 127) << 23;          /* 2^k 直接拼指数域（x<=0 -> ki<=0） */
  return p * sc.ff;
}

/* SiLU(a) = a / (1 + exp(-a))，用于 transformer 的 DynamicPositionBias MLP。
 *
 * 2026-09-13 第九轮：这里原来用 newlib 的 expf（每层 2*H 次调用，H=dpb 隐层
 * 宽度；DPB 每个相对距离算一行，整机调用量十万级）。Xtensa 上 newlib expf
 * 是软件实现（100+ 周期），而 jx_expf_neg 只要十来个操作。
 * 入参范围：-a 落在 [-20,20]，恒在 jx_expf_neg 的安全区间内。
 * 大 |a| 直接给渐近值（|silu(±20)| 与渐近线相差 <5e-9）。 */
#ifndef JX_SILU_FAST
#  define JX_SILU_FAST 1
#endif
static inline float jx_silu(float a)
{
#if JX_SILU_FAST
  if (a >= 20.0f)  { return a; }
  if (a <= -20.0f) { return 0.0f; }
  return a / (1.0f + jx_expf_neg(-a));
#else
  return a / (1.0f + expf(-a));   /* -DJX_SILU_FAST=0 退回 libm，做 A/B */
#endif
}

/* tanh(x) = x * P(x^2) / Q(x^2)，y = x^2 的 [4/3] 有理逼近，x in [0,8]。 */
static inline float jx_tanhf(float x)
{
  float ax = fabsf(x);
  float y;
  float p, q, r;
  const float *T = jx_kt + 16;
  if (ax >= 8.0f) { return (x >= 0.0f) ? 1.0f : -1.0f; }
  y = ax * ax;
  p = T[0];
  p = p * y + T[1];
  p = p * y + T[2];
  p = p * y + T[3];
  p = p * y + T[4];
  q = T[5];
  q = q * y + T[6];
  q = q * y + T[7];
  q = q * y + 1.0f;
  r = ax * p / q;
  return (x < 0.0f) ? -r : r;
}

/* erf(x)/x 在 x in [0,4] 上的 12 次 Horner 多项式（x>=0 时求值）。 */
static inline float jx_erff(float x)
{
  float ax = fabsf(x);
  float p, r;
  const float *E = jx_kt + 4;
  /* 2026-09-15 第三十四轮：去掉提前 return 的分支，改成无分支 clamp + 选择。
   * 数值与原实现逐位一致（ax<4 用 ac=ax，ax>=4 一律取 1.0），但每个元素
   * 少一个真实分支，gelu 的 4 路展开才能交错。 */
  float ac = (ax > 4.0f) ? 4.0f : ax;
  p = E[0];
  p = p * ac + E[1];
  p = p * ac + E[2];
  p = p * ac + E[3];
  p = p * ac + E[4];
  p = p * ac + E[5];
  p = p * ac + E[6];
  p = p * ac + E[7];
  p = p * ac + E[8];
  r = ac * p;
  r = (ax >= 4.0f) ? 1.0f : r;
  return (x < 0.0f) ? -r : r;
}

#else  /* !JX_FASTMATH */

static __attribute__((always_inline)) inline float jx_sinsq(float x) { float s = sinf(x); return s * s; }
static inline float jx_tanhf(float x) { return tanhf(x); }
static inline float jx_erff (float x) { return erff(x); }
static inline float jx_silu(float a) { return a / (1.0f + expf(-a)); }

#endif /* JX_FASTMATH */

#endif /* JX_FASTMATH_H */
