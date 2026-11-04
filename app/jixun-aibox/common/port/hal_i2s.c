/****************************************************************************
 * hal_i2s.c — 2026-08-25 重构：从"手写寄存器直写"迁移为
 * NuttX 官方 esp32s3_i2s 驱动的封装层（保持 hal_i2s.h API 不变）。
 *
 * 背景（为什么迁移）：
 *   手写寄存器版把驱动框架自动处理的几件事做错/丢掉：
 *     1) RX_RESET 清掉时钟域帧配置 → RX 48k 错位噪声（"说话不说话一样"）
 *     2) DMA 完成信号用错（I2S TX_DONE=FIFO 空 假完成）→ 播放截断"都的一声"
 *     3) GDMA OUT/IN 共用 CPU 中断线 → 中断风暴卡死
 *     4) 神经网络级降噪缺失 → 手工滤波对付不了 ES7210 宽带底噪
 *   官方驱动（esp32s3_i2s.c）完整实现：时钟自动分频、DMA/EOF 中断、
 *   全双工 SIG_LOOPBACK、配置重放，与 IDF 例程同架构。
 *
 * ⚠️ 2026-09-02 设计定案（最省内存，勿改回流水线/双缓冲/环形队列）：
 *   曾试验"RX 流水线（2 apb 在途仿 TX 连续传输）"——引入复杂度与
 *   新 bug（预提交/补提交区间重叠、死等、回调竞态），且实测录音
 *   提前退出。根因复盘：diag 说话段"非零率恒定 69%"并非 RX 丢数据，
 *   而是 ai_voice.c 的 s_out[1600] 越界（每 4 块写 [1422..1896) 超 1600）
 *   写穿相邻 s_chunk48 + 回调越界读的**统计污染**。修复越界后回退到
 *   官方驱动原生"单 apb 串行"即够——零额外内存、零并发复杂度。
 *   前提：hal_i2s_read_slot 单次调用内部循环收满（块间处理 < I2S RX
 *   FIFO 深度（64B≈0.67ms）不溢出；验证见 diag 非零率 ≥95%）。
 *
 * ⚠️ 2026-09-02 TX/RX 切换（三策略融合定案，修 plant voice loop 失败）：
 *   【现象】录音（预读链在跑）后播放：第一块 hal_i2s_write 就失败，
 *   打印 hal_i2s_write=-5。旧修复只置 s_rx_prefetch_paused 标志——
 *   在途 RX apb（≤4092B≈43ms@24k 立体声）仍占着 RX DMA，TX 与 RX
 *   并发启动即失败（官方驱动单次 DMA 语义下的实测行为）。
 *   【融合】①小智策略：严格的通道生命周期——先停 RX 再发 TX
 *   （等价 i2s_channel_disable(rx) → write(tx)）；②OpenVela 策略：
 *   用官方驱动原语 + 真实 errno（hal_i2s_write 不再把错误折叠成 -EIO，
 *   失败带 I2S 寄存器现场）；③植小伴策略：保留环形 FIFO 预读链保证
 *   录音连续（链在 TX 前同步停、下次 read_slot 自动重启）。
 *   实现：hal_i2s_rx_prefetch_stop() 置标志后【同步等 RX DMA 空闲】
 *   （轮询在途计数 s_rx_prefetch_inflight 归零，≤300ms 兜底），再发 TX。
 *
 * 本封装保持 hal_i2s.h API 不变（voice_agent/ai_voice 无需改调用）：
 *   - TX：官方驱动 2 槽立体声（ES8311 DAC 原生匹配）
 *   - RX：官方驱动 2 槽立体声（ES7210 2 麦标准模式，LRCK 低=左/MIC1），
 *         hal_i2s_read/read_slot 取单声道（slot 0=左 MIC1 / 1=右 MIC2）
 *   - 采样率 24kHz / 16bit（Kconfig: CONFIG_ESP32S3_I2S0_SAMPLE_RATE=24000）
 *   - MCLK=GPIO2 BCLK=GPIO17 WS=GPIO45 DOUT=GPIO15 DIN=GPIO16 PA=GPIO46
 ****************************************************************************/

#include <nuttx/config.h>


/* 2026-10-25: 待机语音侦听用的短超时。
 * 0 = 默认 3 秒阻塞（正式录音）；>0 且 <500 时，超时直接返回，不打印、
 * 不做 RX 自愈。这样 VAD 线程不会在录音期间卡住 3 秒，也不会把
 * hal_i2s_rx_idle() 停掉的 RX 链又拉起来。 */
volatile int g_i2s_read_timeout_ms = 0;
#include <nuttx/arch.h>
#include <nuttx/irq.h>
#include <nuttx/spinlock.h>
#include <nuttx/clock.h>
#include <nuttx/semaphore.h>
#include <nuttx/signal.h>
#include <nuttx/audio/audio.h>
#include <nuttx/audio/i2s.h>
#include <sys/param.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

#include "xtensa.h"
#include "esp32s3_gpio.h"
#include "hardware/esp32s3_i2s.h"
#include "hal_i2s.h"

/* 官方驱动总线初始化（链接时解析） */

extern FAR struct i2s_dev_s *esp32s3_i2sbus_initialize(int port);

/* 采样参数（与 hal_i2s.h 一致；Kconfig 必须配 24000/16bit/2ch） */

#define HAL_RATE        24000
#define HAL_BITS        16
#define HAL_FRAME_BYTES 4   /* 立体声 2 槽 × 16bit */

static FAR struct i2s_dev_s *g_dev;
static sem_t g_tx_done_sem;   /* TX 每块完成 post 一次（窗口化流水线计数） */
static int g_tx_result;
static int g_rx_result;

/* ⚠️ 2026-09-02 TX 流水线：官方驱动单次 i2s_send ≤ 2044B，若逐块
 * "排队→等完成→再排队"，块间 DMA 空转（HPWORK worker 往返 > FIFO
 * 余量）→ 播放断续（纯音"吱吱吱嘟嘟"）。改为窗口内连续排队，EOF 后
 * 官方驱动 ISR 直接续链 pend 队列 → DMA 无缝。 */

#define HAL_TX_MAX_INFLIGHT   3   /* 在途 i2s_send 块上限（驱动 container 池=4，留 1 裕量） */

static int s_tx_inflight = 0;     /* 已排队未完成块数（单 TX 用户，应用线程独占） */

/* ════════════════════════════════════════════════════════════════════
 * RX 环形 FIFO + 预读自动续（2026-09-01 数据完整性修复）
 *
 * 背景：NuttX 官方 esp32s3_i2s.c 的 RX 是"单次传输"（描述符非循环，
 * 每个 apb 收满即停）。原方案"单次预读缓冲（1023 采样/42.6ms）+ read_slot
 * 手动续"在 read_slot 暂停期间（语音块回调/HTTP 上传/WiFi 抖动）DMA 停
 * → FIFO(64B≈0.67ms) 溢出丢数据 → 语音断续缺失（用户实测"听不到人声"）。
 *
 * 修复（对齐小智 IDF 循环 DMA + auto_clear_after_cb 思路的 NuttX 等效）：
 *   1. 环形 FIFO 2048 采样（4KB static）：RX 数据持续入队，read_slot 按需
 *      取走；FIFO 满则覆盖最旧（等效 auto_clear，无残留混叠）。
 *   2. 预读自动续：预读 apb 完成回调里【立即再提交下一轮】→ DMA 连续收，
 *      read_slot/回调/上传期间不再停 → 无 FIFO 溢出丢数据。
 *   3. 回调（worker）写 FIFO、read_slot（调用线程）读，临界区保护指针。
 *
 * 内存代价：+4KB static（原 2KB 预读缓冲删除，净 +2KB .bss）。
 * ════════════════════════════════════════════════════════════════════ */

#define HAL_RX_FIFO_SAMPLES  2048                 /* 环形 FIFO 采样数（4KB） */
#define HAL_RX_FIFO_MASK     (HAL_RX_FIFO_SAMPLES - 1)

/* ⚠️ 2026-09-02 RX 双深预读（"录音一卡一卡"根因修复）：
 * 硬件 RX FIFO 仅 64B≈0.67ms。单深预读 = 收完 1 个 apb（21ms）才由
 * worker 提交下一个 → apb 间 DMA 空窗（worker 往返 >0.67ms）→ 周期性
 * 丢采样（录音墙钟 ≈2× 标称，回放一卡一卡）。深度 2：预读链始终保持
 * 2 个 apb 在途（1 个 act 在收 + 1 个 pend 排队）→ apb EOF 中断（ISR，
 * 微秒级）直接从 pend 装载下一个 → 无缝。硬件 GDMA 单通道串行 + 驱动
 * rx.act 非空保护 → 两 apb 绝不重叠收数。内存增量仅 +1 apb（~4KB 瞬时
 * 堆，段长/ring FIFO 不变）。 */

#define HAL_RX_PREFETCH_DEPTH  2    /* d2: 2 apb in flight, no gap (1011 fix) */

/* RX APB effective payload and allocation padding.  The ESP32-S3 DMA
 * descriptor rounds RX BUFLEN up for PSRAM (up to 64-byte alignment), so
 * a 2044-byte payload can write 2048 bytes.  Keep the driver-visible
 * length at 2044 but reserve enough backing storage for the rounded DMA
 * write; otherwise the final cache line corrupts adjacent heap metadata. */
#define HAL_RX_APB_WANT        2044
#define HAL_RX_APB_PAD         64


static int16_t s_rx_fifo[HAL_RX_FIFO_SAMPLES];    /* 环形缓冲（static 4KB） */
static volatile uint32_t s_rx_fifo_rd;            /* 读指针 */
static volatile uint32_t s_rx_fifo_cnt;           /* 有效采样数 */
static sem_t   s_rx_prefetch_sem;                 /* 有数据通知 */
static struct  hal_i2s_rx_ctx s_rx_prefetch_ctx;  /* 预读槽位上下文（共享 slot） */
static volatile bool s_rx_prefetch_paused;        /* TX 播放时暂停预读链 */

/* ⚠️ 2026-09-02：在途 RX apb 计数（深度 2 后取代单布尔 active）——
 * 目标 ≤ HAL_RX_PREFETCH_DEPTH；TX 前同步等待用计数归零判定。
 * 增/减均在临界区（read_slot 应用线程与 worker 回调并发）。 */

static volatile uint32_t s_rx_prefetch_inflight;  /* 在途 RX apb 数 */

static int hal_i2s_rx_prefetch_submit(int slot);
static void hal_i2s_rx_prefetch_start(int slot);
int hal_i2s_rx_recover(void);


/****************************************************************************
 * 内部：DMA 完成回调（官方驱动异步 API 的同步包装）
 *
 * ⚠️ 2026-08-25 apb 引用竞态修复（UAF → 堆损坏/泄漏）：
 * 官方驱动完成路径 = callback() → apb_free(apb)（释放驱动引用）。
 * 若调用者在 sem 唤醒后才 apb_free，可能抢在官方驱动前面把 crefs
 * 降到 0 释放内存 → 官方驱动随后 apb_free 访问已释放内存（UAF）。
 * 修复：在回调内 apb_free（我们的引用 2→1），官方驱动随后 1→0
 * 最终释放——释放顺序全在 worker 线程内，无竞态；数据提取也移进
 * 回调（回调返回后不再访问 apb）。
 ****************************************************************************/

struct hal_i2s_rx_pool
{
  FAR struct ap_buffer_s *apb[HAL_RX_PREFETCH_DEPTH];
  volatile uint8_t        busy[HAL_RX_PREFETCH_DEPTH];
};

struct hal_i2s_rx_ctx
{
  FAR struct hal_i2s_rx_pool *pool; /* 常驻 RX APB 池 */
  int                        slot;  /* 0=左 MIC1 / 1=右 MIC2 */
  uint32_t                   got;   /* 已提取采样数 */
};

static void hal_i2s_tx_cb(FAR struct i2s_dev_s *dev,
                          FAR struct ap_buffer_s *apb,
                          FAR void *arg, int result)
{
  g_tx_result = result;

  /* 释放我们的引用；官方驱动随后再释放（1→0）完成最终释放 */

  apb_free(apb);
  sem_post(&g_tx_done_sem);
}

/* 预读 apb 完成回调（worker 线程）：环形写 FIFO + post + 自动续（DMA 连续） */

static void hal_i2s_rx_prefetch_cb(FAR struct i2s_dev_s *dev,
                                   FAR struct ap_buffer_s *apb,
                                   FAR void *arg, int result)
{
  FAR struct hal_i2s_rx_ctx *ctx = (FAR struct hal_i2s_rx_ctx *)arg;
  FAR struct hal_i2s_rx_pool *pool = ctx->pool;
  uint32_t nframe = (uint32_t)(apb->nbytes / 2);
  FAR const int16_t *src = (FAR const int16_t *)apb->samp;
  uint32_t i;
  int pi = -1;
  irqstate_t flags;

  (void)dev;
  g_rx_result = result;

  if (pool != NULL)
    {
      for (i = 0; i < HAL_RX_PREFETCH_DEPTH; i++)
        {
          if (pool->apb[i] == apb)
            {
              pi = (int)i;
              break;
            }
        }
    }

  /* 2026-10-13: 被取消（rx_abort）或失败的传输不写环形 FIFO -- 此时
   * apb->nbytes / dmadesc 是残留值，写进去会污染录音。只清在途状态并
   * 唤醒等待者，让上层据此走恢复路径。 */

  if (result != OK)
    {
      flags = enter_critical_section();
      if (pi >= 0) { pool->busy[pi] = 0; }
      if (s_rx_prefetch_inflight > 0) { s_rx_prefetch_inflight--; }
      leave_critical_section(flags);
      sem_post(&s_rx_prefetch_sem);
      return;
    }

  flags = enter_critical_section();
  for (i = 0; i < nframe / 2; i++)
    {
      uint32_t w = (s_rx_fifo_rd + s_rx_fifo_cnt) & HAL_RX_FIFO_MASK;

      s_rx_fifo[w] = src[2 * i + ctx->slot];
      if (s_rx_fifo_cnt < HAL_RX_FIFO_SAMPLES)
        {
          s_rx_fifo_cnt++;
        }
      else
        {
          s_rx_fifo_rd = (s_rx_fifo_rd + 1) & HAL_RX_FIFO_MASK;
        }
    }

  if (pi >= 0)
    {
      pool->busy[pi] = 0;
    }

  if (s_rx_prefetch_inflight > 0)
    {
      s_rx_prefetch_inflight--;
    }

  leave_critical_section(flags);
  sem_post(&s_rx_prefetch_sem);

  /* No apb_free here: the pool keeps one reference forever.  The driver's
   * matching apb_free() runs after this callback and leaves that base ref. */
  if (!s_rx_prefetch_paused)
    {
      hal_i2s_rx_prefetch_submit(ctx->slot);
    }
}

static FAR struct hal_i2s_rx_pool *hal_i2s_rx_pool_get(void)
{
  FAR struct hal_i2s_rx_pool *pool = s_rx_prefetch_ctx.pool;
  int i;

  if (pool != NULL)
    {
      return pool;
    }

  pool = (FAR struct hal_i2s_rx_pool *)calloc(1, sizeof(*pool));
  if (pool == NULL)
    {
      return NULL;
    }

  for (i = 0; i < HAL_RX_PREFETCH_DEPTH; i++)
    {
      struct audio_buf_desc_s desc;
      FAR struct ap_buffer_s *apb = NULL;
      int ret;

      memset(&desc, 0, sizeof(desc));
      desc.numbytes = HAL_RX_APB_WANT + HAL_RX_APB_PAD;
      desc.u.pbuffer = &apb;
      ret = apb_alloc(&desc);
      if (ret < 0 || apb == NULL)
        {
          break;
        }

      apb->nmaxbytes = HAL_RX_APB_WANT;
      apb->nbytes = HAL_RX_APB_WANT;
      apb->curbyte = 0;
      pool->apb[i] = apb;
    }

  if (i != HAL_RX_PREFETCH_DEPTH)
    {
      int j;
      for (j = 0; j < i; j++)
        {
          apb_free(pool->apb[j]);
        }
      free(pool);
      return NULL;
    }

  s_rx_prefetch_ctx.pool = pool;
  return pool;
}

static int hal_i2s_rx_prefetch_submit(int slot)
{
  FAR struct hal_i2s_rx_pool *pool;
  FAR struct ap_buffer_s *apb;
  int pi = -1;
  int i, ret;
  irqstate_t flags;

  pool = hal_i2s_rx_pool_get();
  if (pool == NULL)
    {
      return -ENOMEM;
    }

  flags = enter_critical_section();
  if (s_rx_prefetch_paused || s_rx_prefetch_inflight >= HAL_RX_PREFETCH_DEPTH)
    {
      leave_critical_section(flags);
      return 0;
    }

  for (i = 0; i < HAL_RX_PREFETCH_DEPTH; i++)
    {
      if (!pool->busy[i])
        {
          pi = i;
          break;
        }
    }

  if (pi < 0)
    {
      leave_critical_section(flags);
      return 0;
    }

  pool->busy[pi] = 1;
  s_rx_prefetch_inflight++;
  leave_critical_section(flags);

  apb = pool->apb[pi];
  apb->nmaxbytes = HAL_RX_APB_WANT;
  apb->nbytes = HAL_RX_APB_WANT;
  apb->curbyte = 0;
  s_rx_prefetch_ctx.slot = slot;
  s_rx_prefetch_ctx.got = 0;

  ret = g_dev->ops->i2s_receive(g_dev, apb, hal_i2s_rx_prefetch_cb,
                                &s_rx_prefetch_ctx, SEC2TICK(3));
  if (ret < 0)
    {
      flags = enter_critical_section();
      pool->busy[pi] = 0;
      if (s_rx_prefetch_inflight > 0)
        {
          s_rx_prefetch_inflight--;
        }
      leave_critical_section(flags);
    }

  return ret;
}

/* 启动/补满预读链到深度 2（幂等：已在跑则 submit 因窗口满自动跳过）。
 * read_slot 每次进入都调用，保证链在跑且满。 */

static void hal_i2s_rx_prefetch_start(int slot)
{
  s_rx_prefetch_paused = false;
  s_rx_prefetch_ctx.slot = slot;   /* 链启动时固定槽位（整段录音期不变） */

  while (s_rx_prefetch_inflight < HAL_RX_PREFETCH_DEPTH)
    {
      if (hal_i2s_rx_prefetch_submit(slot) != 0)
        {
          break;   /* 分配失败等：read_slot 下次进入再补 */
        }
    }
}

/* 暂停预读链（TX 播放时调用）：在途 apb 完成后回调不再续链 →
 * RX DMA 停止、apb 池释放给 TX。read_slot 下次调用自动重启。
 *
 * ⚠️ 2026-09-02 三策略融合修复（plant voice loop 播放 -5 根因）：
 * 仅置 paused 不够——【在途 RX apb 仍占着 RX DMA】，此时启动 TX 实测
 * 失败。必须同步等 RX DMA 真正空闲（在途计数归零）再返回：
 *   ① 小智策略：严格通道生命周期，先停 RX 再发 TX（等价
 *      i2s_channel_disable(rx) → i2s_channel_write(tx)）；
 *   ② OpenVela 策略：官方驱动单次 DMA，回调即传输完成；
 *   ③ 植小伴策略：预读链保持录音连续，此处只在 TX 前同步停。
 * 竞态边界：极端情况下两个预读 apb 可能同时在途（active 标志会在
 * 第一个完成时就清 0），故用 s_rx_prefetch_inflight 计数判定——每个
 * apb 入队 +1、回调完成 -1，归零 = 全部 RX DMA 完成。300ms 兜底防
 * RX 死链导致无限等。 */

static void hal_i2s_rx_prefetch_stop(void)
{
  int wait_ms = 0;

  /* ⚠️ 置 paused：回调看到后不再续链；在途 apb 完成后链自然停止。
   * 注意：不能清 active/inflight——它们是"在途 RX apb"的真相，
   * 清掉会让下面的等待失效；回调完成时自己会更新。 */

  s_rx_prefetch_paused = true;

  /* 同步等 RX DMA 空闲：在途计数归零（所有 RX apb 的 DMA 已完成）
   * 前不发 TX */

  while (s_rx_prefetch_inflight > 0 && wait_ms < 300)
    {
      nxsig_usleep(1000);
      wait_ms++;
    }
}

/* v97 诊断：把 RX 预读链置为空闲（同步等 RX DMA 归零）。
 * app 侧在"录音结束 -> 编解码"这段纯计算期调用，避免 RX DMA 与
 * hpwork 的 apb alloc/free 和编解码并发（loop 1 曾在 i2s_rx_worker
 * 的 apb_free 里踩到坏堆 mm_free.c:208 断言）。read_slot 下次调用
 * 会自动把链重启，语义与 TX 前的 stop 完全一致。 */

void hal_i2s_rx_idle(void)
{
  hal_i2s_rx_prefetch_stop();
}

/* 2026-10-13 硬恢复：RX 链卡死时复位并重启。
 *
 * 触发场景：ES7210 没锁上 MCLK/LRCK -> RX DMA 收不满 apb -> EOF 中断不来
 * -> 预读链在途计数停在深度值不归零 -> hal_i2s_rx_prefetch_start 里的
 * while 进不去（inflight >= DEPTH）-> 之后每次 read_slot 都 3 秒超时，
 * 录音拿 0 样本（重试多少次都一样，现场表现就是"提示音响完没反应"）。
 *
 * 步骤：①暂停预读链；②让驱动取消在途/待发（回调把 inflight 归零）；
 * ③等归零（<=300ms），仍不归零就强制清零兜底；④清环形 FIFO 与残留
 * 信号量；⑤重新补满预读链。 */

int hal_i2s_rx_recover(void)
{
  /* 2026-10-13: the previous implementation aborted the driver RX
   * queues and freed containers behind the driver's back. That
   * double-freed APBs and corrupted the heap (mallinfo() then
   * panicked while walking it). Recovery is no longer needed once
   * the real capture bug is fixed, so keep this a no-op stub and
   * leave the driver's own queues untouched. */
  return 0;
}

/****************************************************************************
 * 公共 API
 ****************************************************************************/

int hal_i2s_init(void)
{
  int ret;

  if (g_dev != NULL)
    {
      return 0;
    }

  sem_init(&g_tx_done_sem, 0, 0);
  sem_init(&s_rx_prefetch_sem, 0, 0);
  s_rx_prefetch_paused = false;
  s_rx_prefetch_inflight = 0;

  /* 2026-10-23: 默认关闭功放。官方小智只在放音时把 PA 拉高，
   * 录音/空闲时拉低；旧代码一上电就常开，功放长期工作会经电源和
   * 地线干扰 ES8311 的 MIC 模拟前端。提示音由 ai_voice_play_tone
   * 临时打开。 */

  esp32s3_configgpio(HAL_PA_PIN, OUTPUT);
  esp32s3_gpiowrite(HAL_PA_PIN, false);

  /* 官方驱动初始化（按 Kconfig：master、24k、16bit、2 槽全双工；
   * 内部自动配时钟/DMA/中断，TX+RX 通道已启动） */

  g_dev = esp32s3_i2sbus_initialize(0);
  if (g_dev == NULL)
    {
      return -ENODEV;
    }

  /* 显式确认运行时参数（与 Kconfig 一致，防默认值漂移） */

  if (g_dev->ops->i2s_txsamplerate != NULL)
    {
      g_dev->ops->i2s_txsamplerate(g_dev, HAL_RATE);
    }

  if (g_dev->ops->i2s_rxsamplerate != NULL)
    {
      g_dev->ops->i2s_rxsamplerate(g_dev, HAL_RATE);
    }

  if (g_dev->ops->i2s_txdatawidth != NULL)
    {
      g_dev->ops->i2s_txdatawidth(g_dev, HAL_BITS);
    }

  if (g_dev->ops->i2s_rxdatawidth != NULL)
    {
      g_dev->ops->i2s_rxdatawidth(g_dev, HAL_BITS);
    }

  if (g_dev->ops->i2s_txchannels != NULL)
    {
      g_dev->ops->i2s_txchannels(g_dev, 2);
    }

  if (g_dev->ops->i2s_rxchannels != NULL)
    {
      g_dev->ops->i2s_rxchannels(g_dev, 2);
    }

  /* 2026-10-22 v13: NuttX 的 i2s_rxchannels() 只存了 priv->channels，
   * 没有写 RX_TDM_CTRL。复位值是 TOT_CHAN_NUM=0 / CHAN_BITS=0xFFFF /
   * 通道使能全开，RX 序列化器会把 ES8311 的 L/R 数据帧错位，读出来
   * 全 0。这里按官方 ESP-IDF / i2s_txchannels 的写法补上：
   *   TOT_CHAN_NUM = 1（2 通道）
   *   CHAN0_EN | CHAN1_EN
   *   RX_MONO = 0（立体声）
   * 然后置 RX_UPDATE 生效。 */

  modifyreg32(I2S_RX_TDM_CTRL_REG(0), I2S_RX_TDM_TOT_CHAN_NUM_M,
              FIELD_TO_VALUE(I2S_RX_TDM_TOT_CHAN_NUM, 1));
  {
    uint32_t rxm = getreg32(I2S_RX_TDM_CTRL_REG(0));
    rxm &= 0xffff0000;
    rxm |= (1u << 0) | (1u << 1);   /* CHAN0_EN | CHAN1_EN */
    putreg32(rxm, I2S_RX_TDM_CTRL_REG(0));
  }
  modifyreg32(I2S_RX_CONF_REG(0), I2S_RX_MONO_M, 0);
  modifyreg32(I2S_RX_CONF_REG(0), 0, I2S_RX_UPDATE);

  /* 2026-10-22 v14: 对齐 ESP-IDF i2s_std 的 Philips 帧格式。
   *
   * NuttX 的 i2s_set_datawidth() 在非 PCM 模式下把
   * RX_TDM_WS_WIDTH 写成 1（= 2 个 BCLK）。ESP-IDF 的
   * I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG 里 ws_width = 16bit，
   * 驱动实际写 ws_width-1 = 15（= 16 个 BCLK，半个 32bit 帧）。
   *
   * WS 只有 2 个 BCLK 时，RX 序列化器只在 WS 高电平的 2 个
   * BCLK 内取值，其余 14 bit 全部丢掉 -> 录到的 PCM 幅度只剩
   * 满量程的很小一部分（v13 实测 acrms=178、峰值 0.002）。
   *
   * 官方 xiaozhi 同款板 sp-esp32-s3-1.54-muma：
   *   24kHz / 16bit / stereo / MCLK 256x / Philips / left_align
   * 这里把 RX/TX 的 WS 宽度、MSB 移位、左对齐一次性对齐 IDF。 */

  modifyreg32(I2S_RX_CONF1_REG(0), I2S_RX_TDM_WS_WIDTH_M,
              FIELD_TO_VALUE(I2S_RX_TDM_WS_WIDTH, 15));
  modifyreg32(I2S_RX_CONF1_REG(0), I2S_RX_MSB_SHIFT_M, I2S_RX_MSB_SHIFT);
  modifyreg32(I2S_RX_CONF_REG(0), I2S_RX_LEFT_ALIGN_M, I2S_RX_LEFT_ALIGN);
  modifyreg32(I2S_RX_CONF_REG(0), 0, I2S_RX_UPDATE);

  modifyreg32(I2S_TX_CONF1_REG(0), I2S_TX_TDM_WS_WIDTH_M,
              FIELD_TO_VALUE(I2S_TX_TDM_WS_WIDTH, 15));
  modifyreg32(I2S_TX_CONF1_REG(0), I2S_TX_MSB_SHIFT_M, I2S_TX_MSB_SHIFT);
  modifyreg32(I2S_TX_CONF_REG(0), I2S_TX_LEFT_ALIGN_M, I2S_TX_LEFT_ALIGN);
  modifyreg32(I2S_TX_CONF_REG(0), 0, I2S_TX_UPDATE);

  printf("[I2S] v14 fmt RX_CONF1=%08lx TX_CONF1=%08lx\n",
         (unsigned long)getreg32(I2S_RX_CONF1_REG(0)),
         (unsigned long)getreg32(I2S_TX_CONF1_REG(0)));

  /* 2026-10-22 v12: 显式把 MCLK 锁到 256x = 6.144MHz，和官方 xiaozhi
   * 的 I2S_MCLK_MULTIPLE_256 对齐。打印请求值和实际值，排查时钟不匹配。 */

  if (g_dev->ops->i2s_setmclkfrequency != NULL)
    {
      uint32_t want = (uint32_t)HAL_RATE * 256;
      uint32_t got = g_dev->ops->i2s_setmclkfrequency(g_dev, want);
      printf("[I2S] MCLK want=%lu got=%lu\n",
             (unsigned long)want, (unsigned long)got);

      if (g_dev->ops->i2s_getmclkfrequency != NULL)
        {
          printf("[I2S] MCLK readback=%lu\n",
                 (unsigned long)g_dev->ops->i2s_getmclkfrequency(g_dev));
        }
    }

  /* ⚠️ 2026-08-25：TX FIFO 空时也要保持 BCK/WS 输出（STOP_EN=0）。
   * 官方驱动未配置该位——录音只开 RX 时 TX 无数据流，若 FIFO 空时钟
   * 停 → RX（SIG_LOOPBACK 共享 TX 时钟）收不到数据。 */

  modifyreg32(I2S_TX_CONF_REG(0), I2S_TX_STOP_EN, 0);
  modifyreg32(I2S_TX_CONF_REG(0), 0, I2S_TX_UPDATE);
  {
    int sync_wait = 0;

    while ((getreg32(I2S_TX_CONF_REG(0)) & I2S_TX_UPDATE) &&
           sync_wait++ < 100000)
      {
      }
  }

  /* ⚠️ 2026-08-25 根因修复：TX_START 必须显式置位！
   * 官方驱动只在 TX 数据排队时置 TX_START（esp32s3_i2s.c L708，
   * i2s_txdma_start）→ record 只读 RX（无 TX 数据流）→ TX_START=0 →
   * TX 串行器不工作 → BCLK/WS 不输出 → ES7210 无时钟 → RX 3s 超时
   *（r=-110，实测 TX_CONF=0x08089200 bit2=0）。
   * 置位后 + STOP_EN=0 → FIFO 空也持续输出 BCLK/WS，RX 时钟稳定。 */

  modifyreg32(I2S_TX_CONF_REG(0), 0, I2S_TX_START);

  /* 2026-10-22 v12: 打印 RX/TX 的 TDM/时钟配置，和 ESP-IDF 官方
   * (24k/16bit/stereo/256x MCLK) 对比，排查 slot 位宽/BCLK 分频。 */

  printf("[I2S] CFG RX_CONF=%08lx RX_TDM=%08lx RX_CLKM=%08lx "
         "TX_CLKM=%08lx RXEOF=%lu\n",
         (unsigned long)getreg32(I2S_RX_CONF_REG(0)),
         (unsigned long)getreg32(I2S_RX_TDM_CTRL_REG(0)),
         (unsigned long)getreg32(I2S_RX_CLKM_CONF_REG(0)),
         (unsigned long)getreg32(I2S_TX_CLKM_CONF_REG(0)),
         (unsigned long)getreg32(I2S_RXEOF_NUM_REG(0)));

  /* 排空残留信号量（官方驱动启动时可能已有完成事件） */

  while (nxsem_trywait(&g_tx_done_sem) == 0);

  return 0;
}

int hal_i2s_start_tx_clock(void)
{
  /* 官方驱动初始化时 TX 通道已启动（时钟持续输出，STOP_EN=0） */

  return 0;
}

/****************************************************************************
 * TX 播放（2026-09-02 窗口化流水线 —— "吱吱吱嘟嘟"根因修复）
 *
 * 背景：旧版逐 2044B 块"排队→等 EOF→HPWORK worker 回调→再排队"，
 * 每块之间 DMA 必然空转（worker 往返 > FIFO 余量），1kHz 纯音被切成
 * ~21ms 段、段间掉拍 → 听感"吱吱吱嘟嘟"。08-25 验证干净的 tone 是
 * 一次性大 DMA 链（无块间隙）；09-02 分块 + apb 压 2044B 后音质从未
 * 验证过——本次修复。
 *
 * 机制：排队 HAL_TX_MAX_INFLIGHT 块（官方驱动 container 池=4，留 1
 * 给 RX 边界）再等完成 → 块 EOF 后官方驱动 ISR 直接从 pend 队列续链
 * 下一块（esp32s3_i2s.c i2s_tx_schedule），无 worker 往返空窗 → DMA
 * 连续。窗口满才阻塞等一块腾位（官方驱动池满时 i2s_send 自身也阻塞，
 * 天然流控，不会死锁）。
 *
 * 注意：i2s_send 排队时驱动已把数据 memcpy 进内部缓冲（txdma_setup），
 * 故 hal_i2s_write_async 返回后调用方可安全复用 buf——tone 播放据此
 * 实现"生成下一块时上一块仍在播"，彻底消除块边界空窗。
 ****************************************************************************/

/* 排队一个 TX 块（≤2044B、帧对齐）进官方驱动 pend 队列，不等完成。
 * 返回 OK / 负 errno。container 池满时官方驱动 i2s_send 内部阻塞。 */

static int hal_tx_queue_block(FAR const void *buf, uint32_t bytes)
{
  struct audio_buf_desc_s desc;
  FAR struct ap_buffer_s *apb = NULL;
  int ret;

  memset(&desc, 0, sizeof(desc));
  desc.numbytes = bytes;
  desc.u.pbuffer = &apb;
  ret = apb_alloc(&desc);
  if (ret < 0)
    {
      return ret;
    }

  memcpy(apb->samp, buf, bytes);
  apb->nbytes = bytes;
  apb->curbyte = 0;

  ret = g_dev->ops->i2s_send(g_dev, apb, hal_i2s_tx_cb, NULL, SEC2TICK(5));
  if (ret < 0)
    {
      /* 未入队：释放我们的引用（失败路径驱动可能已 reference） */

      printf("[I2S] i2s_send 入队失败: %d (%u B)\n", ret, (unsigned)bytes);
      apb_free(apb);
      return ret;
    }

  s_tx_inflight++;
  return OK;
}

/* 等 TX 在途块数降到 want（want=0 全部排空）。超时=驱动/时钟异常：
 * 清空信号量防错乱后返回 -ETIMEDOUT。 */

static int hal_tx_wait(int want)
{
  while (s_tx_inflight > want)
    {
      /* 2026-10-29: 5 秒太长。提示音是 3 短 1 长共 4 次调用，
       * 每次卡 5 秒就是 20 秒，用户按完 BOOT 半分钟没反应。
       * 一块 21 ms，1 秒足够判定 DMA 异常。 */
      if (nxsem_tickwait(&g_tx_done_sem, SEC2TICK(1)) < 0)
        {
          printf("[I2S] TX 排空超时 (inflight=%d)!\n", s_tx_inflight);
          while (nxsem_trywait(&g_tx_done_sem) == 0);
          s_tx_inflight = 0;
          return -ETIMEDOUT;
        }

      s_tx_inflight--;
    }

  return OK;
}

int hal_i2s_write_async(const void *buf, uint32_t bytes)
{
  uint32_t sent = 0;
  FAR const uint8_t *p = buf;
  int err;

  /* ⚠️ 2026-09-02：TX 播放前【同步】停预读链（-5 修复，幂等）——
   * 环形 FIFO 预读自动续会让 RX DMA 持续活动，官方驱动 i2s_send 与
   * RX 并发时 TX 启动失败。分时：播放停 RX，录音时 read_slot 自动重启。 */

  hal_i2s_rx_prefetch_stop();

  while (sent < bytes)
    {
      uint32_t chunk = MIN(bytes - sent, 2044);   /* 2044B=堆容纳上限（见预读注释） */

      chunk -= (chunk % HAL_FRAME_BYTES);   /* 对齐到帧（4B） */

      if (chunk == 0)
        {
          break;
        }

      /* 窗口满 → 等一块完成腾位（ISR 已把 pend 下一块无缝接上，无空窗） */

      if (s_tx_inflight >= HAL_TX_MAX_INFLIGHT)
        {
          err = hal_tx_wait(HAL_TX_MAX_INFLIGHT - 1);
          if (err < 0)
            {
              return err;
            }
        }

      err = hal_tx_queue_block(p + sent, chunk);
      if (err == -ENOMEM)
        {
          /* 堆紧：先排空在途（回收 apb/container）再重试一次 */

          hal_tx_wait(0);
          err = hal_tx_queue_block(p + sent, chunk);
        }

      if (err < 0)
        {
          return err;
        }

      sent += chunk;
    }

  return (int)sent;
}

int hal_i2s_write_flush(void)
{
  return hal_tx_wait(0);
}

int hal_i2s_write(const void *buf, uint32_t bytes)
{
  int ret = hal_i2s_write_async(buf, bytes);

  if (ret < 0)
    {
      return ret;
    }

  return hal_i2s_write_flush() < 0 ? -ETIMEDOUT : ret;
}

int hal_i2s_read(void *buf, uint32_t bytes)
{
  return hal_i2s_read_slot(buf, bytes, 0);
}

int hal_i2s_read_slot(void *buf, uint32_t bytes, int slot)
{
  FAR int16_t *p = (FAR int16_t *)buf;
  uint32_t want_samp = bytes / 2;
  uint32_t got = 0;
  irqstate_t flags;

  /* 确保预读链在跑且满（深度 2：首次调用 / 播放后重启自动补满） */

  /* 2026-10-13: full-duplex RX shares TX BCK/WS through SIG_LOOPBACK.
   * If TX_CONF.TX_STOP_EN is 1, an empty TX FIFO stops BCK/WS, the
   * ES7210 loses its clock and RX DMA never completes (read_slot
   * times out with 0 samples). Force the shared clock on before
   * every record loop; idempotent and nearly free. */
  {
    static int s_clk_dbg = 0;
    uint32_t txc = getreg32(I2S_TX_CONF_REG(0));
    if (!s_clk_dbg)
      {
        s_clk_dbg = 1;
        printf("[I2S] read_slot in: TX_CONF=0x%08x STOP_EN=%u TX_START=%u\n",
               (unsigned)txc,
               (unsigned)((txc & I2S_TX_STOP_EN) ? 1 : 0),
               (unsigned)((txc & I2S_TX_START) ? 1 : 0));
      }
    modifyreg32(I2S_TX_CONF_REG(0), I2S_TX_STOP_EN, 0);
    modifyreg32(I2S_TX_CONF_REG(0), 0, I2S_TX_START);
  }

  hal_i2s_rx_prefetch_start(slot);

  /* 从环形 FIFO 取数：先取已有，不够等预读回调（自动续链持续供数） */

  int recover_try = 0;
  uint32_t t_enter = (uint32_t)clock_systime_ticks();

  while (got < want_samp)
    {
      /* 2026-10-25: hard wall-clock cap.  The prefetch ISR can keep
       * posting s_rx_prefetch_sem while delivering no samples (e.g. RX
       * DMA chain stuck), in which case the per-wait timeout never
       * triggers and read_slot spins forever -- the take then never
       * returns and the link never sends last=1.  One read is ~42 ms of
       * audio; 800 ms is generous and still bounds the hang. */
      if ((uint32_t)(clock_systime_ticks() - t_enter) > (uint32_t)MSEC2TICK(800))
        {
          static int s_hard_to;
          if (s_hard_to++ < 8)
            {
              printf("[I2S] read_slot hard timeout: got=%u/%u fifo=%u inflight=%u\n",
                     (unsigned)got, (unsigned)want_samp,
                     (unsigned)s_rx_fifo_cnt,
                     (unsigned)s_rx_prefetch_inflight);
            }
          break;
        }

      flags = enter_critical_section();
      while (s_rx_fifo_cnt > 0 && got < want_samp)
        {
          p[got++] = s_rx_fifo[s_rx_fifo_rd];
          s_rx_fifo_rd = (s_rx_fifo_rd + 1) & HAL_RX_FIFO_MASK;
          s_rx_fifo_cnt--;
        }

      leave_critical_section(flags);

      if (got >= want_samp)
        {
          break;
        }

      {
        int wait_ms = g_i2s_read_timeout_ms;

        if (wait_ms <= 0) { wait_ms = 3000; }

      if (nxsem_tickwait(&s_rx_prefetch_sem, MSEC2TICK(wait_ms)) < 0)
        {
          /* 短超时模式（VAD）：安静返回，不刷屏、不碰 RX 链。 */
          if (g_i2s_read_timeout_ms > 0 && g_i2s_read_timeout_ms < 500)
            {
              break;
            }

          printf("[I2S] RX FIFO 超时（预读链断？TX_CONF=0x%08x "
                 "RX_CONF=0x%08x）\n",
                 (unsigned)getreg32(I2S_TX_CONF_REG(0)),
                 (unsigned)getreg32(I2S_RX_CONF_REG(0)));

          /* 2026-10-13 自愈：RX 链已死（DMA 收不满 -> EOF 中断不来 ->
           * 在途计数不归零 -> start 补不上链），复位并重启一次再等。
           * 最多自救 2 次，仍无数据才放弃（上层据此整段重试录音）。 */

          if (recover_try == 0)
            {
              printf("[I2S] dbg: inflight=%u paused=%d fifocnt=%u rxres=%d\n"
                     "      RX_CONF=0x%08x INT_RAW=0x%08x RXEOF_NUM=%u\n"
                     "      STATE=0x%08x RX_CLKM=0x%08x RX_TDM=0x%08x\n"
                     "      TX_CLKM=0x%08x RX_TIMING=0x%08x\n",
                     (unsigned)s_rx_prefetch_inflight,
                     (int)s_rx_prefetch_paused,
                     (unsigned)s_rx_fifo_cnt, g_rx_result,
                     (unsigned)getreg32(I2S_RX_CONF_REG(0)),
                     (unsigned)getreg32(I2S_INT_RAW_REG(0)),
                     (unsigned)getreg32(I2S_RXEOF_NUM_REG(0)),
                     (unsigned)getreg32(I2S_STATE_REG(0)),
                     (unsigned)getreg32(I2S_RX_CLKM_CONF_REG(0)),
                     (unsigned)getreg32(I2S_RX_TDM_CTRL_REG(0)),
                     (unsigned)getreg32(I2S_TX_CLKM_CONF_REG(0)),
                     (unsigned)getreg32(I2S_RX_TIMING_REG(0)));
            }

          if (recover_try < 2)
            {
              recover_try++;
              printf("[I2S] RX 链自愈：复位并重启预读链（第 %d 次）\n",
                     recover_try);
              hal_i2s_rx_recover();
              continue;
            }

          break;
        }
      }
    }

  return (int)(got * 2);
}

int hal_i2s_rx_set_channel(int slot)
{
  /* 官方驱动 RX 恒为 2 槽立体声；槽选择在 read_slot 完成 */

  (void)slot;
  return 0;
}

/****************************************************************************
 * 诊断：I2S TX 状态 dump（无声排查用）
 *
 * 播放"完成"却无声时判读：
 *  - TX_CONF bit2=TX_START(1=启动)、bit13=TX_STOP_EN(0=时钟保持)、
 *    bit27=SIG_LOOPBACK；
 *  - STATE bit0=TX_IDLE(1=串行器空闲——播放刚结束正常，播放中应为 0)。
 ****************************************************************************/

void hal_i2s_dump_tx(void)
{
  printf("[I2S] TX_CONF=0x%08x TX_CONF1=0x%08x STATE=0x%08x "
         "TX_CLKM=0x%08x\n",
         (unsigned)getreg32(I2S_TX_CONF_REG(0)),
         (unsigned)getreg32(I2S_TX_CONF1_REG(0)),
         (unsigned)getreg32(I2S_STATE_REG(0)),
         (unsigned)getreg32(I2S_TX_CLKM_CONF_REG(0)));
  printf("[I2S] TX_START=%u TX_STOP_EN=%u TX_IDLE=%u SIG_LOOPBACK=%u\n",
         (unsigned)((getreg32(I2S_TX_CONF_REG(0)) & I2S_TX_START) ? 1 : 0),
         (unsigned)((getreg32(I2S_TX_CONF_REG(0)) & I2S_TX_STOP_EN) ? 1 : 0),
         (unsigned)((getreg32(I2S_STATE_REG(0)) & I2S_TX_IDLE) ? 1 : 0),
         (unsigned)((getreg32(I2S_TX_CONF_REG(0)) & I2S_SIG_LOOPBACK) ? 1 : 0));

  /* ⚠️ 2026-09-02 加 RX 帧结构（ES7210 4 槽/64×FS 判据）：
   * RX_TDM_CTRL bits[19:16]=TOT_CHAN_NUM(3=4槽) 低16位=chan 使能；
   * RX_CLKM div 应使 BCLK=1.536MHz（24k×4×16）。 */

  printf("[I2S] RX_TDM_CTRL=0x%08x RX_CONF=0x%08x RX_CLKM=0x%08x\n",
         (unsigned)getreg32(I2S_RX_TDM_CTRL_REG(0)),
         (unsigned)getreg32(I2S_RX_CONF_REG(0)),
         (unsigned)getreg32(I2S_RX_CLKM_CONF_REG(0)));

  /* ⚠️ 2026-09-02 加 I2S 中断原态（播放欠载判据）：
   * INT_RAW bit3=TX_HUNG（TX FIFO 空转超阈值 = 播放期间 DMA 供数
   * 断过）、bit1=TX_DONE、bit0=RX_DONE。播放刚结束 TX_HUNG=1 说明
   * 数据流有洞（块间空窗），连续播放应全 0。 */

  printf("[I2S] INT_RAW=0x%08x (TX_HUNG=%u TX_DONE=%u RX_DONE=%u)\n",
         (unsigned)getreg32(I2S_INT_RAW_REG(0)),
         (unsigned)((getreg32(I2S_INT_RAW_REG(0)) & I2S_TX_HUNG_INT_RAW) ? 1 : 0),
         (unsigned)((getreg32(I2S_INT_RAW_REG(0)) & I2S_TX_DONE_INT_RAW) ? 1 : 0),
         (unsigned)((getreg32(I2S_INT_RAW_REG(0)) & I2S_RX_DONE_INT_RAW) ? 1 : 0));
}
