/****************************************************************************
 * apps/plant-companion/ai_module/ai_voice/ai_voice.c
 *
 * 语音 AI 模块：
 *   - I2C：ES8311(DAC) + ES7210(ADC) 码片初始化
 *   - I2S：通过 hal_i2s（自定义 HAL 层）直接操作 I2S0 外设
 *   - 录音：24kHz DMA → 降采样到 16kHz → WAV 编码
 *   - 播放：WAV 解析 → 24kHz DMA → 喇叭
 *
 * 不再使用 NuttX 的 esp32s3_i2s 驱动（上游未验证，RX DMA 时钟不工作）。
 ****************************************************************************/

#include <nuttx/config.h>
#include <nuttx/clock.h>
#include <nuttx/i2c/i2c_master.h>
#include <nuttx/sched.h>
#include <nuttx/semaphore.h>

#include <errno.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../components/voice_agent/es8311.h"
#include "../../components/voice_agent/es7210.h"
#include "../../components/voice_agent/voice_agent.h"
#include "../../hal/hal_i2c.h"
#include "../../hal/hal_i2s.h"
#include "ai_voice.h"
#include "ai_ns.h"

/* I2C bus initializer (lives in kernel, resolved at link time) */

extern FAR struct i2c_master_s *esp32s3_i2cbus_initialize(int bus);

/* Kernel GPIO API (flat build: symbols link directly) */

typedef uint16_t gpio_pinattr_t;
extern int  esp32s3_configgpio(uint32_t pin, gpio_pinattr_t attr);
extern void esp32s3_gpiowrite(int pin, bool value);
extern bool esp32s3_gpioread(int pin);

#define GPIO_2   (1 << 1)

/****************************************************************************
 * 私有定义
 ****************************************************************************/

/* I2C 总线号 */

#define I2C_BUS  0

/* 编解码器 MCLK（从模式跟随外部时钟；HAL 实际输出 160/26=6.154MHz） */

#define AI_VOICE_MCLK (24000 * 256)

/* 初始化标记（避免重复初始化） */

static bool g_initialized = false;

/* RX 数据槽位/周期（idx%g_rx_period==g_rx_phase），录音时首块自动检测。
 * 周期 1=连续(全量)、2/4=TDM 稀疏槽。REG08 修复后码片帧结构会变，
 * 故用密度自适应检测（见 ai_voice_record）。 */
static int g_rx_phase = -1;
static int g_rx_period = 1;

/* 2026-08-25 路径A：qsort 比较器（幅度分位数统计用） */

static int cmp_int16_abs(const void *a, const void *b)
{
  int32_t x = *(FAR const int16_t *)a;
  int32_t y = *(FAR const int16_t *)b;

  return (x > y) - (x < y);
}

/****************************************************************************
 * WAV 头生成（44 字节标准 RIFF WAV）
 ****************************************************************************/

static int wav_put32(FAR uint8_t *p, uint32_t v)
{
  p[0] = v & 0xff;
  p[1] = (v >> 8) & 0xff;
  p[2] = (v >> 16) & 0xff;
  p[3] = (v >> 24) & 0xff;
  return 4;
}

static int wav_put16(FAR uint8_t *p, uint16_t v)
{
  p[0] = v & 0xff;
  p[1] = (v >> 8) & 0xff;
  return 2;
}

static int wav_header(FAR uint8_t *wav, uint32_t data_bytes)
{
  FAR uint8_t *p = wav;
  uint32_t rate = AI_VOICE_AI_RATE;

  memcpy(p, "RIFF", 4);  p += 4;
  p += wav_put32(p, 36 + data_bytes);
  memcpy(p, "WAVE", 4);  p += 4;
  memcpy(p, "fmt ", 4);  p += 4;
  p += wav_put32(p, 16);
  p += wav_put16(p, 1);              /* PCM */
  p += wav_put16(p, AI_VOICE_CHANNELS);
  p += wav_put32(p, rate);
  p += wav_put32(p, rate * 2);       /* byte rate */
  p += wav_put16(p, 2);              /* block align */
  p += wav_put16(p, AI_VOICE_BITS_PER_SAMPLE);
  memcpy(p, "data", 4);  p += 4;
  p += wav_put32(p, data_bytes);

  return (int)(p - wav);
}

/****************************************************************************
 * 公共 API：初始化
 ****************************************************************************/

int ai_voice_init(void)
{
  int ret;

  if (g_initialized)
    {
      return 0;
    }

  /* 外设层初始化：ES8311 + ES7210 + I2S0（components/voice_agent）
   * AI 应用层不直接操作码片/寄存器 */

  ret = voice_agent_init();
  if (ret < 0)
    {
      return ret;
    }

  g_initialized = true;
  return 0;
}

/****************************************************************************
 * 公共 API：降采样 24kHz → 16kHz（线性插值）
 ****************************************************************************/

uint32_t ai_voice_decimate(FAR int16_t *out, FAR const int16_t *in24,
                           uint32_t in_samples)
{
  uint32_t n = 0;

  /* 24kHz → 16kHz = 2/3 重采样
   * 输出第 j 个样本 = 输入第 j*1.5 个样本（线性插值） */

  for (uint32_t j = 0; ; j++)
    {
      uint32_t pos_fixed = j * 3;
      uint32_t i = pos_fixed / 2;
      uint32_t frac = pos_fixed % 2;

      if (i >= in_samples)
        {
          break;
        }

      if (frac == 0)
        {
          out[n++] = in24[i];
        }
      else if (i + 1 < in_samples)
        {
          out[n++] = (int16_t)(((int32_t)in24[i] + (int32_t)in24[i + 1]) / 2);
        }
      else
        {
          out[n++] = in24[i];
        }
    }

  return n;
}

/****************************************************************************
 * 公共 API：WAV 编码
 ****************************************************************************/

int ai_voice_wav_encode(FAR uint8_t *wav, FAR const int16_t *pcm,
                        uint32_t pcm_bytes)
{
  int hdr = wav_header(wav, pcm_bytes);

  if (pcm != (FAR const int16_t *)(wav + hdr))
    {
      memcpy(wav + hdr, pcm, pcm_bytes);
    }

  return hdr + (int)pcm_bytes;
}

/****************************************************************************
 * 公共 API：录音（麦克风 → 24kHz DMA → 降采样 16kHz → WAV）
 *
 * 流式处理：每次只 malloc 一个 100ms 的 24kHz chunk（~9.6KB），
 * 降采样后直接写入 wav 缓冲区（调用者提供，最大 AI_VOICE_WAV_MAX 字节）。
 *
 * ⚠️ 栈：NSH 任务栈已加大到 32KB（CONFIG_SYSTEM_NSH_STACKSIZE），
 * NSNet2 推理（dl_lib 算子嵌套）在此栈上直接执行，无需独立线程。
 ****************************************************************************/

int ai_voice_record(FAR uint8_t *wav, size_t wav_size, int seconds)
{
  FAR int16_t *chunk48;       /* 24kHz RX 缓冲 */
  FAR int16_t *pcm16;         /* 16kHz PCM 指向 wav+44 */
  uint32_t chunk_samples;     /* 每次 DMA 的采样数 */
  uint32_t want_samples;      /* 总共需要的 24kHz 采样数 */
  uint32_t got = 0;           /* 已收到的 24kHz 采样数 */
  uint32_t n16 = 0;           /* 已产生的 16kHz 采样数 */
  bool first_chunk = true;    /* 首块热身标志（跳过起始瞬态） */
  int ret;

  if (seconds <= 0 || seconds > AI_VOICE_MAX_SECONDS)
    {
      seconds = AI_VOICE_MAX_SECONDS;
    }

  /* 按实际秒数校验缓冲（不要求满 AI_VOICE_WAV_MAX 5s 上限）：
   * 16kHz 单声道 PCM 字节数 = AI_VOICE_AI_RATE×2×seconds，加 44 头。
   * 固件含 LVGL/UI 后堆紧张，按实际秒数分配可省堆（见 app_main.c）。 */

  if (wav_size < 44 + AI_VOICE_AI_RATE * 2 * (size_t)seconds)
    {
      return -ENOSPC;
    }

  /* 诊断：记录堆余量（定位 -ENOMEM / 评估 UI 并发占用） */

  {
    struct mallinfo mi = mallinfo();

  }

  ret = ai_voice_init();
  if (ret < 0)
    {
      return ret;
    }
  /* NSNet2 神经网络降噪初始化（失败不阻断录音，仅降噪失效）
   * ⚠️ PLANT_NO_NS=1（make 二分验证）：完全跳过 NS → 无 NS 版 */

#ifndef PLANT_NO_NS
  if (ai_ns_init() < 0)
    {
      printf("[Voice] NSNet2 init 失败——降噪关闭\n");
    }
#endif


  /* 24kHz 临时缓冲——1000 采样 = 2KB。
   * ⚠️ 2026-08-21 对照实验：与 rate 测试相同的 2000B/读（rate 测试 271 读
   * 全好；旧 1000B/读在 ~12 读后系统死）——先消除"读大小"变量。
   * ⚠️ 每次读取必须 ≤ 2047 采样（4094 字节）：
   * hal_i2s_read 的 RXEOF_NUM 上限 4095 字节，超过会提前返回、
   * 后半缓冲是过期数据。 */

  chunk_samples = 1000;  /* 2000B 块（标准） */

  /* ⚠️ 每块丢弃的采样数（48kHz 流）：
   * 开头 RX_SETTLE_SAMPLES（HPF 阶跃瞬态）+ 末尾 RX_TAIL_SAMPLES
   * （DMA 结束/块边界噪声突发，实测 864-999 区间 ~31 个大值）。 */

  #define RX_SETTLE_SAMPLES 128
  #define RX_TAIL_SAMPLES   160
  chunk48 = (FAR int16_t *)malloc(chunk_samples * 2);
  if (chunk48 == NULL)
    {
      printf("[Voice] chunk48 malloc 失败 (%u B), free=%u\n",
             (unsigned)(chunk_samples * 2),
             (unsigned)mallinfo().fordblks);
      return -ENOMEM;
    }

  /* 16kHz PCM 直接写入 wav+44（省掉一个大缓冲） */

  pcm16 = (FAR int16_t *)(wav + 44);
  want_samples = (uint32_t)seconds * 24000;   /* 官方驱动封装：read_slot 直接返回
                                               * 单声道 MIC1 24k 连续流
                                               *（2026-08-25 迁移官方驱动后） */

  /* 诊断：记录耗时（判断 RX 是否在读真实数据）+ 数据指纹 */

  {
    clock_t t0 = clock_systime_ticks();
    uint32_t want_out = AI_VOICE_AI_RATE * (uint32_t)seconds;  /* 16kHz 输出容量 */

    printf("[Voice] record loop start (want_out=%u)\n", (unsigned)want_out);

    /* ⚠️ 2026-08-21 致命 bug 修复：循环终止条件从"24kHz 输入 got"改为
     * "16kHz 输出 n16"！原逻辑按 got<48000 循环，降采样每块输出 667 个
     * （1000×2/3 向上取整），48 块 = 32016 采样 > wav 容量 32000 →
     * 最后 32 字节写到 malloc 块外 → 堆元数据破坏 → free/malloc 崩溃 →
     * 系统死、回放无声。现在 n16 到 32000 即停，且每块 mi 按剩余容量
     * 限制，保证降采样输出绝不越界。 */

    while (n16 < want_out)
      {
        uint32_t n_raw;
        uint32_t mi;
        uint32_t j;
        uint32_t n3;
        clock_t rt0;
        clock_t rt1;

        /* ⚠️ 第一个 chunk 只做热身（丢弃）——采集开头有起始瞬态
         * （ES7210 HPF/VMID 对输入阶跃的响应），跳过以免污染数据。 */

        if (first_chunk)
          {
            first_chunk = false;
            continue;
          }

        /* 诊断：记录每次读取耗时 + 前几个 chunk 的数据指纹 */

        rt0 = clock_systime_ticks();
        ret = hal_i2s_read_slot(chunk48, chunk_samples * 2, 0);
        rt1 = clock_systime_ticks();

#if 0  /* ⚠️ 诊断打印已关闭：USB-Serial-JTAG 输出积压会阻塞主线程（假卡死） */
        printf("[Voice] RD#%u done ret=%d\n", (unsigned)(got / 1000), ret);
#endif

        if (ret <= 0)
          {
            printf("[Voice] RX error: %d\n", ret);
            break;
          }

#if 0  /* 诊断：前几个 chunk 的数据指纹（输出积压会假卡死，已关闭） */
        if ((got % (32 * 500)) == 0)
          {
            uint32_t csum = 0;
            uint32_t ck;

            for (ck = 0; ck < 64 && ck < (uint32_t)ret / 2; ck++)
              {
                csum += (uint32_t)(uint16_t)chunk48[ck];
              }

            printf("[Voice][DIAG] chunk@%u: 读取耗时=%lu ticks 前64采样指纹=%u\n",
                   (unsigned)got,
                   (unsigned long)(rt1 - rt0),
                   csum);
          }

        /* 心跳（每 32 块）：tick 值证明系统还活着（控制台冻结 ≠ 系统卡死） */

        if ((got % (32 * 500)) == 0)
          {
            printf("[Voice] 心跳: got=%u/%u tick=%lu\n",
                   (unsigned)got, (unsigned)want_samples,
                   (unsigned long)clock_systime_ticks());
          }
#endif

        n_raw = (uint32_t)ret / 2;


        /* ⚠️ 2026-08-21 定案：录音改【单槽】路径！
         * 4 槽全采流带起始 DC 阶跃 + 确定性尖峰（污染槽检测与 WAV）；
         * 单槽（只收 slot0=MIC1）流实测干净（静音均值 0、说话时波形
         * 大幅上升，mictest 验证）。2026-08-25 迁移官方驱动后，
         * read_slot 直接返回单声道 MIC1 24k 连续流，
         * 全量提取后 3:2 线性插值 → 16kHz WAV。 */

        mi = n_raw;

        /* ⚠️ 2026-08-21 修复：每块丢弃开头+末尾瞬态！
         * 1) 开头：每次 hal_i2s_read_slot（RX 复位+重启）产生起始瞬态
         *    （HPF 阶跃响应，实测 7800=+30720 起，持续 ~数十采样）。
         * 2) 末尾：实测每块 864-999 区间有 ~31 个大值（>2000，23% 密度）
         *    ——DMA 传输结束/块边界的噪声突发。
         * 两者都会造成"兹拉/吃啦"咔哒。丢弃开头 128 + 末尾 160 采样。 */

        {
          uint32_t remain_out = want_out - n16;

          /* ⚠️ 2026-09-14 第二十六轮：同 stream 路径，不再每块丢 288/1000
           * 采样（详见 ai_voice_stream_record 里的根因说明）。 */

          mi = n_raw;
          if (mi > remain_out * 3 / 2)
            {
              mi = remain_out * 3 / 2;
            }
        }

        /* 24kHz → 16kHz：线性插值（每 3 输入 → 2 输出）。
         * 输出 j 对应输入位置 j*1.5（2026-08-25 迁移官方驱动后：
         * read_slot 已是连续单声道 24k 流，直接索引即可）。 */

        n3 = mi * 2 / 3;
        for (j = 0; j < n3; j++)
          {
            uint32_t pos2 = j * 3;               /* j*1.5 的 2 倍定点 */
            uint32_t p0 = pos2 / 2;              /* floor(24k 位置) */
            uint32_t frac = pos2 & 1;            /* 0 或 1（对应 .0/.5） */
            int32_t s0 = chunk48[p0];
            int32_t s1 = chunk48[p0 + 1];

            pcm16[n16 + j] = (int16_t)((s0 * (2 - (int32_t)frac) +
                                        s1 * (int32_t)frac) / 2);
          }

        /* ⚠️ 2026-08-24 修复：高通去 DC + esp_sr NSNet2 神经网络降噪。
         * 手工滤波（中值/低通/噪声门）只能对付固定特征噪声，对
         * "语音频段宽带噪声"无效——改用 esp_sr NSNet2（神经网络降噪，
         * esp_nn 优化，S3 实时；RNNoise 纯 C GRU 实测卡死已弃用）。
         * 处理链：高通220Hz（去 DC/电源噪声）→ NSNet2（帧流水线）。 */

        {
          static int32_t hp_y = 0;
          static int32_t hp_prev = 0;
#ifndef PLANT_NO_NS
          static int16_t ns_in[1024];
          static uint32_t ns_in_cnt = 0;
          static uint32_t ns_total = 0;   /* 全局采样计数（写回定位） */
          const int ns_frame = ai_ns_frame_size();  /* NSNet2 帧大小 */
#else
          /* ⚠️ 2026-09-01：无 NS 版已删破坏性滤波链（中值/低通/限幅/软门），
           * 只保留高通 80Hz 去 DC（上面处理），裸 PCM 直传服务器降噪。 */
#endif
          uint32_t hj;

          for (hj = 0; hj < n3; hj++)
            {
              int32_t s = pcm16[n16 + hj];

              /* 高通：α=1/64（截止 ~80Hz），滤 DC + 电源噪声
               * ⚠️ 2026-09-01：原 α=1/16（220Hz）会削男声基频 100-150Hz，
               * 与删破坏性滤波链一起改为 80Hz。 */
              hp_y += s - hp_prev;
              hp_y -= hp_y >> 6;
              s = hp_y;
              hp_prev = s;

#ifndef PLANT_NO_NS
              pcm16[n16 + hj] = (int16_t)s;

              /* 喂入 NSNet2 流水线（ns_frame 采样/帧 @16k） */
              ns_in[ns_in_cnt++] = (int16_t)s;
              ns_total++;

              if (ns_in_cnt == (uint32_t)ns_frame)
                {
                  ai_ns_process_frame(ns_in);
                  /* 降噪结果写回对应输入位置（延迟 ns_frame 采样，原位替换） */
                  memcpy(&pcm16[ns_total - ns_frame], ns_in, ns_frame * 2);
                  ns_in_cnt = 0;
                }
#else
              /* ⚠️ 2026-09-01 删破坏性滤波链（同 stream_record）：11点中值
               * 抹清辅音、软门削词尾、限幅削顶 → 语音变"咕噜"。
               * 只保留高通 80Hz（上面已做，α=1/64），裸 PCM 直传，
               * 降噪由服务器负责（server_bridge.py noisereduce）。 */
              pcm16[n16 + hj] = (int16_t)s;
#endif
            }
        }

        n16 += n3;
        got += mi;
      }

    printf("[Voice] 录音耗时≈%lu ms (got=%u, 期望~%u ms 若RX=24kHz)\n",
           (unsigned long)(clock_systime_ticks() - t0) * 1000 / TICK_PER_SEC,
           (unsigned)got, (unsigned)(seconds * 1000));
  }

  free(chunk48);
  if (n16 == 0)
    {
      return -EIO;
    }

  /* 数据分析：判断录到的是真音频还是全零/静音。
   * 峰值 < 50 → 基本是静音/零（ES7210 没输出/没时钟/通路断）；
   * 峰值上千且有波动 → 真音频。 */

  {
    int32_t ssum = 0;
    int32_t spea = 0;
    uint32_t snonzero = 0;
    uint32_t spike_cnt = 0;
    uint32_t spike_first = 0;
    int16_t  spike_val = 0;
    uint32_t k;

    for (k = 0; k < n16; k++)
      {
        int32_t s = pcm16[k];
        ssum += s;
        if (s != 0)
          {
            snonzero++;
          }

        if (s > spea)
          {
            spea = s;
          }
        else if (-s > spea)
          {
            spea = -s;
          }

        /* 诊断：统计残留尖峰（|s|>2000，限幅后应很少） */
        if (s > 2000 || s < -2000)
          {
            spike_cnt++;
            if (spike_first == 0)
              {
                spike_first = k;
                spike_val = (int16_t)s;
              }
          }
      }

    printf("[Voice] 录音统计: %u 采样 非零=%u 峰值=%d 均值=%ld (%s)\n",
           (unsigned)n16, (unsigned)snonzero, (int)spea,
           (long)(ssum / (int32_t)n16),
           spea > 50 ? "有信号" : "静音/全零 ← ES7210 通路疑点");
    printf("[Voice] 残留尖峰(>2000): %u 个, 首个@采样%u 值=%d\n",
           (unsigned)spike_cnt, (unsigned)spike_first, (int)spike_val);

    /* 2026-08-25 路径A：底噪量化（RMS + |s| 分位数）——定门限的数据依据。
     * 安静时 P99/P999 ≈ 底噪上界；说话时 P50 应远高于 P99。
     * ⚠️ 2026-08-25 优化：qsort 全量排序（malloc 64KB）改为 256 桶直方图
     * （128 步进）——零大分配，与 NS 共存时不挤占内部 RAM（NS 加载后
     * record 的 64KB 分配曾导致内存紧张 → RX 超时）。 */

    {
      static uint32_t hist[256];
      static const uint32_t pidx[4] = {50, 90, 99, 999};
      uint64_t sq = 0;
      uint32_t k2;
      uint32_t cum = 0;
      uint32_t target[4];
      uint32_t pval[4];
      int p;

      memset(hist, 0, sizeof(hist));
      target[0] = n16 / 2;
      target[1] = n16 * 9 / 10;
      target[2] = n16 * 99 / 100;
      target[3] = n16 * 999 / 1000;

      for (k2 = 0; k2 < n16; k2++)
        {
          int32_t s = pcm16[k2];
          int32_t a = (s < 0) ? -s : s;

          sq += (uint64_t)a * (uint64_t)a;
          hist[(uint32_t)a >> 7]++;   /* 128 步进，256 桶覆盖 0-32767 */
        }

      p = 0;
      for (k2 = 0; k2 < 256 && p < 4; k2++)
        {
          cum += hist[k2];

          while (p < 4 && cum > target[p])
            {
              pval[p] = k2 * 128 + 63;   /* 桶中值 */
              p++;
            }
        }

      printf("[Voice] 幅度统计: RMS=%lu | P50=%d P90=%d P99=%d P999=%d\n",
             (unsigned long)((uint32_t)sqrt((double)sq / n16)),
             (int)pval[0], (int)pval[1], (int)pval[2], (int)pval[3]);
    }
  }

  return ai_voice_wav_encode(wav, pcm16, n16 * 2);
}

/****************************************************************************
 * 堆健康守卫（WROOM 512KB SRAM 资源调度策略的一部分）
 *
 * 语音链路本体已全静态化（零大 malloc），唯一堆需求是 DMA apb
 * （~4KB/个）。堆余量 <8KB 时这些分配会偶发失败 → 提前警告，
 * 而不是让用户在"录音没数据/播放失败"里猜原因。
 ****************************************************************************/

static void ai_voice_heap_warn(FAR const char *op)
{
  struct mallinfo mi = mallinfo();

  if (mi.fordblks < 8192)
    {
      printf("[Voice] ⚠️ %s: 堆余量不足 (free=%uB largest=%uB)，"
             "可能偶发失败。建议 plant mem 查看/关闭相机\n",
             op, (unsigned)mi.fordblks, (unsigned)mi.mxordblk);
    }
}

/****************************************************************************
 * 公共 API：流式录音（语音链路，设备零大缓冲）
 *
 * 与 ai_voice_record 同源逻辑（read_slot → 丢头尾 → 3:2 降采样 → 高通
 * → 滤波链），但每 ~100ms 把 16kHz mono PCM 块回调给 cb——不攒 WAV。
 *
 * 实现要点：
 *  - 独立 static 滤波状态（与 record 的函数内 static 互不干扰）；
 *  - 输出块 static（100ms=1600 采样=3200B，不占栈/堆大块）；
 *  - 能量 VAD：回调块 RMS 低于阈值持续 silent_blocks 块 → 提前结束
 *    （silent_blocks=0 禁用，纯固定时长）；
 *  - 外部取消：ai_voice_stream_cancel() 可在任意时刻停止（微信式随放随停）；
 *  - ⚠️ 不用 NSNet2 降噪：其流水线写回依赖连续缓冲，与分块回调冲突
 *    （NS 留待服务器侧或延迟线方案）。
 ****************************************************************************/

static volatile bool g_stream_cancel;
static volatile bool g_stream_finish;

void ai_voice_stream_cancel(void)
{
  g_stream_cancel = true;
}

/* ⚠️ 2026-09-11：把取消标志重新起算。
 * g_stream_cancel 是「播放/录音循环」专用标志，只有循环内部会消费。
 * 录音结束到真正开始播放之间还有一段 TTS 下载，若下载期间用户点了
 * 一下（此时不该算「停止播放」），旧代码会让播放循环第一次判断就
 * break → 播放耗时=0 ticks、喇叭完全没声。开始播放前调本函数清零
 * 即可把「再点=停」的起算点对齐到真正出声那一刻。 */

void ai_voice_stream_reset_cancel(void)
{
  g_stream_cancel = false;
}

/* ⚠️ 2026-08-31 微信式「再点=说完」：结束录音（保留已录内容，正常发送），
 * 区别于 ai_voice_stream_cancel（取消=丢弃）。录音循环看到 finish 提前
 * break，返回已录音时长，voice_worker 照常 finalize 发给 AI。 */

void ai_voice_stream_finish(void)
{
  g_stream_finish = true;
}

int ai_voice_stream_record(ai_voice_stream_cb_t cb, void *arg,
                           int max_seconds, int silent_blocks)
{
  /* 独立滤波状态（stream 专用，与 record 隔离）——
   * ⚠️ 2026-09-01 删破坏性滤波链后仅剩高通去 DC 状态 */
  static int32_t s_hp_y = 0;
  static int32_t s_hp_prev = 0;

  /* 输出块：100ms @16kHz = 1600 采样 = 3200B（static，零分配）
   * ⚠️ 2026-09-02 越界修复：每块输出 n3=474（输入 1000），out_n 序列
   * 0,474,948,1422,1896 → 第 4 块写 s_out[1422..1896) 超 1600 越界 296
   * 个元素（592B）→ 写穿 s_chunk48 头部 → 回调 samples=1896 又越界读
   * → 结构性零值/统计错乱（diag 说话段非零率恒定 69% 的真凶之一）。
   * 容量加大到 2048（≥1896 峰值），回调按实际 out_n（≤1896）统计。 */
  static int16_t s_out[2048];

  /* 24kHz 读缓冲（static：WiFi 连接后堆被动态缓冲吃紧，语音链路零 malloc） */
  static int16_t s_chunk48[1000];

  FAR int16_t *chunk48;
  uint32_t chunk_samples;
  uint32_t out_n = 0;
  uint32_t want_out;
  uint32_t n16 = 0;
  uint32_t got = 0;
  bool first_chunk = true;
  int silent_cnt = 0;
  int ret;

  g_stream_cancel = false;   /* 每次录音开始清取消标志 */
  g_stream_finish = false;   /* 每次录音开始清「说完」标志 */

  if (cb == NULL)
    {
      return -EINVAL;
    }

  ai_voice_heap_warn("录音");

  if (max_seconds <= 0 || max_seconds > AI_VOICE_MAX_SECONDS)
    {
      max_seconds = AI_VOICE_MAX_SECONDS;
    }

  ret = ai_voice_init();
  if (ret < 0)
    {
      return ret;
    }

  chunk_samples = 1000;   /* 24kHz 2000B/读，同 record */
  chunk48 = s_chunk48;    /* static，零 malloc */

  want_out = AI_VOICE_AI_RATE * (uint32_t)max_seconds;

  while (n16 < want_out)
    {
      uint32_t n_raw;
      uint32_t mi;
      uint32_t j;
      uint32_t n3;

      /* 外部取消（取消=丢弃，如 THINKING 阶段再点）：
       * 再点按键 → 立即停止录音 */
      if (g_stream_cancel)
        {
          break;
        }

      /* ⚠️ 2026-08-31 微信式「再点=说完」：结束录音（保留已录内容，
       * 返回正时长 → worker 照常 finalize 发给 AI） */
      if (g_stream_finish)
        {
          break;
        }

      if (first_chunk)
        {
          first_chunk = false;
          continue;
        }

      ret = hal_i2s_read_slot(chunk48, chunk_samples * 2, 0);
      if (ret <= 0)
        {
          break;
        }

      n_raw = (uint32_t)ret / 2;

      /* ⚠️ 2026-09-14 第二十六轮根因修复：这里原来每块丢头 128 + 尾 160
       * 采样（共 288/1000 = 28.8%）——那是 2026-08-21 针对“48kHz 4 槽 TDM
       * 流”的补丁；2026-08-25 迁移官方驱动后 read_slot 已是连续单声道
       * 24k 流，没有块边界瞬态，再丢就是挖掉 29% 的音频。实测铁证：
       * 录“2.00 s”墙钟 2800 ms（= 2 s × 1000/712 = 2.81 s），即录音被时间压缩
       * 1.41×（音调升高 41%）+ 每 41.7 ms 一个接缝 → 听起来“吃啦/杂音”。
       * 修复：整块使用（开头的启动瞬态已由上面 first_chunk 跳过）。 */
      {
        uint32_t remain_out = want_out - n16;

        mi = n_raw;
        if (mi > remain_out * 3 / 2)
          {
            mi = remain_out * 3 / 2;
          }
      }

      /* 24k → 16k 线性插值（同 record） */
      n3 = mi * 2 / 3;
      for (j = 0; j < n3; j++)
        {
          uint32_t pos2 = j * 3;
          uint32_t p0 = pos2 / 2;
          uint32_t frac = pos2 & 1;
          int32_t s0 = chunk48[p0];
          int32_t s1 = chunk48[p0 + 1];

          s_out[out_n + j] = (int16_t)((s0 * (2 - (int32_t)frac) +
                                        s1 * (int32_t)frac) / 2);
        }

      /* ⚠️ 2026-09-01 删破坏性滤波链：11点中值（非线性破坏语音，抹清辅音）
       * + 低通（二次平滑）+ 限幅±30000（37.5dB 增益下削顶失真）+
       * 软噪声门（|s|<500 衰减 1/4，削词首/词尾）——实测把语音变成
       * "咕噜咕噜"（用户听 before.wav 完全听不到字）。
       * 保留：高通 80Hz（α=1/64，去 DC/电源噪声，不伤语音频段）。
       * 裸 PCM 直传服务器，降噪由服务器负责（server_bridge.py noisereduce）。 */
      for (j = 0; j < n3; j++)
        {
          int32_t s = s_out[out_n + j];

          /* 高通：α=1/64（~80Hz 去 DC/电源噪声；220Hz 会削男声基频） */
          s_hp_y += s - s_hp_prev;
          s_hp_y -= s_hp_y >> 6;
          s = s_hp_y;
          s_hp_prev = s;

          s_out[out_n + j] = (int16_t)s;
        }

      out_n += n3;
      n16 += n3;
      got += mi;

      /* 输出块满（~100ms=1600 采样）→ 回调 */
      if (out_n >= 1600)
        {
          /* 能量 VAD（2026-08-31 修复：峰值对偶发尖峰太敏感 → 静音永不触发；
           * 改用均方（mean-square）——多数采样幅度低才算静音，单个尖峰不影响）。
           * 静音判定：mean-square < 300²=90000（对应 RMS<300，
           * record 静音基线 RMS≈284 → mean-square≈80656，说话时远大于 90000）。
           * ⚠️ 2026-08-31 二修：原阈值 300*300/1000=90 是笔误（小 1000 倍），
           * 导致只有 RMS<9.5 才判静音 → 静音永不触发、总是录满 5 秒。
           * 连续 silent_blocks 块（1.5s）→ 说完自动停。 */

          int64_t sq = 0;
          uint32_t k;

          for (k = 0; k < out_n; k++)
            {
              sq += (int64_t)s_out[k] * s_out[k];
            }

          if (silent_blocks > 0)
            {
              int32_t msq = (int32_t)(sq / out_n);   /* 均方（非 RMS） */

              if (msq < 300 * 300)                   /* 静音（RMS 阈值 ≈300） */
                {
                  silent_cnt++;
                }
              else
                {
                  silent_cnt = 0;
                }
            }

          cb(s_out, out_n, arg);
          out_n = 0;

          if (silent_blocks > 0 && silent_cnt >= silent_blocks)
            {
              break;   /* 静音足够久 → 说完自动停 */
            }
        }
    }

  /* 尾部不足一块的数据也回调（不丢尾） */
  if (out_n > 0 && cb != NULL)
    {
      cb(s_out, out_n, arg);
    }

  return (int)(n16 * 1000 / AI_VOICE_AI_RATE);   /* 录音时长 ms */
}


/****************************************************************************
 * 公共 API：播放（WAV → 24kHz DMA → 喇叭）
 *
 * 注意：当前 I2S 以 24kHz 运行，16kHz WAV 播放出来会快 3 倍。
 * 后续需要加 16k→48k 上采样。
 ****************************************************************************/

int ai_voice_play(FAR const uint8_t *wav, size_t size)
{
  FAR const uint8_t *p;
  int data_bytes = 0;
  bool found = false;
  int ret;

  if (size < 44 || memcmp(wav, "RIFF", 4) != 0)
    {
      return -EINVAL;
    }

  /* 找 WAV 的 "data" chunk。
   * ⚠️ 必须从 wav+12 开始（跳过 "RIFF"+size+"WAVE"）！
   * 之前从 wav[0] 开始，把 RIFF 头当 chunk 遍历，
   * 一步跳过整个文件 → 永远找不到 data → 静默 -EINVAL → 无声。 */

  p = wav + 12;
  while (p + 8 <= wav + size)
    {
      uint32_t csize = (uint32_t)p[4] | ((uint32_t)p[5] << 8) |
                       ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);

      if (memcmp(p, "data", 4) == 0)
        {
          data_bytes = (int)csize;
          p += 8;
          found = true;
          break;
        }

      /* 跳过当前 chunk（含奇数字节补齐） */

      p += 8 + csize + (csize & 1);
    }

  if (!found || p + data_bytes > wav + size)
    {
      printf("[Voice] 播放失败: WAV data chunk 未找到/越界 (size=%u)\n",
             (unsigned)size);
      return -EINVAL;
    }


  ret = ai_voice_init();
  if (ret < 0)
    {
      return ret;
    }

  voice_agent_set_dac_mute(0);

  /* 重采样：16kHz 单声道 WAV → 24kHz 立体声帧（L=R），线性插值。
   * I2S TX 以 24kHz 帧 × 2 槽 = 48k 采样/s 输出，原样喂 16k 单声道
   * 会 3 倍速播放（花栗鼠音）。输出帧 j 对应输入位置 j*2/3。 */

  {
    static int16_t in[800];      /* 栈上 1600B 会压栈，改 static */
    static int16_t outbuf[2400]; /* 800 输入采样 → 1200 帧 × 2 槽 */
    int sent = 0;                /* 已消费的输入字节数 */

    while (sent < data_bytes)
      {
        int n_in;
        int n_out = 0;
        int j;

        n_in = (data_bytes - sent) / 2;
        if (n_in > 800)
          {
            n_in = 800;
          }

        /* 防死循环：无有效采样则结束 */

        if (n_in <= 0)
          {
            break;
          }

        memcpy(in, p + sent, (size_t)n_in * 2);

        for (j = 0; j < n_in * 3 / 2; j++)
          {
            uint32_t pos = (uint32_t)j * 2;   /* 24k 帧 j ↔ 16k 位置 2j/3 */
            uint32_t i = pos / 3;
            uint32_t frac = pos % 3;
            int16_t a = in[i];
            int16_t b = (i + 1 < (uint32_t)n_in) ? in[i + 1] : a;
            int16_t s;

            if (frac == 0)
              {
                s = a;
              }
            else
              {
                s = (int16_t)(((int32_t)a * (3 - (int32_t)frac) +
                               (int32_t)b * (int32_t)frac) / 3);
              }

            outbuf[n_out++] = s;   /* L */
            outbuf[n_out++] = s;   /* R */
          }

        /* ⚠️ 2026-09-02 播放"卡成怪兽"修复：逐块 hal_i2s_write（阻塞等
         * 全部发完）→ 块间 DMA 空窗（flush 等 EOF + worker 往返 + 下块
         * 准备时间）→ 真实语音被切段（纯音听不出、人声"卡"）。改 async
         * 排队：排队时驱动已 memcpy 拷走数据，生成下一块时上一块仍在播，
         * DMA 由 ISR 无缝续链；函数退出前统一 flush。 */

        ret = hal_i2s_write_async(outbuf, (uint32_t)n_out * 2);
        if (ret < 0)
          {
            printf("[Voice] 播放失败: hal_i2s_write_async=%d (%s) "
                   "(sent=%d/%d)\n",
                   ret, strerror(-ret), sent, data_bytes);
            hal_i2s_write_flush();
            voice_agent_set_dac_mute(1);
            return ret;
          }

        sent += n_in * 2;

        /* 调试：第一块打印重采样输出的统计（确认播放数据是真实音频） */

        if (sent == n_in * 2)
          {
            uint32_t dnz = 0;
            int32_t dpeak = 0;
            int32_t dsum = 0;
            int dj;

            for (dj = 0; dj < n_out; dj++)
              {
                int32_t ds = outbuf[dj];
                if (ds != 0)
                  {
                    dnz++;
                  }

                dsum += ds;
                if (ds > dpeak)
                  {
                    dpeak = ds;
                  }
                else if (-ds > dpeak)
                  {
                    dpeak = -ds;
                  }
              }

            printf("[Voice] 播放数据: 输出=%d 采样 非零=%u(%u%%) 峰值=%d 均值=%ld\n",
                   n_out, dnz, (unsigned)(dnz * 100 / (uint32_t)n_out),
                   (int)dpeak, (long)(dsum / n_out));
          }
      }

  printf("[Voice] 播放完成\n");
  }

  /* 全部入队后等 DMA 发完（复用缓冲/换方向前必须 flush） */

  hal_i2s_write_flush();
  voice_agent_set_dac_mute(1);

  return 0;
}

/****************************************************************************
 * 公共 API：从文件流式播放 WAV（TTS 音频可达数百 KB，不整段进内存）
 *
 * 与 ai_voice_play 同逻辑（找 data chunk → 16k→24k 重采样 → hal_i2s_write），
 * 但数据从文件分块读（~1.6KB/块），内存占用恒定。
 ****************************************************************************/

int ai_voice_play_file(FAR const char *wav_path)
{
  static int16_t in[800];
  static int16_t outbuf[2400];
  uint8_t hdr[64];
  uint32_t data_off = 0;
  uint32_t data_bytes = 0;
  uint32_t sample_rate = 0;   /* 从 fmt chunk 读，TTS=24000（匹配 I2S 24k）*/
  uint32_t sent = 0;
  FILE *fp;
  int ret;

  if (wav_path == NULL)
    {
      return -EINVAL;
    }

  ai_voice_heap_warn("播放");

  fp = fopen(wav_path, "rb");
  if (fp == NULL)
    {
      return -errno;
    }

  if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "RIFF", 4) != 0)
    {
      fclose(fp);
      return -EINVAL;
    }

  /* 遍历 chunk 找 data + fmt（文件流式，只读 chunk 头 + fmt 采样率） */

  for (;;)
    {
      uint8_t ch[8];

      if (fread(ch, 1, 8, fp) != 8)
        {
          fclose(fp);
          return -EINVAL;
        }

      {
        uint32_t csize = (uint32_t)ch[4] | ((uint32_t)ch[5] << 8) |
                         ((uint32_t)ch[6] << 16) | ((uint32_t)ch[7] << 24);

        if (memcmp(ch, "fmt ", 4) == 0 && csize >= 8)
          {
            uint8_t fhdr[8];

            /* fmt 数据：audio_format(2)+channels(2)+sample_rate(4) */
            if (fread(fhdr, 1, 8, fp) == 8)
              {
                sample_rate = (uint32_t)fhdr[4] | ((uint32_t)fhdr[5] << 8) |
                              ((uint32_t)fhdr[6] << 16) |
                              ((uint32_t)fhdr[7] << 24);
              }

            fseek(fp, (long)csize - 8 + (csize & 1), SEEK_CUR);
          }
        else if (memcmp(ch, "data", 4) == 0)
          {
            data_off = (uint32_t)ftell(fp);
            data_bytes = csize;
            break;
          }
        else
          {
            fseek(fp, (long)csize + (csize & 1), SEEK_CUR);
          }
      }
    }

  /* ⚠️ 2026-08-31 修复：MiMo TTS 输出 24000Hz（check_wav 实测），
   * I2S 也跑 24kHz → 24k 直接播放（L=R 复制，无需重采样）。
   * 旧代码假设 16kHz 做 3:2 重采样 → 24k 数据被错位 → 无声/音调错。
   * 16kHz（如旧 WAV/录音）才走 3:2 插值。 */

  if (sample_rate != 24000 && sample_rate != 16000)
    {
      printf("[Voice] 播放警告: 未知采样率 %u Hz，按 24k 原样播放\n",
             (unsigned)sample_rate);
    }

  printf("[Voice] 播放 %s: %u Hz, %uB data\n",
         wav_path, (unsigned)sample_rate, (unsigned)data_bytes);

  ret = ai_voice_init();
  if (ret < 0)
    {
      fclose(fp);
      return ret;
    }
  voice_agent_set_dac_mute(0);

  fseek(fp, (long)data_off, SEEK_SET);

  {
    uint32_t t0 = clock_systime_ticks();

    while (sent < data_bytes)
      {
      int n_in;
      int n_out = 0;
      int j;
      size_t got;

      /* 播放中也支持取消（微信式随放随停：TTS 播放到一半再点按键 → 立即停） */

      if (g_stream_cancel)
        {
          printf("[Voice] 播放被取消（用户点按停止）\n");
          break;
        }

      n_in = (data_bytes - sent) / 2;
      if (n_in > 800)
        {
          n_in = 800;
        }

      if (n_in <= 0)
        {
          break;
        }

      got = fread(in, 2, (size_t)n_in, fp);
      if (got == 0)
        {
          break;
        }

      n_in = (int)got;

      if (sample_rate == 16000)
        {
          /* 16k → 24k：3:2 线性插值（同 ai_voice_play） */

          for (j = 0; j < n_in * 3 / 2; j++)
            {
              uint32_t pos = (uint32_t)j * 2;
              uint32_t i = pos / 3;
              uint32_t frac = pos % 3;
              int16_t a = in[i];
              int16_t b = (i + 1 < (uint32_t)n_in) ? in[i + 1] : a;
              int16_t s;

              if (frac == 0)
                {
                  s = a;
                }
              else
                {
                  s = (int16_t)(((int32_t)a * (3 - (int32_t)frac) +
                                 (int32_t)b * (int32_t)frac) / 3);
                }

              outbuf[n_out++] = s;
              outbuf[n_out++] = s;
            }
        }
      else
        {
          /* 24k（TTS）：直接 L=R 复制到 24k stereo 帧，无需重采样 */

          for (j = 0; j < n_in; j++)
            {
              outbuf[n_out++] = in[j];
              outbuf[n_out++] = in[j];
            }
        }

      /* ⚠️ 2026-09-02 播放"卡"修复：同 ai_voice_play——逐块阻塞改
       * async 排队（驱动已拷走数据 → 生成下一块时上一块在播，ISR 无缝
       * 续链），消除每 50ms 块边界的 DMA 空窗。文件读取/重采样耗时被
       * async 节流吸收，不再造成可听停顿。 */

      ret = hal_i2s_write_async(outbuf, (uint32_t)n_out * 2);
      if (ret < 0)
        {
          /* ⚠️ hal_i2s_write_async 返回真实错误码（不再折叠成 -EIO），
           * 便于区分 -ETIMEDOUT(DMA 未完成)/-ENOMEM(缓冲池) 等根因 */

          printf("[Voice] 文件播放失败: hal_i2s_write_async=%d (%s)\n",
                 ret, strerror(-ret));
          hal_i2s_write_flush();
          fclose(fp);
          return ret;
        }

      /* ⚠️ 2026-09-02 播放诊断：第一块打印实际播放数据统计——
       * 若此处非零/峰值正常但喇叭无声 → 问题在码片/功放/DMA 数据未
       * 真正串行化（查 [I2S] TX 状态 + ES8311 寄存器，plant voice regs）；
       * 若此处全零 → 问题在录音/WAV 数据（信号统计应为非零）。 */

      if (sent == 0)
        {
          uint32_t dnz = 0;
          int32_t dpeak = 0;
          int32_t dsum = 0;
          int dj;

          for (dj = 0; dj < n_out; dj++)
            {
              int32_t ds = outbuf[dj];

              if (ds != 0)
                {
                  dnz++;
                }

              dsum += ds;
              if (ds > dpeak)
                {
                  dpeak = ds;
                }
              else if (-ds > dpeak)
                {
                  dpeak = -ds;
                }
            }

          printf("[Voice] 播放数据: 输出=%d 采样 非零=%u(%u%%) 峰值=%d "
                 "均值=%ld\n",
                 n_out, dnz, (unsigned)(dnz * 100 / (uint32_t)n_out),
                 (int)dpeak, (long)(dsum / n_out));
        }

      sent += (uint32_t)n_in * 2;
    }

    /* 全部入队后等 DMA 发完（复用缓冲/换方向前必须 flush） */

    hal_i2s_write_flush();
    voice_agent_set_dac_mute(1);

    {
      uint32_t elapsed = clock_systime_ticks() - t0;
      uint32_t expect;

      /* 理论播放时长：data_bytes 采样 @sample_rate Hz → 秒 → ticks。
       * 96000B@16k mono = 48000/16000 = 3s = 300 ticks；24k = 2s = 200。 */

      expect = (uint32_t)(data_bytes / 2) * 100 / sample_rate;

      /* ⚠️ 播放耗时判读（10ms/tick）：实际明显 > 理论+10% → 播放仍有
       * 停顿（SD 读取/worker 延迟/块间隙）；≈ 理论 → 实时连续。 */

      printf("[Voice] 播放耗时=%u ticks (理论≈%u, %s)\n",
             elapsed, expect,
             elapsed <= expect + expect / 10 ? "实时连续" : "有停顿!");
    }
  }

  fclose(fp);
  printf("[Voice] 文件播放完成 (%s, %uB)\n", wav_path, (unsigned)data_bytes);
  return 0;
}

/****************************************************************************
 * 公共 API：播放测试音（正弦波）
 ****************************************************************************/

int ai_voice_play_tone(uint32_t freq_hz, int ms)
{
  /* ⚠️ 2026-09-02 WROOM 内存修复：旧实现 malloc(48KB) 在 512KB SRAM
   * 堆耗尽时失败（实测 tone ret=-12 → 无声假象）。改【小块 static + 分块发送】：
   * 50ms/块 = 4.8KB（比 100ms/块再省 4.8KB），与录音/播放链路同策略。 */

  /* ⚠️ 2026-09-02 v2（"吱吱吱嘟嘟"修复）：分块发送改【异步排队 + 末尾
   * flush】——hal_i2s_write_async 排队即回（驱动已拷走数据，可复用本
   * 缓冲），生成下一块时上一块仍在播；块间由官方驱动 ISR 无缝续链，
   * 消除"等完再发"造成的 50ms 块边界 DMA 空窗（纯音被切段 → 嘟嘟）。
   * 旧 hal_i2s_write（阻塞等完）对纯音必现块间掉拍。 */

  static int16_t tone[1200 * 2];   /* 50ms × 24k × 2ch = 4.8KB */
  uint32_t n_total;
  uint32_t sent_samp = 0;
  int ret;

  ret = ai_voice_init();
  if (ret < 0)
    {
      return ret;
    }
  voice_agent_set_dac_mute(0);

  if (ms <= 0 || ms > 1000)
    {
      ms = 500;
    }

  n_total = (uint32_t)ms * AI_VOICE_SAMPLE_RATE / 1000;

  while (sent_samp < n_total)
    {
      uint32_t n = n_total - sent_samp;
      uint32_t i;

      if (n > 1200)
        {
          n = 1200;
        }

      for (i = 0; i < n; i++)
        {
          double t = (double)(sent_samp + i) / AI_VOICE_SAMPLE_RATE;
          int16_t sample =
              (int16_t)(30000.0 * sin(2.0 * 3.14159265358979 * freq_hz * t));

          tone[i * 2]     = sample;  /* Left */
          tone[i * 2 + 1] = sample;  /* Right */
        }

      ret = hal_i2s_write_async(tone, n * 2 * 2);
      if (ret < 0)
        {
          hal_i2s_write_flush();
          voice_agent_set_dac_mute(1);
          return ret;
        }

      sent_samp += n;
    }

  /* 全部入队后等 DMA 发完（复用 tone 缓冲 / 换方向前必须 flush） */

  ret = hal_i2s_write_flush();
  voice_agent_set_dac_mute(1);
  if (ret < 0)
    {
      return ret;
    }

  return 0;
}

/****************************************************************************
 * 诊断：I2C 通信测试（验证码片是否在总线上）
 ****************************************************************************/

int ai_voice_i2c_test(void)
{
  FAR struct i2c_master_s *i2c;
  uint8_t v = 0;
  int ret;
  int pass = 0;

  i2c = esp32s3_i2cbus_initialize(I2C_BUS);
  if (i2c == NULL)
    {
      printf("[I2C] 总线初始化失败\n");
      return -ENODEV;
    }

  printf("[I2C] === 音频码片 I2C 应答检测 (bus %d) ===\n", I2C_BUS);

  /* ES8311 (0x18): REG01 = 时钟管理/芯片 ID 区 */

  v = 0;
  ret = hal_i2c_read_reg(i2c, 0x18, 0x01, &v, 100000);
  printf("[I2C] ES8311(0x18) REG01=0x%02x (ret=%d)  %s\n",
         v, ret, (ret == 0 && v != 0xff) ? "✓ 有应答" : "✗ 无应答");
  if (ret == 0 && v != 0xff)
    {
      pass++;
    }

  /* ES8311 (0x18): REG00 = 复位/主从 */

  v = 0;
  ret = hal_i2c_read_reg(i2c, 0x18, 0x00, &v, 100000);
  printf("[I2C] ES8311(0x18) REG00=0x%02x (ret=%d)  %s\n",
         v, ret, (ret == 0 && v != 0xff) ? "✓ 有应答" : "✗ 无应答");
  if (ret == 0 && v != 0xff)
    {
      pass++;
    }

  /* ES7210 (0x40): REG00 = 复位/上电默认 0x41 */

  v = 0;
  ret = hal_i2c_read_reg(i2c, 0x40, 0x00, &v, 100000);
  printf("[I2C] ES7210(0x40) REG00=0x%02x (ret=%d)  %s\n",
         v, ret, (ret == 0 && v != 0xff) ? "✓ 有应答" : "✗ 无应答");
  if (ret == 0 && v != 0xff)
    {
      pass++;
    }

  /* ES7210 (0x40): REG01 = 芯片 ID */

  v = 0;
  ret = hal_i2c_read_reg(i2c, 0x40, 0x01, &v, 100000);
  printf("[I2C] ES7210(0x40) REG01=0x%02x (ret=%d)  %s\n",
         v, ret, (ret == 0 && v != 0xff) ? "✓ 有应答" : "✗ 无应答");

  printf("[I2C] 结果: %d/3 次读操作有应答\n", pass);
  return pass >= 2 ? 0 : -ENODEV;
}
