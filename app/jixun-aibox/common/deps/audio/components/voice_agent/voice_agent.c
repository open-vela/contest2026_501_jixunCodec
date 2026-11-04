/****************************************************************************
 * apps/plant-companion/components/voice_agent/voice_agent.c
 *
 * 外设层：音频子系统（ES8311 DAC + ES7210 ADC + I2S0）的初始化与自检。
 *
 * 职责边界（架构约定）：
 *   - 本层只负责"让外设正常运行"：码片寄存器配置、I2S 时钟/DMA、自检
 *   - ai_module/ai_voice（AI 应用层）只调用本层 API，不做外设寄存器操作
 *   - 本层不包含 AI 逻辑（录音诉求 / 语音回复 / 云端交互）
 *
 * 调用链：
 *   voice_agent_init()      → es8311_init/set_format/start + es7210_init + hal_i2s_init
 *   voice_agent_mic_test()  → 采集原始 I2S RX 并统计（外设级麦克风自检）
 ****************************************************************************/

#include <nuttx/config.h>
#include <stdio.h>
#include <errno.h>
#include <stdbool.h>

#include <nuttx/clock.h>
#include <nuttx/i2c/i2c_master.h>

#include "../../hal/hal_i2c.h"
#include "../../hal/hal_i2s.h"
#include "es8311.h"
#include "es7210.h"
#include "voice_agent.h"

/* I2C 总线初始化（内核符号，链接时解析） */

extern FAR struct i2c_master_s *esp32s3_i2cbus_initialize(int bus);


#define VA_I2C_BUS       0
#define VA_SAMPLE_RATE   24000        /* I2S/码片采样率 */
#define VA_MCLK_HZ       (24000 * 256)  /* 6.154MHz = 160MHz/26（整数分频，
                                         * 零抖动；码片从模式跟随外部时钟） */

static FAR struct i2c_master_s *g_i2c;
static bool g_inited;

/* ES7210 是否存在（探测得到）。小智板 SP-ESP32-S3-1.54-MUMA 没有该芯片，
 * 麦克风直接接在 ES8311 内部 ADC 上；此时不让整条音频链路失败。 */
static bool g_es7210_ok;

/* DAC 输出音量（ES8311 REG32）。
 * 标度：0x00 = -95.5dB，0xFF = +32dB，步进 0.5dB → 0xBF(191) 约 0dB。
 * 2026-09-20：为压低扬声器空闲底噪，默认值改为 0xC0(192) ≈ 0dB。
 * 原值 0xD2(210) ≈ +9.5dB；如现场音量不足，可用 NSH 调回 0xC8/0xD2。
 * 可用 NSH `plant voice vol [0-255]` 运行时微调，不必重烧固件。 */

static uint8_t g_dac_volume = 0xC0;

/****************************************************************************
 * 公共 API：音频外设初始化（幂等）
 *
 * ES8311：上电 → I2S 格式/时钟 → 使能 DAC（含解除 DAC SDP MUTE）→ 音量
 * ES7210：复位 → 时钟 → 从模式 → 电源 → 格式 → MIC 增益 → ADC 使能
 * I2S0：  GPIO 矩阵 + 时钟 + DMA（hal_i2s）
 ****************************************************************************/

/* 2026-10-21: 放开 g_inited，让 jixun 每次启动都完整重配 I2S/码片。
 * 不放开的话第二次录音只走 rearm，ADC 直接全 0。 */
void voice_agent_reset_init(void)
{
  g_inited = false;
}

int voice_agent_init(void)
{
  int ret;

  if (g_inited)
    {
      return 0;
    }

  /* ⚠️ 2026-08-25：NS 版必须在 I2S DMA 配置【之前】映射 PSRAM！
   * 原来在 ai_ns_init（hal_i2s_init 之后）lazy 映射 → 运行中改 MMU
   * 破坏 I2S DMA → RX 3s 超时。提前到最前面映射，I2S 配置时 PSRAM
   * 已就绪。PLANT_NO_NS 版不链接 esp_sr，符号不存在（条件编译）。 */

#ifndef PLANT_NO_NS
  extern int esp_sr_psram_early_init(void);
  esp_sr_psram_early_init();
#endif

  g_i2c = esp32s3_i2cbus_initialize(VA_I2C_BUS);
  if (g_i2c == NULL)
    {
      fprintf(stderr, "[VoiceAgent] I2C%d init failed\n", VA_I2C_BUS);
      return -ENODEV;
    }

  /* ES8311：上电配置 */

  /* 2026-10-22: es8311_init() 返回 1 表示芯片已经上电且配置过（ESP32
   * 热复位场景）。此时绝对不能再写任何 codec 寄存器，只重建 I2S。 */
  int codec_ready = 0;

  ret = es8311_init(g_i2c);
  if (ret == 1)
    {
      codec_ready = 1;
      ret = 0;
      printf("[VoiceAgent] ES8311 kept as-is (already configured)\n");
      /* 2026-10-30: keep-as-is 分支不走 es8311_start，REG32 仍是
       * suspend 留下的 0x00(-95.5dB)，必须在这里恢复音量。 */
      es8311_set_volume(g_i2c, g_dac_volume);
      es8311_mute(g_i2c, true);
      printf("[VoiceAgent] ES8311 volume REG32=0x%02X restored\n",
             g_dac_volume);
    }
  else if (ret < 0)
    {
      fprintf(stderr, "[VoiceAgent] ES8311 init failed: %d\n", ret);
      return ret;
    }

  if (!codec_ready)
    {
      /* ES8311：I2S 格式 + 时钟（12.288MHz / 24kHz / 16-bit） */

      ret = es8311_set_format(g_i2c, VA_SAMPLE_RATE);
      if (ret < 0)
        {
          fprintf(stderr, "[VoiceAgent] ES8311 format failed: %d\n", ret);
          return ret;
        }

      /* ES8311：使能 DAC（内部会解除 REG09 bit6 DAC SDP MUTE） */

      ret = es8311_start(g_i2c);
      if (ret < 0)
        {
          fprintf(stderr, "[VoiceAgent] ES8311 start failed: %d\n", ret);
          return ret;
        }

      es8311_set_volume(g_i2c, g_dac_volume);   /* 见 g_dac_volume */
      es8311_mute(g_i2c, true);   /* 空闲静音，播放前由 ai_voice 解除 */

      printf("[VoiceAgent] ES8311 done, ticks=%lu\n",
             (unsigned long)clock_systime_ticks());
    }

  /* ES7210：ADC 初始化 */

  /* 2026-10-21: 这块小智板没有 ES7210（I2C 全总线只有 0x18）。
   * 每次探测 0x40 都会超时，偶尔把 I2C 总线拖住，导致随后的
   * ES8311 rearm 卡死（日志停在 encoder prewarm done）。直接跳过。 */
  g_es7210_ok = false;
  /* 2026-11-01: 原理图确认本板只有 ES8311（两颗 AOS3729A 模拟麦），
   * 没有 ES7210。探测 0x40 只会等 I2C 超时，拖慢提示音。 */
  g_es7210_ok = false;
  printf("[VoiceAgent] no ES7210 on board, use ES8311 ADC\n");

  /* I2S0：GPIO 矩阵 + 时钟 + DMA + PA 使能 */

  printf("[VoiceAgent] hal_i2s_init entry, ticks=%lu\n",
         (unsigned long)clock_systime_ticks());

  ret = hal_i2s_init();
  if (ret < 0)
    {
      fprintf(stderr, "[VoiceAgent] I2S0 init failed: %d\n", ret);
      return ret;
    }

  printf("[VoiceAgent] hal_i2s_init done, ticks=%lu\n",
         (unsigned long)clock_systime_ticks());

  /* 启动 TX 时钟（BCK/WS 持续输出，RX 跟随时钟） */

  ret = hal_i2s_start_tx_clock();
  if (ret < 0)
    {
      fprintf(stderr, "[VoiceAgent] TX clock start failed: %d\n", ret);
      return ret;
    }

  printf("[VoiceAgent] tx clock done, ticks=%lu\n",
         (unsigned long)clock_systime_ticks());

  /* ⚠️ 2026-08-21 根因修复：MCLK 启动后必须恢复 ES7210 模拟配置！
   * es7210_init 在无 MCLK 状态（GPIO2 尚未配成 I2S0_MCLK）下完成，
   * MCLK 突然出现会把 REG40 bit0(LDO_EN) 清掉（0x43→0x42）→ VMID 未
   * 建立 → ADC 输入共模错误 → 极限环噪声（实测确定性满幅峰值，
   * 麦克风录不到真实声音，只有时钟嘶声）。 */

  if (g_es7210_ok)
    {
      ret = es7210_restore_analog(g_i2c);
      if (ret < 0)
        {
          fprintf(stderr, "[VoiceAgent] ES7210 analog restore failed: %d\n",
                  ret);
        }
    }
  else if (!codec_ready)
    {
      /* 无 ES7210：打开 ES8311 内部 ADC，麦克风数据从 ASDOUT 走 I2S DIN。
       * 芯片已经配置过时不能再写这些模拟电源寄存器。 */

      ret = es8311_adc_enable(g_i2c, true);
      if (ret < 0)
        {
          fprintf(stderr, "[VoiceAgent] ES8311 ADC enable failed: %d\n", ret);
        }
    }

  g_inited = true;
  return 0;
}

/****************************************************************************
 * 公共 API：麦克风外设自检（不经过 AI 层）
 *
 * 直接通过 HAL 采集原始 I2S RX，分别测槽0（左=MIC1）和槽1（右=MIC2），
 * 每个槽输出：
 *   - 前 32 个原始采样（hex）→ 看数据模式（全零/削顶/真实波形）
 *   - 非零率/峰值/均值 → 判定信号强度
 * 最后给出结论：麦克风在哪个槽、是否正常、增益是否合适。
 ****************************************************************************/

/****************************************************************************
 * 公共 API：DAC 输出音量（ES8311 REG32）
 *
 * 0x00 = -95.5dB .. 0xFF = +32dB，0.5dB/step，0xBF ≈ 0dB。
 * 初始化前调用只记值；初始化后调用立即写寄存器（免重烧调音）。
 ****************************************************************************/

int voice_agent_set_dac_volume(uint8_t vol)
{
  g_dac_volume = vol;

  if (g_i2c != NULL)
    {
      return es8311_set_volume(g_i2c, vol);
    }

  return 0;
}

int voice_agent_dac_power_on(void)
{
  if (g_i2c == NULL)
    {
      return -EIO;
    }

  return es8311_dac_power_on(g_i2c);
}

int voice_agent_set_dac_mute(int mute)
{
  if (g_i2c != NULL)
    {
      return es8311_mute(g_i2c, mute != 0);
    }

  return 0;
}

int voice_agent_set_reg44(uint8_t v)
{
  extern int es8311_set_reg44(FAR struct i2c_master_s *i2c, uint8_t v);
  if (g_i2c == NULL)
    {
      return -EIO;
    }
  return es8311_set_reg44(g_i2c, v);
}

int voice_agent_dump_regs(void)
{
  if (g_i2c == NULL)
    {
      printf("[ES8311] I2C 未初始化，读寄存器失败\n");
      return -1;
    }

  return es8311_dump_regs(g_i2c);
}

uint8_t voice_agent_get_dac_volume(void)
{
  return g_dac_volume;
}

bool voice_agent_has_es7210(void)
{
  return g_es7210_ok;
}

/* 2026-10-20：每次录音前重新写一遍 ADC 通路寄存器。
 *
 * 现场：ES8311 内部 ADC 会偶发停止输出，原始 PCM 整段全 0
 * （同一块板连续两次录音，一次 peak 两万多、一次返回 0）。
 * 重新写 REG0A/REG14/REG16/REG17 后立即恢复，说明是码片寄存器
 * 状态丢失，而不是麦克风或 I2S 链路坏掉。初始化只做一次不够，
 * 每次录音前重新使能一次。 */

int voice_agent_rearm_adc(void)
{
  static int s_adc_rearmed;
  int ret;

  if (g_i2c == NULL)
    {
      return -EINVAL;
    }

  /* INMP441 模式：ES8311 内部 ADC 不参与录音，这里什么都不做，
   * 保持 DAC 常开。set JX_MIC_INMP441 1 开启。 */

  {
    const char *inmp = getenv("JX_MIC_INMP441");
    if (inmp != NULL && atoi(inmp) != 0)
      {
        return 0;
      }
  }

  /* 2026-10-22 v10：默认走官方“常开”语义——ADC 已经活着就一个寄存器
   * 都不写。原来每次录音前都 suspend→start，等于每录一次就把模拟前端
   * 拔一次电；ES8311 的 VMID/PGA 在这种反复冷启下很容易闩死。
   * set JX_ADC_MINIMAL 0 可退回旧的每次 full_restart。 */

  {
    const char *minimal = getenv("JX_ADC_MINIMAL");
    if (minimal == NULL || atoi(minimal) != 0)
      {
        if (es8311_adc_is_running(g_i2c))
          {
            return 0;
          }
      }
  }

  /* 2026-10-22: default keep the ADC powered and NEVER soft-reset it
   * during a take. voice_agent_init() already called es8311_adc_enable().
   * The old full_restart path writes REG00=0x80/0x1F and leaves ASDOUT
   * dead on this board. set JX_ADC_REARM_FULL 1 to restore it. */
  {
    const char *full = getenv("JX_ADC_REARM_FULL");
    if (full == NULL || atoi(full) == 0)
      {
        /* 2026-10-22: 完全照官方 close/open 语义，每次录音前先 suspend
         * （模拟前端真断电）再 start（CSM_ON 上电 + MCLK + ADC 通路）。
         * full_restart 内部已经包含 250ms 放电等待。 */
        return es8311_full_restart(g_i2c, VA_SAMPLE_RATE);
      }
  }

  if (g_es7210_ok)
    {
      return es7210_restore_analog(g_i2c);
    }

  {
    /* A/B：JX_ADC_CSM=0 时退回“只重写 ADC 寄存器”，不做整片软上电。 */
    const char *csm = getenv("JX_ADC_CSM");
    if (csm != NULL && atoi(csm) == 0)
      {
        return es8311_adc_enable(g_i2c, true);
      }
  }

  /* 2026-10-21: 整片 full_restart 会让 ES8311 的 ADC 在 1～2 秒后再次
   * 停摆（实测 3 秒录音后 1.7 秒整段变 0）。改为只重写 ADC/模拟通路
   * 寄存器，不动 CSM 和 I2S 格式；这样不会打断正在跑的时钟。 */

  /* 提示音播放会把 DAC 打开并保持；DAC 与 ADC 共用模拟参考，实测
   * 提示音之后 ADC 在 0.6 秒左右被压成整段 0。录音前显式静音 DAC，
   * 让 ADC 的模拟前端恢复。 */
  /* 2026-10-21: 轻量重写只写数字寄存器，ES8311 的模拟 PGA 不会真正
   * 上电，实测说话 msq 只有 1~2 万（正常是百万级），听感就是又小又
   * 滋滋响。恢复整片 full_restart，让模拟前端完整上电。 */
  ret = es8311_full_restart(g_i2c, VA_SAMPLE_RATE);
  if (ret < 0)
    {
      printf("[VoiceAgent] ES8311 full restart failed: %d\n", ret);
      return ret;
    }

  (void)es8311_set_volume(g_i2c, g_dac_volume);
  (void)es8311_mute(g_i2c, true);

  s_adc_rearmed = 1;
  return es8311_adc_enable(g_i2c, true);
}

/* 2026-10-21: light ADC re-enable while recording is running.
 * The ES8311 ADC can stop emitting after a few seconds; rewriting the
 * analog/ADC registers brings it back without a full chip restart. */

int voice_agent_power_off(void)
{
  if (g_i2c == NULL)
    {
      return -EINVAL;
    }

  if (g_es7210_ok)
    {
      return 0;
    }

  /* 2026-10-22 v10：默认【不】断电。官方小智的 codec 是从开机常开到
   * 关机，录音结束只关 I2S 流，不碰 ES8311 的模拟电源。原来每次录完
   * 都 suspend，第二次录音再 start，正是“只能用一次”的诱因。
   * set JX_ADC_POWEROFF 1 可恢复旧的录完断电行为。 */
  {
    const char *po = getenv("JX_ADC_POWEROFF");
    if (po == NULL || atoi(po) == 0)
      {
        return 0;
      }
  }

  return es8311_power_off(g_i2c);
}

int voice_agent_kick_adc(void)
{
  if (g_i2c == NULL)
    {
      return -EINVAL;
    }

  if (g_es7210_ok)
    {
      return es7210_restore_analog(g_i2c);
    }

  return es8311_kick_adc(g_i2c);
}

int voice_agent_kick_adc_light(void)
{
  if (g_i2c == NULL)
    {
      return -EINVAL;
    }

  if (g_es7210_ok)
    {
      return es7210_restore_analog(g_i2c);
    }

  return es8311_kick_adc_light(g_i2c);
}
