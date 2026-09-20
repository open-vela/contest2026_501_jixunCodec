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

  ret = es8311_init(g_i2c);
  if (ret < 0)
    {
      fprintf(stderr, "[VoiceAgent] ES8311 init failed: %d\n", ret);
      return ret;
    }

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

  /* ES7210：ADC 初始化 */

  ret = es7210_init(g_i2c, VA_SAMPLE_RATE, VA_MCLK_HZ);
  if (ret < 0)
    {
      fprintf(stderr, "[VoiceAgent] ES7210 init failed: %d\n", ret);
      return ret;
    }

  printf("[VoiceAgent] ES7210 done, ticks=%lu\n",
         (unsigned long)clock_systime_ticks());

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

  ret = es7210_restore_analog(g_i2c);
  if (ret < 0)
    {
      fprintf(stderr, "[VoiceAgent] ES7210 analog restore failed: %d\n",
              ret);
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

int voice_agent_set_dac_mute(int mute)
{
  if (g_i2c != NULL)
    {
      return es8311_mute(g_i2c, mute != 0);
    }

  return 0;
}

uint8_t voice_agent_get_dac_volume(void)
{
  return g_dac_volume;
}
