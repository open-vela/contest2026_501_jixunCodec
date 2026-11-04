/****************************************************************************
 * apps/plant-companion/components/voice_agent/es8311.c
 *
 * ES8311 DAC driver — 完全按 IDF esp_codec_dev 的寄存器配置。
 * 寄存器地址来自 managed_components/espressif__esp_codec_dev/device/es8311/es8311_reg.h
 ****************************************************************************/

#include <nuttx/config.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

#include <nuttx/i2c/i2c_master.h>
#include "../../hal/hal_i2c.h"
#include "es8311.h"

static uint8_t g_i2c_addr = 0x18;
#define I2C_FREQ  100000

/* ---- IDF ES8311 寄存器地址（来自 es8311_reg.h）---- */

#define ES8311_RESET_REG00       0x00
#define ES8311_CLK_MANAGER_REG01 0x01
#define ES8311_CLK_MANAGER_REG02 0x02
#define ES8311_CLK_MANAGER_REG03 0x03
#define ES8311_CLK_MANAGER_REG04 0x04
#define ES8311_CLK_MANAGER_REG05 0x05
#define ES8311_CLK_MANAGER_REG06 0x06
#define ES8311_CLK_MANAGER_REG07 0x07
#define ES8311_CLK_MANAGER_REG08 0x08
#define ES8311_SDPIN_REG09       0x09   /* DAC I2S 格式 */
#define ES8311_SDPOUT_REG0A      0x0A   /* ADC I2S 格式 */
#define ES8311_SYSTEM_REG0B      0x0B
#define ES8311_SYSTEM_REG0C      0x0C
#define ES8311_SYSTEM_REG0D      0x0D   /* 电源 */
#define ES8311_SYSTEM_REG0E      0x0E   /* 电源 */
#define ES8311_SYSTEM_REG10      0x10
#define ES8311_SYSTEM_REG11      0x11
#define ES8311_SYSTEM_REG12      0x12   /* DAC 电源 */
#define ES8311_SYSTEM_REG13      0x13
#define ES8311_SYSTEM_REG14      0x14   /* ADC 配置 */
#define ES8311_ADC_REG15         0x15
#define ES8311_ADC_REG16         0x16
#define ES8311_ADC_REG17         0x17
#define ES8311_ADC_REG18         0x18   /* ALC 使能 / 自动静音使能 */
#define ES8311_ADC_REG1A         0x1A   /* 自动静音窗口 / 噪声门 */
#define ES8311_ADC_REG1B         0x1B
#define ES8311_ADC_REG1C         0x1C
#define ES8311_DAC_REG31         0x31
#define ES8311_DAC_REG32         0x32   /* DAC 音量 */
#define ES8311_DAC_REG37         0x37
#define ES8311_GPIO_REG44        0x44
#define ES8311_GP_REG45          0x45


static uint8_t es8311_mic_gain_reg(void)
{
  const char *g = getenv("JX_ES8311_GAIN");
  if (g != NULL && g[0] != '\0')
    {
      return (uint8_t)strtoul(g, NULL, 0);
    }
  /* 2026-10-20: 0x24(24dB) 实测编码器基本在编噪声（原始幅度 ±60LSB），
   * 解码出来是固定的“嗡嗡”声。0x27=42dB 后同一段语音解码 RMS 提升 4 倍，
   * 且与源语音包络相关 0.965。 */
  /* 2026-10-21: 低 3 位才是 PGA 增益档（0=0dB,4=24dB,7=42dB）。
   * 0x07/0x27 的低 3 位都是 7，实际一直跑在 42dB，静音/大声后
   * 模拟前端过载，ADC 在 0.6~1.7 秒后整段输出 0。降到 24dB 验证。 */
  /* 2026-10-22：42dB 会把 ES8311 模拟 PGA 顶到饱和并锁零，实测
   * 18dB 0.5s 后归零、24dB 直接全 0。默认降到 12dB，先用不锁死的
   * 档位保证连续录音；音量由 ADC 数字增益和上层 AGC 补。 */
  /* 2026-10-22 v17 实测（官方同款板、24k/16bit/Philips）：
   *   0x24(24dB) acrms≈18，基本是 LSB 噪声
   *   0x27(42dB) acrms≈246~1003，连续 3 次稳定
   * 这块板的模拟 MEMS 输出偏弱，24dB 不够，默认定到 42dB。
   * 如果现场说话削顶，可用 set JX_ES8311_GAIN 0x26 (36dB) 回调。 */
  /* 2026-10-23 v25: default to the exact value the official board runs:
   * REG16=0x05 = 30 dB, bit5=0. Measured when bit5=1 (0x26/0x27) or when
   * the PGA is pushed above 30 dB: the ADC starts fine but after ~1.1 s
   * the ASDOUT goes to exact zeros. 0x05 stayed alive for the whole take.
   * Override with `set JX_ES8311_GAIN 0x07` (42 dB) if the room is quiet. */
  /* 2026-10-30: 0x05 (30 dB) on this board measured acrms=40 -- the ADC was
   * picking up almost nothing and the codec encoded pure noise, so the far
   * end heard "no sound".  The same board measured acrms~2100 with
   * REG16=0x26 (36 dB).  Use the field-proven value. */
  /* 2026-11-02: UI 路径曾无条件写 0x06，把这里的 0x05 覆盖掉；
   * 先回到官方稳定值 0x05，现场可用 JX_ES8311_GAIN 覆盖。 */
  return 0x05;
}

/* ---- I2C 读写 ---- */

static int es8311_write_reg(FAR struct i2c_master_s *i2c, uint8_t reg, uint8_t val)
{
  return hal_i2c_write_reg(i2c, g_i2c_addr, reg, val, I2C_FREQ);
}

static int es8311_read_reg(FAR struct i2c_master_s *i2c, uint8_t reg, uint8_t *val)
{
  return hal_i2c_read_reg(i2c, g_i2c_addr, reg, val, I2C_FREQ);
}

/* ---- 公共 API ---- */

int es8311_init(FAR struct i2c_master_s *i2c)
{
  int ret;
  uint8_t regv;
  uint8_t probe = 0;

  /* Some boards strap the ES8311 CE pin high (address 0x19).  Probe both
   * before touching any register, otherwise a perfectly good codec looks
   * dead with -EIO. */

  if (hal_i2c_read_reg(i2c, 0x18, ES8311_CLK_MANAGER_REG01, &probe,
                       I2C_FREQ) == 0)
    {
      g_i2c_addr = 0x18;
    }
  else if (hal_i2c_read_reg(i2c, 0x19, ES8311_CLK_MANAGER_REG01, &probe,
                            I2C_FREQ) == 0)
    {
      g_i2c_addr = 0x19;
    }
  else
    {
      printf("[ES8311] no I2C ACK at 0x18/0x19 (SDA/SCL or power)\n");
      return -EIO;
    }

  printf("[ES8311] I2C addr=0x%02x REG01=0x%02x\n", g_i2c_addr, probe);

  /* 2026-10-22 v10 诊断：每次启动都打印芯片原始电源/ADC 状态。
   * R00=0x80 表示 CSM_ON 还是上一次的，ES8311 没有真正断电；
   * 冷启动（VCC3.3 真断过）应看到 R00=0x00/0x0?，R0D=0xFA/0xFC。 */

  {
    uint8_t d00 = 0, d0d = 0, d0a = 0;
    (void)es8311_read_reg(i2c, ES8311_RESET_REG00, &d00);
    (void)es8311_read_reg(i2c, ES8311_SYSTEM_REG0D, &d0d);
    (void)es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &d0a);
    printf("[ES8311] BOOTSTATE R00=%02x R0D=%02x R0A=%02x%s\n",
           d00, d0d, d0a,
           ((d00 & 0x80) != 0) ? " (NOT POWER-CYCLED)" : " (cold boot)");
  }

  /* 2026-10-22: ES8311 没有硬复位脚，物理断电才真正复位。ESP32 自己
   * 复位（脚本打开串口、reboot）时芯片仍带着完整配置在跑。此时任何
   * 一次寄存器重写（尤其是 REG0D/0E/12/14 模拟电源）都会让 ADC 在
   * 约 1.2 秒后精确输出全 0。官方小智的做法是 codec 只初始化一次、
   * 之后一直常开。这里用 REG00 bit7(CSM_ON) 判断：
   *   bit7=1 -> 芯片已上电并配置过，直接返回 1，一个字都不写；
   *   bit7=0 -> 冷启动，走完整初始化。 */
  {
    uint8_t r00 = 0;
    if (es8311_read_reg(i2c, ES8311_RESET_REG00, &r00) == 0 &&
        (r00 & 0x80) != 0)
      {
        printf("[ES8311] already configured (REG00=0x%02x), keep as-is\n",
               r00);
        /* 2026-10-29: 不再重跑 start。
         * 实测：芯片已上电时任何一次 REG0D/0E/12/14/15 重写，ADC 会在
         * 1.2~2.4 s 后精确输出全 0；连续第二次录音则从第一块起全 0。
         * 官方语义就是 codec 只初始化一次、之后常开，录音只切 I2S 流。
         * 这里只补最必要的一位：若 REG0A bit6(ADC SDP mute) 被置上则清掉。 */
        {
          uint8_t r0a = 0;
          if (es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &r0a) == 0 &&
              (r0a & 0x40) != 0)
            {
              (void)es8311_write_reg(i2c, ES8311_SDPOUT_REG0A,
                                     (uint8_t)(r0a & ~0x40));
              printf("[ES8311] ADC SDP unmuted on keep\n");
            }
        }

        /* 2026-10-30: 芯片一直上电时，旧代码一个字都不写，于是
         * set JX_ES8311_GAIN 只在真正冷启动后才生效，现场没法调。
         * 只写 REG16（麦克风 PGA），不碰 0D/0E/12/14/15，
         * 避免 ADC 在 1.2 s 后输出全 0 的那个已知问题。 */
        {
          uint8_t want = es8311_mic_gain_reg();
          uint8_t cur = 0;

          if (es8311_read_reg(i2c, ES8311_ADC_REG16, &cur) == 0)
            {
              if (cur != want)
                {
                  (void)es8311_write_reg(i2c, ES8311_ADC_REG16, want);
                  printf("[ES8311] mic gain REG16 %02x -> %02x (keep)\n",
                         cur, want);
                }
              else
                {
                  printf("[ES8311] mic gain REG16=%02x (keep, same)\n", cur);
                }
            }
        }
        /* 2026-11-01: the keep path never restored the ADC analog
         * path, so a codec left configured by an earlier take can
         * come back with a dead microphone.  Re-apply the ADC
         * registers the full start() sequence would write. */
        {
          static const uint8_t ka[][2] =
            {
              { ES8311_SYSTEM_REG0E, 0x02 },
              { ES8311_SYSTEM_REG12, 0x02 },
              { ES8311_SYSTEM_REG14, 0x1A },
              { ES8311_ADC_REG15,    0x40 },
              { ES8311_ADC_REG17,    0xBF },
              { ES8311_ADC_REG18,    0x00 },
              { ES8311_ADC_REG1A,    0x00 },
            };
          int i;
          for (i = 0; i < (int)(sizeof(ka) / sizeof(ka[0])); i++)
            {
              (void)es8311_write_reg(i2c, ka[i][0], ka[i][1]);
            }
          printf("[ES8311] ADC path re-applied on keep\n");
        }
        return 1;
      }
  }

  /* ---- 阶段 1：上电配置（来自 IDF es8311_open）---- */

  /* 官方 es8311_open() 的第一步：先把 REG0D 写成 0xFA，把 DAC/ADC
   * 的模拟电源真正拉掉，再开始配置。原来漏了这一步，芯片可能带着
   * 上一次的 VMID/PGA 偏置直接进配置流程，模拟参考不稳。 */

  {
    uint8_t r0d = 0;
    if (es8311_read_reg(i2c, ES8311_SYSTEM_REG0D, &r0d) == 0 &&
        r0d != 0xFA)
      {
        ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0D, 0xFA);
      }
  }

  /* I2C 抗干扰增强 */

  ret = es8311_write_reg(i2c, ES8311_GPIO_REG44, 0x08);
  ret = es8311_write_reg(i2c, ES8311_GPIO_REG44, 0x08);  /* 写两次确保可靠 */

  /* 时钟初始化 */

  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG01, 0x30);
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG02, 0x00);
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG03, 0x10);

  /* 麦克风增益 */

  ret = es8311_write_reg(i2c, ES8311_ADC_REG16, es8311_mic_gain_reg());

  /* 时钟分频 */

  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG04, 0x10);
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG05, 0x00);

  /* 电源配置 */

  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0B, 0x00);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0C, 0x00);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG10, 0x1F);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG11, 0x7F);

  /* 软复位 + 上电命令
   *
   * 2026-10-22: 只在芯片还没上电时做软复位。`jixun` app 每次命令都会
   * 重新走 es8311_init()，原来每次都写 REG00=0x80，播放一次后模拟参考
   * 还没恢复，下一次软复位就把 ADC 锁死。读 REG00 bit7(CSM_ON) 判断：
   * 已上电则跳过软复位，只重写后续配置。 */

  /* 走到这里说明 REG00 bit7=0，是真正的冷启动。 */
  ret = es8311_write_reg(i2c, ES8311_RESET_REG00, 0x80);

  /* ---- 阶段 2：模式配置（来自 IDF es8311_open）---- */

  /* 从模式（Slave） */

  ret = es8311_read_reg(i2c, ES8311_RESET_REG00, &regv);
  regv &= 0xBF;  /* Slave 模式 */
  ret = es8311_write_reg(i2c, ES8311_RESET_REG00, regv);

  /* 时钟源：使用外部 MCLK（use_mclk=true） */

  regv = 0x3F;
  regv &= 0x7F;  /* use_mclk = true → bit7=0 */
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG01, regv);

  /* 不反转 SCLK */

  ret = es8311_read_reg(i2c, ES8311_CLK_MANAGER_REG06, &regv);
  regv &= ~(0x20);  /* invert_sclk = false */
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG06, regv);

  /* ---- 阶段 3：DAC + ADC 接口配置 ---- */

  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG13, 0x10);
  ret = es8311_write_reg(i2c, ES8311_ADC_REG1B, 0x0A);
  ret = es8311_write_reg(i2c, ES8311_ADC_REG1C, 0x6A);

  /* 内部参考信号（ADCL + DACR） */

  /* 2026-10-22 v16: REG44 在官方头文件里的定义是
   * "GPIO, dac2adc for test" —— 打开后 ADC 可能改读 DAC 内部
   * 参考而不是外部 MIC。官方默认写 0x58，这里做成可 A/B：
   *   set JX_REG44 8    -> 0x08（只保留 I2C 抗干扰，不开 dac2adc）
   *   set JX_REG44 0x58 -> 官方原值
   * 不设时保持 0x58，行为与官方一致。 */
  {
    const char *r44 = getenv("JX_REG44");
    uint8_t v44 = (r44 != NULL && r44[0] != '\0')
                  ? (uint8_t)strtoul(r44, NULL, 0) : 0x58;
    ret = es8311_write_reg(i2c, ES8311_GPIO_REG44, v44);
    printf("[ES8311] REG44=0x%02x%s\n", v44,
           (r44 != NULL) ? " (JX_REG44)" : " (default)");
  }

  if (ret < 0)
    {
      return ret;
    }

  return 0;
}

int es8311_set_format(FAR struct i2c_master_s *i2c, int sample_rate)
{
  int ret;
  uint8_t dac_iface = 0;
  uint8_t adc_iface = 0;
  uint8_t regv;

  /* ---- 配置 I2S 格式（来自 IDF es8311_set_fs）---- */

  /* 位宽：16-bit */

  ret = es8311_read_reg(i2c, ES8311_SDPIN_REG09, &dac_iface);
  ret |= es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &adc_iface);
  dac_iface |= 0x0c;  /* 16-bit */
  adc_iface |= 0x0c;
  ret |= es8311_write_reg(i2c, ES8311_SDPIN_REG09, dac_iface);
  ret |= es8311_write_reg(i2c, ES8311_SDPOUT_REG0A, adc_iface);

  /* 格式：I2S Standard（Philips） */

  ret = es8311_read_reg(i2c, ES8311_SDPIN_REG09, &dac_iface);
  ret |= es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &adc_iface);
  dac_iface &= 0xFC;  /* [1:0] = 00: I2S Standard */
  adc_iface &= 0xFC;
  ret |= es8311_write_reg(i2c, ES8311_SDPIN_REG09, dac_iface);
  ret |= es8311_write_reg(i2c, ES8311_SDPOUT_REG0A, adc_iface);

  /* ---- 时钟配置（MCLK=6.144MHz, 24kHz）---- */
  /* coeff_div[24k,6.144M]（IDF 权威表）: pre_div=1, mult=1, adc_div=1, dac_div=1,
   *   fs_mode=0, lrck_h=0, lrck_l=0xff, bclk_div=4, adc_osr=16, dac_osr=16
   * ⚠️ MCLK 已改回 6.144MHz（xiaozhi 配置），pre_div=1 归一化内部时钟；
   *    12.288MHz 时 pre_div 才是 2。 */

  /* REG02: pre_div + pre_multi */

  regv = 0x00;
  regv |= (1 - 1) << 5;  /* pre_div = 1 */
  regv |= (1 - 1) << 3;  /* pre_multi = 0 (x1) */
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG02, regv);

  /* REG05: adc_div + dac_div */

  regv = 0x00;
  regv |= (1 - 1) << 4;  /* adc_div = 0 (1-1) */
  regv |= (1 - 1) << 0;  /* dac_div = 0 (1-1) */
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG05, regv);

  /* REG03: fs_mode + adc_osr */

  regv = 0x00;
  regv |= 0 << 6;  /* fs_mode = 0 (single speed) */
  regv |= 16 << 0;  /* adc_osr = 16 */
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG03, regv);

  /* REG04: dac_osr */

  regv = 16;
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG04, regv);

  /* REG07: lrck_h */

  regv = 0x00;
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG07, regv);

  /* REG08: lrck_l */

  regv = 0xFF;
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG08, regv);

  /* REG06: bclk_div（保留 bit5 SCLK 反转位） */

  ret = es8311_read_reg(i2c, ES8311_CLK_MANAGER_REG06, &regv);
  regv &= 0xE0;
  regv |= (4 - 1) << 0;  /* bclk_div = 4 (4-1) */
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG06, regv);

  if (ret < 0)
    {
      return ret;
    }

  return 0;
}

/* 官方 esp_codec_dev es8311_suspend()：把 ADC/DAC 模拟前端真正断电。
 * REG00=0x00 只关 CSM_ON，VMID/PGA 偏置还挂着；必须把 REG0D/0E/12/14/15/17
 * 一起拉掉，下次上电才是干净的冷启动。 */

static int es8311_suspend_hw(FAR struct i2c_master_s *i2c)
{
  int ret = 0;

  ret |= es8311_write_reg(i2c, ES8311_DAC_REG32, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_ADC_REG17, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG0E, 0xFF);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG12, 0x02);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG14, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG0D, 0xFA);
  ret |= es8311_write_reg(i2c, ES8311_ADC_REG15, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG02, 0x10);
  ret |= es8311_write_reg(i2c, ES8311_RESET_REG00, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_RESET_REG00, 0x1F);
  usleep(20 * 1000);
  ret |= es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG01, 0x30);
  ret |= es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG01, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_GP_REG45, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG0D, 0xFC);
  ret |= es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG02, 0x00);
  return ret < 0 ? -EIO : 0;
}

int es8311_start(FAR struct i2c_master_s *i2c)
{
  int ret;
  uint8_t regv;

  /* ---- DAC + ADC 接口使能（来自 IDF es8311_start）---- */

  /* 官方 es8311_start 的前两步：CSM_ON 上电 + 选择外部 MCLK。
   * 2026-10-22: 这两步原来漏了。suspend 之后 REG00=0x00、REG01=0x00，
   * 不重新上电/配 MCLK，ADC 的 ASDOUT 就一直是 0。 */

  ret = es8311_write_reg(i2c, ES8311_RESET_REG00, 0x80);
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG01, 0x3F);

  /* 先读当前值，只修改控制位（保留 format/bitwidth 设置） */

  ret = es8311_read_reg(i2c, ES8311_SDPIN_REG09, &regv);
  regv &= 0xBF;  /* 清 bit6（DAC SDP MUTE=0，解除 DAC 静音） */
  ret = es8311_write_reg(i2c, ES8311_SDPIN_REG09, regv);

  /* ES8311 内部 ADC（ASDOUT）不用（麦克风走 ES7210），
   * 但按官方流程把 ADC 接口置为静音（bit6=1），避免内部 ADC 数据串扰 */

  /* 2026-10-22 v10 关键修复：这块板没有 ES7210，麦克风直接接
   * ES8311 的 MICIN/MICIP，内部 ADC 就是唯一音源。官方 es8311_start()
   * 在 ADC/BOTH 模式下是【清】REG0A bit6（ADC SDP MUTE=0），
   * 旧代码却置 1，等于每次 start 都把内部 ADC 关掉；之后 DAC 一放音
   * 就独占模拟参考，ADC 再也起不来。这里按官方改成清 bit6。 */

  ret = es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &regv);
  {
    const char *inmp = getenv("JX_MIC_INMP441");
    if (inmp != NULL && atoi(inmp) != 0)
      {
        /* INMP441 模式：ES8311 只做 DAC，内部 ADC 静音，避免它和
         * INMP441 同时驱动 I2S DIN（GPIO10）。硬件上仍需断开
         * ES8311 ASDOUT 到 GPIO10 的走线。 */
        regv |= 0x40;
      }
    else
      {
        regv &= 0xBF;  /* 清 bit6：ADC SDP 不静音 */
      }
  }
  ret = es8311_write_reg(i2c, ES8311_SDPOUT_REG0A, regv);

  /* ⚠️ 关键修复：REG09 bit6 = "DAC SDP MUTE"（1=静音！）
   * 之前误标为"DAC使能"并置位 → DAC 一直被静音 → 无声。
   * 依据：Linux 内核 es8311.c `SOC_SINGLE("DAC SDP MUTE", REG09, 6, 1, 0)`
   *       + 官方 esp_codec_dev es8311_start（DAC 模式清 bit6）。
   * 这里再读一次确保 bit6=0（防止上面第 2 步误写），并打印确认。 */

  ret = es8311_read_reg(i2c, ES8311_SDPIN_REG09, &regv);
  regv &= 0xBF;  /* 清 bit6 = 解除 DAC 静音 */
  ret = es8311_write_reg(i2c, ES8311_SDPIN_REG09, regv);
  printf("[ES8311] DAC SDP MUTE cleared (REG09=0x%02x)\n",
         regv & 0xBF);

  /* DAC 音量 + 电源 */

  /* 2026-10-22 v17: 官方 start 不写 REG16，增益由 set_mic_gain 单独
   * 设置。我们的 already-configured 路径只跑 start，所以 JX_ES8311_GAIN
   * 一直没生效。这里把 PGA 增益也带进 start，保证每次 enable 都用
   * 当前请求的档位。 */
  ret = es8311_write_reg(i2c, ES8311_ADC_REG16, es8311_mic_gain_reg());
  ret = es8311_write_reg(i2c, ES8311_ADC_REG17, 0xBF);

  /* 2026-10-22 v19：关闭 ADC ALC 与自动静音。
   * ES8311 的 REG18 bit6 是 automute 使能，芯片默认可能为 1；输入安静
   * 约 0.5 秒后 ASDOUT 会被精确掐成 0，用户停顿后的字头直接被吞。
   * 这里先读出来打印，再显式清零（ALC 也一起关掉，避免自动增益压字头）。 */
  {
    uint8_t q18 = 0;
    uint8_t q1a = 0;

    (void)es8311_read_reg(i2c, ES8311_ADC_REG18, &q18);
    (void)es8311_read_reg(i2c, ES8311_ADC_REG1A, &q1a);
    printf("[ES8311] ADC automute before: 18=%02x 1A=%02x\n", q18, q1a);
  }
  ret = es8311_write_reg(i2c, ES8311_ADC_REG18, 0x00);
  ret = es8311_write_reg(i2c, ES8311_ADC_REG1A, 0x00);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0E, 0x02);
  /* 2026-10-22 v21: 录音时 DAC 模拟块和 ADC 共用 VREF/VMID。DAC 一开，
   * 高增益 ADC 会被顶到饱和并锁零（0.4~1s 后整段输出 0）。
   * 小黑只负责采集，录音时默认把 DAC 模拟块掉电；要放音时 set JX_DAC_OFF 0。 */
  {
    const char *dacoff = getenv("JX_DAC_OFF");
    uint8_t v12 = (dacoff != NULL && atoi(dacoff) == 0) ? 0x00 : 0x02;
    ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG12, v12);
  }
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG14, 0x1A);

  /* 不用 DMIC */

  ret = es8311_read_reg(i2c, ES8311_SYSTEM_REG14, &regv);
  regv &= ~(0x40);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG14, regv);

  /* 电源使能 */

  /* 2026-10-23 v24: VMIDSEL=2 (normal operation). Official start() writes
   * 0x01, whose VMIDSEL field is 1 = "startup, normal speed". The startup
   * state is transient; after it times out the ADC stops delivering data
   * (observed: noise for ~1.2 s then exact zeros, I2S throughput falls to
   * ~1/5). Bit0 is VMIDSEL[0]: 0x02 selects the steady normal-operation
   * bias instead of the startup ramp. */
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0D, 0x02);
  ret = es8311_write_reg(i2c, ES8311_ADC_REG15, 0x40);
  ret = es8311_write_reg(i2c, ES8311_DAC_REG37, 0x08);
  ret = es8311_write_reg(i2c, ES8311_GP_REG45, 0x00);

  /* 2026-10-30: suspend 会把 REG32(DAC 音量) 写成 0x00(-95.5dB)，
   * 而 start 原来没恢复它，导致所有播放都是静音。这里每次 start 都恢复。 */
  {
    const char *dv = getenv("JX_DAC_VOL");
    uint8_t vol = (dv != NULL && dv[0] != '\0')
                  ? (uint8_t)strtoul(dv, NULL, 0) : 0xC0;
    ret = es8311_write_reg(i2c, ES8311_DAC_REG32, vol);
    printf("[ES8311] DAC volume REG32=0x%02x\n", vol);
  }

  /* 2026-10-22 v16b: already-configured 分支不会走 init 后半段，
   * 所以把 JX_REG44 的应用也放到 start 里，保证 A/B 一定能生效。 */
  {
    const char *r44 = getenv("JX_REG44");
    /* v17: 恢复官方默认 0x58；可用 set JX_REG44 8 做 A/B。 */
    uint8_t v44 = (r44 != NULL && r44[0] != '\0')
                  ? (uint8_t)strtoul(r44, NULL, 0) : 0x58;
    es8311_write_reg(i2c, ES8311_GPIO_REG44, v44);
    printf("[ES8311] REG44 applied in start: 0x%02x\n", v44);
  }

  {
    uint8_t q15 = 0, q16 = 0, q17 = 0, q44 = 0, q0d = 0, q0e = 0;
    uint8_t q12 = 0, q14 = 0, q0a = 0, q13 = 0;
    es8311_read_reg(i2c, ES8311_ADC_REG15, &q15);
    es8311_read_reg(i2c, ES8311_ADC_REG16, &q16);
    es8311_read_reg(i2c, ES8311_ADC_REG17, &q17);
    es8311_read_reg(i2c, ES8311_GPIO_REG44, &q44);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG0D, &q0d);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG0E, &q0e);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG12, &q12);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG14, &q14);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG13, &q13);
    es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &q0a);
    printf("[ES8311] REG13=%02x REG0D_mode=%s\n", q13, (q0d & 0x03) == 2 ? "normal" : "startup");
    printf("[ES8311] start dump 0A=%02x 0D=%02x 0E=%02x 12=%02x "
           "14=%02x 15=%02x 16=%02x 17=%02x 44=%02x\n",
           q0a, q0d, q0e, q12, q14, q15, q16, q17, q44);
  }

  if (ret < 0)
    {
      return ret;
    }

  return 0;
}

/* 2026-10-30: 播放前把 DAC 模拟块从"录音省电"状态拉回来。
 * 录音路径为了不让 DAC 抢 ADC 的 VREF，会把 REG12 写成 0x02（DAC 掉电）。
 * 播放/提示音前如果不写回 0x00，I2S 有数据、PA 也开着，但喇叭一点声都没有。
 * 这里先读 REG12，已经上电就只做一次轻量确认，避免每块 PCM 都白写 I2C。 */
int es8311_dac_power_on(FAR struct i2c_master_s *i2c)
{
  uint8_t val = 0;
  uint8_t vol = 0xC0;
  const char *dv;
  int ret;

  if (i2c == NULL)
    {
      return -EINVAL;
    }

  ret = es8311_read_reg(i2c, ES8311_SYSTEM_REG12, &val);
  if (ret >= 0 && (val & 0x02) == 0)
    {
      /* JX_REG44_QUICKFIX: 芯片不断电时启动走 keep-as-is，REG44 会保留
       * 上一次固件写进去的 0x5X（dac2adc 测试路由），导致 DAC 不出声。
       * 这里每次播放前补写一次官方值 0x08。 */
      static int r44_done;
      if (!r44_done)
        {
          r44_done = 1;
          ret = es8311_write_reg(i2c, ES8311_GPIO_REG44, 0x58);
          printf("[ES8311] quickfix REG44=0x58 (once)\n");
          return ret < 0 ? -EIO : 0;
        }

      return 0;
    }

  dv = getenv("JX_DAC_VOL");
  if (dv != NULL && dv[0] != '\0')
    {
      vol = (uint8_t)strtoul(dv, NULL, 0);
    }

  ret  = es8311_write_reg(i2c, ES8311_SYSTEM_REG12, 0x00);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG0D, 0x02);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG0E, 0x02);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG14, 0x1A);
  ret |= es8311_write_reg(i2c, ES8311_DAC_REG37, 0x08);
  ret |= es8311_write_reg(i2c, ES8311_DAC_REG32, vol);

  if (es8311_read_reg(i2c, ES8311_SDPIN_REG09, &val) == 0)
    {
      val &= 0xBF;                       /* DAC SDP MUTE=0 */
      ret |= es8311_write_reg(i2c, ES8311_SDPIN_REG09, val);
    }

  if (es8311_read_reg(i2c, ES8311_DAC_REG31, &val) == 0)
    {
      val &= 0x9F;                       /* DAC MUTE bit5/6=0 */
      ret |= es8311_write_reg(i2c, ES8311_DAC_REG31, val);
    }

  printf("[ES8311] DAC power on: REG12=00 REG32=0x%02x\n", vol);
  return ret < 0 ? -EIO : 0;
}

int es8311_set_volume(FAR struct i2c_master_s *i2c, uint8_t vol)
{
  return es8311_write_reg(i2c, ES8311_DAC_REG32, vol);
}

int es8311_mute(FAR struct i2c_master_s *i2c, bool mute)
{
  uint8_t val;
  int ret = es8311_read_reg(i2c, ES8311_DAC_REG31, &val);
  if (ret < 0) return ret;

  val &= 0x9F;
  if (mute)
    {
      val |= 0x60;
    }

  return es8311_write_reg(i2c, ES8311_DAC_REG31, val);
}

/****************************************************************************
 * 公共 API：DAC 输出链路关键寄存器（播放无声时判读）
 *
 *  REG09: bit6=DAC MUTE(0=响) bit7=数据源(0=I2S输入)
 *  REG0D/0E: 电源；REG12: DAC 使能；REG31: DAC MUTE(bit5/6)
 *  REG32: DAC 音量(0x00=-95dB..0xFF=+32dB)；REG37: ramprate
 ****************************************************************************/

int es8311_dump_dac(FAR struct i2c_master_s *i2c)
{
  static const uint8_t regs[] = { 0x09, 0x0A, 0x0D, 0x0E, 0x12,
                                  0x31, 0x32, 0x37, 0x45 };
  int i;

  printf("[ES8311] DAC 链路: ");
  for (i = 0; i < (int)(sizeof(regs) / sizeof(regs[0])); i++)
    {
      uint8_t v = 0;

      es8311_read_reg(i2c, regs[i], &v);
      printf("R%02X=%02x ", regs[i], v);
    }

  printf("\n");
  return 0;
}

/****************************************************************************
 * 公共 API：寄存器全览（对比 小智 esp_codec_dev / OpenVela NuttX 参考值）
 *
 * 无声排查时逐项对照：REG00 主从、REG01-08 时钟（24k/6.144M 系数表）、
 * REG09 DAC 格式/MUTE、REG0A ADC 格式、REG0B-14 电源、REG17 音量、
 * REG31 DAC MUTE、REG32 DAC 音量、REG37 ramprate、REG44/45。
 ****************************************************************************/

int es8311_dump_regs(FAR struct i2c_master_s *i2c)
{
  static const uint8_t regs[] =
  {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12,
    0x13, 0x14, 0x15, 0x16, 0x17, 0x1B, 0x1C,
    0x31, 0x32, 0x37, 0x44, 0x45
  };
  int i;

  printf("[ES8311] === 寄存器全览（对比参考：24k/6.144M slave DAC）===\n");
  for (i = 0; i < (int)(sizeof(regs) / sizeof(regs[0])); i++)
    {
      uint8_t v = 0;
      int ret = es8311_read_reg(i2c, regs[i], &v);

      printf("[ES8311] REG%02X=0x%02x%s\n", regs[i], v,
             (ret < 0) ? " (读失败)" : "");
    }

  return 0;
}

/****************************************************************************
 * 内部 ADC（ASDOUT）使能 —— 无 ES7210 的板子（小智板）走这里
 *
 * 有 ES7210 时内部 ADC 被刻意静音（REG0A bit6=1），避免两路 ADC 串扰；
 * 没有 ES7210 时麦克风直接接 ES8311 的 MIC 输入，必须解除该静音并
 * 把 ADC 通路（电源/增益/格式）配好，否则 I2S DIN 上永远没有数据。
 ****************************************************************************/

int es8311_full_restart(FAR struct i2c_master_s *i2c, int sample_rate)
{
  int ret;

  (void)sample_rate;

  if (i2c == NULL)
    {
      return -EINVAL;
    }

  /* 2026-10-22: 完全照官方 esp_codec_dev 的 close/open 语义：
   *   suspend（模拟前端真断电） -> 等 VMID 放电 -> start（CSM_ON 上电
   *   + MCLK + ADC/DAC 通路）。官方固件就是靠这个循环反复录音/放音的。
   * 不要再调 es8311_init()：open 阶段的时钟分频/格式寄存器在 suspend
   * 里不会被清掉，重新写一遍反而会打乱已经锁定的 PLL。 */

  ret = es8311_suspend_hw(i2c);
  if (ret < 0)
    {
      return ret;
    }

  usleep(250 * 1000);

  return es8311_start(i2c);
}

/****************************************************************************
 * 公共 API：判断内部 ADC 是否已经在正常输出
 *
 * 官方小智的 codec 生命周期是 open 一次、之后一直常开，录音/放音只切
 * enable/disable。这块板没有 ES7210，内部 ADC 一旦被关（REG0A bit6=1）
 * 或芯片掉电（REG00 bit7=0），再想拉回来就会撞上模拟前端闩锁。
 * 所以录音前先用这个函数确认“它本来就活着”，活着就一个寄存器都不写。
 ****************************************************************************/

int es8311_adc_is_running(FAR struct i2c_master_s *i2c)
{
  uint8_t r00 = 0;
  uint8_t r0a = 0;
  uint8_t r0d = 0;
  uint8_t r15 = 0;

  if (i2c == NULL)
    {
      return 0;
    }

  if (es8311_read_reg(i2c, ES8311_RESET_REG00, &r00) < 0)
    {
      return 0;
    }

  if (es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &r0a) < 0)
    {
      return 0;
    }

  if (es8311_read_reg(i2c, ES8311_SYSTEM_REG0D, &r0d) < 0)
    {
      return 0;
    }

  if (es8311_read_reg(i2c, ES8311_ADC_REG15, &r15) < 0)
    {
      return 0;
    }

  /* 三个条件同时成立才算“ADC 已经在跑”：
   *   REG00 bit7 = CSM_ON（芯片上电）
   *   REG0D bit0 = 模拟电源已开（init 阶段是 0xFA，必须由 start 拉到 0x01）
   *   REG0A bit6 = 0（ADC SDP 不静音）
   * 只写 0xFA 的 init 阶段会被判成“没跑”，从而正确走一次 start。 */

  /* 2026-10-22 v15: 必须同时检查 REG15(ADC 使能)。官方 start 写
   * 0x40；suspend 会写成 0x00。只查 R00/R0D/R0A 会把"ADC 已关"误判
   * 成"正在跑"，于是永远不重跑 start，表现就是第一次录音正常、之后
   * 每次 probe acrms 只有十几（纯 LSB 噪声）。 */
  /* 2026-10-29: v24 起 start 写的是 REG0D=0x02（VMIDSEL normal），
   * 旧判据 (r0d & 0x01) 恒为 0，导致 rearm 每次都误判“没在跑”并做
   * full_restart，把正在录音的 ADC 反复打断。按 0x02 判。 */
  return ((r00 & 0x80) != 0) && ((r0d & 0x03) == 0x02) &&
         ((r0a & 0x40) == 0) && ((r15 & 0x40) != 0);
}

int es8311_adc_enable(FAR struct i2c_master_s *i2c, bool on)
{
  int ret;
  uint8_t v = 0;

  if (i2c == NULL)
    {
      return -EINVAL;
    }

  if (on)
    {
      /* INMP441 模式：ES8311 只负责 DAC。这里走一次 start 把 DAC 模拟
       * 电源拉起来，start 内部会把 REG0A bit6 置 1 静音内部 ADC。 */
      {
        const char *inmp = getenv("JX_MIC_INMP441");
        if (inmp != NULL && atoi(inmp) != 0)
          {
            return es8311_start(i2c);
          }
      }

      /* 2026-10-22 v10：ADC 已经在跑就不动它，这是官方小智的常开语义。
       * 只有冷启动后第一次（REG0A 静音或芯片没上电）才走 start。 */
      if (es8311_adc_is_running(i2c))
        {
          return 0;
        }

      /* 2026-10-22: 直接走官方 es8311_start()。它会自己完成
       * REG00=0x80 上电、REG01=0x3F MCLK、清 REG09/REG0A bit6、
       * REG0D/0E/12/14/15/17 全套上电。 */
      ret = es8311_start(i2c);
    }
  else
    {
      ret = es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &v);
      ret |= es8311_write_reg(i2c, ES8311_SDPOUT_REG0A,
                              (uint8_t)(v | 0x40));
    }

  {
    uint8_t r0a = 0, r0d = 0, r0e = 0, r12 = 0, r14 = 0, r15 = 0;
    uint8_t r16 = 0, r17 = 0;
    es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &r0a);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG0D, &r0d);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG0E, &r0e);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG12, &r12);
    es8311_read_reg(i2c, ES8311_SYSTEM_REG14, &r14);
    es8311_read_reg(i2c, ES8311_ADC_REG15, &r15);
    es8311_read_reg(i2c, ES8311_ADC_REG16, &r16);
    es8311_read_reg(i2c, ES8311_ADC_REG17, &r17);
    printf("[ES8311] ADC regs: 0A=%02x 0D=%02x 0E=%02x 12=%02x 14=%02x 15=%02x 16=%02x 17=%02x\n",
           r0a, r0d, r0e, r12, r14, r15, r16, r17);
  }

  return ret < 0 ? -EIO : 0;
}

/* 2026-10-22: light recovery. Rewrite ONLY ADC enable + gain.
 * Do not touch REG0D/0E/12/14 (analog supply / VMID): repeated rearm of
 * those latches the front-end to zero about 1.2 s into the take.
 */
int es8311_kick_adc_light(FAR struct i2c_master_s *i2c)
{
  int ret = 0;

  if (i2c == NULL)
    {
      return -EINVAL;
    }

  ret |= es8311_write_reg(i2c, ES8311_SDPOUT_REG0A, 0x0C);
  ret |= es8311_write_reg(i2c, ES8311_ADC_REG16, es8311_mic_gain_reg());
  ret |= es8311_write_reg(i2c, ES8311_ADC_REG17, 0xBF);
  return ret < 0 ? -EIO : 0;
}

int es8311_kick_adc(FAR struct i2c_master_s *i2c)
{
  int ret = 0;

  /* 2026-10-21: 42dB 下 ADC 每 0.4~1s 会自己衰减到 0，只写使能位
   * 拉不回来；必须连模拟电源/增益一起重写。实测每 400ms 做一次能把
   * 第 4 块从 2 万拉回 611 万，所以录音循环里每 100ms 做一次。 */
  ret |= es8311_write_reg(i2c, ES8311_SDPOUT_REG0A, 0x0C);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG0D, 0x01);
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG0E, 0x02);
  {
    const char *dacoff = getenv("JX_DAC_OFF");
    ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG12,
                            (dacoff != NULL && atoi(dacoff) == 0) ? 0x00 : 0x02);
  }
  ret |= es8311_write_reg(i2c, ES8311_SYSTEM_REG14, 0x1A);
  ret |= es8311_write_reg(i2c, ES8311_ADC_REG16, es8311_mic_gain_reg());
  ret |= es8311_write_reg(i2c, ES8311_ADC_REG17, 0xBF);
  return ret < 0 ? -EIO : 0;
}


/****************************************************************************
 * 公共 API：彻底关闭码片（REG00 CSM_ON=0）
 *
 * 2026-10-21: 录音结束后如果保持上电，第二次录音的 ADC 会直接输出全 0，
 * 只有拔插 USB 才恢复。每次录完断电，下次 rearm 就是干净的冷启动。
 ****************************************************************************/

int es8311_power_off(FAR struct i2c_master_s *i2c)
{
  if (i2c == NULL)
    {
      return -EINVAL;
    }

  /* 2026-10-22: 官方 close 的语义——不是只写 REG00=0x00，而是把
   * REG0D/0E/12/14/15/17 一起拉掉，让下次 start 是干净的冷启动。 */
  return es8311_suspend_hw(i2c);
}

int es8311_set_reg44(FAR struct i2c_master_s *i2c, uint8_t v)
{
  if (i2c == NULL)
    {
      return -EINVAL;
    }

  return es8311_write_reg(i2c, ES8311_GPIO_REG44, v);
}
