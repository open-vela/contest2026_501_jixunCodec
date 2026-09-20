/****************************************************************************
 * apps/plant-companion/components/voice_agent/es8311.c
 *
 * ES8311 DAC driver — 完全按 IDF esp_codec_dev 的寄存器配置。
 * 寄存器地址来自 managed_components/espressif__esp_codec_dev/device/es8311/es8311_reg.h
 ****************************************************************************/

#include <nuttx/config.h>
#include <stdio.h>
#include <errno.h>

#include <nuttx/i2c/i2c_master.h>
#include "../../hal/hal_i2c.h"
#include "es8311.h"

#define I2C_ADDR  0x18
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
#define ES8311_ADC_REG1B         0x1B
#define ES8311_ADC_REG1C         0x1C
#define ES8311_DAC_REG31         0x31
#define ES8311_DAC_REG32         0x32   /* DAC 音量 */
#define ES8311_DAC_REG37         0x37
#define ES8311_GPIO_REG44        0x44
#define ES8311_GP_REG45          0x45

/* ---- I2C 读写 ---- */

static int es8311_write_reg(FAR struct i2c_master_s *i2c, uint8_t reg, uint8_t val)
{
  return hal_i2c_write_reg(i2c, I2C_ADDR, reg, val, I2C_FREQ);
}

static int es8311_read_reg(FAR struct i2c_master_s *i2c, uint8_t reg, uint8_t *val)
{
  return hal_i2c_read_reg(i2c, I2C_ADDR, reg, val, I2C_FREQ);
}

/* ---- 公共 API ---- */

int es8311_init(FAR struct i2c_master_s *i2c)
{
  int ret;
  uint8_t regv;

  /* ---- 阶段 1：上电配置（来自 IDF es8311_open）---- */

  /* I2C 抗干扰增强 */

  ret = es8311_write_reg(i2c, ES8311_GPIO_REG44, 0x08);
  ret = es8311_write_reg(i2c, ES8311_GPIO_REG44, 0x08);  /* 写两次确保可靠 */

  /* 时钟初始化 */

  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG01, 0x30);
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG02, 0x00);
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG03, 0x10);

  /* 麦克风增益 */

  ret = es8311_write_reg(i2c, ES8311_ADC_REG16, 0x24);

  /* 时钟分频 */

  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG04, 0x10);
  ret = es8311_write_reg(i2c, ES8311_CLK_MANAGER_REG05, 0x00);

  /* 电源配置 */

  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0B, 0x00);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0C, 0x00);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG10, 0x1F);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG11, 0x7F);

  /* 软复位 + 上电命令 */

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

  ret = es8311_write_reg(i2c, ES8311_GPIO_REG44, 0x58);

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

int es8311_start(FAR struct i2c_master_s *i2c)
{
  int ret;
  uint8_t regv;

  /* ---- DAC + ADC 接口使能（来自 IDF es8311_start）---- */

  /* 先读当前值，只修改控制位（保留 format/bitwidth 设置） */

  ret = es8311_read_reg(i2c, ES8311_SDPIN_REG09, &regv);
  regv &= 0xBF;  /* 清 bit6（DAC SDP MUTE=0，解除 DAC 静音） */
  ret = es8311_write_reg(i2c, ES8311_SDPIN_REG09, regv);

  /* ES8311 内部 ADC（ASDOUT）不用（麦克风走 ES7210），
   * 但按官方流程把 ADC 接口置为静音（bit6=1），避免内部 ADC 数据串扰 */

  ret = es8311_read_reg(i2c, ES8311_SDPOUT_REG0A, &regv);
  regv &= 0xBF;  /* 清 bit6 */
  regv |= 0x40;  /* 置 bit6（ADC SDP MUTE=1，静音内部 ADC） */
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

  ret = es8311_write_reg(i2c, ES8311_ADC_REG17, 0xBF);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0E, 0x02);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG12, 0x00);  /* DAC 电源使能 */
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG14, 0x1A);

  /* 不用 DMIC */

  ret = es8311_read_reg(i2c, ES8311_SYSTEM_REG14, &regv);
  regv &= ~(0x40);
  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG14, regv);

  /* 电源使能 */

  ret = es8311_write_reg(i2c, ES8311_SYSTEM_REG0D, 0x01);
  ret = es8311_write_reg(i2c, ES8311_ADC_REG15, 0x40);
  ret = es8311_write_reg(i2c, ES8311_DAC_REG37, 0x08);
  ret = es8311_write_reg(i2c, ES8311_GP_REG45, 0x00);

  if (ret < 0)
    {
      return ret;
    }

  return 0;
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
