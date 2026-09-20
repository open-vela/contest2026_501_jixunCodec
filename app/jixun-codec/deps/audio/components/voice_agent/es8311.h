/****************************************************************************
 * apps/plant-companion/components/voice_agent/es8311.h
 *
 * ES8311 Audio DAC register map — IDF-verified on CHQ V2.0 hardware.
 * I2C address: 0x18 (AD0=GND, 7-bit)
 * Shared bus: SDA=IO8, SCL=IO18 (I2C0)
 *
 * WARNING: This register map differs from NuttX drivers/audio/es8311.h
 * (which is based on ESP-ADF). The IDF-verified map below matches the
 * actual ES8311 variant on CHQ-ESP32-S3-BOX V2.0.
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_ES8311_H
#define __APPS_PLANT_COMPANION_ES8311_H

#include <stdint.h>
#include <stdbool.h>
#include <nuttx/i2c/i2c_master.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- ES8311 Register Map (IDF-verified on CHQ V2.0) ---- */
#define ES8311_PWR_CTL0       0x02   /* Power control 0 */
#define ES8311_PWR_CTL1       0x03   /* Power control 1 (DAC analog) */
#define ES8311_CLK_CTL0       0x04   /* Clock control: MCLK div + LRCK div */
#define ES8311_CLK_CTL1       0x05   /* ADC clock control */
#define ES8311_I2S_CTL        0x09   /* I2S format: mode/bits/polarity */
#define ES8311_DAC_CTL        0x0C   /* DAC control: bit7=MUTE */
#define ES8311_DAC_VOL_L      0x10   /* Left channel volume (-96~+32 dB) */
#define ES8311_DAC_VOL_R      0x11   /* Right channel volume */

/* PWR_CTL0 (0x02) bits */
#define ES8311_PDN_ALL        (1 << 7)
#define ES8311_PDN_DAC        (1 << 6)
#define ES8311_PDN_ADC        (1 << 5)
#define ES8311_PDN_BIAS       (1 << 4)
#define ES8311_PDN_VREF       (1 << 3)

/* I2S_CTL (0x09) format values */
#define ES8311_I2S_FMT_STANDARD   (0 << 4)
#define ES8311_I2S_FMT_LEFT       (1 << 4)
#define ES8311_I2S_FMT_RIGHT      (2 << 4)
#define ES8311_I2S_FMT_DSP        (3 << 4)
#define ES8311_I2S_BIT_16         (2 << 2)
#define ES8311_I2S_BIT_24         (0 << 2)
#define ES8311_I2S_BIT_32         (3 << 2)
#define ES8311_I2S_MODE_SLAVE     (0 << 0)
#define ES8311_I2S_MODE_MASTER    (1 << 0)

/**
 * @brief Initialize ES8311 DAC via I2C.
 *
 * Power-up sequence: enable DAC+BIAS+VREF, configure I2S slave mode
 * (16-bit, standard Philips), set 0dB volume, unmute.
 *
 * @param i2c   I2C bus handle (I2C0, already initialized by board bringup)
 * @return 0 on success, negative errno on failure
 */
int es8311_init(FAR struct i2c_master_s *i2c);

/**
 * @brief 配置 I2S 格式 + 时钟（IDF 流程的 set_fs）
 * @param i2c   I2C bus handle
 * @param sample_rate  采样率（如 48000）
 */
int es8311_set_format(FAR struct i2c_master_s *i2c, int sample_rate);

/**
 * @brief 使能 DAC + ADC（IDF 流程的 start）
 * @return 0 on success
 */
int es8311_start(FAR struct i2c_master_s *i2c);

/**
 * @brief Set DAC volume for both channels.
 * @param i2c  I2C bus handle
 * @param vol  0x00 (-96dB) to 0xFF (+32dB). 0xC0 = 0dB.
 * @return 0 on success
 */
int es8311_set_volume(FAR struct i2c_master_s *i2c, uint8_t vol);

/**
 * @brief Mute/unmute DAC output.
 * @param i2c   I2C bus handle
 * @param mute  true = mute, false = normal
 * @return 0 on success
 */
int es8311_mute(FAR struct i2c_master_s *i2c, bool mute);

/**
 * @brief 寄存器全览（无声排查：对比 小智 esp_codec_dev / OpenVela NuttX 参考值）
 * @param i2c  I2C bus handle
 * @return 0 on success
 */
int es8311_dump_regs(FAR struct i2c_master_s *i2c);

/**
 * @brief DAC 输出链路关键寄存器（播放无声判读：MUTE/音量/电源/数据源）
 * @param i2c  I2C bus handle
 * @return 0 on success
 */
int es8311_dump_dac(FAR struct i2c_master_s *i2c);

#ifdef __cplusplus
}
#endif

#endif /* __APPS_PLANT_COMPANION_ES8311_H */
