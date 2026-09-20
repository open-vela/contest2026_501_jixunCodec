/****************************************************************************
 * apps/plant-companion/components/voice_agent/es7210.h
 *
 * ES7210 4-ch Audio ADC register map — IDF-verified on CHQ V2.0 hardware.
 * I2C address: 0x40 (7-bit)
 * Shared bus: SDA=IO8, SCL=IO18 (I2C0)
 *
 * Register addresses corrected per espressif__esp_codec_dev es7210_reg.h
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_ES7210_H
#define __APPS_PLANT_COMPANION_ES7210_H

#include <stdint.h>
#include <stdbool.h>
#include <nuttx/i2c/i2c_master.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- ES7210 Register Map (IDF-verified from es7210_reg.h) ---- */
#define ES7210_RESET          0x00
#define ES7210_CLK_OFF        0x01
#define ES7210_MAINCLK        0x02
#define ES7210_MASTER_CLK     0x03
#define ES7210_LRCK_DIVH      0x04
#define ES7210_LRCK_DIVL      0x05
#define ES7210_POWER_DOWN     0x06
#define ES7210_OSR            0x07
#define ES7210_MODE_CFG       0x08
#define ES7210_TIME_CTRL0     0x09
#define ES7210_TIME_CTRL1     0x0A
#define ES7210_SDP_INTERFACE1 0x11
#define ES7210_SDP_INTERFACE2 0x12
#define ES7210_ADC_MUTE       0x13
#define ES7210_ADC_EN         0x17
#define ES7210_ANALOG_SYS     0x40
#define ES7210_MIC12_BIAS     0x41
#define ES7210_MIC34_BIAS     0x42
#define ES7210_MIC1_GAIN      0x43
#define ES7210_MIC2_GAIN      0x44
#define ES7210_MIC3_GAIN      0x45
#define ES7210_MIC4_GAIN      0x46
#define ES7210_MIC1_POWER     0x47
#define ES7210_MIC2_POWER     0x48
#define ES7210_MIC3_POWER     0x49
#define ES7210_MIC4_POWER     0x4A
#define ES7210_MIC12_POWER    0x4B
#define ES7210_MIC34_POWER    0x4C

/* ⚠️ 2026-08-21 修复：ADC12 HPF 寄存器地址曾写反（HPF2=0x22/HPF1=0x23），
 * 导致 es7210.c 的快速设置值 0x2A/0x0A 落到错误的寄存器 → HPF 未正确
 * 高通 → 录音含大 DC 偏移（均值 ~500）→ 不对称削顶。
 * 与 IDF es7210_reg.h 对齐：REG22=ADC12_HPF1(0x0A)、REG23=ADC12_HPF2(0x2A)。 */
#define ES7210_ADC34_HPF2     0x20
#define ES7210_ADC34_HPF1     0x21
#define ES7210_ADC12_HPF1     0x22
#define ES7210_ADC12_HPF2     0x23

/* ANALOG_SYS bits */
#define ES7210_LDO_EN         (1 << 0)
#define ES7210_REF_EN         (1 << 1)
#define ES7210_VMID_EN        (1 << 2)

/**
 * @brief Initialize ES7210 ADC via I2C (register config only, no I2S).
 *
 * Full init sequence from espressif__esp_codec_dev: analog power,
 * MIC bias, PGA gain (30dB), enable ADC channels 1+2, reset seq.
 *
 * @param i2c          I2C bus handle (I2C0)
 * @param sample_rate  Sample rate in Hz (e.g. 16000)
 * @param mclk_hz      MCLK frequency in Hz (e.g. 12288000)
 * @return 0 on success, negative errno on failure
 */
int es7210_init(FAR struct i2c_master_s *i2c,
                uint32_t sample_rate, uint32_t mclk_hz);

/**
 * @brief 恢复 ES7210 模拟配置（MCLK 启动后必须调用）。
 *
 * es7210_init 在无 MCLK 状态下配置码片；hal_i2s_start_tx_clock 启动
 * MCLK 后，REG40 bit0(LDO_EN) 可能被清（0x43→0x42）→ VMID 未建立 →
 * ADC 输入共模错误 → 极限环噪声（麦克风无声）。本函数重写全部模拟
 * 寄存器并读回验证 REG40。
 *
 * @param i2c I2C 总线句柄
 * @return 0 成功，负 errno 失败
 */
int es7210_restore_analog(FAR struct i2c_master_s *i2c);

/**
 * @brief ES7210 断电/恢复（rxprobe 变量分离实验用）
 */
int es7210_power_down(FAR struct i2c_master_s *i2c);
int es7210_power_up(FAR struct i2c_master_s *i2c);

#ifdef __cplusplus
}
#endif

#endif /* __APPS_PLANT_COMPANION_ES7210_H */
