/****************************************************************************
 * apps/plant-companion/components/voice_agent/voice_agent.h
 *
 * Combined audio subsystem public API.
 *
 * 外设层职责：让 ES8311（播放）+ ES7210（录音）+ I2S0 正常运行。
 * AI 应用层（ai_module/ai_voice）只调用本层 API。
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_VOICE_AGENT_H
#define __APPS_PLANT_COMPANION_VOICE_AGENT_H

#include "es8311.h"
#include "es7210.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* 初始化音频外设（ES8311 + ES7210 + I2S0），幂等，可重复调用。
 * @return 0 成功，<0 失败（errno 风格） */
int voice_agent_init(void);

/* DAC 输出音量（ES8311 REG32，0x00=-95.5dB .. 0xFF=+32dB，0xBF≈0dB）。
 * 初始化后调用立即生效；初始化前调用仅记录，init 时套用。
 * @return 0 成功，<0 写寄存器失败（errno 风格） */
int voice_agent_set_dac_volume(uint8_t vol);

/* 读取当前 DAC 音量寄存器值 */
uint8_t voice_agent_get_dac_volume(void);

/* 空闲/播放时控制 ES8311 DAC mute（1=静音，0=输出）。 */
int voice_agent_set_dac_mute(int mute);

/* 麦克风外设自检：采集原始 I2S RX（24kHz 单声道）并统计非零率/峰值。
 * @param seconds 采集秒数（1~10，默认 2）
 * @return 0 通路正常，<0 失败/无数据 */

#ifdef __cplusplus
}
#endif

#endif /* __APPS_PLANT_COMPANION_VOICE_AGENT_H */
