/****************************************************************************
 * ai_ns.h — esp_sr NSNet2 神经网络降噪封装接口
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_AI_VOICE_AI_NS_H
#define __APPS_PLANT_COMPANION_AI_VOICE_AI_NS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化降噪器（NSNet2，模型内嵌固件）。返回 0 成功。 */
int ai_ns_init(void);

/* 处理一帧（ai_ns_frame_size() 个采样 @16kHz），int16 in-place。返回 0 成功。 */
int ai_ns_process_frame(int16_t *samples);

/* 返回每帧采样数（NSNet2，运行时确定，通常 480/512）。 */
int ai_ns_frame_size(void);

/* 返回采样率（16000）。 */
int ai_ns_samp_rate(void);

#ifdef __cplusplus
}
#endif

#endif /* __APPS_PLANT_COMPANION_AI_VOICE_AI_NS_H */
