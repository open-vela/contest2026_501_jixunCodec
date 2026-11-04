/****************************************************************************
 * apps/plant-companion/hal/hal_log.h
 *
 * Thin abstraction over NuttX logging (printf/syslog).
 * Mirrors IDF ESP_LOGI/ESP_LOGE semantics.
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_HAL_HAL_LOG_H
#define __APPS_PLANT_COMPANION_HAL_HAL_LOG_H

#include <stdio.h>

/* Tags are embedded in the format string for simplicity */
#define HAL_LOGI(tag, fmt, ...)  printf(tag ": " fmt "\n", ##__VA_ARGS__)
#define HAL_LOGE(tag, fmt, ...)  printf(tag ": ERROR: " fmt "\n", ##__VA_ARGS__)

#endif /* __APPS_PLANT_COMPANION_HAL_HAL_LOG_H */
