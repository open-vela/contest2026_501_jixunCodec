/****************************************************************************
 * apps/plant-companion/hal/hal_delay.h
 *
 * Thin abstraction over NuttX delay/sleep functions.
 * Mirrors IDF vTaskDelay / ets_delay_us semantics.
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_HAL_HAL_DELAY_H
#define __APPS_PLANT_COMPANION_HAL_HAL_DELAY_H

#include <nuttx/config.h>
#include <nuttx/arch.h>
#include <nuttx/signal.h>

/**
 * @brief Busy-wait for the specified number of microseconds.
 *
 * Uses up_udelay() — CPU-spin, no context switch.
 */
static inline void hal_delay_us(uint32_t us)
{
  up_udelay(us);
}

/**
 * @brief Sleep for the specified number of milliseconds.
 *
 * Uses nxsig_usleep() — may yield to other threads.
 */
static inline void hal_delay_ms(uint32_t ms)
{
  nxsig_usleep(ms * 1000);
}

#endif /* __APPS_PLANT_COMPANION_HAL_HAL_DELAY_H */
