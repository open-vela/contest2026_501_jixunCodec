/****************************************************************************
 * apps/plant-companion/hal/hal_gpio.h
 *
 * Thin abstraction over NuttX esp32s3 GPIO API.
 * Maps IDF gpio_set_level/gpio_get_level semantics to NuttX.
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_HAL_HAL_GPIO_H
#define __APPS_PLANT_COMPANION_HAL_HAL_GPIO_H

#include <nuttx/config.h>
#include <stdbool.h>
#include "esp32s3_gpio.h"

/* GPIO direction and attribute flags (mirror IDF gpio_mode_t) */
#define HAL_GPIO_INPUT        INPUT
#define HAL_GPIO_OUTPUT       OUTPUT
#define HAL_GPIO_OPEN_DRAIN   OPEN_DRAIN
#define HAL_GPIO_PULLUP       PULLUP
#define HAL_GPIO_PULLDOWN     PULLDOWN

/**
 * @brief Configure a GPIO pin.
 *
 * @param pin   GPIO number (e.g. 4 for IO4)
 * @param attr  Attribute flags (INPUT | OUTPUT | OPEN_DRAIN | PULLUP | ...)
 */
static inline void hal_gpio_config(int pin, int attr)
{
  esp32s3_configgpio(pin, attr);
}

/**
 * @brief Write a boolean value to a GPIO pin.
 *
 * @param pin  GPIO number
 * @param val  true = high, false = low
 */
static inline void hal_gpio_write(int pin, bool val)
{
  esp32s3_gpiowrite(pin, val);
}

/**
 * @brief Read a boolean value from a GPIO pin.
 *
 * @param pin  GPIO number
 * @return true = high, false = low
 */
static inline bool hal_gpio_read(int pin)
{
  return esp32s3_gpioread(pin);
}

#endif /* __APPS_PLANT_COMPANION_HAL_HAL_GPIO_H */
