/****************************************************************************
 * apps/plant-companion/hal/hal_i2c.h
 *
 * Thin abstraction over NuttX I2C_TRANSFER().
 * Mirrors IDF i2c_master.h semantics but uses NuttX I2C API.
 *
 * All plant-companion components use this header — no direct I2C driver calls.
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_HAL_HAL_I2C_H
#define __APPS_PLANT_COMPANION_HAL_HAL_I2C_H

#include <nuttx/config.h>
#include <nuttx/i2c/i2c_master.h>

/* I2C bus handle type */
typedef struct i2c_master_s *hal_i2c_dev_t;

/**
 * @brief Write a register value to an I2C device.
 *
 * @param dev    I2C bus handle (from esp32s3_i2cbus_initialize)
 * @param addr   7-bit I2C address (do NOT left-shift)
 * @param reg    Register address
 * @param val    Value to write
 * @param freq   Bus frequency in Hz (e.g. 100000)
 * @return 0 on success, negative errno on failure
 */
static inline int hal_i2c_write_reg(hal_i2c_dev_t dev, uint8_t addr,
                                     uint8_t reg, uint8_t val,
                                     uint32_t freq)
{
  uint8_t buf[] = { reg, val };
  struct i2c_msg_s msg =
  {
    .frequency = freq,
    .addr      = addr,
    .flags     = 0,
    .buffer    = buf,
    .length    = sizeof(buf)
  };
  return I2C_TRANSFER(dev, &msg, 1);
}

/**
 * @brief Read a register from an I2C device.
 *
 * @param dev    I2C bus handle
 * @param addr   7-bit I2C address
 * @param reg    Register address
 * @param val    Output buffer for register value
 * @param freq   Bus frequency in Hz
 * @return 0 on success, negative errno on failure
 */
static inline int hal_i2c_read_reg(hal_i2c_dev_t dev, uint8_t addr,
                                    uint8_t reg, uint8_t *val,
                                    uint32_t freq)
{
  uint8_t reg_buf[] = { reg };
  struct i2c_msg_s msgv[2] =
  {
    {
      .frequency = freq,
      .addr      = addr,
      .flags     = 0,
      .buffer    = reg_buf,
      .length    = sizeof(reg_buf)
    },
    {
      .frequency = freq,
      .addr      = addr,
      .flags     = I2C_M_READ,
      .buffer    = val,
      .length    = 1
    }
  };
  return I2C_TRANSFER(dev, msgv, 2);
}

/**
 * @brief Burst-read multiple registers from an I2C device.
 *
 * @param dev        I2C bus handle
 * @param addr       7-bit I2C address
 * @param start_reg  Starting register address (auto-increment)
 * @param buf        Output buffer
 * @param len        Number of bytes to read
 * @param freq       Bus frequency in Hz
 * @return 0 on success, negative errno on failure
 */
static inline int hal_i2c_read_burst(hal_i2c_dev_t dev, uint8_t addr,
                                      uint8_t start_reg, uint8_t *buf,
                                      int len, uint32_t freq)
{
  uint8_t reg_buf[] = { start_reg };
  struct i2c_msg_s msgv[2] =
  {
    {
      .frequency = freq,
      .addr      = addr,
      .flags     = 0,
      .buffer    = reg_buf,
      .length    = sizeof(reg_buf)
    },
    {
      .frequency = freq,
      .addr      = addr,
      .flags     = I2C_M_READ,
      .buffer    = buf,
      .length    = len
    }
  };
  return I2C_TRANSFER(dev, msgv, 2);
}

#endif /* __APPS_PLANT_COMPANION_HAL_HAL_I2C_H */
