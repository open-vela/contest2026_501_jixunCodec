#ifndef K7_NATIVE_SPI_H
#define K7_NATIVE_SPI_H
#include <stdint.h>
struct spi_ioc_transfer { unsigned long tx_buf,rx_buf;uint32_t len,speed_hz;uint8_t bits_per_word,cs_change; };
#define GPIO_V2_LINE_FLAG_BIAS_DISABLED 0
int k7_native_spi_open(void);
int k7_native_spi_transfer(const struct spi_ioc_transfer *tr);
void k7_native_spi_close(void);
#endif
