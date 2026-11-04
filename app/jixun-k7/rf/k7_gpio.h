
/* k7_gpio.h - Linux GPIO chardev (v2 ioctl) 小封装, 供软件 SPI 使用 */
#ifndef K7_GPIO_H
#define K7_GPIO_H
#include <stdint.h>

typedef struct {
    void    *native_handle;
    int      chip_fd;
    int      line_fd;
    uint32_t offset;
    int      is_output;
    char     chip_name[16];
} k7_line_t;

/* chip_name 形如 "gpio3"; offset 为 bank 内偏移 (0..31) */
int  k7_line_open(k7_line_t *l, const char *chip_name, uint32_t offset,
                  int is_output, const char *consumer);
/* bias_flag: GPIO_V2_LINE_FLAG_BIAS_PULL_UP / _PULL_DOWN / 0 */
int  k7_line_open_bias(k7_line_t *l, const char *chip_name, uint32_t offset,
                       int is_output, const char *consumer, uint64_t bias_flag);
int  k7_line_set(const k7_line_t *l, int value);
int  k7_line_get(const k7_line_t *l, int *value);
void k7_line_close(k7_line_t *l);

#endif
