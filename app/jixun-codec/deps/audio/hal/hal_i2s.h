/****************************************************************************
 * apps/plant-companion/hal/hal_i2s.h
 *
 * 最小 I2S HAL 层：直接操作 ESP32-S3 I2S0 外设寄存器
 * （绕过 NuttX 的 esp32s3_i2s 驱动，避免 RX DMA 无时钟的问题）
 *
 * 设计要点：
 * - master 模式，TX 和 RX 共享 BCK/WS
 * - 启动时先开 TX（静音）再开 RX，保证 BCK 始终有输出
 * - TX/RX 都走 DMA（复用 NuttX esp32s3_dma API，该驱动已验证 TX 可用）
 * - RX EOF 中断由 I2S 硬件触发（DMA 收满 I2S_RXEOF_NUM 个字节）
 *
 * 时钟：PLL_F160M(160MHz) / mclk_div=13 = ~12.307MHz MCLK
 *       bclk = rate * 2ch * 16bit = 1.536MHz (48kHz)
 *
 * GPIO 引脚（ESP32-S3-BOX-3 实测）：
 *   MCLK=GPIO2   BCLK=GPIO17   WS=GPIO45   DOUT=GPIO15   DIN=GPIO16
 *   PA_CTRL=GPIO46（高电平使能喇叭功放）
 ****************************************************************************/

#ifndef __APPS_PLANT_COMPANION_HAL_HAL_I2S_H
#define __APPS_PLANT_COMPANION_HAL_HAL_I2S_H

#include <nuttx/config.h>
#include <nuttx/arch.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 引脚定义（板级硬件） */

#define HAL_I2S_PIN_MCLK   2
#define HAL_I2S_PIN_BCLK  17
#define HAL_I2S_PIN_WS    45
#define HAL_I2S_PIN_DOUT  15
#define HAL_I2S_PIN_DIN   16
#define HAL_PA_PIN        46

/* 采样参数 */

#define HAL_I2S_RATE       48000
#define HAL_I2S_BITS       16
#define HAL_I2S_CHANNELS   1   /* mono */

/**
 * @brief 初始化 I2S0 外设 + GPIO 矩阵 + PA（启动 TX 让 BCK 持续输出）
 * @return 0 成功
 */
int hal_i2s_init(void);

/**
 * @brief 启动 TX 通道（静音数据，仅用于产生 BCK/WS 时钟）
 * @return 0 成功
 */
int hal_i2s_start_tx_clock(void);

/**
 * @brief DMA 收 RX 数据（阻塞，无中断依赖；适用于已知 BCK 跑着的场景）
 * @param buf  接收缓冲（int16_t *），大小 >= bytes
 * @param bytes  要收的字节数（整数个采样）
 * @return 实际收到的字节数，<0 失败
 */
int hal_i2s_read(void *buf, uint32_t bytes);

/**
 * @brief 单槽 RX 读取（录音专用）——只收指定槽（MIC1/MIC2）
 * @param buf  接收缓冲（int16_t *），大小 >= bytes
 * @param bytes  要收的字节数（整数个采样）
 * @param slot  槽号（0=左/MIC1，1=右/MIC2）
 * @return 实际收到的字节数，<0 失败
 */
int hal_i2s_read_slot(void *buf, uint32_t bytes, int slot);

/**
 * @brief 同时发+收（loopback 专用）：先启 RX DMA，再启 TX DMA
 *        TX 数据通过 SIG_LOOPBACK 到达 RX FIFO → RX 收满后返回。
 *        适用于内部回环测试。
 */

/**
 * @brief DMA 发 TX 数据（阻塞，等全部发完）
 * @param buf  发送缓冲
 * @param bytes  字节数
 * @return 实际发送字节数，<0 失败
 */
int hal_i2s_write(const void *buf, uint32_t bytes);

/**
 * @brief DMA 发 TX 数据（异步：排队即返回，不等发完）
 *        内部窗口化连续排队（≤3 块在途），官方驱动 ISR 无缝续链，
 *        消除"逐块等完成"造成的块间 DMA 空窗（纯音"吱吱"根因）。
 *        ⚠️ 驱动排队时已 memcpy 拷走数据，返回后调用方可复用 buf。
 * @param buf  发送缓冲
 * @param bytes  字节数
 * @return 实际入队字节数，<0 失败
 */
int hal_i2s_write_async(const void *buf, uint32_t bytes);

/**
 * @brief 等所有已排队的 TX 数据发完（hal_i2s_write_async 后调用；
 *        调用方要复用 buf / 换方向（录音）前必须 flush）
 * @return 0 成功；-ETIMEDOUT 驱动异常（已清状态）
 */
int hal_i2s_write_flush(void);

/**
 * @brief 选择 RX 采集槽位（诊断/自检用）
 * @param slot  0 = 左声道(MIC1)，1 = 右声道(MIC2)
 * @return 0 成功
 */
int hal_i2s_rx_set_channel(int slot);

/**
 * @brief 诊断：打印 I2S TX 状态（TX_CONF/TX_CONF1/STATE/TX_CLKM）
 *        无声排查用——播放"完成"却无声时看 TX_START/STOP_EN/IDLE
 */
void hal_i2s_dump_tx(void);

/**
 * @brief 运行时开关 SIG_LOOPBACK（TX/RX 内部共享 BCK/WS）——对照实验，
 *        无需重编译即可切换时钟来源（内部共享 vs 引脚回环）。
 * @param on  1=置位（共享），0=清零（引脚回环）
 * @return 0 成功
 */

#ifdef __cplusplus
}
#endif

#endif /* __APPS_PLANT_COMPANION_HAL_HAL_I2S_H */
