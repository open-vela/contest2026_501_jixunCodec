
/* sx1262.h - SX1262 (SX126x) 软件 SPI 驱动, FSK 38.4kbps, K7 用 */
#ifndef SX1262_H
#define SX1262_H
#include <stdint.h>
#include "k7_gpio.h"

/* 引脚规格字符串: "gpio3:0" (chip 名 : bank 内偏移), 未用传 NULL */
typedef struct {
    k7_line_t sck, mosi, miso, nss, rst, busy, dio1;
    int       has_dio1;
    int       has_rst;
    int       udelay;        /* 半周期忙等(微秒级微调), 默认 0 */
    k7_line_t sw_txen, sw_rxen;   /* 天线收发切换脚(模块 TXEN/RXEN), 不接则收发链路是断的 */
    int       has_sw;
    int       has_sw_txen, has_sw_rxen;
    int       sw_active_high;
    int       spifd;         /* >=0 表示用硬件 SPI(spidev); -1 表示软件位操作 SPI */
    uint32_t  spi_hz;
    int       no_busy_poll;      /* 1=不轮询 BUSY, 改用固定延时 */
    int       busy_delay_us;     /* 固定延时(us), 默认 2000 */
} sx1262_t;

/* 打开引脚并复位芯片 */
int  sx1262_init(sx1262_t *d, const char *sck, const char *mosi, const char *miso,
                 const char *nss, const char *rst, const char *busy, const char *dio1);
/* 绑定天线收发切换脚: active_high=1 表示 TXEN/RXEN 高有效 */
int  sx1262_bind_sw(sx1262_t *d, const char *txen_spec, const char *rxen_spec, int active_high);
/* 硬件 SPI 模式: SPI 收发走 spidev(片选由内核管理), 只需 BUSY / RST(可选) 两个 GPIO */
int  sx1262_init_spi(sx1262_t *d, const char *spidev_path, uint32_t speed_hz,
                     const char *busy_spec, const char *rst_spec, const char *dio1_spec);
/* 裸传输(调试用): 同时收发 n 字节, 不做 BUSY 之外的处理 */
int  sx1262_raw_transfer(sx1262_t *d, const uint8_t *tx, uint8_t *rx, int n);
int  sx1262_last_error(void);   /* 1=BUSY 超时, 2=SPI 传输失败 */
void sx1262_clear_error(void);
void sx1262_sw_tx(sx1262_t *d);   /* 切到发射 */
void sx1262_sw_rx(sx1262_t *d);   /* 切到接收 */
void sx1262_deinit(sx1262_t *d);

int  sx1262_reset(sx1262_t *d);
int  sx1262_wait_busy(sx1262_t *d, int timeout_ms);
int  sx1262_get_status(sx1262_t *d, uint8_t *st);
int  sx1262_calibrate(sx1262_t *d, uint8_t blocks);
int  sx1262_get_rssi_inst(sx1262_t *d, int *dbm);
int  sx1262_set_rf_freq(sx1262_t *d, uint32_t freq_hz);
/* 通用命令读写(供调试工具用) */
int  sx1262_cmd_write_public(sx1262_t *d, uint8_t op, const uint8_t *p, int n);
int  sx1262_cmd_read_public(sx1262_t *d, uint8_t op, const uint8_t *p, int n, int dummy_tail, uint8_t *out, int nout);
int  sx1262_get_device_errors(sx1262_t *d, uint16_t *err);
int  sx1262_clear_device_errors(sx1262_t *d);
int  sx1262_set_standby(sx1262_t *d);
int  sx1262_set_standby_xosc(sx1262_t *d);
int  sx1262_set_fallback(sx1262_t *d, uint8_t mode);
int  sx1262_set_tcxo(sx1262_t *d, int voltage_code, uint32_t delay_us);
int  sx1262_calibrate_image(sx1262_t *d, uint8_t band);
int  sx1262_config_fsk(sx1262_t *d, uint32_t freq_hz, int tx_dbm,
                       uint32_t bitrate_bps, uint32_t fdev_hz);
int  sx1262_config_lora(sx1262_t *d, uint32_t freq_hz, int tx_dbm,
                        uint8_t sf, uint8_t bw, uint8_t cr, uint8_t ldro);
int  sx1262_set_lora_pkt_params(sx1262_t *d, uint16_t preamble_syms, uint8_t hdr,
                                uint8_t plen, uint8_t crc, uint8_t iq);
int  sx1262_set_lora_payload_len(sx1262_t *d, uint8_t plen);
int  sx1262_set_payload_len(sx1262_t *d, uint8_t len);
int  sx1262_set_preamble(sx1262_t *d, uint16_t bits, uint8_t len);
/* 完整包参数: hdr_type 0=定长 1=变长; crc: 0x00=1B 0x02=2B 0x01=关; white: 0=关 1=开 */
int  sx1262_set_pkt_params_full(sx1262_t *d, uint16_t preamble_bits, uint8_t len, uint8_t sync_bits, uint8_t addr, uint8_t hdr, uint8_t crc);
/* RF 用: 变长包 + CRC2B + 指定白化 (dc_free: 1=开) */
int  sx1262_set_pkt_rf(sx1262_t *d, uint8_t len, uint8_t dc_free);
int  sx1262_set_mod_params_ex(sx1262_t *d, uint32_t bitrate, uint32_t fdev, uint8_t shape, uint8_t bw);
int  sx1262_set_pkt_params_ex(sx1262_t *d, uint16_t preamble_bits, uint8_t len,
                              uint8_t hdr_type, uint8_t crc_type, uint8_t white);
int  sx1262_set_sync_word(sx1262_t *d, const uint8_t sync[8]);
/* DIO2 自动驱动模块 TXEN(芯片在 TX 期间拉高) */
int  sx1262_set_dio2_rfswitch(sx1262_t *d, int on);
int  sx1262_write_buffer(sx1262_t *d, uint8_t off, const uint8_t *data, uint8_t len);
int  sx1262_read_buffer(sx1262_t *d, uint8_t off, uint8_t *data, uint8_t len);
/* 读取 RX 载荷, 自动处理 256 字节缓冲区绕回 */
int  sx1262_read_rx_payload(sx1262_t *d, uint8_t off, uint8_t *data, int len);
int  sx1262_start_tx(sx1262_t *d, uint32_t timeout_ms);
int  sx1262_start_rx(sx1262_t *d, uint32_t timeout_ms);
int  sx1262_get_irq(sx1262_t *d, uint16_t *irq);
int  sx1262_clear_irq(sx1262_t *d, uint16_t mask);
int  sx1262_get_rx_info(sx1262_t *d, uint8_t *len, uint8_t *start, int16_t *rssi_dbm);

/* 命令码 */
#define SX_CMD_SET_SLEEP          0x84
#define SX_CMD_SET_STANDBY        0x80
#define SX_CMD_SET_FS             0xC1
#define SX_CMD_SET_TX             0x83
#define SX_CMD_SET_RX             0x82
#define SX_CMD_SET_REGULATOR_MODE 0x96
#define SX_CMD_CALIBRATE          0x89
#define SX_CMD_CALIBRATE_IMAGE    0x98
#define SX_CMD_SET_PA_CFG         0x95
#define SX_CMD_SET_RX_TX_FALLBACK 0x93
#define SX_CMD_WRITE_REGISTER     0x0D
#define SX_CMD_READ_REGISTER      0x1D
#define SX_CMD_WRITE_BUFFER       0x0E
#define SX_CMD_READ_BUFFER        0x1E
#define SX_CMD_SET_DIO_IRQ_PARAMS 0x08
#define SX_CMD_GET_IRQ_STATUS     0x12
#define SX_CMD_CLR_IRQ_STATUS     0x02
#define SX_CMD_SET_DIO2_RFSWITCH  0x9D
#define SX_CMD_SET_DIO3_TCXO      0x97
#define SX_CMD_SET_RF_FREQUENCY   0x86
#define SX_CMD_SET_PACKET_TYPE    0x8A
#define SX_CMD_GET_PACKET_TYPE    0x11
#define SX_CMD_SET_TX_PARAMS      0x8E
#define SX_CMD_SET_MODULATION_PARAMS 0x8B
#define SX_CMD_SET_PACKET_PARAMS  0x8C
#define SX_CMD_SET_BUFFER_BASE    0x8F
#define SX_CMD_GET_STATUS         0xC0
#define SX_CMD_GET_RX_BUFFER_STAT 0x13
#define SX_CMD_GET_PACKET_STATUS  0x14
#define SX_CMD_GET_RSSI_INST      0x15
#define SX_CMD_GET_DEVICE_ERRORS  0x17
#define SX_CMD_CLR_DEVICE_ERRORS  0x07

#define SX_IRQ_TX_DONE    (1u << 0)
#define SX_IRQ_RX_DONE    (1u << 1)
#define SX_IRQ_PREAMBLE   (1u << 2)
#define SX_IRQ_SYNC_VALID (1u << 3)
#define SX_IRQ_HEADER_VALID (1u << 4)
#define SX_IRQ_HEADER_ERR (1u << 5)
#define SX_IRQ_CRC_ERR    (1u << 6)
#define SX_IRQ_TIMEOUT    (1u << 9)

#endif
