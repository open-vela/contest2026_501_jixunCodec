
/* sx1262.c - SX1262 软件 SPI (GPIO 位操作) 驱动
 * 目标: K7 排针 3.3V GPIO, FSK 38.4kbps @433MHz
 * 参考: Semtech Lora-net sx126x_driver (命令帧格式/参数编码一致)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "k7_native_spi.h"

#include "sx1262.h"

#define XTAL_HZ 32000000ULL
int g_sx1262_last_snr_q4 = 0;
int g_sx1262_last_err = 0;   /* 1=BUSY 超时, 2=SPI 传输失败 */
int sx1262_last_error(void) { return g_sx1262_last_err; }
void sx1262_clear_error(void) { g_sx1262_last_err = 0; }

static void bb_delay(const sx1262_t *d)
{
    if (d->udelay <= 0) return;
    for (volatile int i = 0; i < d->udelay * 20; i++) { }
}

/* SPI 模式0: SCK 空闲低, 上升沿采样 */
static void bb_tx_byte(sx1262_t *d, uint8_t b)
{
    for (int i = 7; i >= 0; i--) {
        k7_line_set(&d->mosi, (b >> i) & 1);
        bb_delay(d);
        k7_line_set(&d->sck, 1);
        bb_delay(d);
        k7_line_set(&d->sck, 0);
    }
}

static uint8_t bb_rx_byte(sx1262_t *d)
{
    uint8_t r = 0;
    for (int i = 7; i >= 0; i--) {
        k7_line_set(&d->sck, 1);
        int m = 0;
        k7_line_get(&d->miso, &m);
        r = (uint8_t)((r << 1) | (m & 1));
        bb_delay(d);
        k7_line_set(&d->sck, 0);
        bb_delay(d);
    }
    return r;
}

static void bb_write(sx1262_t *d, const uint8_t *p, int n)
{
    for (int i = 0; i < n; i++) bb_tx_byte(d, p[i]);
}

static void bb_read(sx1262_t *d, uint8_t *p, int n)
{
    for (int i = 0; i < n; i++) p[i] = bb_rx_byte(d);
}

int sx1262_wait_busy(sx1262_t *d, int timeout_ms)
{
    if (d->no_busy_poll) {   /* 不轮询 BUSY(线不干净时用固定延时替代) */
        struct timespec w = {0, (long)d->busy_delay_us * 1000L};
        nanosleep(&w, NULL);
        return 0;
    }
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        int v = 0;
        if (k7_line_get(&d->busy, &v) < 0) return -2;
        /* 毛刺过滤: 实测 BUSY 上有 ~50ns 串扰脉冲; 任一采样为低即视为空闲, 连续 5 次全高才算忙 */
        if (v) {
            int all_high = 1;
            for (int i = 0; i < 5; i++) {
                struct timespec w = {0, 2000};
                nanosleep(&w, NULL);
                int v2 = 0;
                if (k7_line_get(&d->busy, &v2) < 0) return -2;
                if (!v2) { all_high = 0; break; }
            }
            v = all_high;
        }
        if (!v) return 0;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long el = (now.tv_sec - t0.tv_sec) * 1000 + (now.tv_nsec - t0.tv_nsec) / 1000000;
        if (el > timeout_ms) return -1;
    }
}

/* 裸传输: 逐字节边发边收(调试/核对芯片 SPI 时序用) */
int sx1262_raw_transfer(sx1262_t *d, const uint8_t *tx, uint8_t *rx, int n)
{
    if (sx1262_wait_busy(d, 200) != 0) return -1;
    if (d->spifd >= 0) {
        struct spi_ioc_transfer tr;
        memset(&tr, 0, sizeof(tr));
        tr.tx_buf = (unsigned long)tx; tr.rx_buf = (unsigned long)rx;
        tr.len = (uint32_t)n; tr.speed_hz = d->spi_hz; tr.bits_per_word = 8;
        return (k7_native_spi_transfer(&tr) >= 0) ? 0 : -1;
    }
    for (int i = 0; i < n; i++) {
        uint8_t r = 0, b = tx ? tx[i] : 0;
        for (int k = 7; k >= 0; k--) {
            k7_line_set(&d->mosi, (b >> k) & 1);
            bb_delay(d);
            k7_line_set(&d->sck, 1);
            bb_delay(d);
            int m = 0;
            k7_line_get(&d->miso, &m);
            r = (uint8_t)((r << 1) | (m & 1));
            k7_line_set(&d->sck, 0);
            bb_delay(d);
        }
        if (rx) rx[i] = r;
    }
    return 0;
}

static int cmd_w(sx1262_t *d, uint8_t op, const uint8_t *p, int n)
{
    if (sx1262_wait_busy(d, 200) != 0) { g_sx1262_last_err = 1; return -1; }
    if (d->spifd >= 0) {
        uint8_t tx[300];
        if (1 + n > (int)sizeof(tx)) return -1;
        tx[0] = op;
        if (n > 0) memcpy(tx + 1, p, (size_t)n);
        struct spi_ioc_transfer tr;
        memset(&tr, 0, sizeof(tr));
        tr.tx_buf = (unsigned long)tx; tr.rx_buf = 0;
        tr.len = (uint32_t)(1 + n); tr.speed_hz = d->spi_hz;
        tr.bits_per_word = 8; tr.cs_change = 0;
        if (k7_native_spi_transfer(&tr) < 0) { g_sx1262_last_err = 2; return -1; }
        return 0;
    }
    k7_line_set(&d->nss, 0);
    bb_tx_byte(d, op);
    if (n > 0) bb_write(d, p, n);
    k7_line_set(&d->nss, 1);
    return 0;
}

/* dummy_tail: 命令缓冲末尾补 1 个 NOP (与官方驱动 SIZE 一致) */
static int cmd_r(sx1262_t *d, uint8_t op, const uint8_t *p, int n,
                 int dummy_tail, uint8_t *out, int nout)
{
    if (sx1262_wait_busy(d, 200) != 0) { g_sx1262_last_err = 1; return -1; }
    if (d->spifd >= 0) {
        /* 单次传输完成整个命令: [op][params][dummy][0*nout], 数据在 rx 尾部 */
        uint8_t tx[300], rx[300];
        int head = 1 + n + (dummy_tail ? 1 : 0);
        int total = head + nout;
        if (total > (int)sizeof(tx)) return -1;
        memset(tx, 0, (size_t)total);
        tx[0] = op;
        if (n > 0) memcpy(tx + 1, p, (size_t)n);
        struct spi_ioc_transfer tr;
        memset(&tr, 0, sizeof(tr));
        tr.tx_buf = (unsigned long)tx; tr.rx_buf = (unsigned long)rx;
        tr.len = (uint32_t)total; tr.speed_hz = d->spi_hz;
        tr.bits_per_word = 8; tr.cs_change = 0;
        if (k7_native_spi_transfer(&tr) < 0) {
            g_sx1262_last_err = 2;
            return -1;
        }
        if (nout > 0 && out) memcpy(out, rx + head, (size_t)nout);
        return 0;
    }
    k7_line_set(&d->nss, 0);
    bb_tx_byte(d, op);
    if (n > 0) bb_write(d, p, n);
    if (dummy_tail) bb_tx_byte(d, 0x00);
    if (nout > 0) bb_read(d, out, nout);
    k7_line_set(&d->nss, 1);
    return 0;
}

static int pin_spec_open(k7_line_t *l, const char *spec, int is_out, const char *consumer)
{
    char chip[16];
    unsigned off = 0;
    if (!spec) return -1;
    const char *colon = strchr(spec, ':');
    if (!colon) return -1;
    snprintf(chip, sizeof(chip), "%.*s", (int)(colon - spec), spec);
    off = (unsigned)strtoul(colon + 1, NULL, 10);
    if (is_out) return k7_line_open(l, chip, off, 1, consumer);
    return k7_line_open_bias(l, chip, off, 0, consumer, GPIO_V2_LINE_FLAG_BIAS_DISABLED);
}

/* 天线收发切换: 模块靠 TXEN/RXEN 两个脚选通 PA 或 LNA, 不接则收发都不通 */
int sx1262_init_spi(sx1262_t *d,const char *path,uint32_t hz,const char *busy,const char *rst,const char *dio1) {
 memset(d,0,sizeof(*d));d->spifd=-1;d->busy_delay_us=2000;
 int fd=k7_native_spi_open();if(fd<0)return -1;
 d->spifd=fd;d->spi_hz=hz?hz:1000000;
 if(pin_spec_open(&d->busy,busy,0,"sx-busy")<0)return -1;
 if(rst&&pin_spec_open(&d->rst,rst,1,"sx-rst")==0)d->has_rst=1;
 if(dio1&&pin_spec_open(&d->dio1,dio1,0,"sx-dio1")==0)d->has_dio1=1;
 return 0;
}

int sx1262_bind_sw(sx1262_t *d, const char *txen_spec, const char *rxen_spec, int active_high)
{
    if (!txen_spec && !rxen_spec) return -1;   /* 允许只接一根(另一根硬件上固定电平) */
    if (txen_spec && pin_spec_open(&d->sw_txen, txen_spec, 1, "sx-txen") == 0) d->has_sw_txen = 1;
    if (rxen_spec && pin_spec_open(&d->sw_rxen, rxen_spec, 1, "sx-rxen") == 0) d->has_sw_rxen = 1;
    if (!d->has_sw_txen && !d->has_sw_rxen) return -1;
    d->has_sw = 1;
    d->sw_active_high = active_high ? 1 : 0;
    sx1262_sw_rx(d);
    return 0;
}

void sx1262_sw_tx(sx1262_t *d)
{
    if (!d->has_sw) return;
    if (d->has_sw_txen) k7_line_set(&d->sw_txen, d->sw_active_high ? 1 : 0);
    if (d->has_sw_rxen) k7_line_set(&d->sw_rxen, d->sw_active_high ? 0 : 1);
}

void sx1262_sw_rx(sx1262_t *d)
{
    if (!d->has_sw) return;
    if (d->has_sw_txen) k7_line_set(&d->sw_txen, d->sw_active_high ? 0 : 1);
    if (d->has_sw_rxen) k7_line_set(&d->sw_rxen, d->sw_active_high ? 1 : 0);
}

int sx1262_init(sx1262_t *d, const char *sck, const char *mosi, const char *miso,
                const char *nss, const char *rst, const char *busy, const char *dio1)
{
    memset(d, 0, sizeof(*d));
    d->spifd = -1;
    if (pin_spec_open(&d->sck, sck, 1, "sx-sck") < 0) return -1;
    if (pin_spec_open(&d->mosi, mosi, 1, "sx-mosi") < 0) return -1;
    if (pin_spec_open(&d->miso, miso, 0, "sx-miso") < 0) return -1;
    if (pin_spec_open(&d->nss, nss, 1, "sx-nss") < 0) return -1;
    if (pin_spec_open(&d->busy, busy, 0, "sx-busy") < 0) return -1;
    d->has_rst = 0;
    if (rst && pin_spec_open(&d->rst, rst, 1, "sx-rst") == 0) d->has_rst = 1;
    d->has_dio1 = 0;
    if (dio1 && pin_spec_open(&d->dio1, dio1, 0, "sx-dio1") == 0) d->has_dio1 = 1;
    k7_line_set(&d->nss, 1);
    k7_line_set(&d->sck, 0);
    return 0;
}

void sx1262_deinit(sx1262_t *d)
{
    k7_line_close(&d->sck); k7_line_close(&d->mosi); k7_line_close(&d->miso);
    k7_line_close(&d->nss); k7_line_close(&d->busy);
    if (d->spifd >= 0) { k7_native_spi_close(); d->spifd = -1; }
    if (d->has_rst) k7_line_close(&d->rst);
    if (d->has_dio1) k7_line_close(&d->dio1);
    if (d->has_sw_txen) k7_line_close(&d->sw_txen);
    if (d->has_sw_rxen) k7_line_close(&d->sw_rxen);
}

int sx1262_reset(sx1262_t *d)
{
    if (!d->has_rst) return -1;
    k7_line_set(&d->rst, 0);
    struct timespec ts = {0, 200 * 1000 * 1000};
    nanosleep(&ts, NULL);
    k7_line_set(&d->rst, 1);
    ts.tv_nsec = 50 * 1000 * 1000;
    nanosleep(&ts, NULL);
    return 0;
}

int sx1262_get_status(sx1262_t *d, uint8_t *st)
{
    return cmd_r(d, SX_CMD_GET_STATUS, NULL, 0, 0, st, 1);
}

int sx1262_get_device_errors(sx1262_t *d, uint16_t *err)
{
    uint8_t b[2] = {0};
    if (cmd_r(d, SX_CMD_GET_DEVICE_ERRORS, NULL, 0, 1, b, 2) != 0) return -1;
    *err = (uint16_t)((b[0] << 8) | b[1]);
    return 0;
}

int sx1262_clear_device_errors(sx1262_t *d)
{
    return cmd_w(d, SX_CMD_CLR_DEVICE_ERRORS, NULL, 0);
}

int sx1262_set_standby(sx1262_t *d)
{
    uint8_t p = 0x00;   /* STDBY_RC */
    return cmd_w(d, SX_CMD_SET_STANDBY, &p, 1);
}

int sx1262_set_standby_xosc(sx1262_t *d)
{
    uint8_t p = 0x01;   /* STDBY_XOSC: 晶体保持运行, TDD快速切换 */
    return cmd_w(d, SX_CMD_SET_STANDBY, &p, 1);
}

int sx1262_set_fallback(sx1262_t *d, uint8_t mode)
{
    return cmd_w(d, SX_CMD_SET_RX_TX_FALLBACK, &mode, 1);
}

int sx1262_get_rssi_inst(sx1262_t *d, int *dbm)
{
    uint8_t out[2] = {0, 0};
    if (cmd_r(d, SX_CMD_GET_RSSI_INST, NULL, 0, 1, out, 1) != 0) return -1;
    *dbm = -(int)out[0] / 2;
    return 0;
}

int sx1262_calibrate(sx1262_t *d, uint8_t blocks)
{
    return cmd_w(d, SX_CMD_CALIBRATE, &blocks, 1);
}

int sx1262_set_rf_freq(sx1262_t *d, uint32_t freq_hz)
{
    uint64_t frf = ((uint64_t)freq_hz << 25) / XTAL_HZ;
    uint8_t p[4] = { (uint8_t)(frf >> 24), (uint8_t)(frf >> 16),
                     (uint8_t)(frf >> 8), (uint8_t)frf };
    return cmd_w(d, SX_CMD_SET_RF_FREQUENCY, p, 4);
}

static int write_register(sx1262_t *d, uint16_t addr, const uint8_t *data, int n)
{
    uint8_t buf[16];
    if (n > 12) return -1;
    buf[0] = (uint8_t)(addr >> 8);
    buf[1] = (uint8_t)(addr & 0xFF);
    memcpy(buf + 2, data, (size_t)n);
    return cmd_w(d, SX_CMD_WRITE_REGISTER, buf, n + 2);
}

int sx1262_set_tcxo(sx1262_t *d, int voltage_code, uint32_t delay_us)
{
    /* delay 单位 15.625us */
    uint32_t ticks = (uint32_t)((delay_us * 64) / 1000);
    uint8_t p[4] = { (uint8_t)voltage_code,
                     (uint8_t)(ticks >> 16), (uint8_t)(ticks >> 8), (uint8_t)ticks };
    return cmd_w(d, SX_CMD_SET_DIO3_TCXO, p, 4);
}

int sx1262_calibrate_image(sx1262_t *d, uint8_t band)
{
    uint8_t p[2] = { band, band == 0x6B ? 0x6F : 0x00 };
    return cmd_w(d, SX_CMD_CALIBRATE_IMAGE, p, 2);
}

/* DIO2 作为射频开关控制: 芯片在 TX 期间自动把 DIO2 拉高(模块用法见手册 4.2) */
int sx1262_set_dio2_rfswitch(sx1262_t *d, int on)
{
    uint8_t p[1] = { (uint8_t)(on ? 1 : 0) };
    return cmd_w(d, SX_CMD_SET_DIO2_RFSWITCH, p, 1);
}

int sx1262_set_sync_word(sx1262_t *d, const uint8_t sync[8])
{
    return write_register(d, 0x06C0, sync, 8);
}

int sx1262_write_buffer(sx1262_t *d, uint8_t off, const uint8_t *data, uint8_t len)
{
    uint8_t buf[2 + 255];
    buf[0] = off;
    memcpy(buf + 1, data, len);
    return cmd_w(d, SX_CMD_WRITE_BUFFER, buf, len + 1);
}

int sx1262_read_buffer(sx1262_t *d, uint8_t off, uint8_t *data, uint8_t len)
{
    uint8_t p = off;
    return cmd_r(d, SX_CMD_READ_BUFFER, &p, 1, 1, data, len);
}

int sx1262_config_fsk(sx1262_t *d, uint32_t freq_hz, int tx_dbm,
                      uint32_t bitrate_bps, uint32_t fdev_hz)
{
    uint8_t p[16];

    /* 1) 包类型 = GFSK */
    p[0] = 0x00;
    if (cmd_w(d, SX_CMD_SET_PACKET_TYPE, p, 1) != 0) return -1;

    /* 2) 频率 */
    uint64_t frf = ((uint64_t)freq_hz << 25) / XTAL_HZ;
    p[0] = (uint8_t)(frf >> 24); p[1] = (uint8_t)(frf >> 16);
    p[2] = (uint8_t)(frf >> 8);  p[3] = (uint8_t)(frf);
    if (cmd_w(d, SX_CMD_SET_RF_FREQUENCY, p, 4) != 0) return -1;

    /* 3) PA 配置 (SX1262, 14dBm 以内标准参数) */
    p[0] = 0x02;  /* paDutyCycle */
    p[1] = 0x02;  /* hpMax */
    p[2] = 0x00;  /* deviceSel: SX1262 */
    p[3] = 0x01;  /* paLut */
    if (cmd_w(d, SX_CMD_SET_PA_CFG, p, 4) != 0) return -1;

    /* 4) 发射功率 + 斜坡 200us */
    p[0] = (uint8_t)(int8_t)tx_dbm;
    p[1] = 0x04;
    if (cmd_w(d, SX_CMD_SET_TX_PARAMS, p, 2) != 0) return -1;

    /* 5) 调制参数: 码率/成形/带宽/频偏 */
    uint32_t br = (uint32_t)((32ULL * XTAL_HZ) / bitrate_bps);
    uint32_t fd = (uint32_t)(((uint64_t)fdev_hz << 25) / XTAL_HZ);
    p[0] = (uint8_t)(br >> 16); p[1] = (uint8_t)(br >> 8); p[2] = (uint8_t)(br);
    p[3] = 0x09;   /* 高斯成形 BT=0.5 */
    p[4] = 0x0B;   /* RX 带宽 117.3 kHz (DSB) */
    p[5] = (uint8_t)(fd >> 16); p[6] = (uint8_t)(fd >> 8); p[7] = (uint8_t)(fd);
    if (cmd_w(d, SX_CMD_SET_MODULATION_PARAMS, p, 8) != 0) return -1;

    /* 6) 包参数: 前导 32bit / 检测 32bit / 同步字 24bit / 变长 / CRC16 / 白化 */
    p[0] = 0x00; p[1] = 0x20;   /* 前导长度 32 bit */
    p[2] = 0x07;                /* 前导检测门限 32bit */
    p[3] = 24;                  /* 同步字长度 bit */
    p[4] = 0x00;                /* 不做地址过滤 */
    p[5] = 0x01;                /* 变长包 */
    p[6] = 255;                 /* 最大载荷 */
    p[7] = 0x02;                /* CRC 2 字节 */
    p[8] = 0x01;                /* 数据白化开 */
    if (cmd_w(d, SX_CMD_SET_PACKET_PARAMS, p, 9) != 0) return -1;

    /* 7) 缓冲基址 */
    p[0] = 0x00; p[1] = 0x00;
    if (cmd_w(d, SX_CMD_SET_BUFFER_BASE, p, 2) != 0) return -1;

    /* 8) DIO1 中断源 */
    uint16_t irq_mask = SX_IRQ_TX_DONE | SX_IRQ_RX_DONE | SX_IRQ_TIMEOUT |
                        SX_IRQ_CRC_ERR | SX_IRQ_HEADER_ERR | SX_IRQ_PREAMBLE | SX_IRQ_SYNC_VALID;
    uint16_t dio1_mask = SX_IRQ_TX_DONE | SX_IRQ_RX_DONE | SX_IRQ_TIMEOUT | SX_IRQ_CRC_ERR;
    p[0] = (uint8_t)(irq_mask >> 8); p[1] = (uint8_t)irq_mask;
    p[2] = (uint8_t)(dio1_mask >> 8); p[3] = (uint8_t)dio1_mask;
    p[4] = 0x00; p[5] = 0x00;
    p[6] = 0x00; p[7] = 0x00;
    if (cmd_w(d, SX_CMD_SET_DIO_IRQ_PARAMS, p, 8) != 0) return -1;

    /* 9) 校准 433MHz 频段镜像 */
    if (sx1262_calibrate_image(d, 0x6B) != 0) return -1;

    /* 10) 收发完成回落 STDBY_RC */
    p[0] = 0x20;
    if (cmd_w(d, SX_CMD_SET_RX_TX_FALLBACK, p, 1) != 0) return -1;

    return sx1262_clear_irq(d, 0xFFFF);
}

int sx1262_config_lora(sx1262_t *d, uint32_t freq_hz, int tx_dbm,
                       uint8_t sf, uint8_t bw, uint8_t cr, uint8_t ldro)
{
    uint8_t p[16];
    p[0] = 0x01;   /* LoRa */
    if (cmd_w(d, SX_CMD_SET_PACKET_TYPE, p, 1) != 0) return -1;
    if (sx1262_calibrate(d, 0x7F) != 0) return -1;   /* RC64K/RC13M/PLL/ADC 全校准 */
    uint64_t frf = ((uint64_t)freq_hz << 25) / XTAL_HZ;
    p[0] = (uint8_t)(frf >> 24); p[1] = (uint8_t)(frf >> 16);
    p[2] = (uint8_t)(frf >> 8);  p[3] = (uint8_t)frf;
    if (cmd_w(d, SX_CMD_SET_RF_FREQUENCY, p, 4) != 0) return -1;
    p[0] = 0x02;   /* SX1262 14dBm 标准 PA 配置 */
    p[1] = 0x02;
    p[2] = 0x00; p[3] = 0x01;
    if (cmd_w(d, SX_CMD_SET_PA_CFG, p, 4) != 0) return -1;
    p[0] = (uint8_t)(int8_t)tx_dbm;
    p[1] = 0x07;   /* 3400us ramp (更长, 兼容 LoRa/高功率) */
    if (cmd_w(d, SX_CMD_SET_TX_PARAMS, p, 2) != 0) return -1;
    p[0] = sf; p[1] = bw; p[2] = cr; p[3] = ldro;
    if (cmd_w(d, SX_CMD_SET_MODULATION_PARAMS, p, 4) != 0) return -1;
    if (sx1262_set_lora_pkt_params(d, 8, 0, 255, 1, 0) != 0) return -1;
    p[0] = 0x00; p[1] = 0x00;
    if (cmd_w(d, SX_CMD_SET_BUFFER_BASE, p, 2) != 0) return -1;
    uint16_t irq_mask = SX_IRQ_TX_DONE | SX_IRQ_RX_DONE | SX_IRQ_TIMEOUT |
                        SX_IRQ_CRC_ERR | SX_IRQ_HEADER_ERR | SX_IRQ_PREAMBLE | SX_IRQ_SYNC_VALID;
    uint16_t dio1_mask = SX_IRQ_TX_DONE | SX_IRQ_RX_DONE | SX_IRQ_TIMEOUT | SX_IRQ_CRC_ERR;
    p[0] = (uint8_t)(irq_mask >> 8); p[1] = (uint8_t)irq_mask;
    p[2] = (uint8_t)(dio1_mask >> 8); p[3] = (uint8_t)dio1_mask;
    p[4] = 0x00; p[5] = 0x00; p[6] = 0x00; p[7] = 0x00;
    if (cmd_w(d, SX_CMD_SET_DIO_IRQ_PARAMS, p, 8) != 0) return -1;
    if (sx1262_calibrate_image(d, 0x6B) != 0) return -1;
    p[0] = 0x20;
    if (cmd_w(d, SX_CMD_SET_RX_TX_FALLBACK, p, 1) != 0) return -1;
    return sx1262_clear_irq(d, 0xFFFF);
}

int sx1262_set_lora_pkt_params(sx1262_t *d, uint16_t preamble_syms, uint8_t hdr,
                               uint8_t plen, uint8_t crc, uint8_t iq)
{
    uint8_t p[6] = { (uint8_t)(preamble_syms >> 8), (uint8_t)preamble_syms,
                     hdr, plen, crc, iq };
    return cmd_w(d, SX_CMD_SET_PACKET_PARAMS, p, 6);
}

int sx1262_set_lora_payload_len(sx1262_t *d, uint8_t plen)
{
    return sx1262_set_lora_pkt_params(d, 8, 0, plen, 1, 0);
}

int sx1262_set_payload_len(sx1262_t *d, uint8_t len)
{
    uint8_t p[9] = { 0x00, 0x20, 0x07, 24, 0x00, 0x01, len, 0x02, 0x01 };
    return cmd_w(d, SX_CMD_SET_PACKET_PARAMS, p, 9);
}

/* 设置前导码长度(bit) 与载荷长度, 其余包参数不变 */
int sx1262_set_preamble(sx1262_t *d, uint16_t bits, uint8_t len)
{
    uint8_t p[9] = { (uint8_t)(bits >> 8), (uint8_t)bits, 0x07, 24, 0x00, 0x01, len, 0x02, 0x01 };
    return cmd_w(d, SX_CMD_SET_PACKET_PARAMS, p, 9);
}

int sx1262_set_pkt_params_ex(sx1262_t *d, uint16_t preamble_bits, uint8_t len,
                             uint8_t hdr_type, uint8_t crc_type, uint8_t white)
{
    uint8_t p[9] = { (uint8_t)(preamble_bits >> 8), (uint8_t)preamble_bits, 0x07, 24,
                     0x00, hdr_type, len, crc_type, white };
    return cmd_w(d, SX_CMD_SET_PACKET_PARAMS, p, 9);
}

int sx1262_set_mod_params_ex(sx1262_t *d, uint32_t bitrate, uint32_t fdev, uint8_t shape, uint8_t bw)
{
    uint32_t br = (uint32_t)((32ULL * XTAL_HZ) / bitrate);
    uint32_t fd = (uint32_t)(((uint64_t)fdev << 25) / XTAL_HZ);
    uint8_t p[8] = { (uint8_t)(br >> 16), (uint8_t)(br >> 8), (uint8_t)br,
                     shape, bw,
                     (uint8_t)(fd >> 16), (uint8_t)(fd >> 8), (uint8_t)fd };
    return cmd_w(d, SX_CMD_SET_MODULATION_PARAMS, p, 8);
}

/* RX 载荷读取: SX126x 缓冲区 256 字节, 载荷可能跨越末尾绕回, 分两段读 */
int sx1262_read_rx_payload(sx1262_t *d, uint8_t off, uint8_t *data, int len)
{
    if (len <= 0 || len > 256) return -1;
    int first = 256 - (int)off;
    if (first >= len) return sx1262_read_buffer(d, off, data, (uint8_t)len);
    if (sx1262_read_buffer(d, off, data, (uint8_t)first) != 0) return -1;
    return sx1262_read_buffer(d, 0x00, data + first, (uint8_t)(len - first));
}

int sx1262_set_pkt_rf(sx1262_t *d, uint8_t len, uint8_t dc_free)
{
    uint8_t p[9] = { 0x00, 64, 0x07, 32, 0x00, 0x01, len, 0x02, dc_free };
    return cmd_w(d, SX_CMD_SET_PACKET_PARAMS, p, 9);
}

int sx1262_cmd_write_public(sx1262_t *d, uint8_t op, const uint8_t *p, int n)
{
    return cmd_w(d, op, p, n);
}
int sx1262_cmd_read_public(sx1262_t *d, uint8_t op, const uint8_t *p, int n, int dummy_tail,
                           uint8_t *out, int nout)
{
    return cmd_r(d, op, p, n, dummy_tail, out, nout);
}

int sx1262_set_pkt_params_full(sx1262_t *d, uint16_t preamble_bits, uint8_t len, uint8_t sync_bits,
                               uint8_t addr, uint8_t hdr, uint8_t crc)
{
    uint8_t p[9] = { (uint8_t)(preamble_bits >> 8), (uint8_t)preamble_bits, 0x07, sync_bits,
                     addr, hdr, len, crc, 0x00 };
    return cmd_w(d, SX_CMD_SET_PACKET_PARAMS, p, 9);
}

int sx1262_start_tx(sx1262_t *d, uint32_t timeout_ms)
{
    /* timeout 单位 15.625us */
    uint32_t t = (uint32_t)((uint64_t)timeout_ms * 64);
    uint8_t p[3] = { (uint8_t)(t >> 16), (uint8_t)(t >> 8), (uint8_t)t };
    return cmd_w(d, SX_CMD_SET_TX, p, 3);
}

int sx1262_start_rx(sx1262_t *d, uint32_t timeout_ms)
{
    uint32_t t = (timeout_ms == 0) ? 0xFFFFFFu
                                   : (uint32_t)((uint64_t)timeout_ms * 64);
    uint8_t p[3] = { (uint8_t)(t >> 16), (uint8_t)(t >> 8), (uint8_t)t };
    return cmd_w(d, SX_CMD_SET_RX, p, 3);
}

int sx1262_get_irq(sx1262_t *d, uint16_t *irq)
{
    uint8_t b[2] = {0};
    if (cmd_r(d, SX_CMD_GET_IRQ_STATUS, NULL, 0, 1, b, 2) != 0) return -1;
    *irq = (uint16_t)((b[0] << 8) | b[1]);
    return 0;
}

int sx1262_clear_irq(sx1262_t *d, uint16_t mask)
{
    uint8_t p[2] = { (uint8_t)(mask >> 8), (uint8_t)mask };
    return cmd_w(d, SX_CMD_CLR_IRQ_STATUS, p, 2);
}

int sx1262_get_rx_info(sx1262_t *d, uint8_t *len, uint8_t *start, int16_t *rssi_dbm)
{
    uint8_t b[2] = {0};
    if (cmd_r(d, SX_CMD_GET_RX_BUFFER_STAT, NULL, 0, 1, b, 2) != 0) return -1;
    if (len)   *len = b[0];
    if (start) *start = b[1];

    uint8_t ps[3] = {0};
    if (cmd_r(d, SX_CMD_GET_PACKET_STATUS, NULL, 0, 1, ps, 3) != 0) return -1;
    uint8_t type=0;
    if(cmd_r(d,SX_CMD_GET_PACKET_TYPE,NULL,0,1,&type,1)!=0)return -1;
    if(type>1)return -1;
    g_sx1262_last_snr_q4 = type==1 ? (int)(int8_t)ps[1] : 0;
    /* LoRa packet RSSI is byte 0; GFSK sync RSSI is byte 1. */
    if (rssi_dbm) *rssi_dbm = (int16_t)(-(ps[type==1?0:1] / 2));
    return 0;
}
