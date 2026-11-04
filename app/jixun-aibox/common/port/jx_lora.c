#include "jx_wire_v2.h"
#include "jx_wire_ll.h"
/****************************************************************************
 * apps/jixun-doc/port/jx_lora.c
 *
 * SX1262 LoRa 驱动（ESP32-S3 / NuttX）
 *   SPI3: CS=13 CLK=17 MOSI=18 MISO=21  (1 MHz, mode 0)
 *   GPIO: BUSY=40 TXEN=11 RXEN=12
 *
 * 空口参数与乙板 K7 对齐：
 *   433.175 MHz, SF7, BW125(0x04), CR4/5(1), LDRO=0,
 *   前导 8 symbols, 显式头, CRC 开, 52 字节定长
 ****************************************************************************/

#include <nuttx/config.h>

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>

int jx_decode(const int64_t *idx, int Tt, int origT, float **out_pcm);
#include <poll.h>

#include <nuttx/spi/spi_transfer.h>

#include "esp32s3_gpio.h"
#include "stream_codec.h"
#include "model_config.h"

extern int jx_serial_send_pcm_i16(const char *kind, const int16_t *pcm,
                                  int nsamples, int rate);

#define LORA_SPI_PATH   "/dev/spi3"
#define LORA_PIN_BUSY   40
#define LORA_PIN_TXEN   11
#define LORA_PIN_RXEN   12
#define LORA_SPI_HZ     1000000
#define LORA_PKT_LEN    52

#define XTAL_HZ         32000000ULL

/* SX1262 命令码 */
#define SX_CMD_SET_SLEEP            0x84
#define SX_CMD_SET_STANDBY          0x80
#define SX_CMD_SET_TX               0x83
#define SX_CMD_SET_RX               0x82
#define SX_CMD_CALIBRATE            0x89
#define SX_CMD_CALIBRATE_IMAGE      0x98
#define SX_CMD_SET_PA_CFG           0x95
#define SX_CMD_SET_RX_TX_FALLBACK   0x93
#define SX_CMD_WRITE_REGISTER       0x0D
#define SX_CMD_READ_REGISTER        0x1D
#define SX_CMD_WRITE_BUFFER         0x0E
#define SX_CMD_READ_BUFFER          0x1E
#define SX_CMD_SET_DIO_IRQ_PARAMS   0x08
#define SX_CMD_GET_IRQ_STATUS       0x12
#define SX_CMD_CLR_IRQ_STATUS       0x02
#define SX_CMD_SET_RF_FREQUENCY     0x86
#define SX_CMD_SET_PACKET_TYPE      0x8A
#define SX_CMD_SET_TX_PARAMS        0x8E
#define SX_CMD_SET_MODULATION_PARAMS 0x8B
#define SX_CMD_SET_PACKET_PARAMS    0x8C
#define SX_CMD_SET_BUFFER_BASE      0x8F
#define SX_CMD_GET_STATUS           0xC0
#define SX_CMD_GET_RX_BUFFER_STAT   0x13
#define SX_CMD_GET_PACKET_STATUS    0x14
#define SX_CMD_GET_RSSI_INST        0x15
#define SX_CMD_GET_DEVICE_ERRORS    0x17
#define SX_CMD_CLR_DEVICE_ERRORS    0x07

#define SX_IRQ_TX_DONE      (1u << 0)
#define SX_IRQ_RX_DONE      (1u << 1)
#define SX_IRQ_PREAMBLE     (1u << 2)
#define SX_IRQ_SYNC_VALID   (1u << 3)
#define SX_IRQ_HEADER_VALID (1u << 4)
#define SX_IRQ_HEADER_ERR   (1u << 5)
#define SX_IRQ_CRC_ERR      (1u << 6)
#define SX_IRQ_TIMEOUT      (1u << 9)

static int g_fd = -1;
static pid_t g_fd_owner = -1;

/* 2026-10-29: BUSY 飞线接触不良时，GPIO40 被内部上拉读成永久高，
 * 所有 SPI 命令都会在 lora_wait_busy() 里超时，芯片明明还能收能发。
 * 连续 3 次超时后切到「盲发」：不再看 BUSY，按固定 2 ms 延时后直接
 * 发命令。SX1262 单条命令 < 1 ms，2 ms 足够。 */
static int g_busy_blind;
static int g_busy_timeouts;

/* jx_main.c 的发送占用标志：置 1 时自动接收任务必须让出射频。 */
extern volatile int g_lora_tx_busy;

extern void jx_ui_set_status(const char *s);
extern void jx_ui_set_rx(int pkt, int rssi);
extern void jx_ui_set_tx(int pkt);

static void tiny_delay(void)
{
  for (volatile int i = 0; i < 400; i++)
    {
      __asm__ __volatile__("nop");
    }
}

static int lora_wait_busy(int timeout_ms)
{
  int i;

  if (g_busy_blind)
    {
      usleep(2000);
      return 0;
    }

  /* 2026-10-29: 不再用 clock_gettime 算超时。现场出现过「卡住后
   * 一百多秒不返回」，怀疑时钟读数异常导致 el 永远不增长。
   * 改成固定 1 ms 步进的计数循环，最坏情况也一定退出。 */
  for (i = 0; i <= timeout_ms; i++)
    {
      int v = esp32s3_gpioread(LORA_PIN_BUSY) ? 1 : 0;

      /* BUSY 上有约 50ns 串扰毛刺：连续 5 次全高才认为忙 */
      if (v)
        {
          int all_high = 1;
          for (int i = 0; i < 5; i++)
            {
              tiny_delay();
              if (!esp32s3_gpioread(LORA_PIN_BUSY))
                {
                  all_high = 0;
                  break;
                }
            }
          v = all_high;
        }

      if (!v)
        {
          return 0;
        }

      usleep(1000);
    }

  if (++g_busy_timeouts >= 3)
    {
      g_busy_blind = 1;
      printf("[lora] BUSY stuck high x%d -> blind mode (fixed 2ms delay)\n",
             g_busy_timeouts);
    }

  printf("[txdbg] busy timeout after %d ms (total=%d)\n",
         (int)timeout_ms, (int)g_busy_timeouts);
  return -1;
}

static sem_t g_spi_sem;
static int g_spi_sem_ready;

static int lora_xfer_raw(const uint8_t *tx, uint8_t *rx, size_t n)
{
  struct spi_trans_s tr;
  struct spi_sequence_s seq;

  memset(&tr, 0, sizeof(tr));
  memset(&seq, 0, sizeof(seq));

  tr.deselect = true;
  tr.nwords   = n;
  tr.txbuffer = tx;
  tr.rxbuffer = rx;

  seq.dev       = SPIDEV_USER(0);
  seq.mode      = SPIDEV_MODE0;
  seq.nbits     = 8;
  seq.ntrans    = 1;
  seq.frequency = LORA_SPI_HZ;
  seq.trans     = &tr;

  {
    int ret = ioctl(g_fd, SPIIOC_TRANSFER, &seq);
    if (ret < 0)
      {
        static int once;
        if (once < 6)
          {
            once++;
            printf("[lora] spi ioctl ret=%d errno=%d (%s) n=%u freq=%u mode=%u bits=%u dev=0x%08lx\n",
                   ret, errno, strerror(errno), (unsigned)n,
                   (unsigned)seq.frequency, (unsigned)seq.mode,
                   (unsigned)seq.nbits, (unsigned long)seq.dev);
          }
      }
    return ret;
  }
}

/* 2026-10-29: 发送任务和接收任务共用 /dev/spi3。半双工切换时接收
 * 线程可能正卡在一次 sx_get_irq 里，发送线程立刻开始 lora_tx_packet，
 * 两个线程同时 ioctl(SPIIOC_TRANSFER) 会让 NuttX SPI 驱动永久阻塞
 * —— 表现就是连发到第 3 次、发到一半整机卡死，串口也不响应。
 *
 * 用进程级 sem + 500 ms 超时，而不是 pthread_mutex：两个任务是
 * task_create 出来的不同 task group，pthread_mutex 跨 group 不可靠；
 * 而且一旦某个线程在 ioctl 里卡住，没有超时的锁会让另一个线程
 * 永久等下去。拿不到锁就返回 -1，让上层走失败/重试。 */
static void spi_lock_init(void)
{
  if (!g_spi_sem_ready)
    {
      sem_init(&g_spi_sem, 0, 1);
      g_spi_sem_ready = 1;
    }
}

static int lora_xfer(const uint8_t *tx, uint8_t *rx, size_t n)
{
  struct timespec ts;
  int ret;

  spi_lock_init();

  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_nsec += 500 * 1000 * 1000;
  if (ts.tv_nsec >= 1000000000L)
    {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000L;
    }

  if (sem_timedwait(&g_spi_sem, &ts) != 0)
    {
      printf("[lora] spi lock timeout\n");
      return -1;
    }

  {
    uint32_t _t0 = clock_systime_ticks();
    ret = lora_xfer_raw(tx, rx, n);
    {
      uint32_t _dt = (clock_systime_ticks() - _t0) * 1000u / TICK_PER_SEC;
      if (_dt >= 300)
        {
          printf("[txdbg] spi op=%02x n=%u took %u ms ret=%d\n",
                 (unsigned)(n > 0 ? tx[0] : 0), (unsigned)n,
                 (unsigned)_dt, ret);
        }
    }
  }
  sem_post(&g_spi_sem);
  return ret;
}

static int cmd_w(uint8_t op, const uint8_t *p, int n)
{
  uint8_t tx[300];

  if (lora_wait_busy(200) != 0)
    {
      return -1;
    }

  if (1 + n > (int)sizeof(tx))
    {
      return -1;
    }

  tx[0] = op;
  if (n > 0)
    {
      memcpy(tx + 1, p, (size_t)n);
    }

  return lora_xfer(tx, NULL, (size_t)(1 + n));
}

static int cmd_r(uint8_t op, const uint8_t *p, int n, int dummy_tail,
                 uint8_t *out, int nout)
{
  uint8_t tx[300];
  uint8_t rx[300];
  int head = 1 + n + (dummy_tail ? 1 : 0);
  int total = head + nout;

  if (lora_wait_busy(200) != 0)
    {
      return -1;
    }

  if (total > (int)sizeof(tx))
    {
      return -1;
    }

  memset(tx, 0, (size_t)total);
  memset(rx, 0, (size_t)total);
  tx[0] = op;
  if (n > 0)
    {
      memcpy(tx + 1, p, (size_t)n);
    }

  if (lora_xfer(tx, rx, (size_t)total) < 0)
    {
      return -1;
    }

  if (nout > 0 && out)
    {
      memcpy(out, rx + head, (size_t)nout);
    }

  return 0;
}

static void sw_tx(void)
{
  /* 2026-10-27 对照：TXEN 低有效实验 */
  esp32s3_gpiowrite(LORA_PIN_TXEN, 0);
  esp32s3_gpiowrite(LORA_PIN_RXEN, 0);
}

static void sw_rx(void)
{
  /* 2026-10-27 对照：TXEN 低有效实验（RX 时拉高关闭 TX） */
  esp32s3_gpiowrite(LORA_PIN_TXEN, 1);
  esp32s3_gpiowrite(LORA_PIN_RXEN, 1);
}

static int sx_get_status(uint8_t *st)
{
  return cmd_r(SX_CMD_GET_STATUS, NULL, 0, 0, st, 1);
}

static int sx_get_device_errors(uint16_t *err)
{
  uint8_t b[2] = {0, 0};

  if (cmd_r(SX_CMD_GET_DEVICE_ERRORS, NULL, 0, 1, b, 2) != 0)
    {
      return -1;
    }

  *err = (uint16_t)((b[0] << 8) | b[1]);
  return 0;
}

static int sx_clear_device_errors(void)
{
  return cmd_w(SX_CMD_CLR_DEVICE_ERRORS, NULL, 0);
}

static int sx_set_standby(void)
{
  uint8_t p = 0x00;
  return cmd_w(SX_CMD_SET_STANDBY, &p, 1);
}

static int sx_set_standby_xosc(void)
{
  uint8_t p = 0x01;
  return cmd_w(SX_CMD_SET_STANDBY, &p, 1);
}

static int sx_calibrate(uint8_t blocks)
{
  return cmd_w(SX_CMD_CALIBRATE, &blocks, 1);
}

static int sx_calibrate_image(uint8_t band)
{
  uint8_t p[2] = {band, 0x00};
  return cmd_w(SX_CMD_CALIBRATE_IMAGE, p, 2);
}

static int sx_set_rf_freq(uint32_t freq_hz)
{
  uint64_t frf = ((uint64_t)freq_hz << 25) / XTAL_HZ;
  uint8_t p[4];

  p[0] = (uint8_t)(frf >> 24);
  p[1] = (uint8_t)(frf >> 16);
  p[2] = (uint8_t)(frf >> 8);
  p[3] = (uint8_t)frf;

  return cmd_w(SX_CMD_SET_RF_FREQUENCY, p, 4);
}

static int sx_set_lora_pkt_params(uint16_t preamble_syms, uint8_t hdr,
                                  uint8_t plen, uint8_t crc, uint8_t iq)
{
  uint8_t p[6];

  p[0] = (uint8_t)(preamble_syms >> 8);
  p[1] = (uint8_t)preamble_syms;
  p[2] = hdr;
  p[3] = plen;
  p[4] = crc;
  p[5] = iq;

  return cmd_w(SX_CMD_SET_PACKET_PARAMS, p, 6);
}

static int sx_write_buffer(uint8_t off, const uint8_t *data, uint8_t len)
{
  uint8_t buf[2 + 255];

  buf[0] = off;
  memcpy(buf + 1, data, len);
  return cmd_w(SX_CMD_WRITE_BUFFER, buf, len + 1);
}

static int sx_read_buffer(uint8_t off, uint8_t *data, uint8_t len)
{
  return cmd_r(SX_CMD_READ_BUFFER, &off, 1, 1, data, len);
}

static int sx_read_rx_payload(uint8_t off, uint8_t *data, int len)
{
  int first = 256 - (int)off;

  if (len <= 0 || len > 256)
    {
      return -1;
    }

  if (first >= len)
    {
      return sx_read_buffer(off, data, (uint8_t)len);
    }

  if (sx_read_buffer(off, data, (uint8_t)first) != 0)
    {
      return -1;
    }

  return sx_read_buffer(0x00, data + first, (uint8_t)(len - first));
}

static int sx_start_tx(uint32_t timeout_ms)
{
  uint32_t t = (uint32_t)((uint64_t)timeout_ms * 64);
  uint8_t p[3];

  p[0] = (uint8_t)(t >> 16);
  p[1] = (uint8_t)(t >> 8);
  p[2] = (uint8_t)t;

  return cmd_w(SX_CMD_SET_TX, p, 3);
}

static int sx_start_rx(uint32_t timeout_ms)
{
  uint32_t t = (timeout_ms == 0) ? 0xFFFFFFu
                                 : (uint32_t)((uint64_t)timeout_ms * 64);
  uint8_t p[3];

  p[0] = (uint8_t)(t >> 16);
  p[1] = (uint8_t)(t >> 8);
  p[2] = (uint8_t)t;

  return cmd_w(SX_CMD_SET_RX, p, 3);
}

static int sx_get_irq(uint16_t *irq)
{
  uint8_t b[2] = {0, 0};

  if (cmd_r(SX_CMD_GET_IRQ_STATUS, NULL, 0, 1, b, 2) != 0)
    {
      return -1;
    }

  *irq = (uint16_t)((b[0] << 8) | b[1]);
  return 0;
}

static int sx_clear_irq(uint16_t mask)
{
  uint8_t p[2];

  p[0] = (uint8_t)(mask >> 8);
  p[1] = (uint8_t)mask;
  return cmd_w(SX_CMD_CLR_IRQ_STATUS, p, 2);
}

static int sx_get_rx_info(uint8_t *len, uint8_t *start, int16_t *rssi_dbm)
{
  uint8_t b[2] = {0, 0};
  uint8_t ps[3] = {0, 0, 0};

  if (cmd_r(SX_CMD_GET_RX_BUFFER_STAT, NULL, 0, 1, b, 2) != 0)
    {
      return -1;
    }

  if (len)
    {
      *len = b[0];
    }
  if (start)
    {
      *start = b[1];
    }

  if (cmd_r(SX_CMD_GET_PACKET_STATUS, NULL, 0, 1, ps, 3) != 0)
    {
      return -1;
    }

  if (rssi_dbm)
    {
      /* SX1262 GetPacketStatus 返回 3 字节：
       *   ps[0] = RssiPkt (0.5 dBm/LSB)
       *   ps[1] = SnrPkt  (0.25 dB/LSB)
       *   ps[2] = SignalRssiPkt
       * 之前取 ps[1] 把 SNR 当成了 RSSI（-25 实际是 12 dB 信噪比）。
       */
      *rssi_dbm = (int16_t)(-(ps[0] / 2));

      /* 2026-10-30: 把 GetPacketStatus 三字节原样打出来，现场日志
       * 就能自证 ps[0] 是 dBm、ps[1] 是 SNR，防止再把 SNR 当 RSSI。 */
      {
        static int dbg;
        if (dbg < 3)
          {
            dbg++;
            printf("[rssi] raw rssiPkt=%u snrPkt=%u sigRssi=%u -> %d dBm (snr %d.%02d dB)\n",
                   (unsigned)ps[0], (unsigned)ps[1], (unsigned)ps[2],
                   (int)(-(ps[0] / 2)),
                   (int)((int8_t)ps[1] / 4),
                   (int)(((int8_t)ps[1] % 4) * 25));
          }
      }
    }

  return 0;
}

static int sx_get_snr(void)
{
  uint8_t ps[3] = {0, 0, 0};

  if (cmd_r(SX_CMD_GET_PACKET_STATUS, NULL, 0, 1, ps, 3) != 0)
    {
      return 0;
    }

  return (int)((int8_t)ps[1] / 4);
}

static int jx_lora_dbm(void)
{
  const char *e = getenv("JX_LORA_DBM");
  if (e != NULL && e[0] != 0)
    {
      int v = atoi(e);
      if (v < -9) { v = -9; }
      if (v > 22) { v = 22; }
      return v;
    }
  return 22;
}

/* 2026-10-22: LoRa 带宽可配置，两侧必须一致。
 * 0x05=41.67k(原值), 0x06=62.5k, 0x07=125k, 0x08=250k, 0x09=500k。
 * 只影响空口时间，不影响包格式。 */
static uint8_t jx_lora_sf(void)
{
  const char *e = getenv("JX_LORA_SF");
  if (e != NULL && e[0] != '\0')
    {
      int v = atoi(e);
      if (v < 5) { v = 5; }
      if (v > 12) { v = 12; }
      return (uint8_t)v;
    }
  return 7;
}

static uint8_t jx_lora_bw(void)
{
  const char *e = getenv("JX_LORA_BW");
  long v;
  if (e == NULL || e[0] == '\0') { return 0x06; }
  v = strtol(e, NULL, 0);
  if (v < 0 || v > 0x09) { return 0x05; }
  return (uint8_t)v;
}

static int sx_config_lora(uint32_t freq_hz, int tx_dbm,
                          uint8_t sf, uint8_t bw, uint8_t cr, uint8_t ldro)
{
  uint8_t p[16];

  printf("[lora] config sf=%u bw=0x%02x cr=%u ldro=%u dbm=%d\n",
         (unsigned)sf, (unsigned)bw, (unsigned)cr, (unsigned)ldro, tx_dbm);

  p[0] = 0x01;
  if (cmd_w(SX_CMD_SET_PACKET_TYPE, p, 1) != 0)
    {
      return -1;
    }

  if (sx_calibrate(0x7F) != 0)
    {
      return -1;
    }

  if (sx_set_rf_freq(freq_hz) != 0)
    {
      return -1;
    }

  p[0] = 0x04;
  p[1] = 0x07;
  p[2] = 0x00;
  p[3] = 0x01;
  if (cmd_w(SX_CMD_SET_PA_CFG, p, 4) != 0)
    {
      return -1;
    }

  p[0] = (uint8_t)(int8_t)tx_dbm;
  p[1] = 0x07;
  if (cmd_w(SX_CMD_SET_TX_PARAMS, p, 2) != 0)
    {
      return -1;
    }

  p[0] = sf;
  p[1] = bw;
  p[2] = cr;
  p[3] = ldro;
  if (cmd_w(SX_CMD_SET_MODULATION_PARAMS, p, 4) != 0)
    {
      return -1;
    }

  if (sx_set_lora_pkt_params(8, 0, 255, 1, 0) != 0)
    {
      return -1;
    }

  p[0] = 0x00;
  p[1] = 0x00;
  if (cmd_w(SX_CMD_SET_BUFFER_BASE, p, 2) != 0)
    {
      return -1;
    }

  {
    uint16_t irq_mask = SX_IRQ_TX_DONE | SX_IRQ_RX_DONE | SX_IRQ_TIMEOUT |
                        SX_IRQ_CRC_ERR | SX_IRQ_HEADER_ERR |
                        SX_IRQ_PREAMBLE | SX_IRQ_SYNC_VALID;
    uint16_t dio1_mask = SX_IRQ_TX_DONE | SX_IRQ_RX_DONE |
                         SX_IRQ_TIMEOUT | SX_IRQ_CRC_ERR;

    p[0] = (uint8_t)(irq_mask >> 8);
    p[1] = (uint8_t)irq_mask;
    p[2] = (uint8_t)(dio1_mask >> 8);
    p[3] = (uint8_t)dio1_mask;
    p[4] = 0x00;
    p[5] = 0x00;
    p[6] = 0x00;
    p[7] = 0x00;
    if (cmd_w(SX_CMD_SET_DIO_IRQ_PARAMS, p, 8) != 0)
      {
        return -1;
      }
  }

  if (sx_calibrate_image(0x6B) != 0)
    {
      return -1;
    }

  p[0] = 0x20;
  if (cmd_w(SX_CMD_SET_RX_TX_FALLBACK, p, 1) != 0)
    {
      return -1;
    }

  return sx_clear_irq(0xFFFF);
}

/* 打开 SPI + GPIO，并把芯片置到已知状态 */
static int lora_open(void)
{
  /* 2026-10-30: NuttX 的 fd 表是按任务分的，g_fd 却是文件级静态变量。
   * 发送任务退出时内核关掉它自己的 /dev/spi3，但 g_fd 的数值还留着；
   * 接收任务拿同一个数值 fcntl 可能正好命中它自己打开的另一个 fd
   * （console / i2s），ioctl(SPIIOC_TRANSFER) 就打到错误设备上，
   * 表现是整机瞬间失声、LCD 不动、串口也不再响应。
   * 这里按任务属主判断，换了任务一律重开。 */
  if (g_fd >= 0 && g_fd_owner != getpid())
    {
      g_fd = -1;
    }

  if (g_fd >= 0)
    {
      /* 2026-10-29: NuttX 的 fd 表是任务级的。发送任务 "lora-tx" 退出时，
       * 内核会把它打开的 /dev/spi3 关掉，但 g_fd 是文件级静态变量，数值
       * 还留着。接收任务下一轮直接复用这个号，ioctl 立刻 EBADF(errno=9)，
       * 日志就是 "[lora] spi ioctl ret=-1 errno=9" + "[srx2] config failed"。
       * 所以每次复用前先用 fcntl 确认这个 fd 在当前任务里真的有效。 */
      if (fcntl(g_fd, F_GETFD) >= 0)
        {
          return 0;
        }

      g_fd = -1;
    }

  esp32s3_configgpio(LORA_PIN_TXEN, OUTPUT);
  esp32s3_configgpio(LORA_PIN_RXEN, OUTPUT);
  esp32s3_configgpio(LORA_PIN_BUSY, INPUT | PULLUP);
  sw_rx();

  g_fd = open(LORA_SPI_PATH, O_RDWR);
  if (g_fd < 0)
    {
      printf("[lora] open %s failed: %d\n", LORA_SPI_PATH, errno);
      return -1;
    }

  g_fd_owner = getpid();
  return 0;
}

static int lora_close(void)
{
  if (g_fd >= 0 && g_fd_owner == getpid())
    {
      close(g_fd);
      g_fd = -1;
      g_fd_owner = -1;
    }
  return 0;
}

static uint8_t sx_read_reg8(uint16_t addr, int *ok)
{
  uint8_t a[2];
  uint8_t v = 0;

  a[0] = (uint8_t)(addr >> 8);
  a[1] = (uint8_t)(addr & 0xFF);

  if (cmd_r(SX_CMD_READ_REGISTER, a, 2, 1, &v, 1) != 0)
    {
      *ok = 0;
      return 0;
    }

  *ok = 1;
  return v;
}

static uint16_t crc16_ccitt(const uint8_t *data, int len)
{
  uint16_t crc = 0xFFFF;

  for (int i = 0; i < len; i++)
    {
      crc ^= (uint16_t)data[i] << 8;
      for (int b = 0; b < 8; b++)
        {
          crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                               : (uint16_t)(crc << 1);
        }
    }

  return crc;
}

static void build_jx_packet(uint8_t *pkt, uint32_t seq,
                            uint32_t token_idx, uint16_t token_cnt)
{
  memset(pkt, 0, LORA_PKT_LEN);
  pkt[0] = 'J';
  pkt[1] = 'X';
  pkt[2] = 0x21;
  pkt[3] = 0x00;

  pkt[4] = (uint8_t)seq;
  pkt[5] = (uint8_t)(seq >> 8);
  pkt[6] = (uint8_t)(seq >> 16);
  pkt[7] = (uint8_t)(seq >> 24);

  pkt[8] = (uint8_t)token_idx;
  pkt[9] = (uint8_t)(token_idx >> 8);
  pkt[10] = (uint8_t)(token_idx >> 16);
  pkt[11] = (uint8_t)(token_idx >> 24);

  pkt[12] = (uint8_t)token_cnt;
  pkt[13] = (uint8_t)(token_cnt >> 8);

  /* 36 字节净荷：先填一个可辨识的斜坡，再对净荷算 CRC */
  for (int i = 0; i < 36; i++)
    {
      pkt[16 + i] = (uint8_t)((seq + (uint32_t)i) & 0xFF);
    }

  {
    uint16_t crc = crc16_ccitt(pkt + 16, 36);
    pkt[14] = (uint8_t)crc;
    pkt[15] = (uint8_t)(crc >> 8);
  }
}

static int lora_tx_packet(const uint8_t *pkt, uint32_t timeout_ms)
{
  sw_tx();
  if (sx_set_standby_xosc() != 0)
    {
      return -1;
    }

  if (sx_set_lora_pkt_params(8, 0, LORA_PKT_LEN, 1, 0) != 0)
    {
      return -1;
    }

  if (sx_write_buffer(0, pkt, LORA_PKT_LEN) != 0)
    {
      return -1;
    }

  if (sx_clear_irq(0xFFFF) != 0)
    {
      return -1;
    }

  if (sx_start_tx(timeout_ms) != 0)
    {
      return -1;
    }

  /* 2026-10-30: 这里原来用 clock_gettime 算超时。现场 ps 看到
   * lora-tx 的 sender 线程一直 Ready/忙等，主任务卡在 join 上，
   * TX_DONE 却迟迟不来 —— 时钟不往前跳时 el 永远不增长，
   * 这个 for(;;) 退不出去，整轮发送永久挂住。
   * 改成和 lora_wait_busy 一样的固定 1 ms 计数超时。 */
  for (uint32_t tick = 0; ; tick++)
    {
      uint16_t irq = 0;

      if (sx_get_irq(&irq) == 0)
        {
          if (irq & SX_IRQ_TX_DONE)
            {
              /* TX_DONE 产生时芯片还在切回 STDBY，BUSY 可能仍是高。
               * 先等 BUSY 落下再清中断，否则 clear 会白等 200 ms/包
               * （32 包白白多 6 秒），还会把状态机顶死。 */
              if (lora_wait_busy(100) == 0)
                {
                  sx_clear_irq(0xFFFF);
                }

              return 0;
            }

          if (irq & SX_IRQ_TIMEOUT)
            {
              sx_clear_irq(0xFFFF);
              return -4;
            }
        }

      if (tick > timeout_ms)
        {
          sx_clear_irq(0xFFFF);
          return -2;
        }

      usleep(1000);
    }
}


int jx_lora_streamrx_live(int secs);

/* Any byte on the console ends PTT.  Ctrl-C cannot be used here: the
 * command runs inside the NSH task, so SIGINT would kill the shell. */

/* 返回 0 = 无事，1 = 退出，2 = 触发一次发送。
 * 除了 BOOT 键，控制台敲任意键也能触发一次（现场没按键/远程测试用），
 * 'q' 或 Ctrl-C 退出。 */
static int ptt_poll_key(void)
{
  struct pollfd pfd;
  char c;

  pfd.fd = 0;
  pfd.events = POLLIN;
  pfd.revents = 0;

  if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN) != 0)
    {
      if (read(0, &c, 1) == 1)
        {
          if (c == (char)3 || c == 'q' || c == 'Q') { return 1; }
          return 2;
        }
    }

  return 0;
}

/* 2026-10-29 PTT: hold BOOT to talk, release to send.
 * Keeps the codec warm between takes and reuses the normal streaming path. */

int jx_lora_ptt(int maxsec)
{
  extern int jx_streamtx_run_lora(int secs);
  extern void jx_ui_set_boot_enable(int on);
  extern void jx_ui_set_status(const char *s);
extern void jx_ui_set_rx(int pkt, int rssi);
extern void jx_ui_set_tx(int pkt);

  if (maxsec < 1)  { maxsec = 1; }
  if (maxsec > 20) { maxsec = 20; }

  jx_ui_set_boot_enable(0);
  setenv("JX_PTT", "1", 1);
  setenv("JX_BEEP", "0", 1);
  setenv("JX_PREARM_MS", "300", 1);
  esp32s3_configgpio(0, INPUT | PULLUP);

  printf("[ptt] ready: hold BOOT to talk, release to send (max %d s)\n",
         maxsec);
  jx_ui_set_status("PTT IDLE");

  for (;;)
    {
      int by_key = 0;

      /* 等一次触发：BOOT 按下，或控制台敲一个键。 */
      for (;;)
        {
          int k = ptt_poll_key();

          if (k == 1)
            {
              printf("[ptt] quit, BOOT handed back to UI\n");
              jx_ui_set_boot_enable(1);
              jx_ui_set_status("PTT OFF");
              return 0;
            }

          if (k == 2)
            {
              printf("[ptt] console trigger -> record\n");
              break;
            }

          if (esp32s3_gpioread(0) == 0)
            {
              usleep(30000);
              if (esp32s3_gpioread(0) == 0)
                {
                  by_key = 1;
                  printf("[ptt] BOOT down -> record\n");
                  break;
                }
            }

          usleep(10000);
        }

      printf("[ptt] take %d s\n", maxsec);
      jx_ui_set_status("PTT REC");
      (void)jx_streamtx_run_lora(maxsec);
      printf("[ptt] take done\n");
      jx_ui_set_status("PTT IDLE");

      if (by_key)
        {
          /* 等按键松开，避免按住时连着录第二段。 */
          while (esp32s3_gpioread(0) == 0)
            {
              int k = ptt_poll_key();
              if (k == 1)
                {
                  printf("[ptt] quit, BOOT handed back to UI\n");
                  jx_ui_set_boot_enable(1);
                  jx_ui_set_status("PTT OFF");
                  return 0;
                }

              usleep(20000);
            }

          usleep(200000);
        }
    }

  return 0;
}

int jx_lora_cmd(int argc, char **argv)
{
  if (argc < 2)
    {
      printf("usage: jixun lora probe|tx [n]|rx [secs]\n");
      return 0;
    }

  if (strcmp(argv[1], "probe") == 0)
    {
      uint8_t st = 0;
      uint16_t err = 0;
      int ok1 = 0;
      int ok2 = 0;
      int rc;

      if (lora_open() < 0)
        {
          return -1;
        }

      printf("[lora] %s opened, BUSY=%d\n", LORA_SPI_PATH,
             esp32s3_gpioread(LORA_PIN_BUSY) ? 1 : 0);

      rc = sx_get_status(&st);
      printf("[lora] GetStatus rc=%d st=0x%02x\n", rc, st);

      rc = sx_get_device_errors(&err);
      printf("[lora] DeviceErrors rc=%d err=0x%04x\n", rc, err);

      {
        uint8_t v320 = sx_read_reg8(0x0320, &ok1);
        uint8_t v321 = sx_read_reg8(0x0321, &ok2);
        printf("[lora] REG 0x0320=0x%02x(ok=%d) 0x0321=0x%02x(ok=%d)\n",
               v320, ok1, v321, ok2);
      }

      lora_close();
      return 0;
    }

  if (strcmp(argv[1], "tx") == 0)
    {
      int n = (argc >= 3) ? atoi(argv[2]) : 1;
      int gap_ms = (argc >= 4) ? atoi(argv[3]) : 500;
      uint8_t pkt[LORA_PKT_LEN];
      int tx_ok = 0;
      int tx_fail = 0;

      if (lora_open() < 0)
        {
          return -1;
        }

      if (sx_set_standby() != 0 || sx_config_lora(433175000, jx_lora_dbm(), jx_lora_sf(),
                                                  jx_lora_bw(), 1, 0) != 0)
        {
          printf("[lora] config failed\n");
          lora_close();
          return -1;
        }

      sx_clear_device_errors();

      for (int i = 0; i < n; i++)
        {
          struct timespec t0;
          struct timespec t1;
          int rc;

          build_jx_packet(pkt, (uint32_t)i, (uint32_t)(i * 16),
                          (uint16_t)((n > 0 && n <= 0xffff) ? n : 0));
          clock_gettime(CLOCK_MONOTONIC, &t0);
          rc = lora_tx_packet(pkt, 1000);
          clock_gettime(CLOCK_MONOTONIC, &t1);

          printf("[lora] tx %d/%d rc=%d el=%ldms\n", i + 1, n, rc,
                 (t1.tv_sec - t0.tv_sec) * 1000L +
                 (t1.tv_nsec - t0.tv_nsec) / 1000000L);
          if (rc == 0)
            {
              tx_ok++;
            }
          else
            {
              tx_fail++;
            }

          if (gap_ms > 0)
            {
              usleep((useconds_t)gap_ms * 1000);
            }
        }

      printf("[lora] TX_SUMMARY count=%d ok=%d fail=%d gap_ms=%d\n",
             n, tx_ok, tx_fail, gap_ms);
      sw_rx();
      lora_close();
      return 0;
    }

  if (strcmp(argv[1], "rx") == 0)
    {
      int secs = (argc >= 3) ? atoi(argv[2]) : 10;
      int expected = (argc >= 4) ? atoi(argv[3]) : 0;
      struct timespec t0;
      struct timespec now;
      static uint8_t seen[4096];
      int valid = 0;
      int duplicate = 0;
      int out_of_range = 0;
      int bad_hdr = 0;
      int bad_crc = 0;
      int radio_crc = 0;
      int radio_hdr = 0;
      int rssi_min = 0;
      int rssi_max = 0;
      int rssi_sum = 0;
      int snr_min = 0;
      int snr_max = 0;
      int snr_sum = 0;
      uint32_t max_seq = 0;
      int have_seq = 0;

      if (expected < 0)
        {
          expected = 0;
        }
      if (expected > (int)sizeof(seen))
        {
          expected = (int)sizeof(seen);
        }
      memset(seen, 0, sizeof(seen));

      if (lora_open() < 0)
        {
          return -1;
        }

      if (sx_set_standby() != 0 || sx_config_lora(433175000, jx_lora_dbm(), jx_lora_sf(),
                                                  jx_lora_bw(), 1, 0) != 0)
        {
          printf("[lora] config failed\n");
          lora_close();
          return -1;
        }

      sx_clear_device_errors();
      sw_rx();
      sx_clear_irq(0xFFFF);
      sx_start_rx(0);

      printf("[lora] rx %d s ...\n", secs);
      clock_gettime(CLOCK_MONOTONIC, &t0);

      for (;;)
        {
          uint16_t irq = 0;

          if (sx_get_irq(&irq) == 0)
            {
              if (irq & SX_IRQ_RX_DONE)
                {
                  uint8_t len = 0;
                  uint8_t start = 0;
                  int16_t rssi = 0;
                  uint8_t buf[128];
                  int snr;
                  int hdr_ok = 0;
                  int crc_ok = 0;
                  uint32_t seq = 0;

                  sx_get_rx_info(&len, &start, &rssi);
                  snr = sx_get_snr();
                  if (len > sizeof(buf))
                    {
                      len = sizeof(buf);
                    }

                  sx_read_rx_payload(start, buf, len);

                  if (len == LORA_PKT_LEN && buf[0] == 'J' && buf[1] == 'X' &&
                      buf[2] == 0x21)
                    {
                      hdr_ok = 1;
                      seq = (uint32_t)buf[4] |
                            ((uint32_t)buf[5] << 8) |
                            ((uint32_t)buf[6] << 16) |
                            ((uint32_t)buf[7] << 24);
                      crc_ok = (crc16_ccitt(buf + 16, 36) ==
                                (uint16_t)(buf[14] | ((uint16_t)buf[15] << 8)));
                    }

                  sx_clear_irq(0xFFFF);

                  printf("[lora] rx seq=%u len=%d rssi=%d snr=%d crc=%d\n",
                         (unsigned)seq, len, rssi, snr, crc_ok);

                  if (!hdr_ok)
                    {
                      bad_hdr++;
                    }
                  else if (!crc_ok)
                    {
                      bad_crc++;
                    }
                  else
                    {
                      if (expected > 0 && seq < (uint32_t)expected)
                        {
                          if (seen[seq])
                            {
                              duplicate++;
                            }
                          else
                            {
                              seen[seq] = 1;
                              valid++;
                            }
                        }
                      else if (expected > 0)
                        {
                          out_of_range++;
                        }
                      else
                        {
                          valid++;
                        }

                      if (!have_seq || seq > max_seq)
                        {
                          max_seq = seq;
                          have_seq = 1;
                        }
                      if (valid == 1 || rssi < rssi_min)
                        {
                          rssi_min = rssi;
                        }
                      if (valid == 1 || rssi > rssi_max)
                        {
                          rssi_max = rssi;
                        }
                      if (valid == 1 || snr < snr_min)
                        {
                          snr_min = snr;
                        }
                      if (valid == 1 || snr > snr_max)
                        {
                          snr_max = snr;
                        }
                      rssi_sum += rssi;
                      snr_sum += snr;
                    }

                  sx_start_rx(0);
                }
              else if (irq & (SX_IRQ_CRC_ERR | SX_IRQ_HEADER_ERR))
                {
                  if (irq & SX_IRQ_CRC_ERR)
                    {
                      radio_crc++;
                    }
                  if (irq & SX_IRQ_HEADER_ERR)
                    {
                      radio_hdr++;
                    }
                  printf("[lora] rx err irq=0x%04x\n", irq);
                  sx_clear_irq(0xFFFF);
                  sx_start_rx(0);
                }
            }

          clock_gettime(CLOCK_MONOTONIC, &now);
          if ((now.tv_sec - t0.tv_sec) >= secs)
            {
              break;
            }

          usleep(1000);
        }

      {
        int missing = expected > valid ? expected - valid : 0;
        int loss_pm = expected > 0 ? (missing * 1000) / expected : 0;
        int rssi_avg = valid > 0 ? rssi_sum / valid : 0;
        int snr_avg = valid > 0 ? snr_sum / valid : 0;

        printf("[lora] RX_SUMMARY expected=%d valid=%d duplicate=%d "
               "out_of_range=%d missing=%d loss_pm=%d bad_hdr=%d bad_crc=%d "
               "radio_crc=%d radio_hdr=%d max_seq=%u rssi_min=%d rssi_avg=%d "
               "rssi_max=%d snr_min=%d snr_avg=%d snr_max=%d\n",
               expected, valid, duplicate, out_of_range, missing, loss_pm,
               bad_hdr, bad_crc, radio_crc, radio_hdr, (unsigned)max_seq,
               rssi_min, rssi_avg, rssi_max, snr_min, snr_avg, snr_max);
      }
      lora_close();
      return 0;
    }

  if (strcmp(argv[1], "ptt") == 0)
    {
      int maxsec = (argc >= 3) ? atoi(argv[2]) : 8;

      return jx_lora_ptt(maxsec);
    }

  if (strcmp(argv[1], "streamrx") == 0)
    {
      int secs = (argc >= 3) ? atoi(argv[2]) : 30;
      return jx_lora_streamrx_live(secs);
    }

  if (strcmp(argv[1], "stream") == 0)
    {
      int secs = (argc >= 3) ? atoi(argv[2]) : 3;
      extern int jx_streamtx_run_lora(int secs);
      return jx_streamtx_run_lora(secs);
    }

  printf("unknown lora cmd: %s\n", argv[1]);
  return -1;
}



/****************************************************************************
 * LoRa 语音发送接口（供 jx_streamtx 调用）
 ****************************************************************************/

static void put_u32le(uint8_t *p, uint32_t v)
{
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

static void put_u16le(uint8_t *p, uint16_t v)
{
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}

/* 乙板 pack_tokens 是 LSB-first，每 token 18 bit */
static int g_wire_v2=0;
static int g_planned_seconds;
void jx_lora_set_plan(int samples) { g_planned_seconds=samples/16000; }
static uint32_t g_wire_session, g_wire_counter;
static int build_lora_packet_v2(uint8_t *pkt,uint32_t seq,const int64_t *tok,int n,int last,int samples) {
  const char *e=getenv("JX_CONV_HALO");int halo=e?atoi(e):1;if(halo<0)halo=0;
  JxWireFrame f;memset(&f,0,sizeof(f));f.session=g_wire_session;f.seq=seq;
  f.valid=(unsigned)n;f.halo=5u;f.model=JXW_MODEL_LIGHT_A_3K;f.eos=last;
  f.samples=last?(uint32_t)samples:0;
  for(int i=0;i<n;i++)f.tokens[i]=tok[i];
  return jxw_pack(pkt,&f);
}

static void build_lora_packet(uint8_t *pkt, uint32_t seq, uint32_t tokidx,
                              const int64_t *tok, int ntok, int last_flag)
{
  int i;
  int b;

  memset(pkt, 0, LORA_PKT_LEN);
  pkt[0] = 'J';
  pkt[1] = 'X';
  pkt[2] = 0x21;
  pkt[3] = last_flag ? 0x01 : 0x00;
  {
    const char *e=getenv("JX_CONV_HALO"); int h=e?atoi(e):1; if(h<0)h=0;
    if(h!=5) pkt[3] |= (uint8_t)(0x80u | ((unsigned)h << 1));
    if(seq==0) printf("[lora] encoder_profile halo=%d flags=%02x\n",h,pkt[3]);
  }
  put_u32le(pkt + 4, seq);
  put_u32le(pkt + 8, tokidx);
  put_u16le(pkt + 12, 16);

  for (i = 0; i < ntok; i++)
    {
      uint32_t v = (uint32_t)tok[i] & 0x3FFFFu;

      for (b = 0; b < 18; b++)
        {
          int bit = i * 18 + b;

          if ((v >> b) & 1u)
            {
              pkt[16 + (bit >> 3)] |= (uint8_t)(1u << (bit & 7));
            }
        }
    }

  {
    uint16_t crc = crc16_ccitt(pkt + 16, 36);
    pkt[14] = (uint8_t)crc;
    pkt[15] = (uint8_t)(crc >> 8);
  }
}

static uint32_t g_lora_seq;
static uint32_t g_lora_tokidx;
static int g_lora_ready;

int jx_lora_prepare_tx(void)
{
  { const char *e=getenv("JX_CONV_HALO"); if(e && atoi(e)>5) { printf("[lora] unsupported encoder halo profile\n"); return -EINVAL; } }
  int sb;
  int cf;

  if (lora_open() < 0)
    {
      return -1;
    }

  if (g_lora_ready)
    {
      return 0;
    }

  {
    const char *v=getenv("JX_WIRE_V2");g_wire_v2=v?atoi(v)!=0:0;
    if(!g_wire_counter){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);g_wire_counter=(uint32_t)((uint64_t)ts.tv_sec*1000000u+ts.tv_nsec/1000u);}
    if(++g_wire_counter==0)++g_wire_counter;
    g_wire_session=g_wire_counter;
  }
  sb = sx_set_standby();
  cf = (sb == 0) ? sx_config_lora(433175000, jx_lora_dbm(),
                                  jx_lora_sf(), jx_lora_bw(), 1, 0) : -99;
  if (sb != 0 || cf != 0)
    {
      printf("[lora] stream config failed sb=%d cfg=%d\n", sb, cf);
      return -1;
    }

  sx_clear_device_errors();
  g_lora_ready = 1;
  return 0;
}

int jx_lora_send_idx(int bits, const int64_t *idx, int ntok,
                     int tt, int orig_t, uint32_t chunk, uint32_t last)
{
  uint8_t pkt[LORA_PKT_LEN];
  int off = 0;

  (void)bits;
  (void)tt;
  (void)orig_t;
  (void)chunk;

  if (jx_lora_prepare_tx() != 0)
    {
      return -1;
    }

  while (off < ntok)
    {
      int n = ntok - off;
      int rc;

      if (n > 16)
        {
          n = 16;
        }

      if(g_wire_v2) {
        if(bits!=18 || build_lora_packet_v2(pkt,g_lora_seq,idx+off,n,
              (last && off+n>=ntok)?1:0,orig_t))return -EINVAL;
      } else build_lora_packet(pkt,g_lora_seq,g_lora_tokidx,idx+off,n,
                        (last && off+n>=ntok)?1:0);
      if(!g_wire_v2 && g_planned_seconds>=1 && g_planned_seconds<=8) {
        const char *ll=getenv("JX_LL_WIRE");
        if(!ll || ll[0]!='0')jxll_tag(pkt,(unsigned)g_planned_seconds);
      }
      {
        const char *dbg = getenv("JX_TXDBG");
        if (dbg != NULL && dbg[0] != '0')
          {
            printf("[txdbg] pkt begin seq=%u tok=%u ntok=%d\n",
                   (unsigned)g_lora_seq, (unsigned)g_lora_tokidx, n);
          }
      }
      /* 2026-10-30: 单向实测 37 包随机丢 3~4 包。接收端每收完一包
       * 要重新 sx_start_rx()，这段窗口里下一包已经在空中，整包就丢。
       * 接收端对 seq < next_seq 的重复包直接忽略，所以重发同一包安全。
       * JX_LORA_TX_REPEAT 默认 2（每包发两遍）。 */
      {
        const char *rp = getenv("JX_LORA_TX_REPEAT");
        int rep = (rp != NULL) ? atoi(rp) : 1;

        if (rep < 1) { rep = 1; }
        if (rep > 5) { rep = 5; }
        if(g_wire_v2 && last && off+n>=ntok) {
          const char *er=getenv("JX_END_REPEAT");int ends=er?atoi(er):3;
          if(ends<1)ends=1;if(ends>5)ends=5;if(rep<ends)rep=ends;
          printf("[wire-tx] session=%u seq=%u samples=%d end_copies=%d\n",g_wire_session,g_lora_seq,orig_t,rep);
        }

        for (int k = 0; k < rep; k++)
          {
            rc = lora_tx_packet(pkt, 1000);
            if (rc != 0)
              {
                break;
              }

            if ((k + 1) < rep)
              {
                usleep(20 * 1000);
              }
          }
      }

      if (rc != 0)
        {
          printf("[lora] tx failed rc=%d seq=%u\n", rc, (unsigned)g_lora_seq);
          /* 让下一轮强制重配频点/调制/PA，避免沿用坏状态。 */
          g_lora_ready = 0;
          return rc;
        }

      g_lora_seq++;
      g_lora_tokidx += 16;
      off += 16;

      /* 2026-10-21: K7 接收端是单线程轮询，连续背靠背发包子时它处理
       * 不过来，实测前 5 包能收、后面全丢，小黑每包等 1s 超时，
       * 14 包把发送线程拖死、采集 ring 填满后录音线程也停。
       * 包间留 20ms 让接收端回到 RX 并处理上一包。 */
      /* 2026-10-29: 连发 32 包时偶发 SPI 传输永久阻塞（板子整机卡死，
       * 不是丢包）。把包间隔从 5 ms 放宽到 20 ms，并在每 8 包打一次
       * 心跳，卡住时能直接从串口看出停在第几包。 */
      if (((g_lora_seq - 1) % 8) == 0)
        {
          printf("[txp] seq=%u tokidx=%u\n",
                 (unsigned)(g_lora_seq - 1), (unsigned)g_lora_tokidx);
          jx_ui_set_tx((int)(g_lora_seq - 1));
        }

      /* 2026-10-30: 包间隔 20 ms 时实测丢 8~10%，接收端重新进 RX
       * 的窗口不够；默认放到 40 ms，可用 JX_LORA_GAP_MS 覆盖。 */
      {
        const char *gp = getenv("JX_LORA_GAP_MS");
        int gap = (gp != NULL) ? atoi(gp) : 20;

        if (gap < 5) { gap = 5; }
        if (gap > 200) { gap = 200; }
        usleep(gap * 1000);
      }

      {
        const char *dbg = getenv("JX_TXDBG");
        if (dbg != NULL && dbg[0] != '0')
          {
            printf("[txdbg] pkt done seq=%u\n",
                   (unsigned)(g_lora_seq - 1));
          }
      }
    }

  /* 2026-10-29：发送收尾只把芯片放回 standby，绝不在这里 sw_rx()。
   *
   * 现场日志：最后一包 TX_DONE 后马上 sw_rx() 切天线，SX1262 状态机会
   * 锁死，之后所有命令都 BUSY 超时（sb=-1），整块板再也进不了接收。
   * 天线开关和接收配置统一交给 jx_lora_streamrx_live()，它会在配置前
   * 先 sw_rx()。这里失败也不阻塞：包已经发出去了，标记 need-reinit，
   * 下一轮接收从头 set_standby + config。 */
  {
    int sb = -1;
    int i;

    for (i = 0; i < 20; i++)
      {
        sb = sx_set_standby();
        if (sb == 0)
          {
            break;
          }

        usleep(10 * 1000);
      }

    if (sb != 0)
      {
        printf("[lora] tx tail standby failed sb=%d, rx will re-init\n", sb);
      }
  }

  /* 2026-10-29: 这里不再清零。同一段音频的后续 chunk 沿用已配好的频点、
   * 调制和 PA，每包只做 set_standby + write_buffer + start_tx（约 6 次
   * SPI 事务），而整段重配要 14 次以上并带两次 calibrate。事务越少，
   * 驱动丢 DMA 完成中断的概率越低，连发 30+ 包才稳。只有 tx 失败或
   * jx_lora_release() 才清 0，强制下一轮重配。
   */
  return 0;
}


/* 流式发送结束时释放 SPI 句柄：jx_streamtx 的发送任务退出时，
 * NuttX 会关闭该任务的 fd 表，但本文件的 g_fd 是进程级静态变量，
 * 下一次命令若复用旧 fd 号会得到 EBADF。这里显式关闭并复位状态。 */
void jx_lora_release(void)
{
  /* 2026-10-29：不要 close /dev/spi3。
   *
   * 现场实测：接收端收完一句、重新进入接收后 6~22 秒，整机会卡死
   * （串口写超时、nsh 不响应，关掉播放也一样）。每次收尾都
   * close() + 下次 open() 同一个 SPI 设备，NuttX 的 esp32s3 SPI
   * 驱动在 DMA 反复 attach/detach 后会丢掉完成中断，之后任何
   * ioctl(SPIIOC_TRANSFER) 都永久等下去。
   *
   * 改成：芯片置 standby、清协议状态，fd 一直留着。lora_open()
   * 本身是幂等的（g_fd >= 0 直接返回），发送/接收共用同一个 fd。
   * 串口命令里显式 probe 仍可单独 close，那是人工路径。 */
  /* 这里不做任何 SPI 操作：接收刚结束时芯片还在 RX，任何命令都可能
   * 撞上 BUSY；而且实测在这里 set_standby 会让下一次发送的
   * sx_set_standby/config 直接失败（stream config failed）。
   * 只清协议状态，芯片状态交给下一次发送路径自己处理。 */
  g_lora_ready = 0;
  g_lora_seq = 0;
  g_lora_tokidx = 0;
}


/****************************************************************************
 * 边收边解边播（2026-10-28）
 *
 * 旧 streamrx 是「收齐再解码」，3 秒语音要等整段到齐才出声。
 * 这里改成：每收到一包 16 token 就 sdec_step 一次，PCM 立刻进播放缓冲。
 ****************************************************************************/

#define SRX_TPP        16
#define SRX_NT_CHUNK   16 /* one complete received packet per decode step */
#define SRX_HOP        96
#define SRX_RING_SAMP  (16000 * 10) /* covers the UI maximum 8 s plus model flush */
#define SRX_PCM_CAP    (4096)

struct srx_bdec
{
  const int64_t *tok;
  int   Tt;
  float *out;
  int   rc;
};

static void *srx_block_dec(void *arg)
{
  struct srx_bdec *a = (struct srx_bdec *)arg;
  a->rc = jx_decode(a->tok, a->Tt, a->Tt * SRX_HOP, &a->out);
  return NULL;
}

extern int  jx_voice_stream_play_pcm(const int16_t *pcm, int samples);
extern int  ai_voice_init(void);
extern int  voice_agent_set_dac_mute(int mute);
extern void hal_i2s_write_flush(void);

struct srx_pb
{
  pthread_mutex_t lock;
  pthread_cond_t  cond;
  int16_t        *data;
  int             cap;
  int             head;
  int             tail;
  int             count;
  int             prebuffer;
  int             finished;
  int             failed;
  int             started;
  int             buffered;
  int expected, valid_samples, produced, enqueued, received, plan_seconds;
  unsigned first_rx_tick,last_rx_tick, step_ms[4],step_pos,step_count;
  unsigned required_ms;
};

static int srx_pb_init(struct srx_pb *pb, int cap, int prebuffer)
{
  memset(pb, 0, sizeof(*pb));
  pb->cap = cap;
  { const char *mode = getenv("JX_RX_BUFFERED");
    pb->buffered = mode ? atoi(mode) : 2; }
  if (prebuffer < 0) { prebuffer = 0; }
  if (prebuffer > cap) { prebuffer = cap; }
  pb->prebuffer = prebuffer;
  pb->data = (int16_t *)malloc(sizeof(int16_t) * (size_t)cap);
  if (pb->data == NULL) { return -1; }
  if (pthread_mutex_init(&pb->lock, NULL) != 0) { free(pb->data); return -1; }
  if (pthread_cond_init(&pb->cond, NULL) != 0) { free(pb->data); return -1; }
  return 0;
}

static void srx_pb_free(struct srx_pb *pb)
{
  if (pb->data != NULL) { free(pb->data); pb->data = NULL; }
  pthread_mutex_destroy(&pb->lock);
  pthread_cond_destroy(&pb->cond);
}

static void srx_pb_plan(struct srx_pb *pb,int seconds,int received) {
  pthread_mutex_lock(&pb->lock);
  unsigned tick=clock_systime_ticks();
  if(!pb->received)pb->first_rx_tick=tick;
  pb->last_rx_tick=tick;pb->received=received;
  if(seconds>=1 && seconds<=8 && (!pb->plan_seconds || pb->plan_seconds==seconds)) {
    pb->plan_seconds=seconds;pb->valid_samples=seconds*16000;
    pb->expected=((pb->valid_samples+4112+1535)/1536)*1536;
  } else {pb->plan_seconds=-1;pb->expected=0;}
  pthread_mutex_unlock(&pb->lock);
}
static void srx_pb_timing(struct srx_pb *pb,int tokens,unsigned ticks) {
  unsigned ms=(unsigned)TICK2MSEC(ticks);
  ms=(ms*16u+(unsigned)tokens-1u)/(unsigned)tokens;
  pthread_mutex_lock(&pb->lock);
  pb->step_ms[pb->step_pos++%4]=ms;
  if(pb->step_count<4)pb->step_count++;
  pthread_mutex_unlock(&pb->lock);
}
static int srx_pb_ready(struct srx_pb *pb) {
  if(pb->finished)return 1;
  if(pb->started)return pb->count>0;
  if(!pb->buffered)return pb->count>=pb->prebuffer;
  if(pb->buffered!=2 || !pb->expected || pb->plan_seconds<1 ||
     pb->step_count<3 || pb->count<1600)return 0;
  unsigned cost=0;
  for(unsigned i=0;i<pb->step_count;i++)if(pb->step_ms[i]>cost)cost=pb->step_ms[i];
  unsigned remain=(unsigned)(pb->expected-pb->produced);
  unsigned compute=(remain*cost+1535u)/1536u;
  int packets=(pb->expected+1535)/1536;
  int pending=packets-pb->received;
  unsigned receive=0;
  if(pending>0 && pb->received>1) {
    unsigned elapsed=(unsigned)TICK2MSEC(pb->last_rx_tick-pb->first_rx_tick);
    receive=(elapsed*(unsigned)pending*11u)/(10u*(unsigned)(pb->received-1))+cost;
  }
  if(receive>compute)compute=receive;
  const char *option=getenv("JX_LL_MARGIN_MS");
  unsigned margin=option?(unsigned)atoi(option):180u;
  if(margin<100)margin=100;if(margin>2000)margin=2000;
  pb->required_ms=compute+margin;
  return pb->required_ms<=(unsigned)pb->valid_samples/16u;
}
static int srx_pb_write(struct srx_pb *pb, const int16_t *pcm, int samples)
{
  int first;
  int second;

  if (pb == NULL || pcm == NULL || samples <= 0) { return -1; }

  pthread_mutex_lock(&pb->lock);
  int raw_begin=pb->produced;pb->produced+=samples;
  if(pb->buffered==2 && pb->expected && pb->plan_seconds>0) {
    int skip=raw_begin<4118?4118-raw_begin:0;
    if(skip>samples)skip=samples;
    pcm+=skip;samples-=skip;
    int available=pb->valid_samples-pb->enqueued;
    if(samples>available)samples=available;
    if(samples<=0){pthread_mutex_unlock(&pb->lock);return 0;}
  }
  if (pb->buffered && pb->count + samples > pb->cap)
    {
      printf("[SRX_PLAY] buffer overflow capacity=%d samples; abort playback\n", pb->cap);
      pb->failed = 1;
      pthread_cond_broadcast(&pb->cond);
    }
  while (!pb->failed && pb->count + samples > pb->cap)
    {
      pthread_mutex_unlock(&pb->lock);
      usleep(5000);
      pthread_mutex_lock(&pb->lock);
    }
  if (pb->failed) { pthread_mutex_unlock(&pb->lock); return -1; }

  first = pb->cap - pb->tail;
  if (first > samples) { first = samples; }
  second = samples - first;
  memcpy(&pb->data[pb->tail], pcm, sizeof(int16_t) * (size_t)first);
  if (second > 0)
    {
      memcpy(pb->data, pcm + first, sizeof(int16_t) * (size_t)second);
    }
  pb->tail = (pb->tail + samples) % pb->cap;
  pb->count += samples;pb->enqueued+=samples;
  pthread_cond_broadcast(&pb->cond);
  pthread_mutex_unlock(&pb->lock);
  return 0;
}

static void srx_pb_finish(struct srx_pb *pb)
{
  if (pb == NULL) { return; }
  pthread_mutex_lock(&pb->lock);
  pb->finished = 1;
  pthread_cond_broadcast(&pb->cond);
  pthread_mutex_unlock(&pb->lock);
}

static void *srx_pb_worker(void *arg)
{
  struct srx_pb *pb = (struct srx_pb *)arg;
  int16_t chunk[1600];
  uint32_t play_begin = 0, last_feed = 0, max_feed_gap = 0;
  unsigned int submitted = 0, starvation = 0, starvation_ticks = 0;
  const char *np = getenv("JX_RX_NOPLAY");

  if (np != NULL && atoi(np) != 0)
    {
      for (;;)
        {
          pthread_mutex_lock(&pb->lock);
          while (!pb->finished && pb->count == 0)
            {
              pthread_mutex_unlock(&pb->lock);
              usleep(5000);
              pthread_mutex_lock(&pb->lock);
            }
          if (pb->finished && pb->count == 0)
            {
              pthread_mutex_unlock(&pb->lock);
              break;
            }
          pb->count = 0;
          pb->head = pb->tail;
          pthread_cond_broadcast(&pb->cond);
          pthread_mutex_unlock(&pb->lock);
        }
      return NULL;
    }

  if (ai_voice_init() != 0)
    {
      printf("[srx2] ai_voice_init failed\n");
      pthread_mutex_lock(&pb->lock);
      pb->failed = 1;
      pthread_cond_broadcast(&pb->cond);
      pthread_mutex_unlock(&pb->lock);
      return NULL;
    }

  voice_agent_set_dac_mute(0);

  {
    struct timespec wd_t0;
    int wd_have = 0;
    long wd_limit_ms = 2000;

  for (;;)
    {
      int take;
      int first;
      int second;

      pthread_mutex_lock(&pb->lock);

      /* 2026-10-29: 播放看门狗。上一轮实测收到 37 包 LAST 后命令不返回，
       * 下一轮 nsh 命令被挂着不执行，表现就是“连续第二次没反应”。
       * 收尾后最多再给 2 秒把尾音放完，超时就丢掉剩余强制收尾。 */
      if (pb->finished)
        {
          struct timespec wd_now;
          clock_gettime(CLOCK_MONOTONIC, &wd_now);
          if (!wd_have)
            {
              wd_t0 = wd_now;
              wd_have = 1;
              wd_limit_ms = 2000L + (long)pb->count * 1000L / 16000L;
            }
          else
            {
              long wd_ms = (long)(wd_now.tv_sec - wd_t0.tv_sec) * 1000L +
                           (long)(wd_now.tv_nsec - wd_t0.tv_nsec) / 1000000L;
              if (wd_ms > wd_limit_ms)
                {
                  printf("[srx2] 播放看门狗超时 %ld ms，丢弃剩余 %d 样本\n",
                         wd_ms, pb->count);
                  pb->count = 0;
                  pb->head = pb->tail;
                  pthread_cond_broadcast(&pb->cond);
                  pthread_mutex_unlock(&pb->lock);
                  break;
                }
            }
        }

      /* Slower-than-realtime decoding must finish before continuous playback.
       * The explicit legacy mode remains available for measured comparisons. */
      {
        uint32_t wait_begin = 0;
        int waiting_after_start = 0;
        while (!pb->failed && !srx_pb_ready(pb))
          {
            if (pb->started && !waiting_after_start)
              {
                waiting_after_start = 1;
                wait_begin = clock_systime_ticks();
                starvation++;
              }
            pthread_mutex_unlock(&pb->lock);
            usleep(5000);
            pthread_mutex_lock(&pb->lock);
          }
        if (waiting_after_start)
          starvation_ticks += clock_systime_ticks() - wait_begin;
      }
      if (pb->failed || (pb->finished && pb->count == 0))
        {
          pthread_mutex_unlock(&pb->lock);
          break;
        }
      if (!pb->started)
        {
          pb->started = 1;
          printf("[SRX_START] mode=%d produced=%d expected=%d valid=%d cost_ms=%u required_ms=%u\n",
                 pb->buffered,pb->produced,pb->expected,pb->valid_samples,
                 pb->step_count?pb->step_ms[(pb->step_pos-1)%4]:0,pb->required_ms);
          printf("[srx2] 开始播放，缓冲 %d 样本（%.0f ms）\n",
                 pb->count, (double)pb->count * 1000.0 / 16000.0);
        }
      take = pb->count;
      if (take > (int)(sizeof(chunk) / sizeof(chunk[0])))
        {
          take = (int)(sizeof(chunk) / sizeof(chunk[0]));
        }
      first = pb->cap - pb->head;
      if (first > take) { first = take; }
      second = take - first;
      memcpy(chunk, &pb->data[pb->head], sizeof(int16_t) * (size_t)first);
      if (second > 0)
        {
          memcpy(chunk + first, pb->data, sizeof(int16_t) * (size_t)second);
        }
      pb->head = (pb->head + take) % pb->cap;
      pb->count -= take;
      pthread_cond_broadcast(&pb->cond);
      pthread_mutex_unlock(&pb->lock);

      {
        uint32_t tick = clock_systime_ticks();
        if (submitted == 0) play_begin = tick;
        else if (tick - last_feed > max_feed_gap) max_feed_gap = tick - last_feed;
        last_feed = tick;
      }
      if (jx_voice_stream_play_pcm(chunk, take) != 0)
        {
          pthread_mutex_lock(&pb->lock);
          pb->failed = 1;
          pthread_cond_broadcast(&pb->cond);
          pthread_mutex_unlock(&pb->lock);
          break;
        }
      submitted += (unsigned int)take;
    }
  }

  hal_i2s_write_flush();
  printf("[SRX_PLAY] mode=%s samples=%u expected_ms=%u wall_ms=%u starvation=%u starvation_ms=%u max_feed_gap_ms=%u failed=%d\n",
         pb->buffered==2 ? "adaptive" : (pb->buffered ? "buffered" : "legacy"), submitted,
         submitted * 1000u / 16000u,
         submitted ? (unsigned int)TICK2MSEC(clock_systime_ticks() - play_begin) : 0,
         starvation, (unsigned int)TICK2MSEC(starvation_ticks),
         (unsigned int)TICK2MSEC(max_feed_gap), pb->failed);
  return NULL;
}

static uint16_t srx_rd16(const uint8_t *p)
{
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t srx_rd32(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void srx_unpack(const uint8_t *in, int n, int64_t *tok)
{
  int i;
  int b;

  for (i = 0; i < n; i++)
    {
      uint32_t v = 0;
      for (b = 0; b < 18; b++)
        {
          int bit = i * 18 + b;
          if ((in[bit >> 3] >> (bit & 7)) & 1u)
            {
              v |= (uint32_t)1u << b;
            }
        }
      tok[i] = (int64_t)(v & 0x3FFFFu);
    }
}


/* ---------------- 收包/解码解耦队列 ---------------- */
static void srx_emit(struct srx_pb *pb, const float *fpcm, int ns, int16_t *ipcm);
#define SRX_QBUF 96

struct srx_tq
{
  pthread_mutex_t lock;
  pthread_cond_t  cond;
  SDecState      *dec;
  int             chunk;
  struct srx_pb  *pb;
  int64_t         tok[SRX_QBUF][SRX_TPP];
  int             flag[SRX_QBUF];
  int             head;
  int             tail;
  int             count;
  int             done;
  int             deadline_set;
  uint32_t        deadline;
  int             decoded;
  int16_t        *dump;
  int             dumpn;
  int             dumpcap;
};

int jx_dec_dbg_n = 0;
double jx_dec_dbg_ms = 0.0;

static void srx_tq_init(struct srx_tq *q, SDecState *dec, struct srx_pb *pb, int chunk)
{
  memset(q, 0, sizeof(*q));
  q->chunk=chunk;
  q->dec = dec;
  q->pb = pb;
  q->dumpcap = 16000 * 10;
  q->dump = (int16_t *)malloc(sizeof(int16_t) * (size_t)q->dumpcap);
  q->dumpn = 0;
  pthread_mutex_init(&q->lock, NULL);
  pthread_cond_init(&q->cond, NULL);
}

static void srx_tq_finish(struct srx_tq *q)
{
  pthread_mutex_lock(&q->lock);
  q->done = 1;

  /* 2026-10-29: 收尾不再无限等队列解完。实测 37 包 3 秒语音，最后几包
   * 的解码收尾要 40 s 以上，nsh 命令一直不返回，下一轮命令被挂着不执行。
   * 给解码线程 2 秒宽限把剩余队列解完，超过就丢弃（最多损失尾部一小段）。 */
  if (q->count > 0)
    {
      printf("[srx2] 收尾：剩 %d 块，按块给足时间（60 s 上限）\n", q->count);
    }
  q->deadline_set = 1;
  q->deadline = clock_systime_ticks() + MSEC2TICK(60000);
  pthread_cond_broadcast(&q->cond);
  pthread_mutex_unlock(&q->lock);
}

static int srx_tq_push(struct srx_tq *q, const int64_t *t, int fl)
{
  pthread_mutex_lock(&q->lock);
  while (!q->done && q->count >= SRX_QBUF)
    {
      pthread_mutex_unlock(&q->lock);
      usleep(2000);
      pthread_mutex_lock(&q->lock);
    }
  if (q->done) { pthread_mutex_unlock(&q->lock); return -1; }
  if (t != NULL) { memcpy(q->tok[q->tail], t, sizeof(q->tok[0])); }
  q->flag[q->tail] = fl;
  q->tail = (q->tail + 1) % SRX_QBUF;
  q->count++;
  pthread_cond_broadcast(&q->cond);
  pthread_mutex_unlock(&q->lock);
  return 0;
}

static void *srx_dec_worker(void *arg)
{
  struct srx_tq *q = (struct srx_tq *)arg;

  /* 2026-10-29：收包线程每 ~10 ms 醒一次（NuttX tick=10 ms），若和解码线程
   * 同核，解码的 PSRAM 权重大量 cache 行每次被抢占就全部失效，实测解码
   * 慢 5~6 倍。把解码线程固定到 CPU1，收包留在 CPU0。 */
  {
    cpu_set_t cs;
    CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
  }
  extern int jx_pf_session_begin(int);
  extern void jx_pf_session_end(void);
  int decoder_parallel=0;
  const char *dual=getenv("JX_D_DUAL");
  if(!dual || dual[0]=='1') {
    struct sched_param priority;priority.sched_priority=100;
    (void)pthread_setschedparam(pthread_self(),SCHED_FIFO,&priority);
    decoder_parallel=jx_pf_session_begin(1);
  }
  float fpcm[SRX_PCM_CAP];
  int16_t ipcm[SRX_PCM_CAP];
  int64_t prev[SRX_TPP];
  int have_prev = 0;

  for (;;)
    {
      int64_t cur[SRX_TPP];
      int fl;

      pthread_mutex_lock(&q->lock);
      while (!q->done && q->count == 0)
        {
          pthread_cond_wait(&q->cond, &q->lock);
        }
      if (q->count == 0 && q->done)
        {
          pthread_mutex_unlock(&q->lock);
          break;
        }
      if (q->done && q->deadline_set &&
          (int32_t)(clock_systime_ticks() - q->deadline) >= 0)
        {
          printf("[srx2] 解码收尾超 60 s，丢弃剩余 %d 块\n", q->count);
          q->count = 0;
          q->head = q->tail;
          pthread_mutex_unlock(&q->lock);
          break;
        }
      { extern int jx_dec_dbg_n; extern double jx_dec_dbg_ms; static struct timespec _tb; clock_gettime(CLOCK_MONOTONIC,&_tb); jx_dec_dbg_ms = (double)_tb.tv_sec*1000.0+(double)_tb.tv_nsec/1.0e6; }
      memcpy(cur, q->tok[q->head], sizeof(cur));
      fl = q->flag[q->head];
      q->head = (q->head + 1) % SRX_QBUF;
      q->count--;
      pthread_cond_broadcast(&q->cond);
      pthread_mutex_unlock(&q->lock);

      if (fl == 2)
        {
          if (!have_prev) { continue; }
          memcpy(cur, prev, sizeof(cur));
        }

      for (int io = 0; io < SRX_TPP; io += q->chunk)
        {
          int cn = SRX_TPP - io;
          if (cn > q->chunk) { cn = q->chunk; }
          unsigned bt=clock_systime_ticks();
          int ns = sdec_step(q->dec, cur + io, cn, fpcm);
          srx_pb_timing(q->pb,cn,clock_systime_ticks()-bt);
          if (ns > 0)
            {
              srx_emit(q->pb, fpcm, ns, ipcm);
              q->decoded += ns;
              if (q->dump != NULL && q->dumpn + ns <= q->dumpcap)
                {
                  memcpy(&q->dump[q->dumpn], ipcm,
                         sizeof(int16_t) * (size_t)ns);
                  q->dumpn += ns;
                }
            }
        }

      if (fl == 1) { memcpy(prev, cur, sizeof(prev)); have_prev = 1; }
      { extern int jx_dec_dbg_n; struct timespec _te; clock_gettime(CLOCK_MONOTONIC,&_te); double _now=(double)_te.tv_sec*1000.0+(double)_te.tv_nsec/1.0e6; jx_dec_dbg_n++; { const char *_dbg = getenv("JX_SDDBG"); if (_dbg != NULL && atoi(_dbg) != 0 && jx_dec_dbg_n <= 6) { extern double jx_prof_ms[40]; printf("[sdec blk] n=%d %.1f ms conv=%.0f lin=%.0f\n", jx_dec_dbg_n, _now-jx_dec_dbg_ms, jx_prof_ms[11], jx_prof_ms[12]); } } jx_dec_dbg_ms = _now; }
    }
  if(decoder_parallel>0)jx_pf_session_end();
  return NULL;
}

/* 分段流水线的导出通道：流水线不走 srx_tq，所以自己攒一份 PCM，
 * 收包结束（JX_RX_DUMP=1）时经串口发回主机。上限 20 秒。 */
#define SRX_SEGDUMP_CAP (16000 * 20)
static int16_t *srx_dump_buf;
static int      srx_dump_n;
static int      srx_dump_on;

static void srx_segdump_add(const int16_t *pcm, int ns)
{
  /* 2026-10-29: 之前只在 streamrx 启动时读一次 JX_RX_DUMP，
   * 中途打开不生效。补一条进程内开关，命令设完下一段就生效。 */
  if (!srx_dump_on)
    {
      extern int jx_ui_rx_dump_on(void);
      if (jx_ui_rx_dump_on()) { srx_dump_on = 1; srx_dump_n = 0; }
    }

  if (!srx_dump_on) { return; }
  if (srx_dump_buf == NULL)
    {
      srx_dump_buf = (int16_t *)malloc(sizeof(int16_t) * SRX_SEGDUMP_CAP);
      if (srx_dump_buf == NULL) { srx_dump_on = 0; return; }
    }
  if (srx_dump_n + ns > SRX_SEGDUMP_CAP)
    {
      ns = SRX_SEGDUMP_CAP - srx_dump_n;
      if (ns <= 0) { return; }
    }
  memcpy(srx_dump_buf + srx_dump_n, pcm, sizeof(int16_t) * (size_t)ns);
  srx_dump_n += ns;
}

static void srx_emit(struct srx_pb *pb, const float *fpcm, int ns, int16_t *ipcm)
{
  int i;

  if (ns > SRX_PCM_CAP) { ns = SRX_PCM_CAP; }
  for (i = 0; i < ns; i++)
    {
      float v = fpcm[i];
      if (v > 1.0f) { v = 1.0f; }
      if (v < -1.0f) { v = -1.0f; }
      ipcm[i] = (int16_t)(v >= 0.0f ? v * 32767.0f + 0.5f
                                     : v * 32767.0f - 0.5f);
    }
  srx_segdump_add(ipcm, ns);
  (void)srx_pb_write(pb, ipcm, ns);
}


/****************************************************************************
 * 分段流水线：边收边解边播（2026-10-29）
 *
 * 旧 block 模式要等整段到齐才解，3 秒语音从"说完"到"出声"要 20 s。
 * 这里把"收"和"解"拆到两个核：CPU0 收包，CPU1 按段解码，解完一段就进
 * 播放缓冲。首声 = 第一段到齐 + 第一段解码，不再等整段。
 *
 * 段与段之间带 OV 个 token 的真实历史（默认 32），只取本段输出，边界
 * 有真实上下文；每段内部仍是原来的 jx_decode 整段调用，数值路径不变。
 ****************************************************************************/

#define SRX_SEGQ_MAX 8

struct srx_segjob
{
  int64_t *tok;
  int      n;
  int      skip;
  int      take;
};

struct srx_segpipe
{
  pthread_mutex_t mu;
  pthread_cond_t  cv_put;
  pthread_cond_t  cv_get;
  struct srx_segjob q[SRX_SEGQ_MAX];
  int head;
  int tail;
  int nq;
  int done;
  int rc;
  struct srx_pb *pb;
};

static void srx_segpipe_init(struct srx_segpipe *sp, struct srx_pb *pb)
{
  memset(sp, 0, sizeof(*sp));
  pthread_mutex_init(&sp->mu, NULL);
  pthread_cond_init(&sp->cv_put, NULL);
  pthread_cond_init(&sp->cv_get, NULL);
  sp->pb = pb;
  sp->rc = 0;
}

static void srx_segpipe_free(struct srx_segpipe *sp)
{
  while (sp->nq > 0)
    {
      free(sp->q[sp->head].tok);
      sp->head = (sp->head + 1) % SRX_SEGQ_MAX;
      sp->nq--;
    }
  pthread_mutex_destroy(&sp->mu);
  pthread_cond_destroy(&sp->cv_put);
  pthread_cond_destroy(&sp->cv_get);
}

static void *srx_seg_worker(void *arg)
{
  struct srx_segpipe *sp = (struct srx_segpipe *)arg;
  int16_t *ipcm;
  {
    cpu_set_t cs;
    CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
  }
  ipcm = (int16_t *)malloc(sizeof(int16_t) * SRX_PCM_CAP);
  if (ipcm == NULL) { return NULL; }

  for (;;)
    {
      struct srx_segjob jb;
      float *out = NULL;
      int rc;

      pthread_mutex_lock(&sp->mu);
      while (sp->nq == 0 && !sp->done)
        {
          pthread_cond_wait(&sp->cv_get, &sp->mu);
        }
      if (sp->nq == 0 && sp->done)
        {
          pthread_mutex_unlock(&sp->mu);
          break;
        }
      jb = sp->q[sp->head];
      sp->head = (sp->head + 1) % SRX_SEGQ_MAX;
      sp->nq--;
      pthread_cond_signal(&sp->cv_put);
      pthread_mutex_unlock(&sp->mu);

      rc = jx_decode(jb.tok, jb.n, jb.n * SRX_HOP, &out);
      if (rc == 0 && out != NULL)
        {
          int off = 0;
          while (off < jb.take)
            {
              int ns = jb.take - off;
              if (ns > SRX_PCM_CAP) { ns = SRX_PCM_CAP; }
              srx_emit(sp->pb, out + jb.skip + off, ns, ipcm);
              off += ns;
            }
          free(out);
        }
      else if (sp->rc == 0)
        {
          sp->rc = (rc != 0) ? rc : -1;
        }

      free(jb.tok);
    }

  free(ipcm);
  return NULL;
}

/* 提交一段：tok 指向 alltok 的 t0*tpp 位置，段内长 n，跳过 skip 个样本，
 * 只输出 take 个样本。令牌按值复制，调用方可以继续往后写 alltok。 */
static int srx_seg_submit(struct srx_segpipe *sp, const int64_t *tok,
                          int n, int skip, int take)
{
  int64_t *cp = (int64_t *)malloc(sizeof(int64_t) * (size_t)n * SRX_TPP);
  if (cp == NULL) { return -1; }
  memcpy(cp, tok, sizeof(int64_t) * (size_t)n * SRX_TPP);

  pthread_mutex_lock(&sp->mu);
  while (sp->nq == SRX_SEGQ_MAX && !sp->done)
    {
      pthread_cond_wait(&sp->cv_put, &sp->mu);
    }
  if (sp->done)
    {
      pthread_mutex_unlock(&sp->mu);
      free(cp);
      return -1;
    }
  sp->q[sp->tail].tok  = cp;
  sp->q[sp->tail].n    = n;
  sp->q[sp->tail].skip = skip;
  sp->q[sp->tail].take = take;
  sp->tail = (sp->tail + 1) % SRX_SEGQ_MAX;
  sp->nq++;
  pthread_cond_signal(&sp->cv_get);
  pthread_mutex_unlock(&sp->mu);
  return 0;
}

static void srx_seg_finish(struct srx_segpipe *sp)
{
  pthread_mutex_lock(&sp->mu);
  sp->done = 1;
  pthread_cond_broadcast(&sp->cv_get);
  pthread_cond_broadcast(&sp->cv_put);
  pthread_mutex_unlock(&sp->mu);
}

extern void jx_ui_set_status(const char *s);
extern void jx_ui_set_rx(int pkt, int rssi);
extern void jx_ui_set_tx(int pkt);

int jx_lora_streamrx_live(int secs)
{
  SDecState *dec = NULL;
  int decode_chunk=SRX_NT_CHUNK;
  const char *chunk_option=getenv("JX_D_CHUNK");
  if(chunk_option && atoi(chunk_option)==8)decode_chunk=8;
  struct srx_pb pb;
  struct srx_tq tq;
  pthread_t th;
  pthread_t dth;
  pthread_attr_t attr;
  int64_t tok[SRX_TPP];
  int64_t prev[SRX_TPP];
  int have_prev = 0;
  float fpcm[SRX_PCM_CAP];
  int16_t ipcm[SRX_PCM_CAP];
  uint32_t next_seq = 0;
  int have_seq = 0;
  int npkt = 0;
  int bad = 0;
  int gaps = 0;
  int decoded = 0;
  int last_seen = 0;
  uint32_t last_rx_ms = 0;
  /* 50 ms：首段 16 token 只有 96 ms，预缓冲大了会把首声整整推后一段。 */
  int prebuf_ms = 50;
  int idle_ms = 8000;
  int block_mode = 0;
  int64_t *alltok = NULL;
  int alln = 0;
  struct srx_segpipe spipe;
  pthread_t spt;
  int seg_on = 0;
  int seg_len = 152;
  int seg_ov = 16;
  int seg_first = 16;
  int seg_done = 0;
  int seg_ok = 0;
  struct timespec t0;
  struct timespec now;
  const char *e;

  if (secs < 1) { secs = 1; }

  e = getenv("JX_RX_PREBUF_MS");
  if (e != NULL) { prebuf_ms = atoi(e); }
  if (prebuf_ms < 0) { prebuf_ms = 0; }
  e = getenv("JX_RX_BLOCK");
  if (e != NULL) { int v = atoi(e); if (v > 0) { block_mode = v; } }
  e = getenv("JX_RXIDLE");
  if (e != NULL) { idle_ms = atoi(e); }
  if (idle_ms < 500) { idle_ms = 500; }

  {
    cpu_set_t cs;
    CPU_ZERO(&cs); CPU_SET(0, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
  }

  /* Receiver must preempt CPU0's optional compute worker. */
  { const char *dual=getenv("JX_D_DUAL");
    if(!dual || dual[0]=='1') {
      struct sched_param priority;priority.sched_priority=160;
      (void)pthread_setschedparam(pthread_self(),SCHED_FIFO,&priority);
    }
  }

  if (!block_mode)
    {
      dec = sdec_new_chunk(decode_chunk);
      if (dec == NULL)
        {
          printf("[srx2] sdec_new failed\n");
          return -1;
        }
    }

  if (srx_pb_init(&pb, SRX_RING_SAMP, prebuf_ms * 16) != 0)
    {
      printf("[srx2] playbuf init failed\n");
      if (dec != NULL) { sdec_free(dec); }
      return -1;
    }

  if (pthread_attr_init(&attr) != 0 ||
      pthread_attr_setstacksize(&attr, 32768) != 0 ||
      pthread_create(&th, &attr, srx_pb_worker, &pb) != 0)
    {
      printf("[srx2] play thread create failed\n");
      pthread_attr_destroy(&attr);
      srx_pb_free(&pb);
      if (dec != NULL) { sdec_free(dec); }
      return -1;
    }
  pthread_attr_destroy(&attr);

  srx_tq_init(&tq, dec, &pb, decode_chunk);
  if (block_mode == 2)
    {
      const char *se = getenv("JX_SEG");
      const char *oe = getenv("JX_OV");
      const char *fe = getenv("JX_SEG_FIRST");
      /* 接收端默认分块宽度：1536 比 512 少 30% 的分块 halo 开销，
       * 又不至于像 2048 那样在末段把 1.8 MB 的 blk 顶到 OOM。 */
      if (getenv("JX_CHUNK") == NULL) { setenv("JX_CHUNK", "1536", 1); }
      pthread_attr_t sa;
      if (se != NULL) { int v = atoi(se); if (v >= 32) { seg_len = v; } }
      if (oe != NULL) { int v = atoi(oe); if (v >= 0 && v < seg_len) { seg_ov = v; } }
      if (fe != NULL) { int v = atoi(fe); if (v >= 16 && v <= seg_len) { seg_first = v; } }
      alltok = (int64_t *)malloc(sizeof(int64_t) * (size_t)SRX_TPP * 4096);
      /* 必须在 worker 起来之前把 dump 开关打开：worker 在另一个核上，
       * 首段 emit 可能紧跟着第一次收包就到，晚设会丢掉第一段。 */
      {
        const char *dr = getenv("JX_RX_DUMP");
        if (dr != NULL && atoi(dr) != 0) { srx_dump_on = 1; srx_dump_n = 0; }
      }
      if (alltok != NULL)
        {
          srx_segpipe_init(&spipe, &pb);
          if (pthread_attr_init(&sa) == 0 &&
              pthread_attr_setstacksize(&sa, 262144) == 0 &&
              pthread_create(&spt, &sa, srx_seg_worker, &spipe) == 0)
            {
              seg_on = 1;
            }
          else
            {
              srx_segpipe_free(&spipe);
            }
          pthread_attr_destroy(&sa);
        }
      if (!seg_on)
        {
          printf("[srx2] 分段流水线启动失败，回退收齐再解\n");
          free(alltok); alltok = NULL;
          block_mode = 1;
        }
      else
        {
          printf("[srx2] 分段流水：SEG=%d OV=%d 首段=%d（收包 CPU0 / 解码 CPU1）\n",
                 seg_len, seg_ov, seg_first);
        }
    }
  else if (block_mode)
    {
      alltok = (int64_t *)malloc(sizeof(int64_t) * (size_t)SRX_TPP * 4096);
      if (alltok == NULL) { block_mode = 0; }
    }
  if (!block_mode &&
      (pthread_attr_init(&attr) != 0 ||
       pthread_attr_setstacksize(&attr, 32768) != 0 ||
       pthread_create(&dth, &attr, srx_dec_worker, &tq) != 0))
    {
      printf("[srx2] dec thread create failed\n");
      srx_pb_finish(&pb);
      pthread_join(th, NULL);
      srx_pb_free(&pb);
      if (dec != NULL) { sdec_free(dec); }
      return -1;
    }
  pthread_attr_destroy(&attr);

  if (lora_open() < 0)
    {
      srx_pb_finish(&pb);
      pthread_join(th, NULL);
      srx_pb_free(&pb);
      if (dec != NULL) { sdec_free(dec); }
      return -1;
    }

  /* 2026-10-29: 发送刚结束、芯片还在 TX->STDBY 切换时，直接
   * set_standby/config 会偶发失败（实测发完 1 秒后进入接收就报
   * config failed，之后就再也收不到）。改成重试 5 次、每次隔 200 ms，
   * 让 BUSY 落下来。失败也不再 close fd（反复 open/close 会拖死 SPI）。 */
  {
    int sb = -1;
    int cf = -1;
    int attempt;

    /* 2026-10-29: 先把天线开关切到接收侧，再让芯片进 standby。
     * 顺序反了（先 set_standby 后 sw_rx）会在 TX->RX 过渡时把
     * SX1262 状态机弄死，之后所有命令 BUSY 超时。 */
    sw_rx();

    for (attempt = 0; attempt < 5; attempt++)
      {
        sb = sx_set_standby();
        cf = (sb == 0) ? sx_config_lora(433175000, jx_lora_dbm(),
                                         jx_lora_sf(), jx_lora_bw(), 1, 0) : -99;
        if (sb == 0 && cf == 0) { break; }

        printf("[srx2] config retry %d sb=%d cf=%d busy=%d\n", attempt, sb, cf,
               esp32s3_gpioread(LORA_PIN_BUSY) ? 1 : 0);
        usleep(500 * 1000);
      }

    if (sb != 0 || cf != 0)
      {
        printf("[srx2] config failed sb=%d cfg=%d\n", sb, cf);
        srx_pb_finish(&pb);
        pthread_join(th, NULL);
        srx_pb_free(&pb);
        if (dec != NULL) { sdec_free(dec); }
        usleep(1000 * 1000);
        return -1;
      }
  }

  sx_clear_device_errors();
  sw_rx();
  sx_clear_irq(0xFFFF);
  sx_start_rx(0);

  printf("[srx2] 边收边解边播：预缓冲 %d ms，idle %d ms，上限 %d s\n",
         prebuf_ms, idle_ms, secs);
  printf("[srx2] READY secs=%d\n", secs);
  jx_ui_set_status("RX WAIT");
  clock_gettime(CLOCK_MONOTONIC, &t0);

  for (;;)
    {
      uint16_t irq = 0;

      /* 对讲机是半双工：要发的时候立刻让出射频，避免两个任务抢 SPI。 */
      if (g_lora_tx_busy)
        {
          printf("[srx2] TX busy -> yield radio\n");
          break;
        }

      if (sx_get_irq(&irq) == 0 && irq != 0)
        {
          if (irq & SX_IRQ_RX_DONE)
            {
              uint8_t len = 0;
              uint8_t start = 0;
              int16_t rssi = 0;
              uint8_t pkt[LORA_PKT_LEN];

              sx_get_rx_info(&len, &start, &rssi);
              if (len > LORA_PKT_LEN) { len = LORA_PKT_LEN; }
              sx_read_rx_payload(start, pkt, len);
              sx_clear_irq(0xFFFF);
              sx_start_rx(0);

              if (len == LORA_PKT_LEN && pkt[0] == 'J' && pkt[1] == 'X' &&
                  (pkt[2] == 0x21 || pkt[2]==JX_LL_TYPE) && srx_rd16(pkt + 12) == SRX_TPP)
                {
                  uint32_t seq = srx_rd32(pkt + 4);
                  int seconds=pkt[2]==JX_LL_TYPE?jxll_seconds(pkt,len):0;
                  int crc_ok = pkt[2]==JX_LL_TYPE ? seconds>0 :
                       (srx_rd16(pkt+14)==crc16_ccitt(pkt+16,36));
                  int keep_bad = 0;
                  const char *kb = getenv("JX_RX_KEEP_BAD");
                  if (kb != NULL && atoi(kb) != 0) { keep_bad = 1; }

                  if (!crc_ok && !keep_bad)
                    {
                      bad++;
                    }
                  else
                    {
                      if (!crc_ok) { bad++; }

                      if (!have_seq)
                        {
                          have_seq = 1;
                          next_seq = seq;
                        }

                      if (seq >= next_seq)
                        {
                          int miss = (seq > next_seq) ? (int)(seq - next_seq) : 0;

                          if (miss > 0) { gaps += miss; }
                          if (miss > 8) { miss = 8; }
                          while (miss-- > 0)
                            {
                              srx_tq_push(&tq, NULL, 2);
                            }

                          next_seq = seq + 1;
                          if (pkt[3] & 0x01) { last_seen = 1; }

                          srx_pb_plan(&pb,seconds,(int)seq+1);
                          srx_unpack(pkt + 16, SRX_TPP, tok);
                          if (block_mode && alln < 4096)
                            {
                              memcpy(alltok + (size_t)alln * SRX_TPP, tok,
                                     sizeof(int64_t) * SRX_TPP);
                              alln++;
                              if (block_mode == 2 && seg_on)
                                {
                                  int seg_cur = (seg_done == 0) ? seg_first : seg_len;
                                  while ((alln * SRX_TPP) >= (seg_done + seg_cur))
                                    {
                                      int t0 = (seg_done >= seg_ov) ?
                                               (seg_done - seg_ov) : 0;
                                      int t1 = seg_done + seg_cur;
                                      int nn = t1 - t0;
                                      if (srx_seg_submit(&spipe,
                                              alltok + (size_t)t0 * SRX_TPP,
                                              nn, (seg_done - t0) * SRX_HOP,
                                              (t1 - seg_done) * SRX_HOP) != 0)
                                        {
                                          printf("[srx2] 段提交失败 @%d\n", seg_done);
                                        }
                                      seg_done = t1;
                                      seg_cur = seg_len;
                                      seg_ok++;
                                    }
                                }
                            }
                          else if (!block_mode)
                            {
                              srx_tq_push(&tq, tok, 1);
                            }
                          npkt++;
                          {
                            struct timespec tm;
                            clock_gettime(CLOCK_MONOTONIC, &tm);
                            last_rx_ms = (uint32_t)tm.tv_sec * 1000u +
                                         (uint32_t)(tm.tv_nsec / 1000000u);
                          }
                          if ((npkt % 8) == 0 || last_seen)
                            {
                              printf("[srx2] 收包 %d seq=%u rssi=%d gaps=%d%s\n",
                                     npkt, (unsigned)seq, rssi, gaps,
                                     last_seen ? " LAST" : "");
                              {
                                char ub[32];
                                snprintf(ub, sizeof(ub), "RX %d pkt", npkt);
                                jx_ui_set_status(ub);
                                jx_ui_set_rx(npkt, rssi);
                              }
                            }
                          if (last_seen) { break; }
                        }
                    }
                }
              else
                {
                  bad++;
                }
            }
          else
            {
              sx_clear_irq(0xFFFF);
              sx_start_rx(0);
            }
        }

      clock_gettime(CLOCK_MONOTONIC, &now);

      if (npkt > 0 && last_rx_ms != 0)
        {
          uint32_t now_ms = (uint32_t)now.tv_sec * 1000u +
                            (uint32_t)(now.tv_nsec / 1000000u);
          uint32_t idle = now_ms - last_rx_ms;
          if (idle >= (uint32_t)idle_ms)
            {
              printf("[srx2] 连续 %u ms 没有新包，结束\n", idle);
              break;
            }
        }

      if ((now.tv_sec - t0.tv_sec) >= secs)
        {
          printf("[srx2] timeout %d/%d\n", (int)(now.tv_sec - t0.tv_sec), secs);
          break;
        }

      /* 2026-10-29: 原来是 500us 轮询一次 SX1262 的 IRQ，等于每秒 2000
       * 次 SPI 事务；跑几十秒（十几万次）后 esp32s3 SPI 驱动会把完成
       * 中断丢掉，下一次 ioctl 永久阻塞，整机卡死。发端每包间隔 20 ms，
       * 3 ms 轮询完全够用，事务量降到 1/6。 */
      usleep(3000);
    }

  if (block_mode == 2 && seg_on)
    {
      int Tt2 = alln * SRX_TPP;
      if (seg_done < Tt2)
        {
          int t0 = (seg_done >= seg_ov) ? (seg_done - seg_ov) : 0;
          int nn = Tt2 - t0;
          (void)srx_seg_submit(&spipe, alltok + (size_t)t0 * SRX_TPP,
                               nn, (seg_done - t0) * SRX_HOP,
                               (Tt2 - seg_done) * SRX_HOP);
          seg_ok++;
          seg_done = Tt2;
        }
      {
        struct timespec w0, w1;
        clock_gettime(CLOCK_MONOTONIC, &w0);
        srx_seg_finish(&spipe);
        pthread_join(spt, NULL);
        clock_gettime(CLOCK_MONOTONIC, &w1);
        printf("[srx2] 流水线解码完成 %d 段，rc=%d，尾段等待 %.1f s\n",
               seg_ok, spipe.rc,
               (double)(w1.tv_sec - w0.tv_sec) +
               (double)(w1.tv_nsec - w0.tv_nsec) / 1e9);
      }
      srx_segpipe_free(&spipe);
      tq.decoded = seg_done * SRX_HOP;
      free(alltok); alltok = NULL;
    }
  else if (block_mode)
    {
      float *pcm = NULL;
      int Tt = alln * SRX_TPP;
      struct timespec _b0, _b1; double tb0, tb1;
      clock_gettime(CLOCK_MONOTONIC, &_b0);
      int rc = 0;
      if (Tt > 0)
        {
          /* 2026-10-29：整段解码在 8 MB PSRAM 上放不下（stage2 的 up 要
           * 3.10 MB 连续块）。改成带重叠的分段整段解码：每段 192 token，
           * 前面多带 32 token 的真实历史，只取本段对应的输出。这样每段
           * 的内存需求约 1/3，且段边界有真实上下文，不会出现断层。 */
          int SEG = 128, OV = 32;
          { const char *se = getenv("JX_SEG"); if (se != NULL) { int v = atoi(se); if (v >= 32) { SEG = v; } }
            const char *oe = getenv("JX_OV");  if (oe != NULL) { int v = atoi(oe); if (v >= 0 && v < SEG) { OV = v; } } }
          for (int s0 = 0; s0 < Tt; s0 += SEG)
            {
              int t0 = (s0 >= OV) ? (s0 - OV) : 0;
              int t1 = s0 + SEG; if (t1 > Tt) { t1 = Tt; }
              int n  = t1 - t0;
              int skip = (s0 - t0) * SRX_HOP;
              int take = t1 - s0;
              take *= SRX_HOP;
              pthread_t bt; pthread_attr_t ba;
              struct srx_bdec da;
              da.tok = alltok + (size_t)t0 * SRX_TPP;
              da.Tt = n; da.out = NULL; da.rc = -1;
              if (pthread_attr_init(&ba) == 0 &&
                  pthread_attr_setstacksize(&ba, 262144) == 0 &&
                  pthread_create(&bt, &ba, srx_block_dec, &da) == 0)
                {
                  pthread_join(bt, NULL);
                  pthread_attr_destroy(&ba);
                  if (da.rc == 0 && da.out != NULL)
                    {
                      srx_emit(&pb, da.out + skip, take, ipcm);
                      tq.decoded += take;
                      free(da.out);
                    }
                  else
                    {
                      rc = da.rc;
                      printf("[srx2] 段 %d 解码失败 rc=%d\n", s0 / SEG, da.rc);
                    }
                }
              else
                {
                  rc = -1;
                }
            }
        }
      clock_gettime(CLOCK_MONOTONIC, &_b1);
      tb0 = (double)_b0.tv_sec + (double)_b0.tv_nsec / 1e9;
      tb1 = (double)_b1.tv_sec + (double)_b1.tv_nsec / 1e9;
      printf("[srx2] 分段解码 rc=%d Tt=%d 用时 %.1f s\n", rc, Tt, tb1 - tb0);
      (void)pcm;
      free(alltok);
      alltok = NULL;
    }
  else
    {
      srx_tq_finish(&tq);
      pthread_join(dth, NULL);
    }

  printf("[srx2] 收包结束：%d 包，坏包 %d，丢包 %d，解码 %d 样本（%.2f s）\n",
         npkt, bad, gaps, tq.decoded, (double)tq.decoded / 16000.0);
  { extern void jx_prof_dump(void); extern void jx_ct_dump(void); jx_prof_dump(); jx_ct_dump(); }


  /* Serial PCM export may take many seconds; it must not delay or compete
   * with playback. Join only after all queued audio has been consumed. */
  srx_pb_finish(&pb);
  pthread_join(th, NULL);

  if (srx_dump_on && srx_dump_n > 0)
    {
      printf("[srx2] 流水线 dump pcm %d samples (%.2f s)\n",
             srx_dump_n, (double)srx_dump_n / 16000.0);
      (void)jx_serial_send_pcm_i16("out", srx_dump_buf, srx_dump_n, 16000);
      srx_dump_n = 0;
    }

  {
    const char *dr = getenv("JX_RX_DUMP");
    if (dr != NULL && atoi(dr) != 0 && tq.dumpn > 0)
      {
        printf("[srx2] dump pcm %d samples (%.2f s)\n",
               tq.dumpn, (double)tq.dumpn / 16000.0);
        (void)jx_serial_send_pcm_i16("out", tq.dump, tq.dumpn, 16000);
      }
  }
  free(tq.dump);
  tq.dump = NULL;

  srx_pb_free(&pb);
  if (dec != NULL) { sdec_free(dec); }
  /* 2026-10-29：接收结束时必须把射频状态一起清掉。
   * 只 close(fd) 会留下 g_lora_ready=1，下一次按 BOOT 发送时
   * jx_lora_send_idx 以为芯片还配着，跳过 standby/config，
   * 结果 SX1262 停在 RX 状态，发到中途就等不到 TX_DONE。
   * 这里走 jx_lora_release()，同时清 g_lora_ready / seq / tokidx。 */
  jx_lora_release();
  return 0;
}
