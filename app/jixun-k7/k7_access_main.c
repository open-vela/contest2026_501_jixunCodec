/* Native K7-only JR admission. Reuses the Linux transport setup and the
 * unchanged wire state machine; Linux legacy offline decode helpers are excluded. */
#define _GNU_SOURCE
#include <nuttx/config.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "native_runtime.h"
#include "rf/sx1262.h"
#define PKT_BYTES 52
#define DEF_FREQ_HZ 433175000u
#define DEF_BW 0x06
#define DEF_DBM 22
static int g_rf_sf=8,pair_slave;
static unsigned g_rearm_commands;
static volatile sig_atomic_t g_stop;
static void on_signal(int n){(void)n;g_stop=1;}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double wall_s(void){struct timespec t;clock_gettime(CLOCK_REALTIME,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint16_t crc16(const uint8_t *d, int n)
{
  uint16_t c = 0xffff;
  for (int i = 0; i < n; i++)
    {
      c ^= (uint16_t)d[i] << 8;
      for (int b = 0; b < 8; b++)
        c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
  return c;
}

static uint32_t rd32(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void wr32(uint8_t *p, uint32_t v)
{
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

static int radio_open(sx1262_t *d)
{
  static k7_line_t oe;

  if (k7_line_open(&oe, "gpio3", 21, 1, "master-oe") != 0)
    {
      fprintf(stderr, "[master] oe open failed\n");
      return -1;
    }
  k7_line_set(&oe, 1);

  if (sx1262_init_spi(d, "/dev/spidev4.0", 1000000,
                      "gpio3:8", "gpio3:22", NULL) != 0)
    {
      fprintf(stderr, "[master] spi init failed\n");
      return -1;
    }

  sx1262_reset(d);
  struct timespec ts = {0, 60 * 1000 * 1000};
  nanosleep(&ts, NULL);

  if (sx1262_bind_sw(d, NULL, "gpio2:31", 1) != 0)
    {
      fprintf(stderr, "[master] rf switch bind failed\n");
      return -1;
    }

  if (sx1262_config_lora(d, DEF_FREQ_HZ, DEF_DBM, g_rf_sf, DEF_BW, 1, 0) != 0)
    {
      fprintf(stderr, "[master] lora config failed\n");
      return -1;
    }

  {
    uint8_t pa[4] = {0x04, 0x07, 0x00, 0x01};
    uint8_t tx[2] = {(uint8_t)(int8_t)DEF_DBM, 0x07};
    if (sx1262_cmd_write_public(d, SX_CMD_SET_PA_CFG, pa, 4) != 0 ||
        sx1262_cmd_write_public(d, SX_CMD_SET_TX_PARAMS, tx, 2) != 0)
      {
        fprintf(stderr, "[master] pa 22dbm config failed\n");
        return -1;
      }
  }

  if (sx1262_set_lora_pkt_params(d, 8, 0, PKT_BYTES, 1, 0) != 0)
    {
      fprintf(stderr, "[master] packet params failed\n");
      return -1;
    }
  if (sx1262_set_dio2_rfswitch(d, 1) != 0)
    {
      fprintf(stderr, "[master] dio2 rfswitch failed\n");
      return -1;
    }
  sx1262_clear_device_errors(d);
  sx1262_sw_rx(d);
  sx1262_clear_irq(d, 0xffff);
  if (sx1262_start_rx(d, 0) != 0)
    {
      fprintf(stderr, "[master] start rx failed\n");
      return -1;
    }

  fprintf(stderr,
          "[gateway] READY role=%c freq=%.3f sf=%d bw=0x%02x crc=1 dbm=%d\n",
          pair_slave?'B':'A', (double)DEF_FREQ_HZ / 1e6, g_rf_sf, DEF_BW, DEF_DBM);
  return 0;
}

static int radio_rearm(sx1262_t *d)
{
  g_rearm_commands++;
  sx1262_sw_rx(d);
  sx1262_clear_irq(d, 0xffff);
  return sx1262_start_rx(d, 0);
}

static int send_frame(sx1262_t *d, const uint8_t *pkt, int len)
{
  int rc = -1;

  if (sx1262_set_standby_xosc(d) != 0)
    {
      return -1;
    }
  sx1262_sw_tx(d);
  if (sx1262_set_lora_pkt_params(d, 8, 0, (uint8_t)len, 1, 0) == 0 &&
      sx1262_write_buffer(d, 0, pkt, (uint8_t)len) == 0 &&
      sx1262_clear_irq(d, 0xffff) == 0 &&
      sx1262_start_tx(d, 500) == 0)
    {
      double t0 = now_s();
      while (now_s() - t0 < 1.0)
        {
          uint16_t irq = 0;
          if (sx1262_get_irq(d, &irq) != 0)
            {
              break;
            }
          if (irq & SX_IRQ_TX_DONE)
            {
              rc = 0;
              break;
            }
          if (irq & SX_IRQ_TIMEOUT)
            {
              break;
            }
          struct timespec p = {0, 1000000};
          nanosleep(&p, NULL);
        }
    }

  sx1262_set_lora_pkt_params(d, 8, 0, PKT_BYTES, 1, 0);
  radio_rearm(d);
  return rc;
}
#include "access_sync.h"
int k7_access_main(int argc,char **argv){
 if(k7_native_claim())return 1;g_stop=0;g_rearm_commands=0;g_rf_sf=8;
 signal(SIGINT,on_signal);signal(SIGTERM,on_signal);
 double deadline=now_s()+(argc>1?atof(argv[1]):315360000);sx1262_t d;pair_init();
 if(radio_open(&d)){k7_native_release();return 2;}
 int rc=0;
 while(!g_stop&&now_s()<deadline){
  uint16_t irq=0;
  if(sx1262_get_irq(&d,&irq)){struct timespec pause={0,5000000};nanosleep(&pause,NULL);continue;}
  if(irq&SX_IRQ_RX_DONE){
   uint8_t len=0,off=0,pkt[96];int16_t rssi=0;
   int read_rc=sx1262_get_rx_info(&d,&len,&off,&rssi);
   if(!read_rc&&len<=sizeof(pkt))read_rc=sx1262_read_rx_payload(&d,off,pkt,len);else read_rc=-1;
   radio_rearm(&d);
   if(!read_rc)pair_receive(&d,pkt,len);
  }else if(irq&(SX_IRQ_TIMEOUT|SX_IRQ_CRC_ERR|SX_IRQ_HEADER_ERR))radio_rearm(&d);
  rc=pair_tick(&d,0);if(rc)break;
  struct timespec pause={0,500000};nanosleep(&pause,NULL);
 }
 sx1262_set_standby(&d);sx1262_deinit(&d);k7_native_release();return rc;
}
