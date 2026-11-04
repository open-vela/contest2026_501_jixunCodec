/* K7 header: SPI4 M2 GPIO4_B0/B1/B2 AF9; software CS GPIO1_D0.
 * Sources: KICKPI SPI documentation + Rockchip rk3576-pinctrl.dtsi.
 * Pinmux is a board binding; no Linux device files or GPIO ioctls. */
#include <nuttx/config.h>
#include <nuttx/spi/spi.h>
#include <errno.h>
#include "rk3576_gpio.h"
#include "rk3576_spi.h"
#include "k7_gpio.h"
#include "k7_native_spi.h"
static struct spi_dev_s *bus;
static struct gpio_dev_s *pins[3];
static k7_line_t cs;
int k7_native_spi_open(void){
 static const unsigned map[3]={8,9,10};
 if(bus){errno=EBUSY;return -1;}
 for(int i=0;i<3;i++){
  int rc=rk3576_gpio_get(GPIO_PORT4|map[i],&pins[i]);
  if(rc<0){errno=-rc;goto fail;}
  rk3576_gpio_set_pull(pins[i],RK3576_GPIO_FLOAT);rk3576_gpio_set_af(pins[i],9);
 }
 if(k7_line_open(&cs,"gpio1",24,1,"sx-cs"))goto fail;
 k7_line_set(&cs,1);bus=rk3576_spi_initialize(4);if(!bus){errno=ENODEV;goto fail;}
 return 1;
fail:k7_native_spi_close();return -1;
}
int k7_native_spi_transfer(const struct spi_ioc_transfer *tr){
 if(!bus||!tr||tr->len>300){errno=EINVAL;return -1;}
 int rc=SPI_LOCK(bus,true);if(rc<0){errno=-rc;return -1;}
 SPI_SETFREQUENCY(bus,tr->speed_hz);SPI_SETMODE(bus,SPIDEV_MODE0);SPI_SETBITS(bus,8);
 k7_line_set(&cs,0);SPI_SELECT(bus,0,true);
 SPI_EXCHANGE(bus,(const void*)tr->tx_buf,(void*)tr->rx_buf,tr->len);
 SPI_SELECT(bus,0,false);k7_line_set(&cs,1);SPI_LOCK(bus,false);return tr->len;
}
void k7_native_spi_close(void){
 k7_line_set(&cs,1);k7_line_close(&cs);bus=NULL;
 for(int i=0;i<3;i++)if(pins[i]){rk3576_gpio_set_af(pins[i],0);rk3576_gpio_put(pins[i]);pins[i]=NULL;}
}
