/* Native RK3576 GPIO transport for the existing SX1262 protocol driver. */
#include <nuttx/config.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rk3576_gpio.h"
#include "k7_gpio.h"
static struct gpio_dev_s *claims[32];
int k7_line_open_bias(k7_line_t*l,const char*chip,uint32_t off,int output,const char*name,uint64_t bias){
 if(!l||!chip||strncmp(chip,"gpio",4)||off>31){errno=EINVAL;return -1;}
 char *end;long bank=strtol(chip+4,&end,10);if(*end||bank<0||bank>4){errno=EINVAL;return -1;}
 memset(l,0,sizeof(*l));l->chip_fd=l->line_fd=-1;
 struct gpio_dev_s *handle=NULL;int rc=rk3576_gpio_get((bank<<GPIO_PORT_SHIFT)|(off<<GPIO_PIN_SHIFT),&handle);
 if(rc<0){errno=-rc;return -1;}
 int slot;for(slot=0;slot<32&&claims[slot];slot++);if(slot==32){rk3576_gpio_put(handle);errno=EMFILE;return -1;}
 claims[slot]=handle;l->native_handle=handle;l->offset=off;l->is_output=output;
 rk3576_gpio_set_af(handle,0);rk3576_gpio_set_pull(handle,RK3576_GPIO_FLOAT);
 rk3576_gpio_set_mode(handle,output?RK3576_GPIO_OUTPUT:RK3576_GPIO_INPUT);
 return 0;
}
int k7_line_open(k7_line_t*l,const char*c,uint32_t o,int out,const char*n){return k7_line_open_bias(l,c,o,out,n,0);}
int k7_line_set(const k7_line_t*l,int v){if(!l||!l->native_handle){errno=EBADF;return -1;}rk3576_gpio_write_bit(l->native_handle,v!=0);return 0;}
int k7_line_get(const k7_line_t*l,int*v){if(!l||!l->native_handle||!v){errno=EBADF;return -1;}*v=rk3576_gpio_read_bit(l->native_handle);return 0;}
void k7_line_close(k7_line_t*l){if(!l||!l->native_handle)return;for(int i=0;i<32;i++)if(claims[i]==l->native_handle)claims[i]=NULL;rk3576_gpio_put(l->native_handle);l->native_handle=NULL;}
void k7_gpio_release_all(void){for(int i=0;i<32;i++)if(claims[i]){rk3576_gpio_put(claims[i]);claims[i]=NULL;}}
