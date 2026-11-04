#include <pthread.h>
#include <stdio.h>
#include "native_runtime.h"
#include "rf/k7_native_spi.h"
static pthread_mutex_t owner=PTHREAD_MUTEX_INITIALIZER;
extern void k7_gpio_release_all(void);
int k7_native_claim(void){int r=pthread_mutex_trylock(&owner);if(r)fprintf(stderr,"k7: radio busy; stop the active node first\n");return r;}
void k7_native_release(void){k7_native_spi_close();k7_gpio_release_all();pthread_mutex_unlock(&owner);}
