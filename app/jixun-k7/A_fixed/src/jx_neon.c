#include "jx_neon.h"
#ifdef JX_NEON_ENABLED
#include <pthread.h>
#include <stdlib.h>
/* Immutable model weights are packed once, shared across repeated calls.
 * Four output channels are adjacent; integer conversion is exact in float.
 * Scales remain outside the accumulation, as in the scalar reference. */
struct packed_weight {const int8_t *key;int in,out;float *data;};
static struct packed_weight cache[128];
static int cached;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
const float *jx_neon_pack_i8(const int8_t *weights,int in,int out) {
    if(in<=0||out<4)return NULL;
    pthread_mutex_lock(&lock);
    for(int i=0;i<cached;i++) {
        if(cache[i].key==weights&&cache[i].in==in&&cache[i].out==out) {
            const float *p=cache[i].data;pthread_mutex_unlock(&lock);return p;
        }
    }
    if(cached==128){pthread_mutex_unlock(&lock);return NULL;}
    float *data=malloc((size_t)(out/4)*4*in*sizeof(float));
    if(data) {
        for(int o=0;o+3<out;o+=4)
            for(int j=0;j<in;j++)
                for(int lane=0;lane<4;lane++)data[(size_t)o*in+j*4+lane]=(float)weights[(size_t)(o+lane)*in+j];
        cache[cached++]=(struct packed_weight){weights,in,out,data};
    }
    pthread_mutex_unlock(&lock);return data;
}
#endif
