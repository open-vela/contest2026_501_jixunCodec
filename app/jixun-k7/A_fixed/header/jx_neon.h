#ifndef JX_NEON_H
#define JX_NEON_H
#if defined(JX_NATIVE_NEON) && defined(__aarch64__) && !defined(ACC_DOUBLE)
#define JX_NEON_ENABLED 1
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
const float *jx_neon_pack_i8(const int8_t *weights, int in, int out);
int jx_parallel_start(void);
void jx_parallel_stop(void);
void jx_parallel_run(void (*fn)(void *,int,int),void *arg,int units,int quantum,size_t work);
#endif
#endif
