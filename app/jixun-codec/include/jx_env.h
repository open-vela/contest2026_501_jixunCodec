#ifndef JX_ENV_H
#define JX_ENV_H

/* Codec hot paths run on both ESP32-S3 cores. NuttX getenv() is not
 * reentrant there, so route all codec environment lookups through one
 * short global spinlock. */
char *jx_getenv_safe(const char *name);

#ifdef JX_SAFE_GETENV
#  define getenv(name) jx_getenv_safe(name)
#endif

#endif
