#ifndef JX_ENV_H
#define JX_ENV_H

/* Serialize application environment access without a preemptible spinlock.
 * The same gate covers readers and writers in the two streaming threads. */
char *jx_getenv_safe(const char *name);
int jx_setenv_safe(const char *name, const char *value, int overwrite);
int jx_unsetenv_safe(const char *name);

#ifdef JX_SAFE_GETENV
#  define getenv(name) jx_getenv_safe(name)
#  define setenv(name, value, overwrite) jx_setenv_safe(name, value, overwrite)
#  define unsetenv(name) jx_unsetenv_safe(name)
#endif

#endif
