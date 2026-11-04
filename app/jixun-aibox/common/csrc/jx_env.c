#include "jx_env.h"
#undef getenv
#undef setenv
#undef unsetenv
#include <stdlib.h>
#include <semaphore.h>
#include <errno.h>

/* A blocking semaphore avoids spinning while a lower priority owner is
 * preempted. Cover writes too: streaming setup changes flags while the
 * encoder and sender may already be running on different CPUs. */
static sem_t g_env_gate = SEM_INITIALIZER(1);

static int jx_env_enter(void)
{
  int ret;
  do { ret = sem_wait(&g_env_gate); } while (ret < 0 && errno == EINTR);
  return ret;
}

char *jx_getenv_safe(const char *name)
{
  char *value;
  if (jx_env_enter() < 0) return NULL;
  value = getenv(name);
  sem_post(&g_env_gate);
  return value;
}

int jx_setenv_safe(const char *name, const char *value, int overwrite)
{
  int ret, saved_errno;
  if (jx_env_enter() < 0) return -1;
  ret = setenv(name, value, overwrite);
  saved_errno = errno;
  sem_post(&g_env_gate);
  errno = saved_errno;
  return ret;
}

int jx_unsetenv_safe(const char *name)
{
  int ret, saved_errno;
  if (jx_env_enter() < 0) return -1;
  ret = unsetenv(name);
  saved_errno = errno;
  sem_post(&g_env_gate);
  errno = saved_errno;
  return ret;
}
