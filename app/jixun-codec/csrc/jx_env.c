#include "jx_env.h"

/* Undo the macro before including the libc declaration and calling the
 * real getenv from inside the wrapper. */
#undef getenv
#include <stdlib.h>

char *jx_getenv_safe(const char *name)
{
  static volatile int lock;
  char *value;

  while (__sync_lock_test_and_set((int *)&lock, 1) != 0) { }
  value = getenv(name);
  __sync_lock_release((int *)&lock);
  return value;
}
