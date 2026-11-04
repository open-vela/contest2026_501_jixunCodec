/* ============================================================================
 * model_config.c — rate dispatcher (single compile entry point)
 *
 * All three per-rate configs (model_config_3k.c / _6k.c / _1k.c) define the
 * same set of `extern const` globals declared in model_config.h.  Instead of
 * linking a different .c file by hand (easy to get wrong / duplicate-def), we
 * pick ONE of them at COMPILE TIME via a -D macro:
 *
 *     gcc ... -D SQM_RATE_3K ... model_config.c
 *     gcc ... -D SQM_RATE_6K ... model_config.c
 *     gcc ... -D SQM_RATE_1K ... model_config.c
 *
 * build.sh does this for you:  ./build.sh 1k  test/test_nn_infer.c
 *
 * The rate's numeric config (dims/strides/depths/levels/window) comes entirely
 * from the included body; the weight headers (weights_data.h / golden.h) are
 * selected by build.sh via the -I ordering for that rate.
 * ========================================================================== */
#include "model_config.h"

#if   defined(SQM_RATE_3K) || defined(SQM_RATE_3k)
  #include "model_config_3k.c"
#elif defined(SQM_RATE_6K) || defined(SQM_RATE_6k)
  #include "model_config_6k.c"
#elif defined(SQM_RATE_1K) || defined(SQM_RATE_1k)
  #include "model_config_1k.c"
#else
  #error "Set the codec rate: pass -D SQM_RATE_3K | -D SQM_RATE_6K | -D SQM_RATE_1K"
#endif
