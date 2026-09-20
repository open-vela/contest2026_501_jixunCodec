#ifndef TRANSFORMER_H
#define TRANSFORMER_H
/* ============================================================================
 * transformer.h — local-attention transformer (C port of l3ac local_attention
 * / transformer.py). This is the shared backbone behind BOTH the encoder-side
 * LocalEncoder (en_encoder) and the decoder-side LocalDecoder (en_decoder):
 *   - local_mha_layer() -> LocalMHA (causal windowed attention + dynamic pos bias)
 *   - feed_forward()     -> FeedForward (pre-norm GELU MLP)
 *   - local_transformer()-> LocalTransformer (stack of MHA + FFN residuals)
 * Config below mirrors the 3kbps LocalEncoder/LocalDecoder.
 * ========================================================================== */
#include "nn_ops.h"
/* local attention config is declared in model_config.h (extern) and defined
 * per rate in model_config_*.c (pulled in via nn_ops.h -> model_config.h). */

/* FeedForward(pfx): pre-norm -> Linear(C->H) -> GELU*slice -> Linear(H/2->C),
 * where H (and thus H/2) is READ FROM THE WEIGHT SHAPE (rate-independent). */

/* LocalTransformer stack (depth = number of (MHA, FFN) layers).
 * pfx is e.g. "en_encoder.local_trans" / "en_decoder.local_trans"; tpfx is the
 * transformer root (for the shared DynamicPositionBias MLP). `window` is the
 * causal attention window (rate-specific: e.g. 400 for 3k/6k, 750 for the
 * 1kbps down/up blocks, 250 for its inner local blocks). */
void local_transformer(const T *x, T *out, const char *pfx, int depth, int window);

/* FeedForward(pfx): pre-norm -> Linear(C->H) -> GELU*slice -> Linear(H/2->C), H from weights. */
void feed_forward(const T *x, T *out, const char *pfx);

#endif /* TRANSFORMER_H */
