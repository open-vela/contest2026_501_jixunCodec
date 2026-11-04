/****************************************************************************
 * apps/jixun-codec/port/jx_serial.h
 *
 * USB console transport for the SAFE DEMO mode.
 * The PC receives @@JXTOK1:<hex> lines, displays token statistics, and
 * forwards the same frame to the receiver board.  No PCM crosses USB.
 ****************************************************************************/

#ifndef __APPS_JIXUN_CODEC_PORT_JX_SERIAL_H
#define __APPS_JIXUN_CODEC_PORT_JX_SERIAL_H

#include <stdint.h>

int jx_serial_enabled(void);

int jx_serial_pcm_dump_enabled(void);

int jx_serial_send_pcm_i16(const char *kind, const int16_t *pcm,
                           int nsamples, int rate);

int jx_serial_send_pcm_f32(const char *kind, const float *pcm,
                           int nsamples, int rate);

int jx_serial_send_idx(int rate_id, int bits, const int64_t *idx,
                       int ntok, int tt, int orig_t,
                       uint32_t chunk, uint32_t last);

int jx_serial_recv_idx(int timeout_s, int *rate_id, int *bits,
                       int64_t **out_idx, int *out_ntok, int *out_tt,
                       int *out_orig_t, uint32_t *out_chunk,
                       uint32_t *out_last);

#endif /* __APPS_JIXUN_CODEC_PORT_JX_SERIAL_H */
