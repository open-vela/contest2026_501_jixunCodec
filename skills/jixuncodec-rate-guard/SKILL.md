---
name: jixuncodec-rate-guard
description: Validate an already-flashed Jixun Codec firmware on one serial-connected ESP32-S3 board. Use when checking the active 1k/3k/6k rate, repeated benchmark stability, encode/decode timing, or deterministic output fingerprints.
metadata:
  short-description: Validate Jixun Codec rate, timing, and fingerprint consistency
---

# Jixun Codec Rate Guard

Use this skill to verify that one board is running the intended Jixun Codec rate and that its deterministic benchmark remains stable across repeated rounds.

## Use

Run:

```bash
python scripts/run_rate_guard.py --port COM4 --rounds 3
```

The default checks all three known fingerprints. Pass `--expect-rate 3k` to require one specific build, or `--expect-fingerprint <idx>:<rec>` for a candidate build that is not yet in the known table.

## Rules

- This skill is read-only with respect to firmware. It must not call `write-flash` or `erase_flash`.
- Keep `JX_USB`, `JX_PCM_DUMP`, and `JX_BEEP` at `0` so the benchmark does not drive audio hardware.
- Report a rate as consistent only when every requested round returns the same `idx` and `rec` fingerprint.
- Report the measured minimum, mean, maximum, and standard deviation of total encode-plus-decode time.
- If the serial prompt is missing, an assertion appears, or the reported rate differs from the requested rate, stop and return the raw error.

Read [references/serial-bench.md](references/serial-bench.md) only when diagnosing prompt handling, known fingerprints, or serial output parsing.
