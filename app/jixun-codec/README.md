# jixun-codec application

This directory is linked to `apps/jixun-codec` by the contest manifest.

## Layout

- `csrc/`: neural codec, FSQ quantizer, transformer and INT8/PIE kernels.
- `include/`: codec, UI, environment and math headers.
- `main/`: NSH command entry and embedded PCM/TTS test clips.
- `port/`: NuttX adapters for PCM, USB/Wi-Fi token transport and the LCD UI.
- `weights/`: `1k`, `3k` and `6k` model parameters.
- `Makefile` / `Make.defs`: build integration and three-rate selection.

## Build rate

`JX_RATE=1k`, `JX_RATE=3k` or `JX_RATE=6k` selects the model at build time. The board UI reports the actual measured token rate as `tokens/s`.

The application expects the board-side audio helpers from the openvela plant-companion packages and the board changes under `board/esp32s3-box`.
