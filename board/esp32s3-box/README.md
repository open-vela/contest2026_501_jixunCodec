# ESP32-S3-BOX board overlay

This directory contains the NuttX changes required by the Jixun Codec application.

## Files

- `patches/nuttx-dev-ai-contest-2026-jixun.patch`: changes rebased on the competition branch `dev-ai-contest-2026` (baseline commit `dd92bcf4`).
- `patches/esp-hal-3rdparty-jixun.patch`: two spinlock-initializer compile fixes in ESP HAL.
- `configs/openvela/defconfig`: the working ESP32-S3-BOX openvela configuration. `CONFIG_PLANT_AI_API_KEY` is intentionally empty in the submitted copy.

## Apply

```bash
cd /path/to/contest2026_501_jixunCodec
./scripts/apply_board_patch.sh /path/to/openvela-workspace
```

The script applies the patch under the workspace `nuttx/` project and installs the defconfig. It does not write to the public upstream repositories.

It also applies the ESP HAL compile fix to the first available worktree:

- `nuttx/arch/xtensa/src/esp32s3/esp-hal-3rdparty`
- `nxtmpdir/esp-hal-3rdparty`

The large mbedtls namespace patch is not copied into this repository. NuttX regenerates it by applying `nuttx/patches/components/mbedtls/mbedtls/*.patch` during the build.

## Important fixes

- ESP32-S3 memory layout and PSRAM access changes for stable large-model execution.
- Wi-Fi, I2S, camera and SD-related fixes used by the board bring-up.
- ST7796 LCD initialization and SPI CS keep-active handling.
- LCD and GT911 share GPIO48 as reset on ESP32-S3-BOX. Touch is initialized first and LCD last, preventing the already initialized panel from being reset back to sleep mode.

If a public openvela repository is updated, re-run `git apply --check` before applying. A conflict should be reviewed manually rather than forced.
