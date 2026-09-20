#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /path/to/openvela-workspace" >&2
  exit 2
fi

workspace=$(cd "$1" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)
patch_file="$repo/board/esp32s3-box/patches/nuttx-dev-ai-contest-2026-jixun.patch"
esp_hal_patch="$repo/board/esp32s3-box/patches/esp-hal-3rdparty-jixun.patch"
defconfig="$repo/board/esp32s3-box/configs/openvela/defconfig"
dest="$workspace/nuttx/boards/xtensa/esp32s3/esp32s3-box/configs/openvela/defconfig"

if [ ! -d "$workspace/nuttx" ]; then
  echo "nuttx not found under workspace: $workspace" >&2
  exit 1
fi

if git -C "$workspace/nuttx" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  echo "[board] NuttX patch already applied"
else
  git -C "$workspace/nuttx" apply --check --whitespace=nowarn "$patch_file"
  git -C "$workspace/nuttx" apply --whitespace=nowarn "$patch_file"
  echo "[board] applied $patch_file"
fi

esp_hal_applied=0
for esp_hal in \
  "$workspace/nuttx/arch/xtensa/src/esp32s3/esp-hal-3rdparty" \
  "$workspace/nxtmpdir/esp-hal-3rdparty"; do
  if [ ! -d "$esp_hal/.git" ]; then
    continue
  fi

  if git -C "$esp_hal" apply --reverse --check "$esp_hal_patch" >/dev/null 2>&1; then
    echo "[board] ESP HAL patch already applied: $esp_hal"
  else
    git -C "$esp_hal" apply --check --whitespace=nowarn "$esp_hal_patch"
    git -C "$esp_hal" apply --whitespace=nowarn "$esp_hal_patch"
    echo "[board] applied ESP HAL patch: $esp_hal"
  fi
  esp_hal_applied=1
done

if [ "$esp_hal_applied" -eq 0 ]; then
  echo "[board] warning: no ESP HAL git worktree found; skipped compile-fix patch" >&2
fi

mkdir -p "$(dirname "$dest")"
cp "$defconfig" "$dest"
echo "[board] installed $dest"
echo "[board] done"
