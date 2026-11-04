#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: bash $0 /path/to/disposable-sdk-workspace" >&2
  exit 2
fi
workspace=$(cd "$1" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)
nuttx="$workspace/nuttx"
base=fc3f07c0871f4bdaca0d19e48bd9d8b45fe2314b
patch="$repo/board/esp32s3-aibox/patches/nuttx-fc3f07c-aibox.patch"
if [ "$(git -C "$nuttx" rev-parse HEAD)" != "$base" ]; then
  echo "AiBox reference SDK base mismatch; no patch applied." >&2
  echo "This patch has NOT been rebased onto the official dd92bcf4 baseline." >&2
  exit 1
fi
if git -C "$nuttx" apply --reverse --check "$patch" >/dev/null 2>&1; then
  echo "AiBox reference patch already applied"
else
  git -C "$nuttx" apply --check "$patch"
  git -C "$nuttx" apply "$patch"
fi
