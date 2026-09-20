#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 /path/to/openvela-workspace [1k|3k|6k] [jobs]" >&2
  exit 2
fi

workspace=$(cd "$1" && pwd)
rate=${2:-3k}
jobs=${3:-4}
nuttx="$workspace/nuttx"
app="$workspace/apps/jixun-codec"
log="${workspace}/build_jixun_${rate}.log"

case "$rate" in
  1k|3k|6k) ;;
  *) echo "invalid rate: $rate" >&2; exit 2 ;;
esac

if [ ! -d "$nuttx" ] || [ ! -d "$app" ]; then
  echo "workspace is missing nuttx/ or apps/jixun-codec; run repo sync first" >&2
  exit 1
fi

echo "[build] rate=$rate jobs=$jobs"
echo "[build] log=$log"

find "$app" -name '*.o' -delete 2>/dev/null || true
rm -f "$app/.built" "$nuttx/staging/libc.a" "$nuttx/staging/libapps.a"

make -C "$nuttx" -j"$jobs" JX_RATE="$rate" >"$log" 2>&1
rm -f "$nuttx/staging/libc.a" "$nuttx/staging/libapps.a" "$workspace/apps/builtin/builtin_list"*.o
touch "$workspace/apps/builtin/builtin_list.c"
make -C "$nuttx" -j"$jobs" JX_RATE="$rate" >>"$log" 2>&1

if [ ! -f "$nuttx/nuttx.bin" ]; then
  echo "build failed: nuttx.bin not found" >&2
  exit 1
fi

sha256sum "$nuttx/nuttx.bin"
