#!/usr/bin/env bash
# Use a separate reference SDK copy. The original BOX build entry is unchanged.
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: bash $0 /path/to/disposable-sdk-workspace tx|rx [jobs]" >&2
  exit 2
fi
workspace=$(cd "$1" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)
role=$2
jobs=${3:-4}
case "$role" in tx|rx) ;; *) echo "invalid role" >&2; exit 2 ;; esac
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "invalid jobs" >&2; exit 2; }
if [ ! -d "$workspace/nuttx" ] || [ ! -d "$workspace/apps" ]; then
  echo "workspace must contain nuttx/ and apps/" >&2
  exit 1
fi
if [ "$workspace" = "$repo" ]; then
  echo "use a separate SDK workspace" >&2
  exit 1
fi
for name in jixun-doc jixun-doc_duplex_1028 jixun-codec; do
  path="$workspace/apps/$name"
  if [ -e "$path" ] && [ ! -L "$path" ]; then
    echo "refusing to replace a real application directory: $path" >&2
    exit 1
  fi
done
# NuttX includes every top-level Make.defs, even in old backup directories.
# Reject historical registrations rather than compile another source snapshot.
for definition in "$workspace"/apps/*/Make.defs; do
  [ -f "$definition" ] || continue
  parent=$(basename "$(dirname "$definition")")
  case "$parent" in jixun-doc|jixun-doc_duplex_1028|jixun-codec) continue ;; esac
  if grep -Eq '^CONFIGURED_APPS.*[/]jixun-doc([[:space:]]|$)' "$definition"; then
    echo "legacy application registration: $definition" >&2
    echo "preserve this backup outside apps/ in the disposable SDK copy, then retry" >&2
    exit 1
  fi
done
bash "$repo/scripts/apply_aibox_patch.sh" "$workspace"
output=$(mktemp -d "$workspace/aibox-${role}.XXXXXX")
python3 "$repo/scripts/materialize_aibox.py" --role "$role" --output "$output/app"
for name in jixun-doc jixun-doc_duplex_1028 jixun-codec; do
  path="$workspace/apps/$name"
  if [ -L "$path" ]; then unlink "$path"; fi
done
ln -s "$output/app" "$workspace/apps/jixun-codec"
cp "$repo/board/esp32s3-aibox/configs/$role/defconfig" "$workspace/nuttx/.config"
tool="$workspace/prebuilts/gcc/linux-x86_64/xtensa-esp32s3-elf/bin"
if [ -x "$tool/xtensa-esp32s3-elf-gcc" ]; then export PATH="$tool:$PATH"; fi
if [ -d "$HOME/.local/bin" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
for required in xtensa-esp32s3-elf-gcc xtensa-esp32s3-elf-ar esptool.py; do
  command -v "$required" >/dev/null || { echo "missing build dependency: $required" >&2; exit 1; }
done
# The reference SDK may contain an archive from a previous application name.
# Remove those members before linking the selected role.
if [ -f "$workspace/apps/libapps.a" ]; then
  mapfile -t old < <(xtensa-esp32s3-elf-ar t "$workspace/apps/libapps.a" | sed -n '/jixun/p')
  if [ "${#old[@]}" -gt 0 ]; then
    xtensa-esp32s3-elf-ar d "$workspace/apps/libapps.a" "${old[@]}"
  fi
fi
if ! make -C "$workspace/nuttx" -j"$jobs" JX_RATE=3k \
  TOPDIR="$workspace/nuttx" APPDIR="$workspace/apps" >"$output/build.log" 2>&1; then
  tail -n 40 "$output/build.log" >&2
  echo "build failed; full log: $output/build.log" >&2
  exit 1
fi
cp "$workspace/nuttx/nuttx.bin" "$output/nuttx.bin"
cp "$workspace/nuttx/nuttx" "$output/nuttx"
cp "$workspace/nuttx/System.map" "$output/System.map"
cp "$workspace/nuttx/.config" "$output/config"
sha256sum "$output/nuttx.bin"
echo "role=$role output=$output"
