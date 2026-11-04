#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo 'Usage: build_k7.sh /independent/sdk master|slave [cache] [jobs=2]' >&2
  exit 2
fi
workspace=$(cd -- "$1" && pwd)
role=$2
cache=${3:-"$workspace/downloads"}
jobs=${4:-2}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ "$role" == master || "$role" == slave ]] || exit 2
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || exit 2
toolchain=${K7_TOOLCHAIN_DIR:-"$workspace/prebuilts/gcc/linux-x86_64/aarch64-none-elf/bin"}
hosttools=${K7_HOSTTOOLS_DIR:-"$(dirname -- "$workspace")/hosttools"}
export PATH="$hosttools/bin:$toolchain:$workspace/prebuilts/build-tools/linux-x86_64/bin:$workspace/prebuilts/cmake/linux-x86_64/bin:$hosttools/pack-bin:$PATH"
export LD_LIBRARY_PATH="$hosttools/pack-lib:${LD_LIBRARY_PATH:-}"
export LC_ALL=C LANG=C JX_OFFLINE_SDK=1 CCACHE_DISABLE=1
for command in python3 make git patch unzip genromfs olddefconfig aarch64-none-elf-gcc; do
  command -v "$command" >/dev/null || { echo "Missing host tool: $command" >&2; exit 1; }
done
version=$(aarch64-none-elf-gcc -dumpfullversion)
[[ "$version" == 13.4.0 ]] || { echo "Expected verified GCC 13.4.0; found $version" >&2; exit 1; }
python3 "$repo/scripts/prepare_k7_workspace.py" "$workspace" --role "$role" --cache "$cache" --apply
cd "$workspace"
./build.sh configs/jixun_k7_ui -j"$jobs"
python3 - "$repo" "$workspace" <<'PY'
import hashlib,json,sys
from pathlib import Path
repo=Path(sys.argv[1]);work=Path(sys.argv[2]);profile=work/'configs/jixun_k7_ui'
data={'source_defconfig_sha256':hashlib.sha256((repo/'board/kickpi-k7/configs/openvela/defconfig').read_bytes()).hexdigest(),'profile_sha256':hashlib.sha256((profile/'defconfig').read_bytes()).hexdigest()}
(profile/'.jixun-profile-owned.json').write_text(json.dumps(data))
PY
out="$repo/deliverables/kickpi-k7/rebuilt/$role"
mkdir -p "$out"
cp nuttx/nuttx nuttx/nuttx.bin nuttx/.config "$out/"
cp contest2026_062_PharosTech/boards/rk3576/kickpi-k7/src/etc/init.d/rcS "$out/rcS"
sha256sum "$out/nuttx.bin"
