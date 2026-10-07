#!/usr/bin/env bash
# scripts/get-unitrace.sh - build Intel's unitrace (github.com/intel/pti-gpu) into third_party/pti-gpu: the GPU time of
# every kernel by name, which ./arc profile uses when it is there. Needs the Level Zero headers (apt, sudo) and cmake.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
dir="$ROOT/third_party/pti-gpu"
bin="$dir/tools/unitrace/build/unitrace"
[ -x "$bin" ] && { ok "unitrace: $bin"; exit 0; }
if [ ! -f /usr/include/level_zero/ze_api.h ]; then
    say "Level Zero headers (libze-dev) ..."
    sudo_ apt-get install -y libze-dev >/dev/null 2>&1 || sudo_ apt-get install -y level-zero-dev >/dev/null 2>&1 ||
        die "no Level Zero headers: sudo apt install libze-dev (or level-zero-dev from Intel's repository)"
fi
command -v cmake >/dev/null || sudo_ apt-get install -y cmake >/dev/null
[ -d "$dir/.git" ] || git clone --depth 1 https://github.com/intel/pti-gpu "$dir"
mkdir -p "$dir/tools/unitrace/build"
cd "$dir/tools/unitrace/build"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=0 -DBUILD_WITH_ITT=0 -DBUILD_WITH_XPTI=0 -DBUILD_WITH_OPENCL=0 \
      -DBUILD_WITH_OMP=0 .. >/dev/null
make -j"$(nproc)" >/dev/null
[ -x "$bin" ] || die "unitrace did not build (see $dir/tools/unitrace/build)"
ok "unitrace: $bin"
