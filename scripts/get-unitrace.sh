#!/usr/bin/env bash
# scripts/get-unitrace.sh - build Intel's unitrace (github.com/intel/pti-gpu) into third_party/pti-gpu: the GPU time of
# every kernel by name, which ./arc profile uses when it is there. Needs cmake, git and network: its CMake fetches the
# Level Zero headers it builds against itself (v1.32.0), so no libze-dev - which conflicts with the PPA's libze1 -
# is needed; at run time it loads the installed libze_loader.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
dir="$ROOT/third_party/pti-gpu"
bin="$dir/tools/unitrace/build/unitrace"
[ -x "$bin" ] && { ok "unitrace: $bin"; exit 0; }
command -v cmake >/dev/null || sudo_ apt-get install -y cmake >/dev/null
[ -d "$dir/.git" ] || git clone --depth 1 https://github.com/intel/pti-gpu "$dir"
mkdir -p "$dir/tools/unitrace/build"
cd "$dir/tools/unitrace/build"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=0 -DBUILD_WITH_ITT=0 -DBUILD_WITH_XPTI=0 -DBUILD_WITH_OPENCL=0 \
      -DBUILD_WITH_OMP=0 .. >/dev/null
make -j"$(nproc)" >/dev/null
[ -x "$bin" ] || die "unitrace did not build (see $dir/tools/unitrace/build)"
ok "unitrace: $bin"
