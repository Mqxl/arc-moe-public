#!/usr/bin/env bash
# ./arc build - Strata at the pinned commit + the arc-moe patches, compiled ahead of time for the Arc A770.
#   ARC_AOT=acm-g10        the device to compile for (A770/A750; acm-g12 = A580, acm-g11 = A380/A310; "" = SPIR-V JIT)
#   ARC_JOBS=N             parallel compile jobs (default: CPUs; the AOT link needs ~2 GB of RAM per job)
#   ARC_PARITY=ON          also build the kernel tests (ctest --test-dir third_party/strata/build-sycl-aot)
#   ARC_IGC_EXTRA="..."    extra IGC options, e.g. -ze-opt-large-register-file (experimental on DG2)
#   --clean                start from an empty build directory
set -euo pipefail
. "$(dirname "$0")/lib.sh"
clean=0; [ "${1:-}" = --clean ] && clean=1

step "Strata source"
# only when it is not checked out yet: an update would detach it from the patched branch and every source would
# be rewritten (a full 30-60 min AOT rebuild)
[ -e "$STRATA/.git" ] || git -C "$ROOT" submodule update --init third_party/strata
[ -e "$STRATA/.git" ] || die "third_party/strata is missing (git clone --recursive)"
"$ROOT/scripts/apply-patches.sh"

step "Compiling the engine for $AOT_TARGET (Arc A770) - 20-60 minutes the first time"
oneapi_env
command -v icpx >/dev/null || die "icpx not found after setvars.sh (./arc deps)"
if [ -n "$AOT_TARGET" ] && ! command -v ocloc >/dev/null; then
    die "ocloc is needed for the AOT build (./arc deps, or ARC_AOT= for a JIT build)"
fi
[ $clean -eq 1 ] && rm -rf "$BUILD_DIR"
jobs=${ARC_JOBS:-$(nproc)}
mem_gb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1048576 ))
[ "$jobs" -gt $(( mem_gb / 2 )) ] && jobs=$(( mem_gb / 2 > 0 ? mem_gb / 2 : 1 ))
cmake -S "$STRATA/sycl" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_SYCL_AOT="$AOT_TARGET" -DSTRATA_SYCL_NO_FP64=ON \
    -DSTRATA_SYCL_PARITY="${ARC_PARITY:-OFF}" -DSTRATA_SYCL_IGC_EXTRA="${ARC_IGC_EXTRA:-}" >/dev/null
t0=$(date +%s)
set +eo pipefail
cmake --build "$BUILD_DIR" --target strata -j "$jobs" 2>&1 | tee "$BUILD_DIR/build.log" \
    | grep -E --line-buffered '^\[[0-9]+/[0-9]+\]|error:|FAILED' | sed -u 's/^/  /'
rc=${PIPESTATUS[0]}
set -eo pipefail
[ "$rc" -eq 0 ] && [ -x "$BUILD_DIR/strata" ] || die "the build failed: $BUILD_DIR/build.log"
if [ "${ARC_PARITY:-OFF}" = ON ]; then cmake --build "$BUILD_DIR" -j "$jobs" >/dev/null; fi
ok "engine: $BUILD_DIR/strata ($(( ($(date +%s) - t0) / 60 )) min)"
say "  Next: ./arc setup"
