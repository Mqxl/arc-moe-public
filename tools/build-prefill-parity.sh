#!/usr/bin/env bash
# Build the focused numeric parity test with the current patched kernel library.
set -euo pipefail
. "$(dirname "$0")/../scripts/lib.sh"
oneapi_env
mkdir -p "$ROOT/test-logs"
target=()
if [ -n "$AOT_TARGET" ]; then
    target=(-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
            "-device $AOT_TARGET -options \"-cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required -ze-opt-large-register-file\"")
fi
icpx -O2 -std=c++17 -fsycl -fsycl-device-code-split=per_kernel "${target[@]}" \
    -I"$STRATA/sycl/include" -I"$STRATA/include" -I"$STRATA/third_party/ggml" \
    -I"$BUILD_DIR/_deps/strata_llamacpp-src/ggml/include" \
    "${ARC_PARITY_SOURCE:-$ROOT/tools/prefill_parity.cpp}" -o "${ARC_PARITY_OUTPUT:-$ROOT/test-logs/prefill-parity}" \
    "${ARC_PARITY_KERNEL_OBJECT:-$BUILD_DIR/libstrata_kernels.a}" "$BUILD_DIR/ggml/src/libggml-base.a"
