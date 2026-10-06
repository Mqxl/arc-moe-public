#!/usr/bin/env bash
# ./arc check-fp64 - compile every SYCL translation unit to SPIR-V and list any device function still computing in
# double (tools/check_fp64.py). The Arc A-series has no FP64 unit, so the answer must be "0 with FP64".
set -euo pipefail
. "$(dirname "$0")/lib.sh"
"$ROOT/scripts/apply-patches.sh" >/dev/null
oneapi_env
b=$STRATA/build-fp64-check
cmake -S "$STRATA/sycl" -B "$b" -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSTRATA_SYCL_PARITY=OFF -DSTRATA_SYCL_NO_FP64="${1:-ON}" >/dev/null
exec python3 "$ROOT/tools/check_fp64.py" "$b"
