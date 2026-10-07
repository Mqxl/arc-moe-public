#!/usr/bin/env bash
# ./arc probe - a one-minute check of kernel launches on every GPU (tools/probe_launch.cpp): ordinary vs cooperative
# (use_root_sync), and a kernel reading pinned host memory. No model needed.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
oneapi_env
out=$ROOT/build-probe
mkdir -p "$out"
step "Building the probe"
icpx -fsycl -O2 "$ROOT/tools/probe_launch.cpp" -o "$out/probe_launch"
step "Running it on every Level Zero GPU"
unset ONEAPI_DEVICE_SELECTOR
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ZES_ENABLE_SYSMAN=1
exec "$out/probe_launch"
