#!/usr/bin/env bash
# ./arc setup [setup.py options] - the model, its image encoder and the server config, through Strata's own installer
# (sycl/setup_intel.py, patched for a native run on every Arc in the box). Defaults for 2x A770:
#
#   --model IQ2_XS        Qwen3.8-Flash-Next, every one of its 24,576 experts (not the half-pruned Coder)
#   --context 32768       32K tokens of context (KV int8). 65536+ streams the KV from RAM
#   --vision cpu          pictures through the CPU image encoder: both cards' VRAM stays for experts
#   --gpus 0,1            the layers split across both cards (added by setup_intel.py when it sees two Arcs)
#   --data-dir ./data     where the GGUF files go (~70 GB for IQ2_XS); --gguf-dir DIR adopts files you already have
#
# Anything you pass replaces the default of the same name, e.g. ./arc setup --model IQ3_XXS --context 65536
# ARC_GGUF_DIR=/mnt/ssd/qwen ./arc setup   is the same as --gguf-dir /mnt/ssd/qwen
set -euo pipefail
. "$(dirname "$0")/lib.sh"
[ -x "$BUILD_DIR/strata" ] || die "build the engine first: ./arc build"
"$ROOT/scripts/apply-patches.sh" >/dev/null

has() { local f=$1; shift; for a in "$@"; do [ "$a" = "$f" ] && return 0; done; return 1; }
args=("$@")
has --model "$@"   || args+=(--model "${ARC_MODEL:-IQ2_XS}")
has --context "$@" || args+=(--context "${ARC_CONTEXT:-32768}")
has --vision "$@"  || args+=(--vision cpu)
has --data-dir "$@" || has --models-dir "$@" || args+=(--data-dir "${ARC_DATA_DIR:-$ROOT/data}")
if ! has --gguf-dir "$@" && [ -n "${ARC_GGUF_DIR:-}" ]; then args+=(--gguf-dir "$ARC_GGUF_DIR"); fi
mkdir -p "${ARC_DATA_DIR:-$ROOT/data}"

step "Python environment"
cd "$STRATA"
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then rm -rf .venv; fi
[ -x .venv/bin/python ] || python3 -m venv .venv || die "python3 -m venv failed (sudo apt install python3-venv)"
ok "$STRATA/.venv"

step "Strata setup (Intel Arc, native): ${args[*]}"
export STRATA_SYCL_NATIVE=1 STRATA_SYCL_BIN="$BUILD_DIR/strata"
exec .venv/bin/python sycl/setup_intel.py "${args[@]}"
