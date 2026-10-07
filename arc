#!/usr/bin/env bash
# arc-moe: Qwen3.8-Flash-Next (Strata engine, SYCL) on Intel Arc A-series - one A770 or two.
#
#   ./arc doctor            check the machine: GPUs, driver, ReBAR, PCIe, oneAPI, RAM, disk
#   ./arc deps              install the GPU runtime, oneAPI (DPC++ + oneMKL), ocloc, build tools   (sudo)
#   ./arc build             fetch Strata, apply the A770 patches, compile the engine AOT for this machine's Arc (acm-g10: A770)
#   ./arc setup [opts]      download / adopt the model, image encoder, config; then start      (setup.py options)
#   ./arc start             start the server (http://127.0.0.1:8080, OpenAI + Anthropic APIs);
#                           every Arc by default, ARC_GPUS=0 for one card
#   ./arc bench [opts]      measure prompt and decode speed through the API
#   ./arc profile           where the time goes: three profiled runs into one file to send       (~10 min)
#   ./arc check             does every kind of client get text (greedy, sampled, stream, tools)? (~1 / ~10 min)
#   ./arc service           install a systemd unit that starts the server at boot              (sudo)
#   ./arc check-fp64        prove the device code has no FP64 left (no GPU needed)
#   ./arc probe             one-minute check of kernel launches on every GPU (no model needed)
#   ./arc all               deps + build + setup: everything, from a fresh Ubuntu
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cmd=${1:-help}
shift || true
case "$cmd" in
  doctor)     exec "$ROOT/scripts/doctor.sh" "$@" ;;
  deps)       exec "$ROOT/scripts/install-deps.sh" "$@" ;;
  build)      exec "$ROOT/scripts/build.sh" "$@" ;;
  setup)      exec "$ROOT/scripts/setup-model.sh" "$@" ;;
  start)      exec "$ROOT/scripts/start.sh" "$@" ;;
  bench)      exec "$ROOT/scripts/bench.sh" "$@" ;;
  profile)    exec "$ROOT/scripts/profile.sh" "$@" ;;
  check)      exec bash "$ROOT/scripts/check.sh" "$@" ;;
  service)    exec "$ROOT/scripts/install-service.sh" "$@" ;;
  check-fp64) exec "$ROOT/scripts/check-fp64.sh" "$@" ;;
  probe)      exec "$ROOT/scripts/probe.sh" "$@" ;;
  all)        "$ROOT/scripts/install-deps.sh" && "$ROOT/scripts/build.sh" && exec "$ROOT/scripts/setup-model.sh" "$@" ;;
  help|-h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown command: $cmd (./arc help)" >&2; exit 2 ;;
esac
