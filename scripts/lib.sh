# shellcheck shell=bash
# Shared helpers for the arc-moe scripts (sourced, not run).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STRATA=$ROOT/third_party/strata
# The Strata commit the patches apply to: 0.1.39 plus the two SYCL fixes the 2x Arc Pro B60 report ran on. Every
# later upstream commit (0.1.40, 0.1.40.1) no longer compiles the SYCL port (fused_gr_read_multi, Gemm::bf16,
# load_experts_gguf drifted from the migrated copies in sycl/).
STRATA_PIN=a5682fea1667c87b9a6b9ad0b2cd69dd9898fbaf
STRATA_BRANCH=arc-moe
# The AOT device(s) for the GPU code: ARC_AOT if set, else what the PCI ids of the Arcs in this machine need
# (detect_aot below; acm-g10 when none is found, e.g. building on another machine)
AOT_TARGET=${ARC_AOT-}
BUILD_DIR=$STRATA/build-sycl-aot
ONEAPI=${ONEAPI_ROOT:-/opt/intel/oneapi}
# The engine settings ./arc start adds, each measured on 2x A770 with ./arc profile (docs/ENGINEERING.md section 9):
#   --prefill-borrow --prefill auto   8K prompt chunks in borrowed cache slots: every chunk walks all of GPU 0's experts
#                                     (most stream from the RAM mirror), so few big chunks win (2K 49 -> 116 tok/s)
#   --vram-reserve-mib 768            ~550 more expert slots per card than setup's 1536 (512: decode +9%, but 432 MiB free)
#   --spec 6                          8-token windows (code accepts ~90% of the drafts): +4% on code
#   STRATA_GU_LANES=32                32 lanes per row in the experts' gate/up GEMV (8, tuned on Xe2): code +18%
#   STRATA_PREFILL_FIRST=0            no 256-token first chunk (a whole extra walk over the experts, ~4.7 s)
#   --short-read 768                  a prompt part of up to 768 new tokens (an agent's next turn: a tool result, a
#                                     message) goes through the decode windows (~25 ms a token) instead of a batched
#                                     chunk (~17 s however short, + ~4 ms a token: every expert dequantized); upstream's
#                                     64 assumes CUDA's ~0.3 s chunk. A 311-token turn: 18.7 s batched.
ARC_ENGINE_ARGS_DEFAULT="--prefill-borrow --prefill auto --vram-reserve-mib 768 --spec 6 --short-read 768"
arc_engine_env() {   # export the defaults above; a variable already set (even empty) wins
    export STRATA_EXTRA_ARGS="${STRATA_EXTRA_ARGS-$ARC_ENGINE_ARGS_DEFAULT}"
    export STRATA_PREFILL_FIRST="${STRATA_PREFILL_FIRST-0}"
    export STRATA_GU_LANES="${STRATA_GU_LANES-32}"
}

if [ -t 1 ]; then B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[2m'; N=$'\e[0m'; else B='' G='' Y='' R='' D='' N=''; fi
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
bad()  { printf '  %s✗%s %s\n' "$R" "$N" "$*"; }
die()  { printf '%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

oneapi_env() {
    # setvars.sh trips over `set -u` (OCL_ICD_FILENAMES: unbound variable): source it with -u off
    [ -n "${SETVARS_COMPLETED:-}" ] && return 0
    [ -f "$ONEAPI/setvars.sh" ] || die "oneAPI is not installed at $ONEAPI (./arc deps)"
    set +u
    # shellcheck disable=SC1091
    . "$ONEAPI/setvars.sh" --force >/dev/null 2>&1
    set -u
}

sudo_() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }

# PCI addresses of the Intel discrete GPUs (vendor 8086, class 0x03xxxx, not the iGPU at 00:02.0)
arc_bdfs() {
    local d
    for d in /sys/bus/pci/devices/*; do
        [ "$(cat "$d/vendor" 2>/dev/null)" = 0x8086 ] || continue
        case "$(cat "$d/class" 2>/dev/null)" in 0x03*) ;; *) continue ;; esac
        case "$(basename "$d")" in 0000:00:02.*) continue ;; esac
        basename "$d"
    done
}

# The Alchemist die of an Arc by PCI device id (ocloc's names): ACM-G10 = A770, A750, A580 and the A770M / A730M /
# A550M; ACM-G11 = A380, A310 and the small mobile parts; ACM-G12 = Arc Pro A60 / A60M. Battlemage (B580, B570, Pro
# B50/B60) is Strata's own SYCL target (bmg-g21) - arc-moe's patches are for Alchemist.
arc_die() {
    case "$1" in
        0x56a0|0x56a1|0x56a2|0x5690|0x5691|0x5692) echo acm-g10 ;;
        0x56a3|0x56a4|0x56a5|0x56a6|0x5693|0x5694|0x5695|0x5696|0x5697|0x56b0|0x56b1|0x56ba|0x56bb|0x56bc|0x56bd) echo acm-g11 ;;
        0x56b2|0x56b3) echo acm-g12 ;;
        0xe20*|0xe21*) echo bmg-g21 ;;
        *) echo unknown ;;
    esac
}
detect_aot() {   # every die among the Arcs in this machine, comma separated (acm-g10 when there is none)
    local b dies="" d
    for b in $(arc_bdfs); do
        d=$(arc_die "$(cat "/sys/bus/pci/devices/$b/device" 2>/dev/null)")
        [ "$d" = unknown ] && continue
        case ",$dies," in *",$d,"*) ;; *) dies=${dies:+$dies,}$d ;; esac
    done
    echo "${dies:-acm-g10}"
}
[ -n "$AOT_TARGET" ] || AOT_TARGET=$(detect_aot)
