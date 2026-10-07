#!/usr/bin/env bash
# ./arc start [model] - start the configured server (the run-<model>.sh setup wrote). With several models
# configured: the one named, else $ARC_MODEL, else IQ2_XS, else the newest.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
"$ROOT/scripts/apply-patches.sh" >/dev/null   # a git pull's patch updates (the runner, setup_intel.py) take effect
cd "$STRATA"
want=${1:-${ARC_MODEL:-}}
want=$(echo "$want" | tr '[:upper:]' '[:lower:]')
mapfile -t runs < <(ls -t run-*.sh 2>/dev/null)
[ ${#runs[@]} -gt 0 ] || die "no model configured yet: ./arc setup"
pick=""
for r in "${runs[@]}"; do [ -n "$want" ] && [ "$r" = "run-$want.sh" ] && pick=$r; done
[ -z "$pick" ] && [ -n "$want" ] && die "no run-$want.sh (have: ${runs[*]})"
[ -z "$pick" ] && for r in "${runs[@]}"; do [ "$r" = run-iq2_xs.sh ] && pick=$r; done
[ -z "$pick" ] && pick=${runs[0]}
cfg="strata-${pick#run-}"; cfg="${cfg%.sh}.json"
port=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("port", 8080))' "$cfg" 2>/dev/null || echo 8080)
say "${B}Strata on Intel Arc:${N} ${pick#run-} -> http://127.0.0.1:${port}  (OpenAI: /v1, Anthropic: /v1/messages)"
say "${D}The first start fills the experts into VRAM and pins the overflow in RAM: ~1 minute.${N}"
export STRATA_SYCL_NATIVE=1
# arc-moe defaults, measured on 2x A770 (./arc profile, docs/ENGINEERING.md section 9). Every prompt chunk walks all of
# GPU 0's experts (most stream from the RAM mirror), so few big chunks win: the prompt path borrows expert-cache slots
# for 8K-token chunks (2K prompt 49 -> 116 tok/s, 8K 63 -> 284, decode unchanged), and the 256-token first chunk -
# a whole extra walk, ~4.7 s here - is off. Set either variable (even empty) to override.
export STRATA_EXTRA_ARGS="${STRATA_EXTRA_ARGS---prefill-borrow --prefill auto}"
export STRATA_PREFILL_FIRST="${STRATA_PREFILL_FIRST-0}"
exec "./$pick"
