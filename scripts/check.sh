#!/usr/bin/env bash
# ./arc check [model] [--no-long] - does the server answer with text the way every kind of client asks? tools/check.py sends
# one short question greedy and sampled, thinking off and on, streamed, with penalties and with tools.
#
# A server already running: checked as it is (a few minutes). None running: started once per configuration below with
# STRATA_DBG_NAN=1, each checked, everything written to check-<time>.txt (~15 minutes, each start reloads the model):
#   base          ./arc start's defaults
#   old-layout    STRATA_CACHE_TAIL_LOW=0     only with ARC_CHECK_OLD_LAYOUT=1: the prompt path's borrowed buffers
#                                             7-11 GB into the expert cache, as before patch 27 - a control that MUST
#                                             break the 10K rows (it did on 2026-10-07 and 10-08: the cause confirmed)
#   own-buffers   --no-prefill-borrow         the prompt path in its own allocations (smaller chunks, fewer experts)
#   one-gpu       ARC_GPUS=0                  on a machine with two Arcs: no layer split
set -euo pipefail
. "$(dirname "$0")/lib.sh"
cd "$STRATA"
pos=(); pass=()   # --no-long and the other tools/check.py options go through to it
for a in "$@"; do case "$a" in --*) pass+=("$a") ;; *) pos+=("$a") ;; esac; done
want=${pos[0]:-${ARC_MODEL:-iq2_xs}}
want=$(echo "$want" | tr '[:upper:]' '[:lower:]')
cfg="strata-$want.json"
[ -f "$cfg" ] || die "no $cfg: ./arc setup first (or ./arc check <model>)"
read -r port log < <(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("port", 8080), c.get("log") or "-")' "$cfg")
url="http://127.0.0.1:$port"

if curl -fsS -m 3 "$url/health" >/dev/null 2>&1; then
    say "${B}checking the running server${N} at $url (stop it for the full run over several configurations)"
    exec python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/check.py" --url "$url"  "${pass[@]}"
fi

[ "$log" != "-" ] || die "$cfg has no \"log\": the engine's lines are needed"
case "$log" in /*) ;; *) log="$STRATA/$log" ;; esac
arc_engine_env
out="$ROOT/check-$(date +%Y%m%d-%H%M%S).txt"

srv_pid=""
failures=0
stop_server() {
    [ -n "$srv_pid" ] || return 0
    local py
    py=$(pgrep -g "$srv_pid" -f "serve/server(_intel)?\.py" | head -1 || true)
    [ -n "$py" ] && kill -TERM "$py" 2>/dev/null || true
    for _ in $(seq 1 60); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -TERM -- "-$srv_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -KILL -- "-$srv_pid" 2>/dev/null || true
    srv_pid=""
    sleep 3
}
trap stop_server EXIT

keep='strata prefill|prompt chunk|borrows|own buffers|strata dbg|head sampling|sampl|NaN|non-finite|GPU HANG|rror|xception|terminate|what\(\)|abort|egmentation|DEVICE_LOST|split|extra engine'

run() {   # run <name> <env...>
    local name=$1; shift
    say "${B}check $name${N}: $*"
    local since=0
    [ -f "$log" ] && since=$(stat -c %s "$log")
    env "$@" STRATA_DBG_NAN=1 setsid "$ROOT/scripts/start.sh" "$want" >"$out.$name.start" 2>&1 &
    srv_pid=$!
    local t=0
    until curl -fsS -m 3 "$url/health" 2>/dev/null | grep -q '"loaded": *true'; do
        sleep 5; t=$((t + 5))
        kill -0 "$srv_pid" 2>/dev/null || { warn "the server stopped while loading (see $out.$name.start)"; break; }
        [ $t -lt 900 ] || { warn "not loaded after 15 minutes"; break; }
    done
    echo "================ $name: $*" >>"$out"
    if python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/check.py" --url "$url"  "${pass[@]}" 2>&1 | tee -a "$out"; then
        :
    else
        failures=$((failures + 1))
    fi
    stop_server
    {
        echo "---- engine log"
        [ -f "$log" ] && tail -c +$((since + 1)) "$log" | grep -E "$keep" | sort | uniq -c | sort -rn | head -40 | cut -c1-600 || true
        tail -n 30 "$out.$name.start" | grep -E "$keep" | cut -c1-600 || true
        echo
    } >>"$out"
}

{
    echo "arc-moe check, $(date), $(git -C "$ROOT" rev-parse --short HEAD), model $want, $(arc_bdfs | wc -l) Arc(s)"
    echo "engine source: $(git -C "$STRATA" rev-parse --short HEAD); patch fingerprint: $(git -C "$STRATA" config --get arcmoe.applied)"
    for knob in STRATA_PF_EXPERT_KERNEL STRATA_PF_XMX_GRF STRATA_PF_XMX_MIN_ROWS STRATA_PF_STAGE_BANKS STRATA_PF_STAGE_SLOTS STRATA_PF_GROUPS STRATA_PF_BATCH_ROWS STRATA_PCIE_PINNED_PROBE STRATA_PF_PIN_LOAN STRATA_PF_PIN_REFILL STRATA_PF_PREFETCH_SLOTS STRATA_PF_PIPELINE STRATA_PROMPT_ATTN_XMX STRATA_PF_SPLIT_ROWS STRATA_PF_FUSE_EXPERT STRATA_PF_EP STRATA_VERIFY_COMMIT_OVERLAP STRATA_MTP_CHAIN_SUBMIT STRATA_VERIFY_SKIP_EMPTY STRATA_VERIFY_NATIVE_GRAPH STRATA_MTP_LOOKAHEAD STRATA_NATIVE_UPLOAD_PINNED STRATA_DENSE_UPLOAD_KERNEL STRATA_UPLOAD_VERIFY STRATA_UPLOAD_RELOCATE STRATA_DBG_PF_PHASES STRATA_DBG_WEIGHT_COPY; do
        [ -z "${!knob-}" ] || echo "$knob=${!knob}"
    done
    echo
} >"$out"
BASE_ARGS="${STRATA_EXTRA_ARGS:-$ARC_ENGINE_ARGS_DEFAULT}"
run base ARC_CHECK_RUN=base
if [ "${ARC_CHECK_OLD_LAYOUT:-0}" = 1 ]; then run old-layout STRATA_CACHE_TAIL_LOW=0; fi
run own-buffers "STRATA_EXTRA_ARGS=$BASE_ARGS --no-prefill-borrow"
if [ "$(arc_bdfs | wc -l)" -ge 2 ]; then run one-gpu ARC_GPUS=0; fi
trap - EXIT
rm -f "$out".*.start
if [ "$failures" -gt 0 ]; then
    warn "$failures configuration(s) failed: $out"
    exit 3
fi
ok "written: $out (send this file)"
