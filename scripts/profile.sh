#!/usr/bin/env bash
# ./arc profile [model] [--no-unitrace] - where the time goes: starts the configured server several times with the
# engine's profilers on, sends prompts, stops it, and writes everything worth reading into one file
# (profile-<time>.txt).
#
#   N     the defaults (./arc start's: lib.sh arc_engine_env), nothing traced: the real speeds on short prompts, on
#         2K / 10K prompts and on a sampled request and an agent harness's (tools, streaming); every reply checked
#   PS1   STRATA_PLAN_STAGE=1: the device plan stages mirrored experts in VRAM (patch 21; off by default since round 9)
#   DQ32  STRATA_DQ_WG=32: the prompt path's dequantization in one-superblock work-groups (8 per group by default)
#   SR64  --short-read 64: short prompt parts batched (768 by default: up to 768 new tokens go through the windows)
#   G1    one card only (ARC_GPUS=0), on a machine with two: the single-GPU mode
#   A   the defaults with the prompt path's phases (STRATA_PREFILL_TIMING) and the draft layer's (STRATA_DRAFT_TIMING)
#   D   decode, eager, under unitrace: the GPU time of every kernel by name   (ARC_PROFILE_TRACE=1)
#   E   the 10K prompt under unitrace                                          (ARC_PROFILE_TRACE=1)
#   ARC_PROFILE_SWEEP=1 adds round 4's kernel-variant runs (K*: decode, G*: the GDN recurrence)
#
# D and E need unitrace (Intel's pti-gpu, built once by scripts/get-unitrace.sh: cmake + network, ~5 min).
# The server must be stopped first (the run starts its own). About 25 minutes (each run reloads the model).
set -euo pipefail
. "$(dirname "$0")/lib.sh"
use_unitrace=1
pos=()
for a in "$@"; do case "$a" in --no-unitrace) use_unitrace=0 ;; *) pos+=("$a") ;; esac; done
cd "$STRATA"
want=${pos[0]:-${ARC_MODEL:-iq2_xs}}
want=$(echo "$want" | tr '[:upper:]' '[:lower:]')
cfg="strata-$want.json"
[ -f "$cfg" ] || die "no $cfg: ./arc setup first (or ./arc profile <model>)"
read -r port key log < <(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("port", 8080), c.get("api_key") or "-", c.get("log") or "-")' "$cfg")
url="http://127.0.0.1:$port"
if curl -fsS -m 3 "$url/health" >/dev/null 2>&1; then
    die "a server already answers on $url: stop it first (the profile starts its own)"
fi
[ "$log" != "-" ] || die "$cfg has no \"log\": the engine's lines are needed"
case "$log" in /*) ;; *) log="$STRATA/$log" ;; esac
out="$ROOT/profile-$(date +%Y%m%d-%H%M%S).txt"
keyarg=(); [ "$key" != "-" ] && keyarg=(--api-key "$key")

unitrace=""
if [ $use_unitrace = 1 ] && [ "${ARC_PROFILE_TRACE:-0}" = 1 ]; then
    if "$ROOT/scripts/get-unitrace.sh"; then unitrace="$ROOT/third_party/pti-gpu/tools/unitrace/build/unitrace"
    else warn "no unitrace: runs D and E are skipped (./arc profile --no-unitrace to not try)"; fi
fi

srv_pid=""
stop_server() {   # Ctrl+C to the server only: it sends the engine QUIT, so a traced engine writes its summary
    [ -n "$srv_pid" ] || return 0
    local py
    py=$(pgrep -g "$srv_pid" -f "server.py" | head -1 || true)
    [ -n "$py" ] && kill -INT "$py" 2>/dev/null || true
    for _ in $(seq 1 90); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -TERM -- "-$srv_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -KILL -- "-$srv_pid" 2>/dev/null || true
    srv_pid=""
    sleep 3   # the GPUs' memory is released after the process is gone
}
trap stop_server EXIT

keep='prefill timing|prompt chunk|prompt path|borrow|serve: prompt|verify stages|window: since|split|mirror|expert cache'
keep+='|strata dbg|head sampling|draft kv|VRAM free|NO_HOST|GPU HANG|rror|xception|terminate|what\(\)|abort|egmentation|DEVICE_LOST|extra engine|engine under'

run() {   # run <name> <cases> <max-tokens> <env...>: start, wait for the model, bench, stop, keep the engine's lines
    local name=$1 cases=$2 maxt=$3; shift 3
    say "${B}profile $name${N}: $*"
    local since=0
    [ -f "$log" ] && since=$(stat -c %s "$log")
    env "$@" STRATA_SYCL_NATIVE=1 setsid "$ROOT/scripts/start.sh" "$want" >"$out.$name.start" 2>&1 &
    srv_pid=$!
    local t=0
    until curl -fsS -m 3 "$url/health" 2>/dev/null | grep -q '"loaded": *true'; do
        sleep 5; t=$((t + 5))
        kill -0 "$srv_pid" 2>/dev/null || { warn "the server stopped while loading (see $out.$name.start)"; break; }
        [ $t -lt 900 ] || { warn "not loaded after 15 minutes"; break; }
    done
    echo "================ $name: $*" >>"$out"
    python3 "$ROOT/tools/bench.py" --url "$url" "${keyarg[@]}" --runs "${ARC_PROFILE_RUNS:-2}" --max-tokens "$maxt" --cases "$cases" >>"$out" 2>&1 || true
    stop_server
    {
        echo "---- engine log"
        [ -f "$log" ] && tail -c +$((since + 1)) "$log" | grep -E "$keep" | cut -c1-2000 || true
        echo
    } >>"$out"
}

{
    echo "arc-moe profile, $(date), $(git -C "$ROOT" rev-parse --short HEAD), model $want"
    for b in $(arc_bdfs); do
        p=$(readlink -f "/sys/bus/pci/devices/$b")
        while [ -n "$p" ] && [ "$p" != /sys/devices ]; do   # the card's own link is reported by its upstream bridge
            [ -f "$p/current_link_width" ] && echo "$b via $(basename "$p"): $(cat "$p/current_link_speed") x$(cat "$p/current_link_width")"
            p=$(dirname "$p")
        done | tail -1
    done
    free -g | head -2
    echo
} >"$out"

run N short,prose,1k,2k,10k,sampled,agent 128 ARC_PROFILE_RUN=N
if [ "${ARC_PROFILE_SWEEP:-0}" = 1 ]; then
    run K-sg16 short,prose 128 STRATA_MMVQ_SG=16
    run K-sg3 short,prose 128 STRATA_MMVQ_SG=3
    run K-rpw2 short,prose 128 STRATA_MMVQ_RPW=2
    run K-grv3 short,prose 128 STRATA_GR_V3=1
    run K-sliced0 short,prose 128 STRATA_GR_DOWN_SLICED=0
    run K-normsplit0 short,prose 128 STRATA_GR_NORM_SPLIT=0
    run K-hcsplit1 short,prose 128 STRATA_HC_SPLIT=1
    run K-hcsplit0 short,prose 128 STRATA_HC_SPLIT=0
    run G-keyhead 10k 16 STRATA_GDN_KEYHEAD=1
    run G-heads 10k 16 STRATA_GDN_REC_HEADS=1
    run G-nopipe 10k 16 STRATA_GDN_PIPELINE=0
fi
# ./arc start's flags (STRATA_EXTRA_ARGS replaces them, so a run repeats them; the engine keeps a flag's last value)
BASE_ARGS="$ARC_ENGINE_ARGS_DEFAULT"
run PS1 short,prose 128 STRATA_PLAN_STAGE=1
run DQ32 2k,10k 128 STRATA_DQ_WG=32
run SR64 1k,agent 128 "STRATA_EXTRA_ARGS=$BASE_ARGS --short-read 64"
if [ "$(arc_bdfs | wc -l)" -ge 2 ]; then   # the same box on one card: the single-GPU mode's speeds
    run G1 short,prose,2k,10k 128 ARC_GPUS=0
fi
run A 10k 128 STRATA_PREFILL_TIMING=1 STRATA_DRAFT_TIMING=1
if [ -n "$unitrace" ] && [ "${ARC_PROFILE_TRACE:-0}" = 1 ]; then
    run D short,prose 128 STRATA_VERIFY_EAGER=1 STRATA_EXIT_SIGNAL=10 "STRATA_WRAP=$unitrace -d --teardown-on-signal 10 -o $out.D.unitrace"
    run E 10k 1 STRATA_EXIT_SIGNAL=10 "STRATA_WRAP=$unitrace -d --teardown-on-signal 10 -o $out.E.unitrace"
    for f in "$out".D.unitrace* "$out".E.unitrace*; do
        [ -f "$f" ] || continue
        { echo "================ $(basename "$f")"; head -c 60000 "$f"; echo; } >>"$out"
    done
fi
trap - EXIT
python3 "$ROOT/tools/profile_summary.py" "$out" || true
rm -f "$out".*.start
ok "written: $out (send this file)"
