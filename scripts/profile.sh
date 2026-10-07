#!/usr/bin/env bash
# ./arc profile [model] [--no-unitrace] - where the time goes: starts the configured server several times with the
# engine's profilers on, sends prompts, stops it, and writes everything worth reading into one file
# (profile-<time>.txt).
#
#   N  the config as it is, nothing traced: the real decode speed (the baseline of V2 and IO)
#   V2 the same with the Level Zero v2 adapter (SYCL_UR_USE_LEVEL_ZERO_V2=1)
#   IO the same with the driver's in-order command lists (UR_L0_USE_DRIVER_INORDER_LISTS=1)
#   A  the config as it is: the prompt path's phases (STRATA_PREFILL_TIMING) on the 2K and 8K prompts
#   B  decode windows run eagerly with a host clock at every stage (STRATA_VERIFY_EAGER + _PROFILE): ms per stage
#      of a verify window (slower than real decode: each stage waits for the GPU)
#   C  bigger prompt chunks: the prompt path borrows expert-cache slots (--prefill-borrow --prefill auto)
#   D  decode, eager, under unitrace: the GPU time of every kernel by name
#   E  the 8K prompt under unitrace
#
# D and E need unitrace (Intel's pti-gpu, built once by scripts/get-unitrace.sh: libze-dev from apt, ~3 min).
# The server must be stopped first (the run starts its own). About 25 minutes.
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
if [ $use_unitrace = 1 ]; then
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
keep+='|VRAM free|NO_HOST|GPU HANG|rror|xception|terminate|what\(\)|abort|egmentation|DEVICE_LOST|extra engine|engine under'

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
    python3 "$ROOT/tools/bench.py" --url "$url" "${keyarg[@]}" --runs 1 --max-tokens "$maxt" --cases "$cases" >>"$out" 2>&1 || true
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

run N short,prose,2k 128 ARC_PROFILE_RUN=N
run V2 short,prose,2k 128 SYCL_UR_USE_LEVEL_ZERO_V2=1
run IO short,prose,2k 128 UR_L0_USE_DRIVER_INORDER_LISTS=1
run A 2k,8k 128 STRATA_PREFILL_TIMING=1
run B short,prose 128 STRATA_VERIFY_EAGER=1 STRATA_VERIFY_PROFILE=1
run C 2k,8k 128 STRATA_PREFILL_TIMING=1 "STRATA_EXTRA_ARGS=--prefill-borrow --prefill auto"
if [ -n "$unitrace" ]; then
    run D short,prose 128 STRATA_VERIFY_EAGER=1 "STRATA_WRAP=$unitrace -d -o $out.D.unitrace"
    run E 8k 1 "STRATA_WRAP=$unitrace -d -o $out.E.unitrace"
    for f in "$out".D.unitrace* "$out".E.unitrace*; do
        [ -f "$f" ] || continue
        { echo "================ $(basename "$f")"; head -c 60000 "$f"; echo; } >>"$out"
    done
fi
trap - EXIT
python3 "$ROOT/tools/profile_summary.py" "$out" || true
rm -f "$out".*.start
ok "written: $out (send this file)"
