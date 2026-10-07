#!/usr/bin/env bash
# ./arc profile [model] - where the time goes: starts the configured server three times with the engine's profilers
# on, sends prompts, stops it, and writes everything worth reading into one file (profile-<time>.txt).
#
#   A  the config as it is: the prompt path's phases (STRATA_PREFILL_TIMING) on the 2K and 8K prompts
#   B  the same, decode windows run eagerly with a host clock at every stage (STRATA_VERIFY_EAGER + _PROFILE):
#      ms per stage of a verify window (slower than real decode: each stage waits for the GPU)
#   C  bigger prompt chunks: the prompt path borrows expert-cache slots (--prefill-borrow --prefill auto)
#
# The server must be stopped first (the run starts its own). About 10 minutes.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
cd "$STRATA"
want=${1:-${ARC_MODEL:-iq2_xs}}
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

srv_pid=""
stop_server() {
    [ -n "$srv_pid" ] || return 0
    kill -TERM -- "-$srv_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$srv_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$srv_pid" 2>/dev/null || true
    srv_pid=""
    sleep 3   # the GPUs' memory is released after the process is gone
}
trap stop_server EXIT

run() {   # run <name> <cases> <env...>: start, wait for the model, bench, stop, keep the engine's lines
    local name=$1 cases=$2; shift 2
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
    {
        echo "================ $name: $*"
        python3 "$ROOT/tools/bench.py" --url "$url" "${keyarg[@]}" --runs 1 --max-tokens 128 --cases "$cases" 2>&1 || true
        echo "---- engine log"
        if [ -f "$log" ]; then
            tail -c +$((since + 1)) "$log" | grep -E "prefill timing|prompt chunk|prompt path|borrow|serve: prompt|verify stages|window: since|split|mirror|expert cache|VRAM free|NO_HOST|GPU HANG|error|failed" || true
        fi
        echo
    } >>"$out"
    stop_server
}

{
    echo "arc-moe profile, $(date), $(git -C "$ROOT" rev-parse --short HEAD), model $want"
    for d in /sys/class/drm/card*/device; do
        [ -f "$d/vendor" ] && [ "$(cat "$d/vendor")" = 0x8086 ] || continue
        echo "$(basename "$(readlink -f "$d")"): link $(cat "$d/current_link_speed" 2>/dev/null) x$(cat "$d/current_link_width" 2>/dev/null)"
    done
    free -g | head -2
    echo
} >"$out"

run A 2k,8k STRATA_PREFILL_TIMING=1
run B short,prose STRATA_VERIFY_EAGER=1 STRATA_VERIFY_PROFILE=1
run C 2k,8k STRATA_PREFILL_TIMING=1 "STRATA_EXTRA_ARGS=--prefill-borrow --prefill auto"
trap - EXIT
ok "written: $out (send this file)"
