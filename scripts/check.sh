#!/usr/bin/env bash
# ./arc check [model] - does the server answer with text the way every kind of client asks? tools/check.py sends
# one short question greedy and sampled, thinking off and on, streamed, with penalties and with tools.
#
# A server already running: checked as it is (a minute). None running: started once per configuration below with
# STRATA_DBG_NAN=1, each checked, everything written to check-<time>.txt (~10 minutes, each start reloads the model):
#   base          ./arc start's defaults
#   one-block     STRATA_SAMPLER_ONE_BLOCK=1  the sampled path's one-kernel variant (no split scratch)
#   old-sampler   STRATA_OLD_SAMPLER=1        the original sampled kernel
#   no-spec       --spec 0                    no drafter (no coupled draft sampling)
#   eager         STRATA_VERIFY_EAGER=1       the verify window without SYCL graphs
#   one-gpu       ARC_GPUS=0                  on a machine with two Arcs: no layer split
set -euo pipefail
. "$(dirname "$0")/lib.sh"
cd "$STRATA"
want=${1:-${ARC_MODEL:-iq2_xs}}
want=$(echo "$want" | tr '[:upper:]' '[:lower:]')
cfg="strata-$want.json"
[ -f "$cfg" ] || die "no $cfg: ./arc setup first (or ./arc check <model>)"
read -r port key log < <(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("port", 8080), c.get("api_key") or "-", c.get("log") or "-")' "$cfg")
url="http://127.0.0.1:$port"
keyarg=(); [ "$key" != "-" ] && keyarg=(--api-key "$key")

if curl -fsS -m 3 "$url/health" >/dev/null 2>&1; then
    say "${B}checking the running server${N} at $url (stop it for the full run over several configurations)"
    exec python3 "$ROOT/tools/check.py" --url "$url" "${keyarg[@]}"
fi

[ "$log" != "-" ] || die "$cfg has no \"log\": the engine's lines are needed"
case "$log" in /*) ;; *) log="$STRATA/$log" ;; esac
out="$ROOT/check-$(date +%Y%m%d-%H%M%S).txt"

srv_pid=""
stop_server() {
    [ -n "$srv_pid" ] || return 0
    local py
    py=$(pgrep -g "$srv_pid" -f "server.py" | head -1 || true)
    [ -n "$py" ] && kill -INT "$py" 2>/dev/null || true
    for _ in $(seq 1 60); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -TERM -- "-$srv_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -KILL -- "-$srv_pid" 2>/dev/null || true
    srv_pid=""
    sleep 3
}
trap stop_server EXIT

keep='strata dbg|head sampling|sampl|NaN|non-finite|GPU HANG|rror|xception|terminate|what\(\)|abort|egmentation|DEVICE_LOST|split|extra engine'

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
    python3 "$ROOT/tools/check.py" --url "$url" "${keyarg[@]}" 2>&1 | tee -a "$out" || true
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
    echo
} >"$out"
BASE_ARGS="$ARC_ENGINE_ARGS_DEFAULT"
run base ARC_CHECK_RUN=base
run one-block STRATA_SAMPLER_ONE_BLOCK=1
run old-sampler STRATA_OLD_SAMPLER=1
run no-spec "STRATA_EXTRA_ARGS=$BASE_ARGS --spec 0"
run eager STRATA_VERIFY_EAGER=1
if [ "$(arc_bdfs | wc -l)" -ge 2 ]; then run one-gpu ARC_GPUS=0; fi
trap - EXIT
rm -f "$out".*.start
ok "written: $out (send this file)"
