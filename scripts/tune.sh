#!/usr/bin/env bash
# ./arc tune [model] - find this machine's fastest CORRECT settings and keep them for ./arc start.
#
# For one card (ARC_GPUS=0) and, with two or more Arcs, for the layer split, the server is started once per candidate
# below; each candidate must first answer the 12 short checks of tools/check.py, then tools/bench.py measures it
# (short code, Russian prose, a 10K prompt and an agent turn, two runs each). A candidate is kept when it beats the
# base by >= 2% on the number it targets (code decode for decode switches, 10K prompt tok/s for prompt ones) and
# loses no more than 3% anywhere else. The kept ones are then combined and the combination has to pass the full
# check (with the two 10K prompts) and stay at least as fast as the base - otherwise the best single candidate is
# used. The result goes to tune-dual.env / tune-single.env in the repository root; ./arc start (lib.sh) applies the
# one that matches the cards it runs on. A variable set by the caller still wins. Everything is written to
# tune-<time>.txt. About 35 minutes per mode (each start reloads the model). The server must be stopped first.
#
#   ARC_TUNE_MODES="dual single"   which modes to tune (default: both when two Arcs are present, else single)
#   ARC_TUNE_RUNS=2                bench runs per case
set -euo pipefail
. "$(dirname "$0")/lib.sh"
cd "$STRATA"
want=${1:-${ARC_MODEL:-iq2_xs}}
want=$(echo "$want" | tr '[:upper:]' '[:lower:]')
cfg="strata-$want.json"
[ -f "$cfg" ] || die "no $cfg: ./arc setup first"
read -r port log < <(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("port", 8080), c.get("log") or "-")' "$cfg")
url="http://127.0.0.1:$port"
curl -fsS -m 3 "$url/health" >/dev/null 2>&1 && die "a server answers on $url: stop it first (./arc tune starts its own)"

n_arc=$(arc_bdfs | wc -l)
modes=${ARC_TUNE_MODES:-$([ "$n_arc" -ge 2 ] && echo "dual single" || echo single)}
runs=${ARC_TUNE_RUNS:-2}
stamp=$(date +%Y%m%d-%H%M%S)
out="$ROOT/tune-$stamp.txt"
work="$ROOT/.tune-$stamp"
mkdir -p "$work"
: >"$out"

# the tuned files must not steer the measurements: every run starts from lib.sh's own defaults
export ARC_TUNE_IGNORE=1
arc_engine_env
BASE_ARGS="$STRATA_EXTRA_ARGS"

srv_pid=""
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

# measure <tag> <full-check 0|1> <env...>: prints "ok" or "fail" into $work/<tag>.status, bench rows into <tag>.json
measure() {
    local tag=$1 full=$2; shift 2
    say "${B}tune $tag${N}: $*"
    echo "================ $tag: $*" >>"$out"
    env "$@" setsid "$ROOT/scripts/start.sh" "$want" >"$work/$tag.start" 2>&1 &
    srv_pid=$!
    local t=0 loaded=1
    until curl -fsS -m 3 "$url/health" 2>/dev/null | grep -q '"loaded": *true'; do
        sleep 5; t=$((t + 5))
        kill -0 "$srv_pid" 2>/dev/null || { loaded=0; break; }
        [ $t -lt 900 ] || { loaded=0; break; }
    done
    if [ $loaded = 0 ]; then
        echo "did not load" | tee -a "$out"; tail -n 20 "$work/$tag.start" >>"$out"
        echo fail >"$work/$tag.status"; stop_server; return 0
    fi
    local long=(--no-long); [ "$full" = 1 ] && long=()
    if python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/check.py" --url "$url" "${long[@]}" >>"$out" 2>&1; then
        python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/bench.py" --url "$url" --runs "$runs" \
            --max-tokens 128 --cases short,prose,10k,agent --json "$work/$tag.json" >>"$out" 2>&1 || true
        if [ -s "$work/$tag.json" ]; then echo ok >"$work/$tag.status"; else echo fail >"$work/$tag.status"; fi
    else
        echo "check failed: rejected" | tee -a "$out"
        echo fail >"$work/$tag.status"
    fi
    stop_server
}

# metrics <tag>: "decode_code prompt_10k agent_wall" medians (0 when missing)
metrics() {
    python3 - "$work/$1.json" <<'EOF'
import json, statistics, sys
try:
    rows = json.load(open(sys.argv[1]))
except Exception:
    print("0 0 0"); sys.exit()
def med(case, key):
    v = [r[key] for r in rows if r["case"] == case and isinstance(r.get(key), (int, float))]
    return statistics.median(v) if v else 0
print(f'{med("short code", "decode_tps"):.2f} {med("10K code", "prompt_tps"):.1f} {med("agent (stream)", "wall"):.2f}')
EOF
}

# the candidates: <tag> <targets decode|prompt> <env...>   (all opt-in switches this repository has, plus window/chunk)
candidates() {
    cat <<EOF
commit_overlap decode STRATA_VERIFY_COMMIT_OVERLAP=1
chain_submit decode STRATA_MTP_CHAIN_SUBMIT=1
lookahead decode STRATA_MTP_LOOKAHEAD=1
skip_empty decode STRATA_VERIFY_SKIP_EMPTY=1
native_graph decode STRATA_VERIFY_NATIVE_GRAPH=1
spec5 decode STRATA_EXTRA_ARGS=${BASE_ARGS// --spec 6/} --spec 5
spec4 decode STRATA_EXTRA_ARGS=${BASE_ARGS// --spec 6/} --spec 4
chunk4096 prompt STRATA_EXTRA_ARGS=${BASE_ARGS// --prefill auto/} --prefill 4096
split_rows8 prompt STRATA_PF_SPLIT_ROWS=8
EOF
}

for mode in $modes; do
    menv=(); [ "$mode" = single ] && menv=(ARC_GPUS=0)
    say "${B}=== mode $mode${N}"
    measure "$mode-base" 0 "${menv[@]}"
    [ "$(cat "$work/$mode-base.status")" = ok ] || { warn "$mode: the base itself fails - not tuned"; continue; }
    read -r bd bp ba < <(metrics "$mode-base")
    echo "$mode base: decode $bd tok/s, 10K prompt $bp tok/s, agent $ba s" | tee -a "$out"
    kept=(); best_tag=""; best_gain=0
    declare -A envof=()
    while read -r tag target rest; do
        [ -n "$tag" ] || continue
        # one env assignment per candidate; STRATA_EXTRA_ARGS keeps its spaces
        measure "$mode-$tag" 0 "${menv[@]}" "$rest"
        [ "$(cat "$work/$mode-$tag.status")" = ok ] || continue
        read -r d p a < <(metrics "$mode-$tag")
        verdict=$(python3 -c "
bd,bp,ba,d,p,a=map(float,'$bd $bp $ba $d $p $a'.split()); t='$target'
gain = d/bd-1 if t=='decode' else p/bp-1
keep = gain >= 0.02 and d >= bd*0.97 and p >= bp*0.97 and (a <= ba*1.05 or ba == 0)
print(f'{\"keep\" if keep else \"drop\"} {gain:.4f}')")
        echo "$mode $tag: decode $d, 10K $p, agent $a -> $verdict" | tee -a "$out"
        set -- $verdict
        if [ "$1" = keep ]; then
            envof[$tag]=$rest
            # two STRATA_EXTRA_ARGS candidates cannot both be kept: keep the larger gain
            if [[ "$rest" == STRATA_EXTRA_ARGS=* ]]; then
                for k in "${!kept[@]}"; do
                    [[ "${envof[${kept[$k]}]}" == STRATA_EXTRA_ARGS=* ]] && unset 'kept[k]'
                done
            fi
            kept+=("$tag")
            if python3 -c "import sys; sys.exit(0 if $2 > $best_gain else 1)"; then best_gain=$2; best_tag=$tag; fi
        fi
    done < <(candidates)

    final=()
    for tag in "${kept[@]}"; do final+=("${envof[$tag]}"); done
    chosen=()
    if [ ${#final[@]} -gt 0 ]; then
        measure "$mode-combined" 1 "${menv[@]}" "${final[@]}"
        if [ "$(cat "$work/$mode-combined.status")" = ok ]; then
            read -r d p a < <(metrics "$mode-combined")
            echo "$mode combined (${kept[*]}): decode $d, 10K $p, agent $a" | tee -a "$out"
            if python3 -c "import sys; sys.exit(0 if $d >= $bd and $p >= $bp*0.98 else 1)"; then chosen=("${final[@]}"); fi
        fi
        if [ ${#chosen[@]} -eq 0 ] && [ -n "$best_tag" ]; then
            measure "$mode-best" 1 "${menv[@]}" "${envof[$best_tag]}"
            [ "$(cat "$work/$mode-best.status")" = ok ] && chosen=("${envof[$best_tag]}")
        fi
    fi
    f="$ROOT/tune-$mode.env"
    {
        echo "# ./arc tune $stamp ($mode): base decode $bd tok/s, 10K prompt $bp tok/s, agent $ba s"
        echo "# measured candidates and verdicts: tune-$stamp.txt"
        for kv in "${chosen[@]}"; do
            k=${kv%%=*}; v=${kv#*=}
            printf 'export %s="${%s-%s}"\n' "$k" "$k" "$v"
        done
    } >"$f"
    if [ ${#chosen[@]} -gt 0 ]; then ok "$mode: kept ${chosen[*]} -> $f"; else ok "$mode: the defaults are the fastest -> $f (empty)"; fi
    unset envof
done
trap - EXIT
ok "written: $out (send this file); the tuned settings apply from the next ./arc start"
