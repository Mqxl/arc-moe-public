#!/usr/bin/env bash
# ./arc tune [model] - find this machine's fastest CORRECT settings and keep them for ./arc start.
#
# Resolve the current measured configuration independently for dual and single.
# Baseline and final runs use full 14-request checks and all eight benchmark cases;
# screens use 12 checks and six cases including prose, post-10K/64K and vision.
# Keep >=2% target gain, <=2% loss in every metric (agent screen <=3%).
# Recheck combined and fallback settings against the full baseline. Failed or
# incomplete JSON never qualifies. Save tune-dual.env/tune-single.env atomically;
# retain the verified baseline when no candidate qualifies. Caller values win.
# Reports are tune-<time>.txt; a server must be stopped before this starts.
#
#   ARC_TUNE_MODES="dual single"   which modes to tune (default: both when two Arcs are present, else single)
#   ARC_TUNE_RUNS=2                bench runs per case
#   ARC_TUNE_CANDIDATES="spec5 split_rows8" optional focused shortlist
# The baseline now includes all eight context/vision cases. Candidate screens
# include prose, post-10K and 64K decode; final combinations and fallback must
# pass all cases within 2%. Agent screen tolerance is 3%, final tolerance 2%.
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

# Each mode resolves its current tune file in a fresh shell. Tuning must not
# discard an already verified machine configuration or leak dual args to single.
all_cases=short,prose,10k,32k,64k,120k,agent,vision
screen_cases=short,prose,10k,64k,agent,vision
baseline_env=()

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
    local cases=$screen_cases
    [ "$full" = 1 ] && cases=$all_cases
    say "${B}tune $tag${N}: $*"
    echo "================ $tag: $*" >>"$out"
    env "${baseline_env[@]}" "$@" setsid "$ROOT/scripts/start.sh" "$want" >"$work/$tag.start" 2>&1 &
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
    if python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/check.py" --url "$url" "${long[@]}" >"$work/$tag.check.txt" 2>&1; then
        cat "$work/$tag.check.txt" >>"$out"
        if python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/bench.py" --url "$url" --runs "$runs" \
            --max-tokens 128 --cases "$cases" --ttft --vram --seed "tune-$stamp" --json "$work/$tag.json" >>"$out" 2>&1 \
            && python3 "$ROOT/tools/tune_metrics.py" validate "$work/$tag.json" --cases "$cases" --runs "$runs" >>"$out" 2>&1; then
            echo ok >"$work/$tag.status"
        else
            echo "incomplete, erroneous or broken benchmark: rejected" >>"$out"
            echo fail >"$work/$tag.status"
        fi
    else
        cat "$work/$tag.check.txt" >>"$out"
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
expert_tail decode STRATA_EXPERT_TAIL_BATCH=1
EOF
}

for mode in $modes; do
    case "$mode" in dual) menv=(ARC_GPUS=0,1) ;; single) menv=(ARC_GPUS=0) ;; *) die "invalid tune mode: $mode" ;; esac
    mapfile -d '' -t baseline_env < <(env "${menv[@]}" bash -c '
        . "$1"; arc_engine_env
        python3 - "$2" <<"PY_ENV"
import json, os, sys
values = {k: v for k, v in os.environ.items() if k.startswith("STRATA_")}
cfg = json.load(open(sys.argv[1]))
if os.environ.get("STRATA_CONFIG_ENV_OVERRIDE") != "1":
    values.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items() if str(k).startswith("STRATA_")})
else:
    for k, v in (cfg.get("env") or {}).items():
        if str(k).startswith("STRATA_"): values.setdefault(str(k), str(v))
values.pop("STRATA_API_KEY", None)
values["STRATA_CONFIG_ENV_OVERRIDE"] = "1"
for k, v in sorted(values.items()):
    sys.stdout.buffer.write((k + "=" + v + "\0").encode())
PY_ENV
        ' _ "$ROOT/scripts/lib.sh" "$cfg")
    BASE_ARGS=""
    for kv in "${baseline_env[@]}"; do [[ "$kv" != STRATA_EXTRA_ARGS=* ]] || BASE_ARGS=${kv#*=}; done
    [ -n "$BASE_ARGS" ] || die "failed to resolve baseline arguments for $mode"
    say "${B}=== mode $mode${N}"
    measure "$mode-base" 1 "${menv[@]}"
    [ "$(cat "$work/$mode-base.status")" = ok ] || { warn "$mode: the base itself fails - not tuned"; continue; }
    read -r bd bp ba < <(metrics "$mode-base")
    echo "$mode base: decode $bd tok/s, 10K prompt $bp tok/s, agent $ba s" | tee -a "$out"
    kept=(); best_tag=""; best_gain=0
    declare -A envof=()
    while read -r tag target rest; do
        [ -n "$tag" ] || continue
        if [ -n "${ARC_TUNE_CANDIDATES:-}" ]; then
            case " $ARC_TUNE_CANDIDATES " in *" $tag "*) ;; *) continue ;; esac
        fi
        # one env assignment per candidate; STRATA_EXTRA_ARGS keeps its spaces
        measure "$mode-$tag" 0 "${menv[@]}" "$rest"
        [ "$(cat "$work/$mode-$tag.status")" = ok ] || continue
        read -r d p a < <(metrics "$mode-$tag")
        verdict=$(python3 "$ROOT/tools/tune_metrics.py" compare "$work/$mode-base.json" "$work/$mode-$tag.json" \
            --baseline-cases "$all_cases" --cases "$screen_cases" --runs "$runs" --target "$target" --agent-tolerance .03)
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
            verdict=$(python3 "$ROOT/tools/tune_metrics.py" compare "$work/$mode-base.json" "$work/$mode-combined.json" \
                --cases "$all_cases" --runs "$runs" --target any)
            echo "$mode combined gate: $verdict" >>"$out"
            [[ "$verdict" != keep* ]] || chosen=("${final[@]}")
        fi
        if [ ${#chosen[@]} -eq 0 ] && [ -n "$best_tag" ]; then
            measure "$mode-best" 1 "${menv[@]}" "${envof[$best_tag]}"
            if [ "$(cat "$work/$mode-best.status")" = ok ]; then
                verdict=$(python3 "$ROOT/tools/tune_metrics.py" compare "$work/$mode-base.json" "$work/$mode-best.json" \
                    --cases "$all_cases" --runs "$runs" --target any)
                echo "$mode fallback gate: $verdict" >>"$out"
                [[ "$verdict" != keep* ]] || chosen=("${envof[$best_tag]}")
            fi
        fi
    fi
    f="$ROOT/tune-$mode.env"
    # Retain the measured base when all candidates lose. Escape defaults as
    # shell literals; secrets are excluded above. Replace atomically.
    python3 - "$f" "$stamp" "$mode" "${baseline_env[@]}" "${chosen[@]}" <<'PY_SAVE'
import os, re, sys
from pathlib import Path
path, stamp, mode, *assignments = sys.argv[1:]
values = dict(a.split('=', 1) for a in assignments)
lines = [f'# ./arc tune {stamp} ({mode}); see tune-{stamp}.txt']
for key, value in sorted(values.items()):
    if not re.fullmatch(r'STRATA_[A-Z0-9_]+', key) or key == 'STRATA_API_KEY':
        raise ValueError('unexpected tune variable')
    escaped = ''.join('\\' + c if c in '\\"`$}' else c for c in value)
    lines.append('export ' + key + '="${' + key + '-' + escaped + '}"')
tmp = Path(path + '.tmp')
tmp.write_text('\n'.join(lines) + '\n')
os.replace(tmp, path)
PY_SAVE
    if [ ${#chosen[@]} -gt 0 ]; then ok "$mode: kept ${chosen[*]} -> $f"; else ok "$mode: retained verified base -> $f"; fi
    unset envof
done
trap - EXIT
ok "written: $out (send this file); the tuned settings apply from the next ./arc start"
