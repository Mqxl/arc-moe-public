#!/usr/bin/env bash
# ./arc profile [model] [--no-unitrace] - where the time goes: starts the configured server several times with the
# engine's profilers on, sends prompts, stops it, and writes everything worth reading into one file
# (profile-<time>.txt).
#
#   N     the defaults (./arc start's: lib.sh arc_engine_env), nothing traced: the real speeds on short prompts, on
#         2K / 10K prompts and on a sampled request and an agent harness's (tools, streaming); every reply checked
#   ARC_PROFILE_EP_SWEEP=1: prompt expert parallelism, greedy/top-20 parity, both card modes
#   PS1   STRATA_PLAN_STAGE=1: the device plan stages mirrored experts in VRAM (patch 21; off by default since round 9)
#   PG0   STRATA_PF_GROUPED=0: the prompt path's experts dequantized to FP16 + oneMKL (before patch 30)
#   SR768 --short-read 768: old threshold against the current 64-token default
#   SR64  --short-read 64: upstream's threshold (short prompt parts batched)
#   G1    one card only (ARC_GPUS=0), on a machine with two: the single-GPU mode
#   A   the defaults with the prompt path's phases (STRATA_PREFILL_TIMING) and the draft layer's (STRATA_DRAFT_TIMING)
#   D   decode, eager, under unitrace: the GPU time of every kernel by name   (ARC_PROFILE_TRACE=1)
#   E   the 10K prompt under unitrace                                          (ARC_PROFILE_TRACE=1)
#   ARC_PROFILE_SHORT_READ_SWEEP=1: thresholds 64/256/512/1536 on agent/730/1425/2125 tokens
#   ARC_PROFILE_EXPERT_SWEEP=1: DP4A baseline versus 8/16-row prefill tiles
#   ARC_PROFILE_XMX_SWEEP=1: DP4A, tile16, SG8 XMX (128/256 GRF), small groups
#   ARC_PROFILE_COPY_SWEEP=1: staging banks/slots, group/row caps, then phase timing
#   ARC_PROFILE_LAYER_SWEEP=1: layer boundaries on this model/context and one-card control
#   ARC_PROFILE_LAYER_SPLITS="auto 30 32": optional boundaries, not global defaults
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
read -r port log < <(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("port", 8080), c.get("log") or "-")' "$cfg")
url="http://127.0.0.1:$port"
if curl -fsS -m 3 "$url/health" >/dev/null 2>&1; then
    die "a server already answers on $url: stop it first (the profile starts its own)"
fi
[ "$log" != "-" ] || die "$cfg has no \"log\": the engine's lines are needed"
case "$log" in /*) ;; *) log="$STRATA/$log" ;; esac
profile_args_set=${STRATA_EXTRA_ARGS+x}
profile_user_args=${STRATA_EXTRA_ARGS-}
declare -A profile_user_knobs=()
for knob in ${!STRATA_@}; do profile_user_knobs[$knob]=1; done
arc_engine_env
profile_auto_unset=()
for knob in ${!STRATA_@}; do
    [[ -v profile_user_knobs[$knob] ]] || profile_auto_unset+=(-u "$knob")
done
out="$ROOT/profile-$(date +%Y%m%d-%H%M%S).txt"

unitrace=""
if [ $use_unitrace = 1 ] && [ "${ARC_PROFILE_TRACE:-0}" = 1 ]; then
    if "$ROOT/scripts/get-unitrace.sh"; then unitrace="$ROOT/third_party/pti-gpu/tools/unitrace/build/unitrace"
    else warn "no unitrace: runs D and E are skipped (./arc profile --no-unitrace to not try)"; fi
fi

srv_pid=""
failures=0
stop_server() {   # SIGTERM to Python only sends the engine QUIT; background jobs may inherit SIGINT ignored
    [ -n "$srv_pid" ] || return 0
    local py
    py=$(pgrep -g "$srv_pid" -f "serve/server(_intel)?\.py" | head -1 || true)
    [ -n "$py" ] && kill -TERM "$py" 2>/dev/null || true
    for _ in $(seq 1 90); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -TERM -- "-$srv_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do pgrep -g "$srv_pid" >/dev/null || break; sleep 1; done
    kill -KILL -- "-$srv_pid" 2>/dev/null || true
    srv_pid=""
    sleep 3   # the GPUs' memory is released after the process is gone
}
trap stop_server EXIT

keep='PCIe probe|prefill copy|prefill timing|prompt chunk|prompt path|borrow|serve: prompt|verify stages|window: since|split|mirror|expert cache'
keep+='|prompt EP|mtp chain|mtp lookahead|gate/up split weight reuse|integer XMX|pin loan|early expert pool|prompt attention|strata decode timing|strata decode GPU|strata dbg|head sampling|draft kv|VRAM free|NO_HOST|GPU HANG|rror|xception|terminate|what\(\)|abort|egmentation|DEVICE_LOST|extra engine|engine under'

run() {   # run <name> <cases> <max-tokens> <env...>: start, wait for the model, bench, stop, keep the engine's lines
    local name=$1 cases=$2 maxt=$3; shift 3
    # Resume only completed, matching decode arms from an explicitly supplied
    # prior profile. Interrupted arms are rerun and provenance stays in the report.
    case "$name" in dual-DC|dual-D0|dual-DQ|dual-DL)
        if [ -n "${ARC_PROFILE_DECODE_RESUME:-}" ] && [ -f "$ARC_PROFILE_DECODE_RESUME.$name.json" ]; then
            if { [ "$name" = dual-DC ] || [ -f "$ARC_PROFILE_DECODE_RESUME.$name.parity.json" ]; } && python3 - "$ARC_PROFILE_DECODE_RESUME.$name.json" <<'PY_CHECK'
import json,sys
rows=json.load(open(sys.argv[1]))
assert len(rows)==6 and all(not r.get('error') and r.get('tokens',0)>0 and r.get('prompt_tps') and r.get('decode_tps') for r in rows)
PY_CHECK
            then
                cp "$ARC_PROFILE_DECODE_RESUME.$name.json" "$out.$name.json"
                [ ! -f "$ARC_PROFILE_DECODE_RESUME.$name.parity.json" ] || cp "$ARC_PROFILE_DECODE_RESUME.$name.parity.json" "$out.$name.parity.json"
                echo "================ $name: reused completed arm from $ARC_PROFILE_DECODE_RESUME; interrupted arms are rerun" >>"$out"
                return
            fi
        fi ;;
    esac
    # Screening keeps both cold/warm prompts; decoding speed is validated separately.
    if [ "${ARC_PROFILE_SCREEN:-0}" = 1 ]; then
        case "$name" in
            *SPEC*|*SR*|*Dgraph*|*Deager*|*DT*|RP*|RG*) ;;
            *) case ",$cases," in *,10k,*) case ",$cases," in *,2k,*) cases=2k,10k ;; *) cases=10k ;; esac ;; esac; maxt=32 ;;
        esac
    fi
    if [ "${ARC_PROFILE_SCREEN:-0}" = 1 ]; then
        local candidate="" baseline=""
        case "$name" in
            G1-FX4|G1-FX8|G1-FX16) candidate=${name#G1-}; baseline=FX0 ;;
            single-Gkey|single-Gheads|single-Gserial) candidate=dual-${name#single-}; baseline=dual-G0 ;;
        esac
        if [ -n "$candidate" ] && python3 "$ROOT/tools/screen-reject.py" "$out.$baseline.json" "$out.$candidate.json"; then
            echo "================ $name: skipped: dual-GPU prefill loses >5% in every measured case" >>"$out"
            return
        fi
    fi
    say "${B}profile $name${N}: $*"
    local since=0
    [ -f "$log" ] && since=$(stat -c %s "$log")
    env "${profile_auto_unset[@]}" "$@" STRATA_SYCL_NATIVE=1 setsid "$ROOT/scripts/start.sh" "$want" >"$out.$name.start" 2>&1 &
    srv_pid=$!
    local t=0
    until curl -fsS -m 3 "$url/health" 2>/dev/null | grep -q '"loaded": *true'; do
        sleep 5; t=$((t + 5))
        kill -0 "$srv_pid" 2>/dev/null || { warn "the server stopped while loading (see $out.$name.start)"; break; }
        [ $t -lt 900 ] || { warn "not loaded after 15 minutes"; break; }
    done
    echo "================ $name: $*; cases=$cases max_tokens=$maxt runs=${ARC_PROFILE_RUNS:-2}" >>"$out"
    # First request after loading: prior decode requests can adapt expert placement.
    # Keep the prompt-path parity comparison independent of that changing state.
    case "$name" in PIPE0|PIPE1|G1-PIPE0|G1-PIPE1|EP0|EP1|dual-D0|dual-DQ|dual-DN|single-D0|single-DQ|single-DN|dual-DL|single-DL)
        if ! python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/prefill-response-parity.py" --url "$url" --output "$out.$name.parity.json" >>"$out" 2>&1; then
            failures=$((failures + 1))
            echo "---- arm stopped after failed 10K correctness request" >>"$out"
            stop_server
            [ ! -f "$log" ] || { tail -c +$((since + 1)) "$log" | grep -E "$keep" | cut -c1-2000 >>"$out" || true; }
            return
        fi ;;
    esac
    case "$name" in RP*|RG*)
        if ! python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/prefill-response-parity.py" --url "$url" --max-tokens "${ARC_PARITY_MAX_TOKENS:-1}" --output "$out.$name.parity.json" >>"$out" 2>&1; then
            failures=$((failures + 1))
        fi ;;
    esac
    local observers=(); [ "${ARC_PROFILE_CONTEXT_TIMING:-0}" != 1 ] || observers=(--ttft --vram)
    if python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/bench.py" --url "$url" --runs "${ARC_PROFILE_RUNS:-2}" --max-tokens "$maxt" --cases "$cases" "${observers[@]}" --json "$out.$name.json" >>"$out" 2>&1; then
        :
    else
        failures=$((failures + 1))
    fi
    if [ "$name" = LANG-S6 ]; then
        # The API supports per-request confidence: compare without two model reloads.
        for prob in 0.35 0.65; do
            echo "================ LANG-P$prob: --spec 6, per-request spec_min_p=$prob; cases=$cases max_tokens=$maxt runs=${ARC_PROFILE_RUNS:-2}" >>"$out"
            if ! python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/bench.py" --url "$url" --runs "${ARC_PROFILE_RUNS:-2}" --max-tokens "$maxt" --cases "$cases" --spec-min-p "$prob" --json "$out.LANG-P$prob.json" >>"$out" 2>&1; then
                failures=$((failures + 1))
            fi
        done
    fi
    if [ "$name" = C128 ]; then
        local context_args=()
        [ "${ARC_CONTEXT_COMBINED:-0}" != 1 ] || context_args+=(--combined-agent --decode-tokens 256)
        if ! python3 "$ROOT/tools/run-authorized.py" "$cfg" "$ROOT/tools/context-check.py" --url "$url" --json "$out.$name.context.json" "${context_args[@]}" >>"$out" 2>&1; then
            failures=$((failures + 1))
        fi
    fi
    stop_server
    {
        echo "---- engine log"
        [ -f "$log" ] && tail -c +$((since + 1)) "$log" | grep -E "$keep" | cut -c1-2000 || true
        echo
    } >>"$out"
}

{
    echo "arc-moe profile, $(date), $(git -C "$ROOT" rev-parse --short HEAD), model $want"
    echo "screen=${ARC_PROFILE_SCREEN:-0}; runs=${ARC_PROFILE_RUNS:-2}; screen decode rates are not final speed measurements"
    echo "engine source: $(git -C "$STRATA" rev-parse --short HEAD); patch fingerprint: $(git -C "$STRATA" config --get arcmoe.applied)"
    for knob in STRATA_LAYER_SPLIT STRATA_PF_EXPERT_KERNEL STRATA_PF_XMX_GRF STRATA_PF_XMX_MIN_ROWS STRATA_PF_STAGE_BANKS STRATA_PF_STAGE_SLOTS STRATA_PF_GROUPS STRATA_PF_BATCH_ROWS STRATA_PCIE_PINNED_PROBE STRATA_PF_PIN_LOAN STRATA_PF_PIN_REFILL STRATA_PF_COPY_TIMING STRATA_PF_PREFETCH_SLOTS STRATA_PF_PIPELINE STRATA_PROMPT_ATTN_XMX STRATA_PF_SPLIT_ROWS STRATA_PF_FUSE_EXPERT STRATA_PF_GU_I8_XMX STRATA_PF_DOWN_I8_XMX STRATA_PF_I8_SLAB STRATA_PF_I8_PAD STRATA_PF_I8_GROUP_MIN STRATA_PF_GU_SPLIT STRATA_PF_GU_SPLIT_LANES STRATA_PF_EP STRATA_VERIFY_COMMIT_OVERLAP STRATA_MTP_CHAIN_SUBMIT STRATA_VERIFY_SKIP_EMPTY STRATA_VERIFY_NATIVE_GRAPH STRATA_MTP_LOOKAHEAD STRATA_NATIVE_UPLOAD_PINNED STRATA_DENSE_UPLOAD_KERNEL STRATA_UPLOAD_VERIFY STRATA_UPLOAD_RELOCATE STRATA_DBG_PF_PHASES STRATA_DBG_WEIGHT_COPY; do
        [ -z "${!knob-}" ] || echo "$knob=${!knob}"
    done
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

# A focused sweep uses the same loader, reply checks and final report as the full profile.
if [ "${ARC_PROFILE_LAYER_SWEEP:-0}" = 1 ]; then
    ARC_PROFILE_CONTEXT_TIMING=1
    context_cases=${ARC_PROFILE_CONTEXT_CASES:-short,prose,10k,32k,64k,120k,agent,vision}
    for boundary in ${ARC_PROFILE_LAYER_SPLITS:-auto 30 32}; do
        [[ "$boundary" == auto || "$boundary" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid layer boundary: $boundary"
        run "LAY-${boundary//,/_}" "$context_cases" 128 ARC_GPUS=0,1 "STRATA_LAYER_SPLIT=$boundary"
    done
    # Resolve single-card arguments in a fresh shell: the global dual tuning
    # exports would otherwise override an empty tune-single.env (auto prefill).
    if [ "$profile_args_set" = x ]; then
        single_args=$profile_user_args
    else
    single_args=$(env "${profile_auto_unset[@]}" ARC_GPUS=0 bash -c '. "$1"; arc_engine_env; printf "%s" "$STRATA_EXTRA_ARGS"' _ "$ROOT/scripts/lib.sh")
    fi
    run G1-LAY "$context_cases" 128 ARC_GPUS=0 "STRATA_EXTRA_ARGS=$single_args" STRATA_LAYER_SPLIT=auto
elif [ "${ARC_PROFILE_DECODE_TRACE_ONLY:-0}" = 1 ]; then
    [ -n "$unitrace" ] || die "decode trace needs ARC_PROFILE_TRACE=1 and unitrace"
    run DT short,prose-ru-long,10k 128 STRATA_PF_EP=0 STRATA_DECODE_TIMING=1 STRATA_SPLIT_TIMING=1
    run D short,prose-ru-long 64 STRATA_PF_EP=0 STRATA_DECODE_TIMING=1 STRATA_EXIT_SIGNAL=10 "STRATA_WRAP=$unitrace -d -s --chrome-kernel-logging --teardown-on-signal 10 -o $out.D.unitrace"
    run D-eager short,prose-ru-long 64 STRATA_PF_EP=0 STRATA_DECODE_TIMING=1 STRATA_VERIFY_EAGER=1 STRATA_EXIT_SIGNAL=10 "STRATA_WRAP=$unitrace -d -s --chrome-kernel-logging --teardown-on-signal 10 -o $out.D-eager.unitrace"
    for f in "$out".D*.unitrace*; do
        [ -f "$f" ] || continue
        { echo "================ $(basename "$f")"; head -c 60000 "$f"; echo; } >>"$out"
        trace_pid="${f%.unitrace}"; trace_pid="${trace_pid##*.}"
        case "$trace_pid" in ""|*[!0-9]*) continue ;; esac
        timeline="$STRATA/strata.$trace_pid.json"
        if [ -f "$timeline" ]; then
            python3 "$ROOT/tools/analyze-decode-trace.py" "$timeline" >>"$out" 2>&1 || warn "GPU interval analysis unavailable: $timeline"
        fi
    done
elif [ "${ARC_PROFILE_DECODE_SWEEP:-0}" = 1 ]; then
    # Matched wall-clock API runs, with observers off. Positive min_p keeps the original MTP path.
    for gpu in dual single; do
        card=(); [ "$gpu" = single ] && card=(ARC_GPUS=0)
        run "$gpu-DC" short,prose-ru-long,10k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --spec-min-p 0.5" STRATA_PF_EP=0 STRATA_VERIFY_NATIVE_GRAPH=0 STRATA_VERIFY_COMMIT_OVERLAP=0 STRATA_MTP_CHAIN_SUBMIT=0 STRATA_VERIFY_SKIP_EMPTY=0 STRATA_MTP_LOOKAHEAD=0
        run "$gpu-D0" short,prose-ru-long,10k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --spec-min-p 0" STRATA_PF_EP=0 STRATA_VERIFY_NATIVE_GRAPH=0 STRATA_VERIFY_COMMIT_OVERLAP=0 STRATA_MTP_CHAIN_SUBMIT=0 STRATA_VERIFY_SKIP_EMPTY=0 STRATA_MTP_LOOKAHEAD=0
        run "$gpu-DQ" short,prose-ru-long,10k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --spec-min-p 0" STRATA_PF_EP=0 STRATA_VERIFY_NATIVE_GRAPH=0 STRATA_VERIFY_COMMIT_OVERLAP=1 STRATA_MTP_CHAIN_SUBMIT=1 STRATA_VERIFY_SKIP_EMPTY=1 STRATA_MTP_LOOKAHEAD=0
        run "$gpu-DL" short,prose-ru-long,10k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --spec-min-p 0" STRATA_PF_EP=0 STRATA_VERIFY_NATIVE_GRAPH=0 STRATA_VERIFY_COMMIT_OVERLAP=1 STRATA_MTP_CHAIN_SUBMIT=1 STRATA_VERIFY_SKIP_EMPTY=1 STRATA_MTP_LOOKAHEAD=1
        if [ "${ARC_PROFILE_NATIVE_GRAPH:-0}" = 1 ]; then
        run "$gpu-DN" short,prose-ru-long,10k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --spec-min-p 0" STRATA_PF_EP=0 STRATA_VERIFY_NATIVE_GRAPH=1 STRATA_VERIFY_COMMIT_OVERLAP=1 STRATA_MTP_CHAIN_SUBMIT=1 STRATA_VERIFY_SKIP_EMPTY=1 STRATA_MTP_LOOKAHEAD=0
            if ! python3 "$ROOT/tools/compare-prefill-response.py" "$out.$gpu-D0.parity.json" "$out.$gpu-DN.parity.json" >>"$out" 2>&1; then failures=$((failures + 1)); fi
        else
            echo "================ $gpu-DN: skipped; native recording requires ARC_PROFILE_NATIVE_GRAPH=1" >>"$out"
        fi
        for candidate in DQ DL; do
            if ! python3 "$ROOT/tools/compare-prefill-response.py" "$out.$gpu-D0.parity.json" "$out.$gpu-$candidate.parity.json" >>"$out" 2>&1; then failures=$((failures + 1)); fi
        done
    done
    run A-DQ short,prose-ru-long,10k 128 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --spec-min-p 0" STRATA_PF_EP=0 STRATA_VERIFY_NATIVE_GRAPH=0 STRATA_VERIFY_COMMIT_OVERLAP=1 STRATA_MTP_CHAIN_SUBMIT=1 STRATA_VERIFY_SKIP_EMPTY=1 STRATA_DECODE_TIMING=1 STRATA_SPLIT_TIMING=1 STRATA_VERIFY_WINDOW_TIMING=1
elif [ "${ARC_PROFILE_EP_SWEEP:-0}" = 1 ]; then
    run EP0 2k,10k 128 STRATA_PF_EP=0
    run EP1 2k,10k 128 STRATA_PF_EP=1
    if ! python3 "$ROOT/tools/compare-prefill-response.py" "$out.EP0.parity.json" "$out.EP1.parity.json" >>"$out" 2>&1; then failures=$((failures + 1)); fi
    run G1-EP0 2k,10k 128 ARC_GPUS=0 STRATA_PF_EP=0
    run G1-EP1 2k,10k 128 ARC_GPUS=0 STRATA_PF_EP=1
    run A-EP1 10k 128 STRATA_PF_EP=1 STRATA_PREFILL_TIMING=1
elif [ "${ARC_PROFILE_FINAL_FAST:-0}" = 1 ]; then
    # Normal-length final validation; alternatives and observers were screened separately.
    if [ "${ARC_PROFILE_FINAL_CONTROL:-0}" = 1 ]; then run BASE32 agent,1k,2k,10k 128 STRATA_PF_PIN_LOAN=0 STRATA_PF_PIN_REFILL=0 STRATA_PF_PREFETCH_SLOTS=0 STRATA_PF_PIPELINE=1 STRATA_PROMPT_ATTN_XMX=0 STRATA_PF_SPLIT_ROWS=4 STRATA_PF_FUSE_EXPERT=0 "STRATA_EXTRA_ARGS=--prefill-borrow --prefill auto --vram-reserve-mib 768 --spec 6 --short-read 64"; fi
    run N short,prose,1k,2k,10k,sampled,agent 128
    if [ "${ARC_PROFILE_FINAL_CONTROL:-0}" = 1 ]; then run G1-BASE32 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_PIN_LOAN=0 STRATA_PF_PIN_REFILL=0 STRATA_PF_PREFETCH_SLOTS=0 STRATA_PF_PIPELINE=1 STRATA_PROMPT_ATTN_XMX=0 STRATA_PF_SPLIT_ROWS=4 STRATA_PF_FUSE_EXPERT=0 "STRATA_EXTRA_ARGS=--prefill-borrow --prefill auto --vram-reserve-mib 768 --spec 6 --short-read 64"; fi
    run G1 short,prose,1k,2k,10k,sampled,agent 128 ARC_GPUS=0
elif [ "${ARC_PROFILE_DRAFT_VOCAB_SINGLE:-0}" = 1 ]; then
    run G1-DV-CJK short,prose-ru-long,prose-en,prose-zh 256 ARC_GPUS=0
    dir="$ROOT/test-logs/mtp-union"
    quoted=$(python3 -c 'import shlex,sys;print(shlex.quote(sys.argv[1]))' "$dir")
    run G1-DV-union short,prose-ru-long,prose-en,prose-zh 256 ARC_GPUS=0 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --mtp $quoted"
elif [ "${ARC_PROFILE_DRAFT_VOCAB_SWEEP:-0}" = 1 ]; then
    run DV-CJK short,prose-ru-long,prose-en,prose-zh 256
    for kind in union full; do
        dir="$ROOT/test-logs/mtp-$kind"
        [ -d "$dir" ] || die "prepare $dir with the existing MTP files first"
        quoted=$(python3 -c 'import shlex,sys;print(shlex.quote(sys.argv[1]))' "$dir")
        run "DV-$kind" short,prose-ru-long,prose-en,prose-zh 256 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --mtp $quoted"
    done
elif [ "${ARC_PROFILE_LANGUAGE_SWEEP:-0}" = 1 ]; then
    for spec in 2 6; do
        run "LANG-S$spec" short,prose-ru-long,prose-en,prose-zh 256 "STRATA_EXTRA_ARGS=--prefill-borrow --prefill auto --vram-reserve-mib 768 --spec $spec --short-read 64"
    done
elif [ "${ARC_PROFILE_GU_REUSE_LANES_SWEEP:-0}" = 1 ]; then
    nc=${ARC_PROFILE_GU_ROWS:-4}
    run GUL32 2k,10k 128 STRATA_PF_GU_SPLIT=1 STRATA_PF_SPLIT_ROWS="$nc" STRATA_PF_GU_SPLIT_LANES=32
    run GUL8 2k,10k 128 STRATA_PF_GU_SPLIT=1 STRATA_PF_SPLIT_ROWS="$nc" STRATA_PF_GU_SPLIT_LANES=8
    run GUL16 2k,10k 128 STRATA_PF_GU_SPLIT=1 STRATA_PF_SPLIT_ROWS="$nc" STRATA_PF_GU_SPLIT_LANES=16
    lane_winners=()
    for lanes in 8 16; do
        if python3 "$ROOT/tools/screen-improvement.py" "$out.GUL32.json" "$out.GUL$lanes.json"; then lane_winners+=("$lanes"); fi
    done
    if [ ${#lane_winners[@]} -gt 0 ]; then
        run G1-GUL32 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_SPLIT=1 STRATA_PF_SPLIT_ROWS="$nc" STRATA_PF_GU_SPLIT_LANES=32
        for lanes in "${lane_winners[@]}"; do run "G1-GUL$lanes" 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_SPLIT=1 STRATA_PF_SPLIT_ROWS="$nc" STRATA_PF_GU_SPLIT_LANES="$lanes"; done
    fi
elif [ "${ARC_PROFILE_GU_REUSE_SWEEP:-0}" = 1 ]; then
    run GUBASE 2k,10k 128 STRATA_PF_GU_SPLIT=0 STRATA_PF_SPLIT_ROWS=4
    for nc in 4 8 16; do run "GUR$nc" 2k,10k 128 STRATA_PF_GU_SPLIT=1 STRATA_PF_SPLIT_ROWS="$nc"; done
    gu_winners=()
    for nc in 4 8 16; do
        if python3 "$ROOT/tools/screen-improvement.py" "$out.GUBASE.json" "$out.GUR$nc.json"; then gu_winners+=("$nc"); fi
    done
    if [ ${#gu_winners[@]} -gt 0 ]; then
        run G1-GUBASE 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_SPLIT=0 STRATA_PF_SPLIT_ROWS=4
        for nc in "${gu_winners[@]}"; do run "G1-GUR$nc" 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_SPLIT=1 STRATA_PF_SPLIT_ROWS="$nc"; done
    fi
elif [ "${ARC_PROFILE_I8_PAD_SWEEP:-0}" = 1 ]; then
    run I8PADBASE 2k,10k 128 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=0
    run I8GUPAD 2k,10k 128 STRATA_PF_GU_I8_XMX=1 STRATA_PF_DOWN_I8_XMX=0 STRATA_PF_I8_SLAB=256 STRATA_PF_I8_PAD=16 STRATA_PF_I8_GROUP_MIN=32
    run I8DPAD 2k,10k 128 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=1 STRATA_PF_I8_SLAB=256 STRATA_PF_I8_PAD=16 STRATA_PF_I8_GROUP_MIN=32
    i8_winners=()
    for candidate in I8GUPAD I8DPAD; do
        if python3 "$ROOT/tools/screen-improvement.py" "$out.I8PADBASE.json" "$out.$candidate.json"; then i8_winners+=("$candidate"); fi
    done
    if [ ${#i8_winners[@]} -gt 0 ]; then
        run G1-I8PADBASE 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=0
        for candidate in "${i8_winners[@]}"; do
            if [ "$candidate" = I8GUPAD ]; then run G1-I8GUPAD 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=1 STRATA_PF_DOWN_I8_XMX=0 STRATA_PF_I8_SLAB=256 STRATA_PF_I8_PAD=16 STRATA_PF_I8_GROUP_MIN=32
            else run G1-I8DPAD 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=1 STRATA_PF_I8_SLAB=256 STRATA_PF_I8_PAD=16 STRATA_PF_I8_GROUP_MIN=32; fi
        done
    fi
elif [ "${ARC_PROFILE_I8_SLAB_SWEEP:-0}" = 1 ]; then
    run I8B 2k,10k 128 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=0
    run I8GU256 2k,10k 128 STRATA_PF_GU_I8_XMX=1 STRATA_PF_DOWN_I8_XMX=0 STRATA_PF_I8_SLAB=256
    run I8D256 2k,10k 128 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=1 STRATA_PF_I8_SLAB=256
    i8_winners=()
    for candidate in I8GU256 I8D256; do
        if ! python3 "$ROOT/tools/screen-reject.py" "$out.I8B.json" "$out.$candidate.json"; then i8_winners+=("$candidate"); fi
    done
    if [ ${#i8_winners[@]} -gt 0 ]; then
        run G1-I8B 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=0
        for candidate in "${i8_winners[@]}"; do
            if [ "$candidate" = I8GU256 ]; then run G1-I8GU256 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=1 STRATA_PF_DOWN_I8_XMX=0 STRATA_PF_I8_SLAB=256
            else run G1-I8D256 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=1 STRATA_PF_I8_SLAB=256; fi
        done
    fi
elif [ "${ARC_PROFILE_I8_SWEEP:-0}" = 1 ]; then
    run I8BASE 2k,10k 128 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=0
    run I8GU 2k,10k 128 STRATA_PF_GU_I8_XMX=1 STRATA_PF_DOWN_I8_XMX=0
    run I8XMX 2k,10k 128 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=1
    i8_winners=()
    for candidate in I8GU I8XMX; do
        if ! python3 "$ROOT/tools/screen-reject.py" "$out.I8BASE.json" "$out.$candidate.json"; then i8_winners+=("$candidate"); fi
    done
    if [ ${#i8_winners[@]} -gt 0 ]; then
        run G1-I8BASE 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=0
        for candidate in "${i8_winners[@]}"; do
            if [ "$candidate" = I8GU ]; then run G1-I8GU 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=1 STRATA_PF_DOWN_I8_XMX=0
            else run G1-I8XMX 2k,10k 128 ARC_GPUS=0 STRATA_PF_GU_I8_XMX=0 STRATA_PF_DOWN_I8_XMX=1; fi
        done
    fi
elif [ "${ARC_PROFILE_CONTEXT128:-0}" = 1 ]; then
    run C128 short,prose,2k,10k,sampled,agent 128
elif [ "${ARC_PROFILE_BALANCE_SWEEP:-0}" = 1 ]; then
    # Same precision and copy candidate; only stage placement and redundant dense tensors vary.
    run BAUTO short,prose,2k,10k 128
    run B34T short,prose,2k,10k 128 STRATA_STAGE_TRIM=1 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --layer-split 34"
    run B28T short,prose,2k,10k 128 STRATA_STAGE_TRIM=1 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --layer-split 28"
    run B24T short,prose,2k,10k 128 STRATA_STAGE_TRIM=1 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --layer-split 24"
elif [ "${ARC_PROFILE_PARITY_REPEAT:-0}" != 0 ]; then
    if [ "$ARC_PROFILE_PARITY_REPEAT" != single ]; then
        run RP0a short 1 STRATA_PF_PIPELINE=0
        run RP0b short 1 STRATA_PF_PIPELINE=0
        run RP1a short 1 STRATA_PF_PIPELINE=1
        run RP1b short 1 STRATA_PF_PIPELINE=1
    fi
    run RG1a short 1 ARC_GPUS=0 STRATA_PF_PIPELINE=1
    run RG1b short 1 ARC_GPUS=0 STRATA_PF_PIPELINE=1
elif [ "${ARC_PROFILE_SINGLE_COPY_SWEEP:-0}" = 1 ]; then
    run G1-CP2 2k,10k 128 ARC_GPUS=0 STRATA_PF_STAGE_BANKS=2
    run G1-CP3 2k,10k 128 ARC_GPUS=0 STRATA_PF_STAGE_BANKS=3
    run G1-A-CP2 10k 128 ARC_GPUS=0 STRATA_PF_STAGE_BANKS=2 STRATA_PREFILL_TIMING=1
    run G1-A-CP3 10k 128 ARC_GPUS=0 STRATA_PF_STAGE_BANKS=3 STRATA_PREFILL_TIMING=1
elif [ "${ARC_PROFILE_GDN_SWEEP:-0}" = 1 ]; then
    for gpu in dual single; do
        card=(); [ "$gpu" = single ] && card=(ARC_GPUS=0)
        run "$gpu-G0" 10k 128 "${card[@]}"
        run "$gpu-Gkey" 10k 128 "${card[@]}" STRATA_GDN_KEYHEAD=1
        run "$gpu-Gheads" 10k 128 "${card[@]}" STRATA_GDN_REC_HEADS=1
        run "$gpu-Gserial" 10k 128 "${card[@]}" STRATA_GDN_PIPELINE=0
    done
elif [ "${ARC_PROFILE_SHORT_READ_SWEEP:-0}" = 1 ]; then
    BASE_ARGS="$ARC_ENGINE_ARGS_DEFAULT"
    for sr in 64 256 512 1536; do
        run "SR$sr" agent,1k,1.5k,2k 128 "STRATA_EXTRA_ARGS=$BASE_ARGS --short-read $sr"
    done
elif [ "${ARC_PROFILE_EXPERT_SWEEP:-0}" = 1 ]; then
    for kernel in dp4a tile8 tile16; do
        run "PF-$kernel" 2k,10k 128 "STRATA_PF_EXPERT_KERNEL=$kernel"
    done
elif [ "${ARC_PROFILE_XMX_SWEEP:-0}" = 1 ]; then
    run PF4 agent,1k,2k,10k 128 STRATA_PF_EXPERT_KERNEL=dp4a
    run PF16 agent,1k,2k,10k 128 STRATA_PF_EXPERT_KERNEL=tile16
    run XM256 agent,1k,2k,10k 128 STRATA_PF_EXPERT_KERNEL=xmx STRATA_PF_XMX_GRF=256 STRATA_PF_XMX_MIN_ROWS=16
    run XM128 agent,1k,2k,10k 128 STRATA_PF_EXPERT_KERNEL=xmx STRATA_PF_XMX_GRF=128 STRATA_PF_XMX_MIN_ROWS=16
    run XM128-M1 agent,1k,2k,10k 128 STRATA_PF_EXPERT_KERNEL=xmx STRATA_PF_XMX_GRF=128 STRATA_PF_XMX_MIN_ROWS=1
elif [ "${ARC_PROFILE_NB_CONTROL:-0}" = 1 ]; then
    run NB0 2k 128 STRATA_PF_PIN_LOAN=0 STRATA_PF_PIN_REFILL=0 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --prefill 2048 --no-prefill-borrow"
elif [ "${ARC_PROFILE_EARLY_COMBO:-0}" = 1 ]; then
    run EC0 2k,10k 128 STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=1 STRATA_PF_PREFETCH_SLOTS=0
    run EC128 2k,10k 128 STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=1 STRATA_PF_PREFETCH_SLOTS=128
    if python3 "$ROOT/tools/screen-improvement.py" "$out.EC0.json" "$out.EC128.json"; then
        run G1-EC0 2k,10k 128 ARC_GPUS=0 STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=1 STRATA_PF_PREFETCH_SLOTS=0
        run G1-EC128 2k,10k 128 ARC_GPUS=0 STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=1 STRATA_PF_PREFETCH_SLOTS=128
    else
        echo "================ early-combo: no convincing dual gain; single repeat skipped" >>"$out"
    fi
elif [ "${ARC_PROFILE_DMA_SWEEP:-0}" = 1 ]; then
    for gpu in dual single; do
        card=(); [ "$gpu" = single ] && card=(ARC_GPUS=0)
        run "$gpu-DMA0" 10k 128 "${card[@]}" STRATA_PREFILL_TIMING=1 STRATA_PF_COPY_TIMING=1 STRATA_PF_PIN_LOAN=0 STRATA_PF_PIN_REFILL=0
        run "$gpu-DMA2" 10k 128 "${card[@]}" STRATA_PREFILL_TIMING=1 STRATA_PF_COPY_TIMING=1 STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=1
    done
elif [ "${ARC_PROFILE_REFILL_SWEEP:-0}" = 1 ]; then
    for gpu in dual single; do
        card=(); [ "$gpu" = single ] && card=(ARC_GPUS=0)
        run "$gpu-RF0" agent,2k,10k 128 "${card[@]}" STRATA_PF_PIN_LOAN=1 STRATA_PF_PIN_REFILL=0
        run "$gpu-RF1" agent,2k,10k 128 "${card[@]}" STRATA_PF_PIN_LOAN=1 STRATA_PF_PIN_REFILL=1
    done
elif [ "${ARC_PROFILE_EAGER_PIN_SWEEP:-0}" = 1 ]; then
    for gpu in dual single; do
        card=(); [ "$gpu" = single ] && card=(ARC_GPUS=0)
        run "$gpu-P0" agent,2k,10k 128 "${card[@]}" STRATA_PF_PIN_LOAN=0 STRATA_PF_PIN_REFILL=0
        run "$gpu-P2R0" agent,2k,10k 128 "${card[@]}" STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=0
        run "$gpu-P2R1" agent,2k,10k 128 "${card[@]}" STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=1
    done
    run A-P2R1 10k 128 STRATA_PF_PIN_LOAN=2 STRATA_PF_PIN_REFILL=1 STRATA_PREFILL_TIMING=1
elif [ "${ARC_PROFILE_PIN_SWEEP:-0}" = 1 ]; then
    run PIN0 agent,2k,10k 128 STRATA_PF_PIN_LOAN=0
    run PIN1 agent,2k,10k 128 STRATA_PF_PIN_LOAN=1
    run G1-PIN0 agent,2k,10k 128 ARC_GPUS=0 STRATA_PF_PIN_LOAN=0
    run G1-PIN1 agent,2k,10k 128 ARC_GPUS=0 STRATA_PF_PIN_LOAN=1
    run A-PIN0 10k 128 STRATA_PF_PIN_LOAN=0 STRATA_PREFILL_TIMING=1
    run A-PIN1 10k 128 STRATA_PF_PIN_LOAN=1 STRATA_PREFILL_TIMING=1
elif [ "${ARC_PROFILE_EARLY_SWEEP:-0}" = 1 ]; then
    run EP0 2k,10k 128 STRATA_PF_PREFETCH_SLOTS=0
    run EP64 2k,10k 128 STRATA_PF_PREFETCH_SLOTS=64
    run EP128 2k,10k 128 STRATA_PF_PREFETCH_SLOTS=128
    run EP256 2k,10k 128 STRATA_PF_PREFETCH_SLOTS=256
    run G1-EP0 2k,10k 128 ARC_GPUS=0 STRATA_PF_PREFETCH_SLOTS=0
    run G1-EP128 2k,10k 128 ARC_GPUS=0 STRATA_PF_PREFETCH_SLOTS=128
    run A-EP128 10k 128 STRATA_PF_PREFETCH_SLOTS=128 STRATA_PREFILL_TIMING=1
elif [ "${ARC_PROFILE_PIPELINE_SWEEP:-0}" = 1 ]; then
    run PIPE0 10k 128 STRATA_PF_PIPELINE=0
    run PIPE1 10k 128 STRATA_PF_PIPELINE=1
    run PIPE4K 10k 128 STRATA_PF_PIPELINE=1 "STRATA_EXTRA_ARGS=$ARC_ENGINE_ARGS_DEFAULT --prefill 4096"
    run G1-PIPE0 10k 128 ARC_GPUS=0 STRATA_PF_PIPELINE=0
    run G1-PIPE1 10k 128 ARC_GPUS=0 STRATA_PF_PIPELINE=1
    for mode in PIPE G1-PIPE; do
        if ! python3 "$ROOT/tools/compare-prefill-response.py" "$out.${mode}0.parity.json" "$out.${mode}1.parity.json" >>"$out" 2>&1; then failures=$((failures + 1)); fi
    done
elif [ "${ARC_PROFILE_ATTN_SWEEP:-0}" = 1 ]; then
    run AT0 2k,10k 128 STRATA_PROMPT_ATTN_XMX=0
    run AT16 2k,10k 128 STRATA_PROMPT_ATTN_XMX=16
    run AT32 2k,10k 128 STRATA_PROMPT_ATTN_XMX=32
    run G1-AT0 2k,10k 128 ARC_GPUS=0 STRATA_PROMPT_ATTN_XMX=0
    run G1-AT16 2k,10k 128 ARC_GPUS=0 STRATA_PROMPT_ATTN_XMX=16
    run G1-AT32 2k,10k 128 ARC_GPUS=0 STRATA_PROMPT_ATTN_XMX=32
    run A-AT0 10k 128 STRATA_PROMPT_ATTN_XMX=0 STRATA_PREFILL_TIMING=1
    run A-AT32 10k 128 STRATA_PROMPT_ATTN_XMX=32 STRATA_PREFILL_TIMING=1
elif [ "${ARC_PROFILE_SPLIT_SWEEP:-0}" = 1 ]; then
    for nr in 4 8 16 auto; do
        run "SP$nr" agent,1k,2k,10k 128 "STRATA_PF_SPLIT_ROWS=$nr"
    done
    run G1-SP4 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_SPLIT_ROWS=4
    run G1-SP8 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_SPLIT_ROWS=8
    run G1-SPauto agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_SPLIT_ROWS=auto
    run G1-SP16 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_SPLIT_ROWS=16
elif [ "${ARC_PROFILE_FUSED_SWEEP:-0}" = 1 ]; then
    for nr in 0 4 8 16; do
        run "FX$nr" agent,2k,10k 128 "STRATA_PF_FUSE_EXPERT=$nr"
    done
    run G1-FX0 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_FUSE_EXPERT=0
    run G1-FX4 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_FUSE_EXPERT=4
    run G1-FX8 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_FUSE_EXPERT=8
    run G1-FX16 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_FUSE_EXPERT=16
elif [ "${ARC_PROFILE_CARD_SWEEP:-0}" = 1 ]; then
    BASE_ARGS="$ARC_ENGINE_ARGS_DEFAULT"
    for gpu in dual single; do
        card=(); [ "$gpu" = single ] && card=(ARC_GPUS=0)
        # 64 and 256 select the same batched path for all cases here (minimum 311 tokens).
        for sr in 64 768; do
            run "$gpu-SR$sr" agent,1k,2k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$BASE_ARGS --short-read $sr"
        done
        run "$gpu-PF4K" 2k,10k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$BASE_ARGS --prefill 4096"
        run "$gpu-NB" 2k,10k 128 "${card[@]}" "STRATA_EXTRA_ARGS=$BASE_ARGS --prefill 2048 --no-prefill-borrow"
        for spec in 2 4 6 8; do
            run "$gpu-SPEC$spec" short,prose 128 "${card[@]}" "STRATA_EXTRA_ARGS=$BASE_ARGS --spec $spec"
        done
    done
elif [ "${ARC_PROFILE_PCIE_SWEEP:-0}" = 1 ]; then
    run PC0 short,prose,2k,10k 128 STRATA_PCIE_PINNED_PROBE=0
    run PC1 short,prose,2k,10k 128 STRATA_PCIE_PINNED_PROBE=1
    run G1-PC0 short,prose,2k,10k 128 ARC_GPUS=0 STRATA_PCIE_PINNED_PROBE=0
    run G1-PC1 short,prose,2k,10k 128 ARC_GPUS=0 STRATA_PCIE_PINNED_PROBE=1
    run G1-PF4 2k,10k 128 ARC_GPUS=0 STRATA_PCIE_PINNED_PROBE=0 STRATA_PF_EXPERT_KERNEL=dp4a
    run G1-XM128 2k,10k 128 ARC_GPUS=0 STRATA_PCIE_PINNED_PROBE=0 STRATA_PF_EXPERT_KERNEL=xmx STRATA_PF_XMX_GRF=128
    run A-PC1 10k 128 STRATA_PCIE_PINNED_PROBE=1 STRATA_PREFILL_TIMING=1
elif [ "${ARC_PROFILE_COPY_SWEEP:-0}" = 1 ]; then
    run CP2 2k,10k 128 STRATA_PF_STAGE_BANKS=2 STRATA_PF_STAGE_SLOTS=32
    run CP3 2k,10k 128 STRATA_PF_STAGE_BANKS=3 STRATA_PF_STAGE_SLOTS=32
    run CP3-S16 2k,10k 128 STRATA_PF_STAGE_BANKS=3 STRATA_PF_STAGE_SLOTS=16
    run CP3-S64 2k,10k 128 STRATA_PF_STAGE_BANKS=3 STRATA_PF_STAGE_SLOTS=64
    run CP3-G32 2k,10k 128 STRATA_PF_STAGE_BANKS=3 STRATA_PF_STAGE_SLOTS=32 STRATA_PF_GROUPS=32
    run CP3-R2048 2k,10k 128 STRATA_PF_STAGE_BANKS=3 STRATA_PF_STAGE_SLOTS=32 STRATA_PF_BATCH_ROWS=2048
    run A-CP2 10k 128 STRATA_PF_STAGE_BANKS=2 STRATA_PF_STAGE_SLOTS=32 STRATA_PREFILL_TIMING=1
    run A-CP3 10k 128 STRATA_PF_STAGE_BANKS=3 STRATA_PF_STAGE_SLOTS=32 STRATA_PREFILL_TIMING=1
else
run N short,prose,1k,2k,10k,sampled,agent 128 ARC_PROFILE_RUN=N
if [ "${ARC_PROFILE_FINAL_AB:-0}" = 1 ]; then
    run BASE32 agent,1k,2k,10k 128 STRATA_PF_EXPERT_KERNEL=tile16 STRATA_PF_PIN_LOAN=0 STRATA_PF_PIN_REFILL=0 STRATA_PF_PREFETCH_SLOTS=0 STRATA_PF_PIPELINE=1 STRATA_PROMPT_ATTN_XMX=0 STRATA_PF_SPLIT_ROWS=4 STRATA_PF_FUSE_EXPERT=0 "STRATA_EXTRA_ARGS=--prefill-borrow --prefill auto --vram-reserve-mib 768 --spec 6 --short-read 64"
fi
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
if [ "${ARC_PROFILE_STAGE_AB:-0}" = 1 ]; then run PS1 short,prose 128 STRATA_PLAN_STAGE=1; fi
run PG0 2k,10k 128 STRATA_PF_GROUPED=0
run SR768 1k,2k 128 "STRATA_EXTRA_ARGS=$BASE_ARGS --short-read 768"
run SR64 agent 128 "STRATA_EXTRA_ARGS=$BASE_ARGS --short-read 64"
if [ "$(arc_bdfs | wc -l)" -ge 2 ]; then   # the same box on one card: the single-GPU mode's speeds
    run G1 short,prose,1k,2k,10k,sampled,agent 128 ARC_GPUS=0
    if [ "${ARC_PROFILE_FINAL_AB:-0}" = 1 ]; then
        run G1-BASE32 agent,1k,2k,10k 128 ARC_GPUS=0 STRATA_PF_EXPERT_KERNEL=tile16 STRATA_PF_PIN_LOAN=0 STRATA_PF_PIN_REFILL=0 STRATA_PF_PREFETCH_SLOTS=0 STRATA_PF_PIPELINE=1 STRATA_PROMPT_ATTN_XMX=0 STRATA_PF_SPLIT_ROWS=4 STRATA_PF_FUSE_EXPERT=0 "STRATA_EXTRA_ARGS=--prefill-borrow --prefill auto --vram-reserve-mib 768 --spec 6 --short-read 64"
    fi
fi
run A 10k 128 STRATA_PREFILL_TIMING=1 STRATA_DRAFT_TIMING=1
if [ "$(arc_bdfs | wc -l)" -ge 2 ]; then run G1-A 10k 128 ARC_GPUS=0 STRATA_PREFILL_TIMING=1 STRATA_DRAFT_TIMING=1; fi
if [ -n "$unitrace" ] && [ "${ARC_PROFILE_TRACE:-0}" = 1 ]; then
    run D short,prose 128 STRATA_VERIFY_EAGER=1 STRATA_EXIT_SIGNAL=10 "STRATA_WRAP=$unitrace -d --teardown-on-signal 10 -o $out.D.unitrace"
    run E 10k 1 STRATA_EXIT_SIGNAL=10 "STRATA_WRAP=$unitrace -d --teardown-on-signal 10 -o $out.E.unitrace"
    for f in "$out".D*.unitrace* "$out".E*.unitrace*; do
        [ -f "$f" ] || continue
        { echo "================ $(basename "$f")"; head -c 60000 "$f"; echo; } >>"$out"
        trace_pid="${f%.unitrace}"; trace_pid="${trace_pid##*.}"
        case "$trace_pid" in ""|*[!0-9]*) continue ;; esac
        timeline="$STRATA/strata.$trace_pid.json"
        if [ -f "$timeline" ]; then
            python3 "$ROOT/tools/analyze-decode-trace.py" "$timeline" >>"$out" 2>&1 || warn "GPU interval analysis unavailable: $timeline"
        fi
    done
fi
fi
trap - EXIT
python3 "$ROOT/tools/profile_summary.py" "$out" || true
rm -f "$out".*.start
if [ "$failures" -gt 0 ]; then
    warn "$failures profile run(s) failed reply checks: $out"
    exit 3
fi
ok "written: $out (send this file)"
