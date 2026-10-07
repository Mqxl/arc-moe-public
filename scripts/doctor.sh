#!/usr/bin/env bash
# ./arc doctor - everything that decides whether 2x A770 reach 30-40 tok/s, checked without changing anything.
# Exit status: 0 all good, 1 something blocks a run, 2 only warnings.
set -uo pipefail
. "$(dirname "$0")/lib.sh"
fail=0; warns=0
blk() { bad "$*"; fail=1; }
wrn() { warn "$*"; warns=1; }

step "System"
. /etc/os-release 2>/dev/null || true
say "  ${PRETTY_NAME:-unknown Linux}, kernel $(uname -r)"
case "${VERSION_ID:-}" in 24.04|24.10|25.04|25.10|26.04) ok "Ubuntu ${VERSION_ID}" ;; 22.04) wrn "Ubuntu 22.04 works, 24.04+ has the newer GPU stack" ;; *) wrn "not an Ubuntu release this was written for" ;; esac
cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')
say "  CPU: $cpu ($(nproc) threads)"
grep -q ' avx2 ' /proc/cpuinfo && ok "AVX2" || wrn "no AVX2: the image encoder and the CPU helpers run slowly"
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")
[ "$gov" = performance ] && ok "CPU governor: performance" || wrn "CPU governor: $gov (the host side of every decode window is latency-bound: sudo cpupower frequency-set -g performance)"

step "Memory and disk"
mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo); av_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
mem=$((mem_kb / 1048576)); av=$((av_kb / 1048576))
say "  RAM: ${mem} GiB total, ${av} GiB available"
if [ "$mem" -ge 44 ]; then ok "RAM holds the IQ2_XS overflow mirror (~10 GiB pinned) beside the system"
elif [ "$mem" -ge 30 ]; then wrn "RAM ${mem} GiB: IQ2_XS will mirror fewer experts; the Coder (IQ1_M) fits VRAM instead"
else blk "RAM ${mem} GiB is too little for the pinned mirror"; fi
ml=$(ulimit -l)
[ "$ml" = unlimited ] && ok "locked-memory limit: unlimited" || say "  ${D}locked-memory limit: ${ml} KiB (Level Zero pins through the driver; only matters if pinning fails)${N}"
df -BG --output=avail,target "$ROOT" 2>/dev/null | tail -1 | while read -r a t; do
    a=${a%G}; [ "$a" -ge 80 ] && ok "disk: ${a} GB free on $t" || wrn "disk: ${a} GB free on $t (a model with its lookup table needs ~70 GB; use --gguf-dir for files you have)"; done

step "GPUs"
mapfile -t bdfs < <(arc_bdfs)
[ ${#bdfs[@]} -eq 0 ] && blk "no Intel discrete GPU on the PCI bus"
n_a770=0
for b in "${bdfs[@]}"; do
    d=/sys/bus/pci/devices/$b
    id=$(cat "$d/device"); id=${id#0x}
    drv=$(basename "$(readlink -f "$d/driver" 2>/dev/null)" 2>/dev/null || echo none)
    name="Intel GPU $id"
    case "$id" in 56a0) name="Arc A770"; n_a770=$((n_a770 + 1)) ;; 56a1) name="Arc A750" ;; 56a2) name="Arc A580" ;; 56a5|56a6) name="Arc A380/A310" ;; esac
    say "  ${B}$b${N}  $name [8086:$id]  driver: $drv"
    case "$drv" in i915|xe) ok "kernel driver $drv" ;; *) blk "no i915/xe driver bound (kernel too old, or i915.force_probe=$id needed)" ;; esac
    if [ "$drv" = i915 ]; then
        # the engine sizes its caches from the VRAM it holds per /proc/self/fdinfo (drm-total-local*): i915 hides
        # the free size from unprivileged processes, and prints per-client memory stats only on newer kernels
        kv=$(uname -r | cut -d. -f1-2); kmaj=${kv%%.*}; kmin=${kv#*.}; kmin=${kmin%%[^0-9]*}
        if [ "$kmaj" -gt 6 ] || { [ "$kmaj" -eq 6 ] && [ "${kmin:-0}" -ge 8 ]; }; then ok "kernel $kv: i915 per-process VRAM stats (fdinfo)"
        else wrn "kernel $kv: i915 may not print per-process VRAM stats - the engine then cannot see what it holds (./arc probe shows it; 6.8+ recommended)"; fi
    fi
    # ReBAR: BAR2 is the VRAM aperture; without Resizable BAR it is 256 MiB and every upload goes through a window
    bar=0
    if [ -r "$d/resource" ]; then
        read -r s e _ < <(sed -n 3p "$d/resource")
        [ -n "${s:-}" ] && bar=$(( (e - s + 1) / 1048576 ))
    fi
    if [ "$bar" -ge 8192 ]; then ok "Resizable BAR on (VRAM aperture $((bar / 1024)) GiB)"
    elif [ "$bar" -gt 0 ]; then blk "Resizable BAR OFF (aperture ${bar} MiB): enable 'Above 4G decoding' + 'Re-Size BAR' in the BIOS - A-series cards lose much of their speed without it"
    else wrn "could not read the BAR size"; fi
    # PCIe: the card's own functions sit behind an internal switch that reports x1; the slot's link is further up
    best=""; p=$(readlink -f "$d")
    while [ "$p" != / ] && [ "$p" != /sys/devices ]; do
        p=$(dirname "$p")
        [ -r "$p/current_link_width" ] || continue
        w=$(cat "$p/current_link_width" 2>/dev/null); sp=$(cat "$p/current_link_speed" 2>/dev/null | cut -d' ' -f1)
        [ "${w:-1}" -gt 1 ] 2>/dev/null && best="Gen ${sp} GT/s x${w}"
    done
    if [ -n "$best" ]; then
        case "$best" in *"16.0 GT/s x16") ok "PCIe link: $best (4.0 x16: the mirror reads run at ~25 GB/s)" ;;
            *x16|*"16.0 GT/s x8") ok "PCIe link: $best" ;;
            *) wrn "PCIe link: $best - the experts mirrored in RAM are read over this link; x8 Gen4 or better keeps decode near the top of the range" ;; esac
    else say "  ${D}PCIe link: not readable here${N}"; fi
done
if [ "$n_a770" -ge 2 ]; then ok "${n_a770}x Arc A770: the layers are split across them (32 GB of VRAM together)"
elif [ "$n_a770" -eq 1 ]; then wrn "one A770: IQ2_XS mirrors more than half of its experts in RAM (slower); the Coder fits better"; fi
if [ "$(id -u)" -eq 0 ]; then ok "running as root (/dev/dri is open to it)"
elif id -nG | tr ' ' '\n' | grep -qx render; then ok "user $(id -un) is in the render group"
else blk "user $(id -un) is not in the 'render' group: sudo usermod -aG render,video $(id -un), then log in again"; fi

step "Compute runtime and oneAPI"
if [ -f "$ONEAPI/setvars.sh" ]; then
    oneapi_env
    ok "oneAPI: $(icpx --version 2>/dev/null | head -1 || echo 'runtime only')"
    command -v ocloc >/dev/null && ok "ocloc (ahead-of-time compiler for acm-g10)" || wrn "no ocloc: the engine builds as SPIR-V and compiles every kernel at its first start (~1 min)"
    if command -v sycl-ls >/dev/null; then
        lz=$(sycl-ls 2>/dev/null | grep -c 'level_zero:gpu' || true)
        sycl-ls 2>/dev/null | grep -E 'level_zero:gpu' | sed 's/^/    /'
        if [ "$lz" -ge 2 ]; then ok "Level Zero sees $lz GPUs"
        elif [ "$lz" -eq 1 ]; then wrn "Level Zero sees one GPU"
        else blk "Level Zero sees no GPU (libze-intel-gpu1 / intel-level-zero-gpu missing, or not in the render group)"; fi
    fi
else
    blk "oneAPI not installed ($ONEAPI): ./arc deps"
fi

step "Engine"
if [ -x "$BUILD_DIR/strata" ]; then ok "engine: $BUILD_DIR/strata"
else wrn "engine not built yet: ./arc build"; fi
cfgs=$(ls "$STRATA"/strata-*.json 2>/dev/null | head -3)
[ -n "$cfgs" ] && ok "configured: $(echo "$cfgs" | xargs -n1 basename | tr '\n' ' ')" || say "  ${D}no model configured yet: ./arc setup${N}"

echo
if [ $fail -ne 0 ]; then say "${R}${B}Something above blocks a run.${N}"; exit 1; fi
if [ $warns -ne 0 ]; then say "${Y}${B}Runs; the warnings above cost speed.${N}"; exit 2; fi
say "${G}${B}All good.${N}"
