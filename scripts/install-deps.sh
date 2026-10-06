#!/usr/bin/env bash
# ./arc deps - the GPU compute runtime, oneAPI (DPC++ compiler + oneMKL), ocloc and the build tools. Ubuntu 22.04+.
# Safe to run again: apt skips what is installed.
#   ARC_ONEAPI_VERSION=2026.1   pin the oneAPI release (default: the newest in Intel's repository)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
. /etc/os-release
[ "${ID:-}" = ubuntu ] || warn "written for Ubuntu; on ${PRETTY_NAME:-this system} install the same packages by hand"
export DEBIAN_FRONTEND=noninteractive

step "Build tools and Python"
sudo_ apt-get update -qq
sudo_ apt-get install -y -qq build-essential cmake ninja-build git git-lfs pkg-config curl wget gpg ca-certificates \
    python3 python3-venv python3-pip pciutils software-properties-common libcurl4-openssl-dev >/dev/null
ok "compilers, cmake, ninja, git, python3-venv"

step "Intel GPU compute runtime (Level Zero + OpenCL + ocloc)"
case "${VERSION_ID:-}" in
  22.04)
    # Intel's client GPU repository for jammy
    wget -qO- https://repositories.intel.com/gpu/intel-graphics.key | sudo_ gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" \
        | sudo_ tee /etc/apt/sources.list.d/intel-gpu-jammy.list >/dev/null
    sudo_ apt-get update -qq
    sudo_ apt-get install -y -qq intel-opencl-icd intel-level-zero-gpu level-zero intel-ocloc clinfo >/dev/null ;;
  *)
    # 24.04 and later: Intel's documented route for Arc is the kobuk-team PPA (dgpu-docs.intel.com, client GPUs)
    sudo_ add-apt-repository -y ppa:kobuk-team/intel-graphics >/dev/null
    sudo_ apt-get update -qq
    sudo_ apt-get install -y -qq libze-intel-gpu1 libze1 intel-opencl-icd intel-ocloc clinfo intel-gsc >/dev/null
    sudo_ apt-get install -y -qq intel-metrics-discovery >/dev/null 2>&1 || true ;;
esac
ok "Level Zero GPU driver, OpenCL ICD, ocloc"

step "Intel oneAPI: DPC++/C++ compiler and oneMKL (~5 GB)"
if [ ! -f /usr/share/keyrings/oneapi-archive-keyring.gpg ]; then
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | sudo_ gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
fi
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | sudo_ tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
sudo_ apt-get update -qq
v=${ARC_ONEAPI_VERSION:+-$ARC_ONEAPI_VERSION}
sudo_ apt-get install -y -qq "intel-oneapi-compiler-dpcpp-cpp$v" "intel-oneapi-mkl-devel$v" >/dev/null
ok "oneAPI in $ONEAPI"

step "Access to the GPUs"
u=${SUDO_USER:-$(id -un)}
if id -nG "$u" | tr ' ' '\n' | grep -qx render; then ok "$u is in the render group"
else sudo_ usermod -aG render,video "$u"; warn "$u added to render/video: log out and in again (or reboot) before ./arc build"; fi

step "Done"
say "  Next: ./arc doctor, then ./arc build"
