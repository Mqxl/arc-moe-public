#!/usr/bin/env bash
# ./arc service [model] - a systemd unit (arc-moe.service) that starts the server at boot as this user.
#   ./arc service --remove   takes it out again
set -euo pipefail
. "$(dirname "$0")/lib.sh"
unit=/etc/systemd/system/arc-moe.service
if [ "${1:-}" = --remove ]; then
    sudo_ systemctl disable --now arc-moe.service 2>/dev/null || true
    sudo_ rm -f "$unit"; sudo_ systemctl daemon-reload
    ok "arc-moe.service removed"; exit 0
fi
u=${SUDO_USER:-$(id -un)}
model=${1:-${ARC_MODEL:-}}
sudo_ tee "$unit" >/dev/null <<EOF
[Unit]
Description=Qwen3.8-Flash-Next on 2x Intel Arc A770 (Strata SYCL, arc-moe)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$u
Group=render
WorkingDirectory=$ROOT
Environment=ARC_MODEL=$model
Environment=STRATA_SYCL_NATIVE=1
ExecStart=$ROOT/scripts/start.sh $model
Restart=on-failure
RestartSec=20
# the engine pins RAM for the experts mirror
LimitMEMLOCK=infinity
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF
sudo_ systemctl daemon-reload
sudo_ systemctl enable --now arc-moe.service
ok "arc-moe.service enabled and started (journalctl -u arc-moe -f)"
