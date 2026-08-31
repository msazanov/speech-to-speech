#!/usr/bin/env bash
set -euo pipefail

systemctl_bin="${SYSTEMCTL_BIN:-systemctl}"
systemd_user_dir="${SYSTEMD_USER_DIR:-/home/random/.config/systemd/user}"
huggingvoice_unit_source="${HUGGINGVOICE_UNIT_SOURCE:-/home/random/dev/huggingvoice/deploy/systemd/huggingvoice.service}"

mkdir -p "$systemd_user_dir"
ln -sfn "$huggingvoice_unit_source" "${systemd_user_dir}/huggingvoice.service"
"$systemctl_bin" --user daemon-reload
"$systemctl_bin" --user enable --now huggingvoice.service
