#!/usr/bin/env bash
set -euo pipefail

systemctl_bin="${SYSTEMCTL_BIN:-systemctl}"
systemd_user_dir="${SYSTEMD_USER_DIR:-/home/random/.config/systemd/user}"
ornith_unit_source="${ORNITH_UNIT_SOURCE:-/home/random/dev/qwen/freetoken/deploy/systemd/freetoken-ornith.service}"

ln -sfn "$ornith_unit_source" "${systemd_user_dir}/freetoken-ornith.service"
"$systemctl_bin" --user daemon-reload
"$systemctl_bin" --user stop freetoken-ornith.service
"$systemctl_bin" --user disable freetoken-ornith.service
ln -sfn "$ornith_unit_source" "${systemd_user_dir}/freetoken-ornith.service"
"$systemctl_bin" --user daemon-reload
"$systemctl_bin" --user enable --now huggingvoice-gemma.service
"$systemctl_bin" --user enable --now huggingvoice.service
