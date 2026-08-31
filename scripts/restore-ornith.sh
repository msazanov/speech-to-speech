#!/usr/bin/env bash
set -euo pipefail

systemctl_bin="${SYSTEMCTL_BIN:-systemctl}"
systemd_user_dir="${SYSTEMD_USER_DIR:-/home/random/.config/systemd/user}"
ornith_unit_source="${ORNITH_UNIT_SOURCE:-/home/random/dev/qwen/freetoken/deploy/systemd/freetoken-ornith.service}"
gemma_unit_source="${GEMMA_UNIT_SOURCE:-/home/random/dev/huggingvoice/deploy/systemd/huggingvoice-gemma.service}"
huggingvoice_unit_source="${HUGGINGVOICE_UNIT_SOURCE:-/home/random/dev/huggingvoice/deploy/systemd/huggingvoice.service}"

"$systemctl_bin" --user stop huggingvoice.service
"$systemctl_bin" --user stop huggingvoice-gemma.service
"$systemctl_bin" --user disable huggingvoice.service
"$systemctl_bin" --user disable huggingvoice-gemma.service
ln -sfn "$huggingvoice_unit_source" "${systemd_user_dir}/huggingvoice.service"
ln -sfn "$gemma_unit_source" "${systemd_user_dir}/huggingvoice-gemma.service"
ln -sfn "$ornith_unit_source" "${systemd_user_dir}/freetoken-ornith.service"
"$systemctl_bin" --user daemon-reload
"$systemctl_bin" --user enable --now freetoken-ornith.service
