#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
uv_bin="${UV_BIN:-uv}"

cd "$repo_root"
"$uv_bin" sync --locked --extra supertonic --extra silero --extra gigaam --python 3.12
"$uv_bin" run python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
