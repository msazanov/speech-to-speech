#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
speech_to_speech_bin="${HUGGINGVOICE_BIN:-${repo_root}/.venv/bin/speech-to-speech}"

exec "$speech_to_speech_bin" serve "${repo_root}/config/omniroute-ru-en.json"
