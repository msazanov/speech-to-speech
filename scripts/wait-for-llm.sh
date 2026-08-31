#!/usr/bin/env bash
set -euo pipefail

base_url="${1:-http://127.0.0.1:1919/v1}"
model_id="${2:-gemma-4-e2b}"
attempts="${LLM_WAIT_ATTEMPTS:-120}"

for ((attempt = 1; attempt <= attempts; attempt++)); do
  if curl --fail --silent --max-time 2 "${base_url}/models" | grep --fixed-strings --quiet "${model_id}"; then
    exit 0
  fi
  sleep 1
done

echo "LLM ${model_id} did not become ready at ${base_url}" >&2
exit 1
