#!/usr/bin/env bash
set -euo pipefail

url="${1:-http://127.0.0.1:8765}"
attempts="${SPEECH_WAIT_ATTEMPTS:-180}"

for ((attempt = 1; attempt <= attempts; attempt++)); do
  if curl --silent --output /dev/null --max-time 2 "${url}/"; then
    exit 0
  fi
  sleep 1
done

echo "Speech service did not become ready at ${url}" >&2
exit 1
