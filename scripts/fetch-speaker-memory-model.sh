#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
model_name="3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
model_sha256="357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"
model_url="https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/${model_name}"
model_dir="${HUGGINGVOICE_SPEAKER_MODEL_DIR:-${repo_root}/models/speaker-memory}"
model_path="${model_dir}/${model_name}"

mkdir -p -- "$model_dir"
if [[ -f "$model_path" ]]; then
    existing_sha256="$(sha256sum -- "$model_path" | awk '{print $1}')"
    if [[ "$existing_sha256" == "$model_sha256" ]]; then
        printf 'Speaker model already verified: %s\n' "$model_path"
        exit 0
    fi
    printf 'Existing speaker model has an unexpected checksum; refusing to overwrite: %s\n' "$model_path" >&2
    exit 1
fi

temporary_path="$(mktemp "${model_path}.download.XXXXXX")"
cleanup() {
    rm -f -- "$temporary_path"
}
trap cleanup EXIT

curl --fail --location --retry 2 --output "$temporary_path" "$model_url"
downloaded_sha256="$(sha256sum -- "$temporary_path" | awk '{print $1}')"
if [[ "$downloaded_sha256" != "$model_sha256" ]]; then
    printf 'Speaker model checksum mismatch: expected %s, got %s\n' "$model_sha256" "$downloaded_sha256" >&2
    exit 1
fi

chmod 0644 -- "$temporary_path"
mv -f -- "$temporary_path" "$model_path"
trap - EXIT
printf 'Speaker model installed: %s\n' "$model_path"
