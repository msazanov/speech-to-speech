from __future__ import annotations

import json
import subprocess
from pathlib import Path

from speech_to_speech.s2s_pipeline import parse_arguments

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "omniroute-ru-en.json"
FETCH_SCRIPT = ROOT / "scripts" / "fetch-speaker-memory-model.sh"
SMOKE_SCRIPT = ROOT / "scripts" / "speaker_memory_smoke.py"
MODEL_NAME = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
MODEL_SHA256 = "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"


def test_russian_profile_enables_cpu_speaker_memory() -> None:
    profile = json.loads(PROFILE.read_text())
    parsed = parse_arguments([str(PROFILE)])

    assert profile["speaker_memory_enabled"] is True
    assert profile["speaker_memory_threads"] == 1
    assert profile["speaker_memory_model_path"].endswith(MODEL_NAME)
    assert parsed.speaker_memory_kwargs.speaker_memory_enabled is True
    assert parsed.speaker_memory_kwargs.speaker_memory_threads == 1
    assert parsed.speaker_memory_kwargs.speaker_memory_min_audio_ms == 700


def test_fetch_script_pins_hugging_face_artifact_and_checksum() -> None:
    source = FETCH_SCRIPT.read_text()

    assert "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/" in source
    assert MODEL_NAME in source
    assert MODEL_SHA256 in source
    assert "mktemp" in source
    assert "sha256sum" in source
    assert "mv -f" in source
    subprocess.run(["bash", "-n", str(FETCH_SCRIPT)], check=True)


def test_smoke_script_is_source_importable_without_optional_sherpa() -> None:
    completed = subprocess.run(
        ["python", str(SMOKE_SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--sample" in completed.stdout
    assert "--model" in completed.stdout
    assert "--json" in completed.stdout
