from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path

import pytest

from speech_to_speech.s2s_pipeline import parse_arguments

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "omniroute-ru-en.json"
FETCH_SCRIPT = ROOT / "scripts" / "fetch-speaker-memory-model.sh"
SMOKE_SCRIPT = ROOT / "scripts" / "speaker_memory_smoke.py"
MODEL_NAME = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
MODEL_SHA256 = "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"


def test_russian_profile_keeps_cpu_speaker_memory_gated_until_acoustic_proof() -> None:
    profile = json.loads(PROFILE.read_text())
    parsed = parse_arguments([str(PROFILE)])

    assert profile["speaker_memory_enabled"] is False
    assert profile["speaker_memory_threads"] == 1
    assert profile["speaker_memory_model_path"].endswith(MODEL_NAME)
    assert parsed.speaker_memory_kwargs.speaker_memory_enabled is False
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


def test_smoke_requires_two_recordings_for_each_of_two_people(tmp_path) -> None:
    validate_samples = runpy.run_path(str(SMOKE_SCRIPT))["validate_samples"]
    arkady_1 = tmp_path / "arkady-1.wav"
    arkady_2 = tmp_path / "arkady-2.wav"
    andrey_1 = tmp_path / "andrey-1.wav"
    andrey_2 = tmp_path / "andrey-2.wav"
    arkady_1.write_bytes(b"arkady one")
    arkady_2.write_bytes(b"arkady two")
    andrey_1.write_bytes(b"andrey one")
    andrey_2.write_bytes(b"andrey two")

    with pytest.raises(ValueError, match="at least two recordings"):
        validate_samples([("Аркадий", arkady_1), ("Аркадий", arkady_2), ("Андрей", andrey_1)])

    with pytest.raises(ValueError, match="distinct recordings"):
        validate_samples(
            [
                ("Аркадий", arkady_1),
                ("Аркадий", arkady_1),
                ("Андрей", andrey_1),
                ("Андрей", andrey_2),
            ]
        )

    validate_samples(
        [
            ("Аркадий", arkady_1),
            ("Аркадий", arkady_2),
            ("Андрей", andrey_1),
            ("Андрей", andrey_2),
        ]
    )
