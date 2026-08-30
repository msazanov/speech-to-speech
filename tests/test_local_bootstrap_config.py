import os
import subprocess
from pathlib import Path

from speech_to_speech.s2s_pipeline import parse_arguments

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "omniroute-ru-en.json"


def parsed_profile():
    return parse_arguments([str(PROFILE)], command="serve")


def test_profile_routes_chat_completions_to_loopback_omniroute():
    parsed = parsed_profile()

    assert parsed.module_kwargs.llm_backend == "chat-completions"
    assert parsed.llm_backend.config["model_name"] == "kmc/k3-256k"
    assert parsed.llm_backend.config["base_url"] == "http://127.0.0.1:20128/v1"
    assert parsed.llm_backend.config["api_key"] is None
    assert parsed.realtime_server_kwargs.host == "127.0.0.1"
    assert parsed.realtime_server_kwargs.port == 8765


def test_profile_uses_local_ru_en_speech_backends():
    parsed = parsed_profile()

    assert parsed.module_kwargs.stt == "faster-whisper"
    assert parsed.stt_backend.config["device"] == "cpu"
    assert parsed.stt_backend.config["model_name"] == "ammaraldirawi/faster-whisper-small-ru-int8"
    assert parsed.stt_backend.config["compute_type"] == "int8"
    assert parsed.stt_backend.config["gen_kwargs"]["language"] == "auto"
    assert parsed.module_kwargs.enable_live_transcription is False
    assert parsed.vad_handler_kwargs.min_silence_ms == 500
    assert parsed.llm_backend.config["enable_lang_prompt"] is True
    assert parsed.module_kwargs.tts == "supertonic"
    assert parsed.tts_backend.config["voice"] == "M1"
    assert parsed.tts_backend.config["lang"] == "na"


def test_profile_prompt_prefers_russian_without_forcing_english_terms():
    prompt = parsed_profile().llm_backend.config["init_chat_prompt"]

    assert "По умолчанию отвечай по-русски" in prompt
    assert "явно говорит по-английски" in prompt
    assert "термин" in prompt


def make_argv_recorder(tmp_path: Path) -> Path:
    recorder = tmp_path / "record-argv"
    recorder.write_text('#!/usr/bin/env bash\nprintf "ARG=%s\\n" "$@"\n')
    recorder.chmod(0o755)
    return recorder


def test_bootstrap_uses_locked_python_312_environment(tmp_path):
    recorder = make_argv_recorder(tmp_path)
    env = {**os.environ, "UV_BIN": str(recorder)}

    completed = subprocess.run(
        [ROOT / "scripts" / "bootstrap-local.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "ARG=sync",
        "ARG=--locked",
        "ARG=--extra",
        "ARG=supertonic",
        "ARG=--extra",
        "ARG=faster-whisper",
        "ARG=--python",
        "ARG=3.12",
        "ARG=run",
        "ARG=python",
        "ARG=-m",
        "ARG=nltk.downloader",
        "ARG=punkt_tab",
        "ARG=averaged_perceptron_tagger_eng",
    ]


def test_launcher_runs_the_committed_profile(tmp_path):
    recorder = make_argv_recorder(tmp_path)
    env = {**os.environ, "HUGGINGVOICE_BIN": str(recorder)}

    completed = subprocess.run(
        [ROOT / "scripts" / "run-omniroute-ru-en.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "ARG=serve",
        f"ARG={PROFILE}",
    ]
