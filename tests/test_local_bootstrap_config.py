import json
import os
import subprocess
from configparser import ConfigParser
from pathlib import Path

from speech_to_speech.s2s_pipeline import parse_arguments

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "omniroute-ru-en.json"


def test_local_profile_does_not_persist_conversation_text_without_explicit_opt_in() -> None:
    profile = json.loads(PROFILE.read_text())

    assert profile.get("log_transcripts", False) is False
HUGGINGVOICE_UNIT = ROOT / "deploy" / "systemd" / "huggingvoice.service"
HUGGINGVOICE_WEB_UNIT = ROOT / "deploy" / "systemd" / "huggingvoice-web.service"


def parsed_profile():
    return parse_arguments([str(PROFILE)], command="serve")


def test_profile_routes_chat_completions_to_local_gemma():
    parsed = parsed_profile()

    assert parsed.module_kwargs.llm_backend == "chat-completions"
    assert parsed.llm_backend.config["model_name"] == "gemma-4-e2b"
    assert parsed.llm_backend.config["base_url"] == "http://127.0.0.1:1919/v1"
    assert parsed.llm_backend.config["api_key"] == "local"
    assert parsed.llm_backend.config["request_timeout_s"] == 120.0
    assert parsed.llm_backend.config["max_retries"] == 0
    assert parsed.llm_backend.config["disable_thinking"] is True
    assert parsed.llm_backend.config["reasoning_effort"] is None
    assert parsed.llm_backend.config["gen_kwargs"] == {"max_tokens": 64, "temperature": 0.2}
    assert parsed.realtime_server_kwargs.host == "127.0.0.1"
    assert parsed.realtime_server_kwargs.port == 8765


def test_profile_uses_local_ru_en_speech_backends():
    parsed = parsed_profile()

    assert parsed.module_kwargs.stt == "gigaam-onnx"
    assert parsed.stt_backend.config["model_name"] == "gigaam-multilingual-ctc"
    assert parsed.stt_backend.config["quantization"] == "int8"
    assert parsed.stt_backend.config["provider"] == "CPUExecutionProvider"
    assert parsed.stt_backend.config["threads"] == 6
    assert parsed.stt_backend.config["language"] == "auto"
    assert parsed.module_kwargs.enable_live_transcription is True
    assert parsed.module_kwargs.live_transcription_update_interval == 0.25
    assert parsed.vad_handler_kwargs.min_silence_ms == 500
    assert parsed.llm_backend.config["enable_lang_prompt"] is True
    assert parsed.module_kwargs.tts == "silero"
    assert parsed.tts_backend.config["speaker"] == "xenia"
    assert parsed.tts_backend.config["sample_rate"] == 24000
    assert parsed.tts_backend.config["threads"] == 6
    assert parsed.tts_backend.config["english_fallback"] is True


def test_profile_prompt_prefers_russian_without_forcing_english_terms():
    prompt = parsed_profile().llm_backend.config["init_chat_prompt"]

    assert "По умолчанию отвечай по-русски" in prompt
    assert "явно говорит по-английски" in prompt
    assert "термин" in prompt
    assert "вопросительные и восклицательные знаки" in prompt
    assert "ха-ха" in prompt
    assert "шёпот" in prompt


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
        "ARG=silero",
        "ARG=--extra",
        "ARG=gigaam",
        "ARG=--extra",
        "ARG=speaker-memory",
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


def test_huggingvoice_unit_uses_arbiter_without_owning_an_llm_service():
    unit = ConfigParser(interpolation=None, strict=False)
    unit.read(HUGGINGVOICE_UNIT)

    assert unit["Unit"].get("Requires") is None
    assert "huggingvoice-gemma.service" not in unit["Unit"].get("After", "")
    assert unit["Service"]["ExecStartPre"].split()[-2:] == [
        "http://127.0.0.1:1919/v1",
        "gemma-4-e2b",
    ]


def test_web_unit_pins_local_realtime_and_loads_private_search_environment():
    unit = ConfigParser(interpolation=None, strict=False)
    unit.read(HUGGINGVOICE_WEB_UNIT)

    assert unit["Unit"]["Requires"] == "huggingvoice.service"
    assert unit["Service"]["EnvironmentFile"] == "-%h/.config/huggingvoice/web.env"
    assert "SPEECH_TO_SPEECH_URL=ws://127.0.0.1:8765/v1/realtime" in unit["Service"].get(
        "Environment", ""
    )
    assert "SERPER" not in HUGGINGVOICE_WEB_UNIT.read_text()
    assert "--host 127.0.0.1 --port 7860" in unit["Service"]["ExecStart"]


def test_wait_for_llm_defaults_to_arbiter_gemma_model(tmp_path):
    curl = tmp_path / "curl"
    curl.write_text('#!/usr/bin/env bash\nprintf \'%s\' \'{"data":[{"id":"gemma-4-e2b"}]}\'\n')
    curl.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "LLM_WAIT_ATTEMPTS": "1",
    }

    completed = subprocess.run(
        [ROOT / "scripts" / "wait-for-llm.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_voice_stack_activation_manages_only_huggingvoice(tmp_path):
    systemd_user_dir = tmp_path / "systemd-user"
    systemd_user_dir.mkdir()
    huggingvoice_unit_source = tmp_path / "huggingvoice.source.service"
    huggingvoice_unit_source.write_text("[Service]\nExecStart=/bin/true\n")
    recorder = tmp_path / "record-systemctl"
    recorder.write_text('#!/usr/bin/env bash\nprintf "ARG=%s\\n" "$@"\n')
    recorder.chmod(0o755)
    env = {
        **os.environ,
        "SYSTEMCTL_BIN": str(recorder),
        "SYSTEMD_USER_DIR": str(systemd_user_dir),
        "HUGGINGVOICE_UNIT_SOURCE": str(huggingvoice_unit_source),
    }

    activated = subprocess.run(
        [ROOT / "scripts" / "activate-voice-stack.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert (systemd_user_dir / "huggingvoice.service").resolve() == huggingvoice_unit_source
    assert not (systemd_user_dir / "huggingvoice-gemma.service").exists()
    assert not (systemd_user_dir / "freetoken-ornith.service").exists()

    assert activated.stdout.splitlines() == [
        "ARG=--user",
        "ARG=daemon-reload",
        "ARG=--user",
        "ARG=enable",
        "ARG=huggingvoice.service",
        "ARG=--user",
        "ARG=restart",
        "ARG=huggingvoice.service",
    ]
