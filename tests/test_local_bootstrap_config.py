import os
import subprocess
from pathlib import Path

from speech_to_speech.s2s_pipeline import parse_arguments

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "omniroute-ru-en.json"


def parsed_profile():
    return parse_arguments([str(PROFILE)], command="serve")


def test_profile_routes_chat_completions_to_local_gemma():
    parsed = parsed_profile()

    assert parsed.module_kwargs.llm_backend == "chat-completions"
    assert parsed.llm_backend.config["model_name"] == "gemma4-e2b-q4_0"
    assert parsed.llm_backend.config["base_url"] == "http://127.0.0.1:1919/v1"
    assert parsed.llm_backend.config["api_key"] == "local"
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
    assert parsed.module_kwargs.enable_live_transcription is False
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


def test_voice_stack_switches_services_without_leaving_ornith_enabled(tmp_path):
    systemd_user_dir = tmp_path / "systemd-user"
    systemd_user_dir.mkdir()
    ornith_unit_source = tmp_path / "freetoken-ornith.source.service"
    ornith_unit_source.write_text("[Service]\nExecStart=/bin/true\n")
    gemma_unit_source = tmp_path / "huggingvoice-gemma.source.service"
    gemma_unit_source.write_text("[Service]\nExecStart=/bin/true\n")
    huggingvoice_unit_source = tmp_path / "huggingvoice.source.service"
    huggingvoice_unit_source.write_text("[Service]\nExecStart=/bin/true\n")
    (systemd_user_dir / "freetoken-ornith.service").symlink_to(ornith_unit_source)
    (systemd_user_dir / "huggingvoice-gemma.service").symlink_to(gemma_unit_source)
    (systemd_user_dir / "huggingvoice.service").symlink_to(huggingvoice_unit_source)
    recorder = tmp_path / "record-systemctl"
    recorder.write_text(
        '#!/usr/bin/env bash\nprintf "ARG=%s\\n" "$@"\n'
        'if [[ "$*" == *" stop "* && ! -e "$SYSTEMD_USER_DIR/${@: -1}" ]]; then\n'
        '  printf "cannot stop missing unit\\n" >&2\n'
        "  exit 1\n"
        "fi\n"
        'if [[ "$*" == *" disable --now "* ]]; then\n'
        '  rm -f "$SYSTEMD_USER_DIR/${@: -1}"\n'
        '  printf "disable --now removed linked unit before stop\\n" >&2\n'
        "  exit 1\n"
        "fi\n"
        'if [[ "$*" == *" disable "* ]]; then\n'
        '  rm -f "$SYSTEMD_USER_DIR/${@: -1}"\n'
        "fi\n"
    )
    recorder.chmod(0o755)
    env = {
        **os.environ,
        "SYSTEMCTL_BIN": str(recorder),
        "SYSTEMD_USER_DIR": str(systemd_user_dir),
        "ORNITH_UNIT_SOURCE": str(ornith_unit_source),
        "GEMMA_UNIT_SOURCE": str(gemma_unit_source),
        "HUGGINGVOICE_UNIT_SOURCE": str(huggingvoice_unit_source),
    }

    activated = subprocess.run(
        [ROOT / "scripts" / "activate-voice-stack.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    restored = subprocess.run(
        [ROOT / "scripts" / "restore-ornith.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    restored_link = systemd_user_dir / "freetoken-ornith.service"
    assert restored_link.is_symlink()
    assert restored_link.resolve() == ornith_unit_source
    assert (systemd_user_dir / "huggingvoice-gemma.service").resolve() == gemma_unit_source
    assert (systemd_user_dir / "huggingvoice.service").resolve() == huggingvoice_unit_source

    assert activated.stdout.splitlines() == [
        "ARG=--user",
        "ARG=daemon-reload",
        "ARG=--user",
        "ARG=stop",
        "ARG=freetoken-ornith.service",
        "ARG=--user",
        "ARG=disable",
        "ARG=freetoken-ornith.service",
        "ARG=--user",
        "ARG=daemon-reload",
        "ARG=--user",
        "ARG=enable",
        "ARG=--now",
        "ARG=huggingvoice-gemma.service",
        "ARG=--user",
        "ARG=enable",
        "ARG=--now",
        "ARG=huggingvoice.service",
    ]
    assert restored.stdout.splitlines() == [
        "ARG=--user",
        "ARG=stop",
        "ARG=huggingvoice.service",
        "ARG=--user",
        "ARG=stop",
        "ARG=huggingvoice-gemma.service",
        "ARG=--user",
        "ARG=disable",
        "ARG=huggingvoice.service",
        "ARG=--user",
        "ARG=disable",
        "ARG=huggingvoice-gemma.service",
        "ARG=--user",
        "ARG=daemon-reload",
        "ARG=--user",
        "ARG=enable",
        "ARG=--now",
        "ARG=freetoken-ornith.service",
    ]
