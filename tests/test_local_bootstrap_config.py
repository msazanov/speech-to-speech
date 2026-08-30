from pathlib import Path

from speech_to_speech.s2s_pipeline import parse_arguments


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "omniroute-ru-en.json"


def parsed_profile():
    return parse_arguments([str(PROFILE)], command="serve")


def test_profile_routes_chat_completions_to_loopback_omniroute():
    parsed = parsed_profile()

    assert parsed.module_kwargs.llm_backend == "chat-completions"
    assert parsed.llm_backend.config["model_name"] == "auto/chat"
    assert parsed.llm_backend.config["base_url"] == "http://127.0.0.1:20128/v1"
    assert parsed.llm_backend.config["api_key"] is None
    assert parsed.realtime_server_kwargs.host == "127.0.0.1"
    assert parsed.realtime_server_kwargs.port == 8765


def test_profile_uses_local_ru_en_speech_backends():
    parsed = parsed_profile()

    assert parsed.module_kwargs.stt == "parakeet-tdt"
    assert parsed.stt_backend.config["device"] == "cpu"
    assert parsed.stt_backend.config["compute_type"] == "float32"
    assert parsed.stt_backend.config["language"] == "auto"
    assert parsed.llm_backend.config["enable_lang_prompt"] is True
    assert parsed.module_kwargs.tts == "qwen3"
    assert parsed.tts_backend.config["device"] == "cpu"
    assert parsed.tts_backend.config["backend"] == "ggml"
    assert parsed.tts_backend.config["language"] == "auto"


def test_profile_prompt_prefers_russian_without_forcing_english_terms():
    prompt = parsed_profile().llm_backend.config["init_chat_prompt"]

    assert "По умолчанию отвечай по-русски" in prompt
    assert "явно говорит по-английски" in prompt
    assert "термин" in prompt
