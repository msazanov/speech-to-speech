from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watch_voice_log.py"
SPEC = importlib.util.spec_from_file_location("watch_voice_log", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
watch_voice_log = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch_voice_log)


def test_humanize_labels_pipeline_stages() -> None:
    assert watch_voice_log.humanize("Audio route changed route=aec").startswith("[AUDIO]")
    assert watch_voice_log.humanize("GigaAM ONNX transcription: привет").startswith("[STT]")
    assert watch_voice_log.humanize("LLM response model=gemma").startswith("[LLM]")
    assert watch_voice_log.humanize("Silero TTS first audio").startswith("[TTS]")
    assert watch_voice_log.humanize("Audio rejected reason=empty").startswith("[DROP]")


def test_same_voice_has_stable_color_and_different_voices_differ() -> None:
    first = watch_voice_log.voice_color("v_aaa")

    assert watch_voice_log.voice_color("v_aaa") == first
    assert watch_voice_log.voice_color("v_bbb") != first


def test_palette_resolves_hash_collisions_between_people(monkeypatch) -> None:
    palette = watch_voice_log.VoicePalette((39, 45))
    monkeypatch.setattr(watch_voice_log, "_preferred_color_index", lambda _voice_id, _size: 0)

    assert palette.color("v_first") == 39
    assert palette.color("v_second") == 45
    assert palette.color("v_first") == 39


def test_colorize_marks_voice_and_confirmed_name_with_voice_color() -> None:
    line = "Speaker attributed voice=v_aaa person_id=p_123 person='Марат' state=known"
    rendered = watch_voice_log.colorize(line, color=True)
    voice_escape = f"\x1b[38;5;{watch_voice_log.voice_color('v_aaa')}m"

    assert voice_escape in rendered
    assert "voice=v_aaa" in rendered
    assert "person_id=p_123" in rendered
    assert "person='Марат'" in rendered
    assert rendered.endswith("\x1b[0m")


def test_colorize_can_be_disabled_for_redirected_output() -> None:
    line = "Speaker attributed voice=v_aaa state=unknown"

    assert watch_voice_log.colorize(line, color=False) == "[VOICE] " + line
