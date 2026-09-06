import sys
import threading
import time

import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS import glados_tts_handler as glados
from speech_to_speech.TTS.local_tts_handler import LocalTTSHandler, parse_local_voice


@pytest.fixture
def handler(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("""
import base64, io, json, os, sys, time, wave
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["text"] == "slow":
        time.sleep(0.4)
    result = {"id": request["id"]}
    if request["text"] == "fail":
        result["error"] = "ValueError"
    else:
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            wav.writeframes(b"\\x80\\x10" * 4410)
        result["wav"] = base64.b64encode(out.getvalue()).decode()
    print(json.dumps(result), flush=True)
""")
    monkeypatch.setattr(glados, "WORKER_PATH", worker)
    h = glados.GladosTTSHandler.__new__(glados.GladosTTSHandler)
    h.setup(threading.Event(), python=sys.executable, workdir=str(tmp_path), timeout=3, cancel_scope=CancelScope())
    yield h
    h.cleanup()


def test_warm_worker_is_reused_across_styles_and_sessions(handler):
    handler.warmup()
    pid = handler._worker.pid
    assert handler._render("first", "Neutral", 0).startswith(b"RIFF")
    handler.on_session_end()
    assert handler._render("second", "Deep", 0).startswith(b"RIFF")
    assert handler._worker.pid == pid


def test_cancel_discards_pending_result_without_reloading_model(handler):
    handler.warmup()
    pid = handler._worker.pid
    timer = threading.Timer(0.08, handler.cancel_scope.cancel)
    timer.start()
    try:
        start = time.monotonic()
        assert handler._render("slow", "Neutral", 0) is None
        assert time.monotonic() - start < 0.3
        handler.cancel_scope.new_response()
        assert handler._render("next", "Light", 1).startswith(b"RIFF")
        assert handler._worker.pid == pid
    finally:
        timer.join()


def test_worker_error_does_not_poison_next_response(handler):
    with pytest.raises(RuntimeError, match="ValueError"):
        handler._render("fail", "Neutral", 0)
    assert handler._render("next", "Neutral", 0).startswith(b"RIFF")


def test_stale_input_does_not_enter_renderer(handler, monkeypatch):
    monkeypatch.setattr(handler, "_render", lambda *args: pytest.fail("Rendered cancelled text"))
    handler.cancel_scope.cancel()
    assert list(handler.process(TTSInput(text="stale", cancel_generation=0))) == []


def test_cleanup_only_completion_and_stale_turn(handler, monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn", 1)
    handler.speculative_turns = tracker
    monkeypatch.setattr(handler, "_render", lambda *args: pytest.fail("Rendered obsolete revision"))
    assert list(handler.process(TTSInput(text="old", turn_id="turn", turn_revision=0))) == []
    end = EndOfResponse(turn_id="turn", turn_revision=0, response_key="old-response")
    assert list(handler.process(end)) == [AUDIO_RESPONSE_DONE]
    assert end.cleanup_only


def test_local_router_exposes_cancellation_to_base_handler(monkeypatch):
    from speech_to_speech.TTS.local_tts_handler import GladosTTSHandler, RHVoiceTTSHandler, SileroTTSHandler

    for backend in (GladosTTSHandler, RHVoiceTTSHandler, SileroTTSHandler):
        monkeypatch.setattr(backend, "setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(GladosTTSHandler, "warmup", lambda self: None)
    scope = CancelScope()
    h = LocalTTSHandler.__new__(LocalTTSHandler)
    h.setup(threading.Event(), cancel_scope=scope)
    scope.cancel()
    assert not h.should_process_input(TTSInput(text="old", cancel_generation=0))


def test_default_sdk_voice_uses_configured_default_backend():
    assert parse_local_voice("Aiden", "glados", "Deep") == ("glados", "Deep")
