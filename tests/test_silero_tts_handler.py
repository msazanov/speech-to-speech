from __future__ import annotations

from threading import Event
from typing import Any

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS import silero_tts_handler as silero_module
from speech_to_speech.TTS.silero_tts_handler import SILERO_REPOSITORY, SileroTTSHandler, _resolve_silero_repository


class FakeSileroModel:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.calls: list[dict[str, Any]] = []

    def apply_tts(self, **kwargs: Any) -> np.ndarray:
        self.calls.append(kwargs)
        return self.audio


def make_handler(model: FakeSileroModel, *, blocksize: int = 4) -> SileroTTSHandler:
    handler = SileroTTSHandler.__new__(SileroTTSHandler)
    handler.model = model
    handler.speaker = "xenia"
    handler.sample_rate = 24000
    handler.blocksize = blocksize
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.english_fallback = False
    return handler


@pytest.mark.parametrize(
    ("setup_kwargs", "message"),
    [
        ({"blocksize": 0}, "blocksize must be positive"),
        ({"sample_rate": 16000}, "sample_rate must be one of"),
        ({"speaker": "unknown"}, "Unsupported Silero Russian speaker"),
        ({"threads": 0}, "threads must be positive"),
    ],
)
def test_setup_rejects_invalid_configuration_before_loading_model(setup_kwargs, message: str) -> None:
    handler = SileroTTSHandler.__new__(SileroTTSHandler)

    with pytest.raises(ValueError, match=message):
        handler.setup(Event(), **setup_kwargs)


def test_russian_synthesis_uses_silero_and_emits_clipped_padded_16k_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeSileroModel(np.zeros(3, dtype=np.float32))
    handler = make_handler(model)
    monkeypatch.setattr(
        silero_module.scipy.signal,
        "resample_poly",
        lambda *_args: np.array([2.0, -2.0, 0.5], dtype=np.float32),
    )

    chunks = list(handler.process(TTSInput(text="Привет, мир.", language_code="ru")))

    assert model.calls == [
        {
            "text": "Привет, мир.",
            "speaker": "xenia",
            "sample_rate": 24000,
            "put_accent": True,
            "put_yo": True,
        }
    ]
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], np.array([32767, -32768, 16384, 0], dtype=np.int16))


def test_setup_loads_pinned_v55_model_on_cpu_and_warms_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model = FakeSileroModel(np.zeros(8, dtype=np.float32))
    loaded: list[dict[str, Any]] = []
    thread_counts: list[int] = []

    def fake_load(**kwargs: Any):
        loaded.append(kwargs)
        return model, "example"

    monkeypatch.setattr(silero_module.torch.hub, "load", fake_load)
    monkeypatch.setattr(silero_module.torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(silero_module.torch, "set_num_threads", thread_counts.append)
    handler = SileroTTSHandler.__new__(SileroTTSHandler)

    handler.setup(Event(), threads=6)

    assert thread_counts == [6]
    assert loaded == [
        {
            "repo_or_dir": "snakers4/silero-models:d9355348e2781dc8fa25a135d1602c530afae24c",
            "model": "silero_tts",
            "language": "ru",
            "speaker": "v5_5_ru",
            "trust_repo": True,
            "source": "github",
        }
    ]
    assert model.calls[0]["text"] == "Раз, два."
    assert handler.model is model


def test_process_drops_audio_when_cancelled_during_blocking_synthesis() -> None:
    cancel_scope = CancelScope()
    model = FakeSileroModel(np.zeros(8, dtype=np.float32))
    original_apply_tts = model.apply_tts

    def cancelling_apply_tts(**kwargs: Any) -> np.ndarray:
        audio = original_apply_tts(**kwargs)
        cancel_scope.cancel()
        return audio

    model.apply_tts = cancelling_apply_tts  # type: ignore[method-assign]
    handler = make_handler(model)
    handler.cancel_scope = cancel_scope

    assert list(handler.process(TTSInput(text="Привет", language_code="ru"))) == []


def test_stale_keyed_terminal_is_forwarded_as_cleanup_only() -> None:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    model = FakeSileroModel(np.zeros(4, dtype=np.float32))
    handler = make_handler(model)
    handler.speculative_turns = tracker
    terminal = EndOfResponse(
        response_key="response_1",
        turn_id="turn_1",
        turn_revision=0,
        cancel_generation=7,
    )
    tracker.observe("turn_1", 1)

    outputs = list(handler.process(terminal))
    queued = handler.output_for_queue(outputs[0], terminal)

    assert outputs == [AUDIO_RESPONSE_DONE]
    assert terminal.cleanup_only is True
    assert isinstance(queued, AudioOutput)
    assert queued.cleanup_only is True


def test_english_input_uses_prepared_fallback_without_calling_russian_model() -> None:
    model = FakeSileroModel(np.zeros(4, dtype=np.float32))
    handler = make_handler(model)
    handler.english_fallback = True

    class FakeEnglishHandler:
        def process(self, tts_input: TTSInput):
            assert tts_input.language_code == "en-US"
            yield np.array([1, 2, 3, 4], dtype=np.int16)

    handler._english_handler = FakeEnglishHandler()

    chunks = list(handler.process(TTSInput(text="Hello.", language_code="en-US")))

    assert model.calls == []
    np.testing.assert_array_equal(chunks[0], np.array([1, 2, 3, 4], dtype=np.int16))


def test_setup_keeps_english_fallback_lazy_and_reuses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSileroModel(np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(silero_module.torch.hub, "load", lambda **_kwargs: (model, "example"))
    monkeypatch.setattr(silero_module.torch, "set_num_threads", lambda _threads: None)
    handler = SileroTTSHandler.__new__(SileroTTSHandler)
    load_calls = 0

    class FakeEnglishHandler:
        def process(self, _tts_input: TTSInput):
            yield np.zeros(4, dtype=np.int16)

    def fake_load_english_handler():
        nonlocal load_calls
        load_calls += 1
        return FakeEnglishHandler()

    handler._load_english_handler = fake_load_english_handler  # type: ignore[method-assign]
    handler.setup(Event(), english_fallback=True, english_voice="M1")

    assert load_calls == 0
    list(handler.process(TTSInput(text="One.", language_code="en")))
    list(handler.process(TTSInput(text="Two.", language_code="en")))
    assert load_calls == 1


def test_process_drops_stale_speculative_turn_before_synthesis() -> None:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.observe("turn_1", 1)
    model = FakeSileroModel(np.zeros(4, dtype=np.float32))
    handler = make_handler(model)
    handler.speculative_turns = tracker

    outputs = list(
        handler.process(
            TTSInput(text="Старый ответ", language_code="ru", turn_id="turn_1", turn_revision=0)
        )
    )

    assert outputs == []
    assert model.calls == []


def test_repository_resolution_prefers_existing_torch_cache(tmp_path) -> None:
    cached_repository = tmp_path / "snakers4_silero-models_master"
    cached_repository.mkdir()

    assert _resolve_silero_repository(tmp_path) == str(cached_repository)
    assert _resolve_silero_repository(tmp_path / "empty") == SILERO_REPOSITORY
