from __future__ import annotations

import logging

import numpy as np

from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.speaker_memory.handler import SpeakerMemoryHandler
from speech_to_speech.speaker_memory.models import SpeakerAttribution, SpeakerState


class FakeExtractor:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        assert sample_rate == 16000
        if self.fail:
            raise RuntimeError("private audio details must not be logged")
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeTracker:
    def __init__(self) -> None:
        self.calls = []

    def observe(self, embedding: np.ndarray, **kwargs) -> SpeakerAttribution:
        self.calls.append((embedding, kwargs))
        return SpeakerAttribution(voice_id="v_test", speaker_ref="sr_test", state=SpeakerState.UNKNOWN)


def bare_handler(extractor: FakeExtractor, tracker: FakeTracker, *, min_audio_ms: int = 700):
    handler = object.__new__(SpeakerMemoryHandler)
    handler.setup(
        extractor=extractor,
        tracker=tracker,
        sample_rate=16000,
        min_audio_ms=min_audio_ms,
        conversation_id="conv_test",
    )
    return handler


def audio(*, samples: int = 16000, mode: str = "final") -> VADAudio:
    return VADAudio(
        audio=np.full(samples, 0.1, dtype=np.float32),
        mode=mode,
        turn_id="turn_1",
        turn_revision=0,
    )


def test_final_audio_is_attributed_but_progressive_audio_is_not(caplog) -> None:
    extractor = FakeExtractor()
    tracker = FakeTracker()
    handler = bare_handler(extractor, tracker)
    caplog.set_level(logging.INFO)

    progressive = list(handler.process(audio(mode="progressive")))[0]
    final = list(handler.process(audio(mode="final")))[0]

    assert getattr(progressive, "speaker", None) is None
    assert final.speaker.voice_id == "v_test"
    assert final.speaker.speaker_ms > 0
    assert extractor.calls == 1
    assert tracker.calls[0][1]["conversation_id"] == "conv_test"
    assert "Speaker attributed voice=v_test state=unknown" in caplog.text
    assert "sr_test" not in caplog.text


def test_short_final_audio_passes_through_without_inference() -> None:
    extractor = FakeExtractor()
    handler = bare_handler(extractor, FakeTracker(), min_audio_ms=700)

    result = list(handler.process(audio(samples=8000)))[0]

    assert getattr(result, "speaker", None) is None
    assert extractor.calls == 0


def test_low_signal_final_audio_is_rejected_before_embedding() -> None:
    extractor = FakeExtractor()
    tracker = FakeTracker()
    tracker.minimum_quality = 0.5
    handler = bare_handler(extractor, tracker)

    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_silent",
                turn_revision=0,
            )
        )
    )[0]

    assert result.speaker is None
    assert extractor.calls == 0
    assert tracker.calls == []


def test_embedding_failure_keeps_pipeline_usable_without_logging_sensitive_message(caplog) -> None:
    extractor = FakeExtractor(fail=True)
    handler = bare_handler(extractor, FakeTracker())
    caplog.set_level(logging.WARNING)

    original = audio()
    result = list(handler.process(original))[0]

    assert result is original
    assert "private audio details" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_session_end_rotates_reference_scope() -> None:
    handler = bare_handler(FakeExtractor(), FakeTracker())
    previous = handler.conversation_id

    handler.on_session_end()

    assert handler.conversation_id != previous
