"""Guided voice calibration and the acoustic boundary of person merges."""

from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.speaker_memory.enrollment import ENROLL_PHRASES, ENROLL_SAMPLE_TARGET
from speech_to_speech.speaker_memory.models import SpeakerState
from speech_to_speech.speaker_memory.service import SpeakerMemoryService
from speech_to_speech.speaker_memory.store import SpeakerMemoryStore
from speech_to_speech.speaker_memory.tools import TOOLS
from speech_to_speech.speaker_memory.tracker import SpeakerTracker


def unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def runtime(tmp_path):
    store = SpeakerMemoryStore(tmp_path / "memory.sqlite3", clock=lambda: 100.0)
    tracker = SpeakerTracker(store, reference_ttl_s=60.0)
    service = SpeakerMemoryService(store)
    attribution = tracker.observe(
        unit([1.0, 0.0]),
        quality=1.0,
        turn_id="turn_1",
        turn_revision=0,
        conversation_id="conv_1",
    )
    yield store, tracker, service, attribution
    store.close()


def test_enrollment_folds_repeated_phrases_into_the_named_voice(runtime) -> None:
    store, tracker, service, attribution = runtime
    assert any(tool["name"] == "speaker_memory_enroll" for tool in TOOLS)
    session = service.start_enrollment(attribution.speaker_ref, conversation_id="conv_1")
    assert len(session.phrases) == ENROLL_SAMPLE_TARGET
    assert all(phrase in ENROLL_PHRASES for phrase in session.phrases)
    assert session.remaining == ENROLL_SAMPLE_TARGET

    # Phrase repetitions may acoustically drift far from the centroid; the
    # guided path must still fold them into the calibrated voice.
    for index in range(ENROLL_SAMPLE_TARGET):
        enrolled = tracker.enroll(
            unit([0.8, 0.6]),
            voice_id=attribution.voice_id,
            quality=0.9,
            turn_id=f"turn_enroll_{index}",
            turn_revision=0,
            conversation_id="conv_1",
        )
        assert enrolled.voice_id == attribution.voice_id
        remaining = service.note_enrollment_sample("conv_1")
        assert remaining == ENROLL_SAMPLE_TARGET - index - 1

    assert service.active_enrollment("conv_1") is None
    cluster = store.get_voice_cluster(attribution.voice_id)
    assert cluster.sample_count == 1 + ENROLL_SAMPLE_TARGET
    assert len(store.get_voice_prototypes(attribution.voice_id)) > 1


def test_remember_name_never_folds_an_acoustically_foreign_voice_into_a_person(runtime) -> None:
    # A misheard name must not let one speaker's cluster swallow another
    # speaker's voice: identity binds, acoustically foreign clusters stay split.
    store, tracker, service, marat = runtime
    service.remember_name(marat.speaker_ref, "Марат", conversation_id="conv_1")

    wife = tracker.observe(
        unit([0.0, 1.0]),
        quality=1.0,
        turn_id="turn_2",
        turn_revision=0,
        conversation_id="conv_1",
    )
    assert wife.voice_id != marat.voice_id

    # The model mishears the wife's name as "Марат" and remembers it.
    service.remember_name(wife.speaker_ref, "Марат", conversation_id="conv_1")

    assert store.resolve_voice_id(wife.voice_id) == wife.voice_id
    marat_candidates = store.resolve_person_candidates(marat.voice_id)
    wife_candidates = store.resolve_person_candidates(wife.voice_id)
    assert marat_candidates[0].name == "Марат"
    assert wife_candidates[0].name == "Марат"

    # A fresh utterance from the wife's direction must not resolve to Marat's
    # canonical voice cluster.
    again = tracker.observe(
        unit([0.0, 1.0]),
        quality=1.0,
        turn_id="turn_3",
        turn_revision=0,
        conversation_id="conv_1",
    )
    assert again.state is not SpeakerState.BLACKLISTED
    assert store.resolve_voice_id(again.voice_id) != store.resolve_voice_id(marat.voice_id)
