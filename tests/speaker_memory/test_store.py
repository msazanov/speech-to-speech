from __future__ import annotations

import logging

import numpy as np
import pytest

from speech_to_speech.speaker_memory.models import (
    ExpiredSpeakerReference,
    InvalidEmbedding,
    SpeakerReferenceConversationMismatch,
)
from speech_to_speech.speaker_memory.store import SpeakerMemoryStore


class MutableClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def store(tmp_path, clock: MutableClock) -> SpeakerMemoryStore:
    memory = SpeakerMemoryStore(tmp_path / "speaker-memory.sqlite3", clock=clock)
    yield memory
    memory.close()


def test_voice_cluster_round_trips_normalized_float32_centroid(store: SpeakerMemoryStore) -> None:
    cluster = store.create_voice_cluster(np.array([3.0, 4.0], dtype=np.float64), quality=0.75)

    restored = store.get_voice_cluster(cluster.id)

    assert cluster.id.startswith("v_")
    assert restored.centroid.dtype == np.float32
    assert restored.centroid.tolist() == pytest.approx([0.6, 0.8])
    assert restored.sample_count == 1
    assert restored.quality_weight == pytest.approx(0.75)


@pytest.mark.parametrize(
    "embedding",
    [
        np.array([], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, np.nan], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
    ],
)
def test_invalid_embeddings_never_enter_the_store(store: SpeakerMemoryStore, embedding: np.ndarray) -> None:
    with pytest.raises(InvalidEmbedding):
        store.create_voice_cluster(embedding, quality=1.0)


def test_reference_resolves_only_for_its_conversation_before_expiry(
    store: SpeakerMemoryStore,
    clock: MutableClock,
) -> None:
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    observation = store.create_observation(
        voice.id,
        turn_id="turn_1",
        turn_revision=0,
        conversation_id="conv_1",
        quality=0.9,
    )
    speaker_ref = store.issue_reference(observation.id, conversation_id="conv_1", ttl_s=30)

    resolved = store.resolve_reference(speaker_ref, conversation_id="conv_1")

    assert speaker_ref.startswith("sr_")
    assert resolved.voice_id == voice.id
    assert resolved.observation_id == observation.id
    with pytest.raises(SpeakerReferenceConversationMismatch):
        store.resolve_reference(speaker_ref, conversation_id="conv_2")

    clock.now = 131.0
    with pytest.raises(ExpiredSpeakerReference):
        store.resolve_reference(speaker_ref, conversation_id="conv_1")


def test_identity_candidates_aggregate_positive_and_negative_evidence(store: SpeakerMemoryStore) -> None:
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    arkady = store.create_person("Аркадий")
    andrey = store.create_person("Андрей")

    store.add_identity_evidence(voice.id, arkady.id, kind="self_introduction", weight=3.0)
    store.add_identity_evidence(voice.id, arkady.id, kind="agent_rejection", weight=-1.0)
    store.add_identity_evidence(voice.id, andrey.id, kind="passive_match", weight=0.5)

    candidates = store.resolve_person_candidates(voice.id)

    assert [(candidate.person_id, candidate.name, candidate.evidence_score) for candidate in candidates] == [
        (arkady.id, "Аркадий", 2.0),
        (andrey.id, "Андрей", 0.5),
    ]


def test_person_name_is_normalized_for_reuse_without_losing_display_form(store: SpeakerMemoryStore) -> None:
    first = store.create_person("  Аркадий  ")
    second = store.create_person("аркадий")

    assert second.id == first.id
    assert second.name == "Аркадий"


def test_sensitive_values_are_not_written_to_normal_logs(
    store: SpeakerMemoryStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    voice = store.create_voice_cluster(np.array([0.25, 0.75], dtype=np.float32), quality=1.0)
    person = store.create_person("Совершенно Секретное Имя")
    store.add_identity_evidence(voice.id, person.id, kind="manual_link", weight=4.0)

    log_text = caplog.text
    assert "Совершенно Секретное Имя" not in log_text
    assert "0.25" not in log_text
    assert "0.75" not in log_text
