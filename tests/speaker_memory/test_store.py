from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pytest

from speech_to_speech.speaker_memory.models import (
    ExpiredSpeakerReference,
    InvalidEmbedding,
    SpeakerReferenceConversationMismatch,
    SupersededSpeakerReference,
    UnknownSpeakerReference,
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


def test_store_enables_wal_and_foreign_keys(store: SpeakerMemoryStore) -> None:
    assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_adaptive_centroid_weight_is_bounded(store: SpeakerMemoryStore) -> None:
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)

    for _ in range(30):
        store.update_voice_cluster(voice.id, np.array([1.0, 0.01], dtype=np.float32), quality=1.0)

    updated = store.get_voice_cluster(voice.id)
    assert updated.sample_count == 31
    assert updated.quality_weight == pytest.approx(20.0)


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


def test_only_latest_reference_can_authorize_voice_routing_mutation(
    store: SpeakerMemoryStore,
) -> None:
    first_voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    second_voice = store.create_voice_cluster(np.array([0.0, 1.0], dtype=np.float32), quality=1.0)
    first_observation = store.create_observation(
        first_voice.id,
        turn_id="turn_1",
        turn_revision=0,
        conversation_id="conv_1",
        quality=1.0,
    )
    first_ref = store.issue_reference(first_observation.id, conversation_id="conv_1", ttl_s=30)
    second_observation = store.create_observation(
        second_voice.id,
        turn_id="turn_2",
        turn_revision=0,
        conversation_id="conv_1",
        quality=1.0,
    )
    second_ref = store.issue_reference(second_observation.id, conversation_id="conv_1", ttl_s=30)

    with pytest.raises(SupersededSpeakerReference):
        store.set_voice_blocked_by_reference(
            first_ref,
            conversation_id="conv_1",
            blocked=True,
            reason="background_tv",
        )

    voice_id = store.set_voice_blocked_by_reference(
        second_ref,
        conversation_id="conv_1",
        blocked=True,
        reason="background_tv",
    )
    assert voice_id == second_voice.id
    assert store.is_voice_blocked(first_voice.id) is False
    assert store.is_voice_blocked(second_voice.id) is True


def test_unattributed_final_turn_can_invalidate_all_prior_mutation_references(
    store: SpeakerMemoryStore,
) -> None:
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    observation = store.create_observation(
        voice.id,
        turn_id="turn_1",
        turn_revision=0,
        conversation_id="conv_1",
        quality=1.0,
    )
    speaker_ref = store.issue_reference(observation.id, conversation_id="conv_1", ttl_s=30)

    assert store.invalidate_references("conv_1") == 1

    with pytest.raises(UnknownSpeakerReference):
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


def test_aggregate_identity_evidence_is_bounded(store: SpeakerMemoryStore) -> None:
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    person = store.create_person("Аркадий")
    for _ in range(20):
        store.add_identity_evidence(voice.id, person.id, kind="passive_match", weight=1.0)

    assert store.resolve_person_candidates(voice.id)[0].evidence_score == pytest.approx(10.0)


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


def test_expired_references_and_unneeded_observations_are_pruned(tmp_path, clock: MutableClock) -> None:
    store = SpeakerMemoryStore(tmp_path / "retention.sqlite3", clock=clock, observation_retention_days=1)
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    observation = store.create_observation(
        voice.id,
        turn_id="turn_old",
        turn_revision=0,
        conversation_id="conv_old",
        quality=1.0,
    )
    reference = store.issue_reference(observation.id, conversation_id="conv_old", ttl_s=10)
    clock.now += 2 * 86400

    removed = store.prune_expired()

    assert removed == {"references": 1, "observations": 1}
    with pytest.raises(UnknownSpeakerReference):
        store.resolve_reference(reference, conversation_id="conv_old")
    store.close()


def test_retention_prunes_observation_but_preserves_detached_identity_audit(
    tmp_path, clock: MutableClock
) -> None:
    store = SpeakerMemoryStore(tmp_path / "audit-retention.sqlite3", clock=clock, observation_retention_days=1)
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    person = store.create_person("Аркадий")
    observation = store.create_observation(
        voice.id,
        turn_id="turn_old",
        turn_revision=0,
        conversation_id="conv_old",
        quality=1.0,
    )
    store.add_identity_evidence(
        voice.id,
        person.id,
        kind="self_introduction",
        weight=3.0,
        observation_id=observation.id,
    )
    clock.now += 2 * 86400

    removed = store.prune_expired()

    assert removed["observations"] == 1
    assert store.resolve_person_candidates(voice.id)[0].evidence_score == pytest.approx(3.0)
    evidence = store._connection.execute(
        "SELECT observation_id FROM voice_person_evidence WHERE voice_id = ?", (voice.id,)
    ).fetchone()
    assert evidence["observation_id"] is None
    store.close()


def test_schema_v2_database_migrates_to_current_version(tmp_path) -> None:
    path = tmp_path / "migration.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE schema_meta(version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_meta(version) VALUES (2)")
    connection.commit()
    connection.close()

    store = SpeakerMemoryStore(path)

    assert store._connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 4
    assert store._connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='speaker_reference_candidates'"
    ).fetchone()
    store.close()


def test_second_writer_obeys_bounded_busy_timeout(tmp_path) -> None:
    path = tmp_path / "locked.sqlite3"
    first = SpeakerMemoryStore(path, timeout_s=0.01)
    second = SpeakerMemoryStore(path, timeout_s=0.01)
    first._connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            second.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    finally:
        first._connection.rollback()
        first.close()
        second.close()
