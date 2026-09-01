from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.speaker_memory.models import SpeakerState
from speech_to_speech.speaker_memory.store import SpeakerMemoryStore
from speech_to_speech.speaker_memory.tracker import SpeakerMemoryService, SpeakerTracker


def unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def store(tmp_path) -> SpeakerMemoryStore:
    memory = SpeakerMemoryStore(tmp_path / "speaker-memory.sqlite3", clock=lambda: 100.0)
    yield memory
    memory.close()


def tracker(store: SpeakerMemoryStore, **overrides) -> SpeakerTracker:
    options = {
        "match_threshold": 0.82,
        "candidate_threshold": 0.70,
        "ambiguity_margin": 0.08,
        "minimum_quality": 0.5,
        "reference_ttl_s": 60.0,
    }
    options.update(overrides)
    return SpeakerTracker(store, **options)


def observe(memory_tracker: SpeakerTracker, embedding: np.ndarray, turn_id: str = "turn_1"):
    return memory_tracker.observe(
        embedding,
        quality=1.0,
        turn_id=turn_id,
        turn_revision=0,
        conversation_id="conv_1",
    )


def test_first_observation_creates_stable_voice_and_reference(store: SpeakerMemoryStore) -> None:
    result = observe(tracker(store), unit([1.0, 0.0]))

    assert result.state is SpeakerState.UNKNOWN
    assert result.voice_id is not None and result.voice_id.startswith("v_")
    assert result.speaker_ref is not None and result.speaker_ref.startswith("sr_")
    assert len(store.get_voice_clusters()) == 1


def test_decisive_match_reuses_voice_and_updates_centroid(store: SpeakerMemoryStore) -> None:
    memory_tracker = tracker(store)
    first = observe(memory_tracker, unit([1.0, 0.0]))

    second = observe(memory_tracker, unit([0.98, 0.20]), turn_id="turn_2")
    updated = store.get_voice_cluster(first.voice_id)

    assert second.voice_id == first.voice_id
    assert updated.sample_count == 2
    assert updated.centroid[1] > 0
    assert np.linalg.norm(updated.centroid) == pytest.approx(1.0)


def test_mature_unassigned_cluster_reuses_weak_match_and_refines_centroid(
    store: SpeakerMemoryStore,
) -> None:
    memory_tracker = tracker(
        store,
        soft_match_threshold=0.60,
        soft_match_min_samples=2,
        soft_match_weight=0.25,
    )
    first = observe(memory_tracker, unit([1.0, 0.0]))
    observe(memory_tracker, unit([0.98, 0.20]), turn_id="turn_2")
    before = store.get_voice_cluster(first.voice_id)

    weak = observe(memory_tracker, unit([0.60, 0.80]), turn_id="turn_3")
    after = store.get_voice_cluster(first.voice_id)

    assert weak.voice_id == first.voice_id
    assert weak.state is SpeakerState.UNKNOWN
    assert after.sample_count == before.sample_count + 1
    assert not np.array_equal(after.centroid, before.centroid)


def test_decisive_match_to_blacklisted_voice_is_rejected_without_centroid_drift(
    store: SpeakerMemoryStore,
) -> None:
    memory_tracker = tracker(store)
    first = observe(memory_tracker, unit([1.0, 0.0]))
    before = store.get_voice_cluster(first.voice_id)
    store.set_voice_blocked(first.voice_id, blocked=True, reason="background_tv")

    result = observe(memory_tracker, unit([0.99, 0.02]), turn_id="turn_tv")

    after = store.get_voice_cluster(first.voice_id)
    assert result.state is SpeakerState.BLACKLISTED
    assert result.voice_id == first.voice_id
    assert result.speaker_ref is None
    assert result.recommendation == "do_not_learn"
    assert after.sample_count == before.sample_count
    assert np.array_equal(after.centroid, before.centroid)


def test_ambiguous_observation_does_not_move_a_centroid(store: SpeakerMemoryStore) -> None:
    first = store.create_voice_cluster(unit([1.0, 0.0]), quality=1.0)
    store.create_voice_cluster(unit([0.98, 0.2]), quality=1.0)
    before = first.centroid.copy()

    result = observe(tracker(store), unit([0.995, 0.1]), turn_id="turn_3")

    assert result.state is SpeakerState.AMBIGUOUS
    assert result.recommendation == "clarify"
    assert np.array_equal(store.get_voice_cluster(first.id).centroid, before)


def test_low_quality_observation_is_not_persisted(store: SpeakerMemoryStore) -> None:
    result = tracker(store).observe(
        unit([1.0, 0.0]),
        quality=0.2,
        turn_id="turn_low",
        turn_revision=0,
        conversation_id="conv_1",
    )

    assert result.state is SpeakerState.UNKNOWN
    assert result.voice_id is None
    assert result.speaker_ref is None
    assert result.recommendation == "do_not_learn"
    assert store.get_voice_clusters() == []


def test_decisive_person_evidence_yields_known_identity(store: SpeakerMemoryStore) -> None:
    memory_tracker = tracker(store)
    first = observe(memory_tracker, unit([1.0, 0.0]))
    person = store.create_person("Аркадий")
    store.add_identity_evidence(first.voice_id, person.id, kind="self_introduction", weight=3.0)

    # Still a plausible match, but below the strict acoustic threshold; the
    # decisive self-introduction evidence must keep the identity known.
    result = observe(memory_tracker, unit([0.80, 0.60]), turn_id="turn_2")

    assert result.state is SpeakerState.KNOWN
    assert result.candidate is not None
    assert result.candidate.person_id == person.id
    assert result.candidate.name == "Аркадий"


def test_competing_person_evidence_yields_conflict(store: SpeakerMemoryStore) -> None:
    memory_tracker = tracker(store)
    first = observe(memory_tracker, unit([1.0, 0.0]))
    arkady = store.create_person("Аркадий")
    andrey = store.create_person("Андрей")
    store.add_identity_evidence(first.voice_id, arkady.id, kind="manual_link", weight=3.0)
    store.add_identity_evidence(first.voice_id, andrey.id, kind="manual_link", weight=2.8)

    result = observe(memory_tracker, unit([1.0, 0.01]), turn_id="turn_2")

    assert result.state is SpeakerState.CONFLICT
    assert result.candidate is not None
    assert result.candidate.person_id == arkady.id
    assert result.recommendation == "clarify"


def test_service_uses_fixed_semantic_evidence_weights(store: SpeakerMemoryStore) -> None:
    first = observe(tracker(store), unit([1.0, 0.0]))
    service = SpeakerMemoryService(store)

    remembered = service.remember_name(first.speaker_ref, "Аркадий", conversation_id="conv_1")
    service.reject(first.speaker_ref, remembered.candidate.person_id, conversation_id="conv_1")
    service.confirm(first.speaker_ref, remembered.candidate.person_id, conversation_id="conv_1")

    candidates = store.resolve_person_candidates(first.voice_id)
    assert candidates[0].name == "Аркадий"
    assert candidates[0].evidence_score == pytest.approx(1.0)


def test_replayed_confirmation_is_idempotent_for_same_observation(store: SpeakerMemoryStore) -> None:
    first = observe(tracker(store), unit([1.0, 0.0]))
    service = SpeakerMemoryService(store)
    remembered = service.remember_name(first.speaker_ref, "Аркадий", conversation_id="conv_1")
    assert remembered.candidate is not None

    service.confirm(first.speaker_ref, remembered.candidate.person_id, conversation_id="conv_1")
    service.confirm(first.speaker_ref, remembered.candidate.person_id, conversation_id="conv_1")

    assert store.resolve_person_candidates(first.voice_id)[0].evidence_score == pytest.approx(5.0)


def test_inspection_cannot_use_reference_from_another_conversation(store: SpeakerMemoryStore) -> None:
    first = observe(tracker(store), unit([1.0, 0.0]))
    service = SpeakerMemoryService(store)

    with pytest.raises(Exception, match="another conversation"):
        service.inspect(first.speaker_ref, conversation_id="conv_2")


def test_similar_new_voice_gets_known_person_as_clarification_candidate(store: SpeakerMemoryStore) -> None:
    memory_tracker = tracker(store, group_threshold=0.55)
    first = observe(memory_tracker, unit([1.0, 0.0]), turn_id="turn_owner")
    person = store.create_person("Михаил")
    store.add_identity_evidence(first.voice_id, person.id, kind="self_introduction", weight=3.0)

    # The sample is deliberately below the ordinary cluster threshold, so it
    # becomes a second voice_id while still being close enough to propose the
    # already-known person for explicit clarification.
    second = observe(memory_tracker, unit([0.58, 0.815]), turn_id="turn_alias")

    assert second.voice_id != first.voice_id
    assert second.state is SpeakerState.CONFLICT
    assert second.recommendation == "clarify"
    assert second.candidate is not None
    assert second.candidate.person_id == person.id
    assert store.reference_allows_candidate(second.speaker_ref, person.id)

    linked = SpeakerMemoryService(store).remember_name(second.speaker_ref, "Михаил", conversation_id="conv_1")
    assert linked.candidate is not None
    assert linked.candidate.person_id == person.id
    assert linked.voice_id == first.voice_id
    assert store.resolve_voice_id(second.voice_id) == first.voice_id

    # A later explicit rejection detaches only this newly merged source and
    # forces clarification instead of leaking the canonical person's memory.
    store.add_identity_evidence(second.voice_id, person.id, kind="passive_match", weight=10.0)
    rejected = SpeakerMemoryService(store).reject(
        second.speaker_ref,
        person.id,
        conversation_id="conv_1",
    )
    assert rejected.state is SpeakerState.UNKNOWN
    assert rejected.voice_id == second.voice_id
    assert store.resolve_voice_id(second.voice_id) == second.voice_id
