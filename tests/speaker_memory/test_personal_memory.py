from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from speech_to_speech.speaker_memory.policy import SPEAKER_MEMORY_POLICY
from speech_to_speech.speaker_memory.service import IdentityNotConfirmed, SpeakerMemoryService
from speech_to_speech.speaker_memory.store import SpeakerMemoryStore
from speech_to_speech.speaker_memory.tracker import SpeakerTracker


@pytest.fixture
def memory(tmp_path):
    store = SpeakerMemoryStore(tmp_path / "memory.sqlite3", clock=lambda: 100.0)
    tracker = SpeakerTracker(store, candidate_threshold=0.7, match_threshold=0.82)
    service = SpeakerMemoryService(store)
    yield store, tracker, service
    store.close()


def attributed(tracker: SpeakerTracker, vector: list[float], *, turn: str, conversation: str = "conv_1"):
    return tracker.observe(
        np.asarray(vector, dtype=np.float32),
        quality=1.0,
        turn_id=turn,
        turn_revision=0,
        conversation_id=conversation,
    )


def known_speaker(memory, *, vector=None, name="Аркадий", turn="turn_1"):
    _store, tracker, service = memory
    result = attributed(tracker, vector or [1.0, 0.0], turn=turn)
    service.remember_name(result.speaker_ref, name, conversation_id="conv_1")
    return result


def test_unknown_voice_cannot_recall_or_store_private_facts(memory) -> None:
    _store, tracker, service = memory
    unknown = attributed(tracker, [1.0, 0.0], turn="turn_1")

    with pytest.raises(IdentityNotConfirmed):
        service.recall(unknown.speaker_ref, query="напиток", conversation_id="conv_1")
    with pytest.raises(IdentityNotConfirmed):
        service.remember_fact(unknown.speaker_ref, "Любит чай", conversation_id="conv_1")


def test_known_person_can_store_and_search_russian_fact(memory) -> None:
    _store, _tracker, service = memory
    speaker = known_speaker(memory)

    saved = service.remember_fact(
        speaker.speaker_ref,
        "Любимый напиток — зелёный чай",
        topic="предпочтения",
        conversation_id="conv_1",
    )
    results = service.recall(
        speaker.speaker_ref,
        query="зелёный чай",
        limit=5,
        conversation_id="conv_1",
    )

    assert saved.id.startswith("f_")
    assert [(fact.id, fact.fact, fact.topic) for fact in results] == [
        (saved.id, "Любимый напиток — зелёный чай", "предпочтения")
    ]


def test_personal_facts_are_isolated_between_people(memory) -> None:
    _store, _tracker, service = memory
    arkady = known_speaker(memory, vector=[1.0, 0.0], name="Аркадий", turn="turn_1")
    andrey = known_speaker(memory, vector=[0.0, 1.0], name="Андрей", turn="turn_2")
    service.remember_fact(arkady.speaker_ref, "Кодовое слово: кедр", conversation_id="conv_1")

    assert service.recall(andrey.speaker_ref, query="кедр", conversation_id="conv_1") == []


def test_explicit_fact_and_all_fact_forgetting(memory) -> None:
    _store, _tracker, service = memory
    speaker = known_speaker(memory)
    first = service.remember_fact(speaker.speaker_ref, "Любит чай", conversation_id="conv_1")
    service.remember_fact(speaker.speaker_ref, "Любит кофе", conversation_id="conv_1")

    assert service.forget(
        speaker.speaker_ref,
        scope="fact",
        fact_id=first.id,
        conversation_id="conv_1",
    ) == 1
    assert [fact.fact for fact in service.recall(speaker.speaker_ref, query="любит", conversation_id="conv_1")] == [
        "Любит кофе"
    ]
    assert service.forget(speaker.speaker_ref, scope="facts", conversation_id="conv_1") == 1
    assert service.recall(speaker.speaker_ref, query="любит", conversation_id="conv_1") == []


def test_fact_text_is_absent_from_normal_logs(memory, caplog) -> None:
    _store, _tracker, service = memory
    speaker = known_speaker(memory)
    caplog.set_level(logging.DEBUG)

    service.remember_fact(
        speaker.speaker_ref,
        "Совершенно секретный личный факт",
        conversation_id="conv_1",
    )
    service.recall(speaker.speaker_ref, query="секретный", conversation_id="conv_1")

    assert "Совершенно секретный личный факт" not in caplog.text
    assert "секретный" not in caplog.text


def test_native_policy_and_skill_cover_conflict_privacy_and_every_tool() -> None:
    root = Path(__file__).resolve().parents[2]
    skill = (root / "skills" / "speaker-memory" / "SKILL.md").read_text()
    combined = f"{SPEAKER_MEMORY_POLICY}\n{skill}"
    for tool_name in (
        "speaker_memory_inspect",
        "speaker_memory_remember_name",
        "speaker_memory_confirm",
        "speaker_memory_reject",
        "speaker_memory_remember_fact",
        "speaker_memory_recall",
        "speaker_memory_forget",
    ):
        assert tool_name in combined
    assert "ambiguous" in combined
    assert "conflict" in combined
    assert "mixed" in combined
    assert "не раскрывай" in combined.casefold()
