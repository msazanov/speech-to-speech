from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.audio_client import ToolResult
from speech_to_speech.s2s_pipeline import build_local_pipeline, parse_arguments
from speech_to_speech.speaker_memory.service import SpeakerMemoryService
from speech_to_speech.speaker_memory.store import SpeakerMemoryStore
from speech_to_speech.speaker_memory.tools import (
    TOOLS,
    configure_tool_service,
    execute_tool,
)
from speech_to_speech.speaker_memory.tracker import SpeakerTracker


@pytest.fixture
def tool_runtime(tmp_path):
    store = SpeakerMemoryStore(tmp_path / "memory.sqlite3", clock=lambda: 100.0)
    tracker = SpeakerTracker(store, reference_ttl_s=60.0)
    attributed = tracker.observe(
        np.array([1.0, 0.0], dtype=np.float32),
        quality=1.0,
        turn_id="turn_1",
        turn_revision=0,
        conversation_id="conv_1",
    )
    service = SpeakerMemoryService(store)
    configure_tool_service(service, conversation_id_provider=lambda: "conv_1")
    yield store, attributed
    configure_tool_service(None)
    store.close()


def tool_schema(name: str) -> dict:
    return next(tool for tool in TOOLS if tool["name"] == name)


def test_mutation_tools_never_accept_raw_voice_id_or_numeric_weight() -> None:
    for name in (
        "speaker_memory_remember_name",
        "speaker_memory_confirm",
        "speaker_memory_reject",
        "speaker_memory_block_voice",
        "speaker_memory_unblock_voice",
    ):
        properties = tool_schema(name)["parameters"]["properties"]
        assert "speaker_ref" in properties
        assert "voice_id" not in properties
        assert "weight" not in properties


@pytest.mark.asyncio
async def test_agent_can_block_and_immediately_unblock_current_voice_by_reference(tool_runtime) -> None:
    store, attributed = tool_runtime

    blocked = await execute_tool(
        "speaker_memory_block_voice",
        {"speaker_ref": attributed.speaker_ref, "reason": "background_tv"},
    )
    assert blocked.output == {"ok": True, "voice_id": attributed.voice_id, "blocked": True}
    assert blocked.create_response is False
    assert store.is_voice_blocked(attributed.voice_id) is True

    unblocked = await execute_tool(
        "speaker_memory_unblock_voice",
        {"speaker_ref": attributed.speaker_ref},
    )
    assert unblocked.output == {"ok": True, "voice_id": attributed.voice_id, "blocked": False}
    assert unblocked.create_response is False
    assert store.is_voice_blocked(attributed.voice_id) is False


@pytest.mark.asyncio
async def test_remember_and_reject_use_fixed_evidence_without_automatic_followup(tool_runtime) -> None:
    store, attributed = tool_runtime

    remembered = await execute_tool(
        "speaker_memory_remember_name",
        {"speaker_ref": attributed.speaker_ref, "name": "Аркадий"},
    )
    person_id = remembered.output["attribution"]["candidate"]["person_id"]
    rejected = await execute_tool(
        "speaker_memory_reject",
        {"speaker_ref": attributed.speaker_ref, "person_id": person_id},
    )

    assert isinstance(remembered, ToolResult)
    assert remembered.create_response is False
    assert rejected.create_response is False
    assert store.resolve_person_candidates(attributed.voice_id) == []


@pytest.mark.asyncio
async def test_confirm_rejects_person_not_proposed_for_reference(tool_runtime) -> None:
    _store, attributed = tool_runtime

    result = await execute_tool(
        "speaker_memory_confirm",
        {"speaker_ref": attributed.speaker_ref, "person_id": "p_hallucinated"},
    )

    assert result.create_response is True
    assert result.output == {
        "ok": False,
        "error": "person_candidate_invalid",
        "recommendation": "clarify",
    }


@pytest.mark.asyncio
async def test_database_lock_returns_bounded_retryable_tool_result(tool_runtime, monkeypatch) -> None:
    store, attributed = tool_runtime
    service = SpeakerMemoryService(store)
    monkeypatch.setattr(service, "inspect", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        sqlite3.OperationalError("database is locked")
    ))
    configure_tool_service(service, conversation_id_provider=lambda: "conv_1")

    result = await execute_tool("speaker_memory_inspect", {"speaker_ref": attributed.speaker_ref})

    assert result.create_response is True
    assert result.output == {"ok": False, "error": "speaker_memory_busy", "retryable": True}


@pytest.mark.asyncio
async def test_inspect_requests_followup_and_returns_no_embedding(tool_runtime) -> None:
    _store, attributed = tool_runtime

    result = await execute_tool("speaker_memory_inspect", {"speaker_ref": attributed.speaker_ref})

    assert result.create_response is True
    assert result.output["ok"] is True
    assert "centroid" not in str(result.output)
    assert "embedding" not in str(result.output)


@pytest.mark.asyncio
async def test_conversation_mismatch_fails_closed_without_mutation(tool_runtime) -> None:
    store, attributed = tool_runtime
    configure_tool_service(SpeakerMemoryService(store), conversation_id_provider=lambda: "conv_2")

    result = await execute_tool(
        "speaker_memory_remember_name",
        {"speaker_ref": attributed.speaker_ref, "name": "Не должен сохраниться"},
    )

    assert result.create_response is True
    assert result.output["ok"] is False
    assert result.output["error"] == "speaker_reference_invalid"
    assert store.resolve_person_candidates(attributed.voice_id) == []


@pytest.mark.asyncio
async def test_unconfigured_tool_runtime_fails_explicitly() -> None:
    configure_tool_service(None)

    with pytest.raises(RuntimeError, match="not configured"):
        await execute_tool("speaker_memory_inspect", {"speaker_ref": "sr_missing"})


def test_packaged_local_client_receives_native_speaker_tools(monkeypatch) -> None:
    from threading import Event
    from types import SimpleNamespace

    fake_speaker_handler = SimpleNamespace(
        service=SimpleNamespace(),
        conversation_id="conv_local",
    )
    monkeypatch.setattr(
        "speech_to_speech.s2s_pipeline.build_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(handlers=[fake_speaker_handler]),
    )

    manager = build_local_pipeline(parse_arguments([], command="local"), Event())
    client = manager.handlers[-1]

    assert {tool["name"] for tool in client.config.tools} >= {
        "speaker_memory_inspect",
        "speaker_memory_remember_name",
        "speaker_memory_confirm",
        "speaker_memory_reject",
    }
    assert client.config.tool_executor is not None
