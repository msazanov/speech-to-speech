"""Native OpenAI Realtime function tools for speaker identity memory."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from .models import SpeakerReferenceError
from .service import IdentityNotConfirmed, InvalidPersonCandidate, SpeakerMemoryService

CREATE_RESPONSE = True

_REFERENCE_PROPERTY = {
    "type": "string",
    "minLength": 4,
    "description": "Short-lived speaker reference from trusted HuggingVoice context.",
}
_PERSON_PROPERTY = {
    "type": "string",
    "minLength": 3,
    "description": "Candidate person ID returned by speaker_memory_inspect.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "speaker_memory_inspect",
        "description": "Inspect the current speaker identity state before clarifying a name.",
        "parameters": {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE_PROPERTY},
            "required": ["speaker_ref"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_remember_name",
        "description": "Remember a name explicitly given by the current speaker.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE_PROPERTY,
                "name": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["speaker_ref", "name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_confirm",
        "description": "Confirm that the current speaker is the proposed person after an affirmative answer.",
        "parameters": {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE_PROPERTY, "person_id": _PERSON_PROPERTY},
            "required": ["speaker_ref", "person_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_reject",
        "description": "Reject a proposed person after the current speaker denies the match.",
        "parameters": {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE_PROPERTY, "person_id": _PERSON_PROPERTY},
            "required": ["speaker_ref", "person_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_block_voice",
        "description": "Ignore the current voice in future turns after the user explicitly identifies it as unwanted background audio.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE_PROPERTY,
                "reason": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["speaker_ref", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_unblock_voice",
        "description": "Remove the current voice from the background-audio blacklist after an explicit correction.",
        "parameters": {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE_PROPERTY},
            "required": ["speaker_ref"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_remember_fact",
        "description": "Remember a personal fact explicitly stated by the confirmed current speaker.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE_PROPERTY,
                "fact": {"type": "string", "minLength": 1, "maxLength": 500},
                "topic": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["speaker_ref", "fact"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_recall",
        "description": "Recall relevant private facts only for a confirmed current speaker.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE_PROPERTY,
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["speaker_ref", "query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_forget",
        "description": "Forget one fact or all facts after an explicit request from the confirmed speaker.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE_PROPERTY,
                "scope": {"type": "string", "enum": ["fact", "facts"]},
                "fact_id": {"type": "string", "minLength": 3},
            },
            "required": ["speaker_ref", "scope"],
            "additionalProperties": False,
        },
    },
]

_service: SpeakerMemoryService | None = None
_conversation_id_provider: Callable[[], str] | None = None


def configure_tool_service(
    service: SpeakerMemoryService | None,
    *,
    conversation_id_provider: Callable[[], str] | None = None,
) -> None:
    """Configure the importable module; local pipelines use isolated closures."""

    global _service, _conversation_id_provider
    _service = service
    _conversation_id_provider = conversation_id_provider


def create_tool_executor(
    service: SpeakerMemoryService,
    *,
    conversation_id_provider: Callable[[], str],
) -> Callable[[str, dict[str, Any]], Any]:
    async def executor(name: str, arguments: dict[str, Any]) -> Any:
        return _execute(service, conversation_id_provider(), name, arguments)

    return executor


def tool_error_output(exc: Exception) -> dict[str, Any] | None:
    """Map domain and transient storage errors consistently across tool transports."""

    if isinstance(exc, IdentityNotConfirmed):
        return {"ok": False, "error": "identity_not_confirmed", "recommendation": "clarify"}
    if isinstance(exc, InvalidPersonCandidate):
        return {"ok": False, "error": "person_candidate_invalid", "recommendation": "clarify"}
    if isinstance(exc, SpeakerReferenceError):
        return {"ok": False, "error": "speaker_reference_invalid", "recommendation": "clarify"}
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).casefold()
        if "locked" in message or "busy" in message:
            return {"ok": False, "error": "speaker_memory_busy", "retryable": True}
    return None


async def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    if _service is None or _conversation_id_provider is None:
        raise RuntimeError("speaker memory tool service is not configured")
    return _execute(_service, _conversation_id_provider(), name, arguments)


def _execute(
    service: SpeakerMemoryService,
    conversation_id: str,
    name: str,
    arguments: dict[str, Any],
) -> Any:
    from speech_to_speech.api.openai_realtime.audio_client import ToolResult

    speaker_ref = arguments.get("speaker_ref")
    if not isinstance(speaker_ref, str):
        raise ValueError("speaker_ref must be a string")
    try:
        if name == "speaker_memory_inspect":
            attribution = service.inspect(speaker_ref, conversation_id=conversation_id)
            create_response = True
        elif name == "speaker_memory_remember_name":
            attribution = service.remember_name(
                speaker_ref,
                _required_string(arguments, "name"),
                conversation_id=conversation_id,
            )
            create_response = False
        elif name == "speaker_memory_confirm":
            attribution = service.confirm(
                speaker_ref,
                _required_string(arguments, "person_id"),
                conversation_id=conversation_id,
            )
            create_response = False
        elif name == "speaker_memory_reject":
            attribution = service.reject(
                speaker_ref,
                _required_string(arguments, "person_id"),
                conversation_id=conversation_id,
            )
            create_response = False
        elif name == "speaker_memory_block_voice":
            voice_id = service.set_voice_blocked(
                speaker_ref,
                blocked=True,
                reason=_required_string(arguments, "reason"),
                conversation_id=conversation_id,
            )
            return ToolResult(
                output={"ok": True, "voice_id": voice_id, "blocked": True},
                create_response=False,
            )
        elif name == "speaker_memory_unblock_voice":
            voice_id = service.set_voice_blocked(
                speaker_ref,
                blocked=False,
                conversation_id=conversation_id,
            )
            return ToolResult(
                output={"ok": True, "voice_id": voice_id, "blocked": False},
                create_response=False,
            )
        elif name == "speaker_memory_remember_fact":
            fact = service.remember_fact(
                speaker_ref,
                _required_string(arguments, "fact"),
                topic=arguments.get("topic"),
                conversation_id=conversation_id,
            )
            return ToolResult(
                output={"ok": True, "fact": fact.model_dump(mode="json")},
                create_response=False,
            )
        elif name == "speaker_memory_recall":
            facts = service.recall(
                speaker_ref,
                query=_required_string(arguments, "query"),
                limit=arguments.get("limit", 5),
                conversation_id=conversation_id,
            )
            return ToolResult(
                output={"ok": True, "facts": [fact.model_dump(mode="json") for fact in facts]},
                create_response=True,
            )
        elif name == "speaker_memory_forget":
            deleted = service.forget(
                speaker_ref,
                scope=_required_string(arguments, "scope"),
                fact_id=arguments.get("fact_id"),
                conversation_id=conversation_id,
            )
            return ToolResult(output={"ok": True, "deleted": deleted}, create_response=False)
        else:
            raise ValueError(f"unknown speaker memory tool: {name}")
    except Exception as exc:
        output = tool_error_output(exc)
        if output is None:
            raise
        return ToolResult(
            output=output,
            create_response=True,
        )
    return ToolResult(
        output={"ok": True, "attribution": attribution.model_dump(mode="json", exclude_none=True)},
        create_response=create_response,
    )


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value
