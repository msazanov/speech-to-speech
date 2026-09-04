"""Native OpenAI Realtime function tools for speaker identity memory."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from .models import SpeakerReferenceError, compact_voice_id
from .service import IdentityNotConfirmed, InvalidPersonCandidate, SpeakerMemoryService

CREATE_RESPONSE = True

_VOICE_PROPERTY = {
    "type": "string",
    "minLength": 4,
    "maxLength": 32,
    "description": "Current 4-byte voice token.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "speaker_memory_inspect",
        "description": "Inspect current voice identity.",
        "parameters": {
            "type": "object",
            "properties": {"voice": _VOICE_PROPERTY},
            "required": ["voice"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_remember_name",
        "description": "Save the speaker's explicit name.",
        "parameters": {
            "type": "object",
            "properties": {
                "voice": _VOICE_PROPERTY,
                "name": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["voice", "name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_enroll",
        "description": "Start guided voice calibration for the current speaker. Returns phrases to ask them to repeat one by one; each repetition strengthens this voice's memory.",
        "parameters": {
            "type": "object",
            "properties": {"voice": _VOICE_PROPERTY},
            "required": ["voice"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_confirm",
        "description": "Confirm the current voice candidate.",
        "parameters": {
            "type": "object",
            "properties": {"voice": _VOICE_PROPERTY},
            "required": ["voice"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_reject",
        "description": "Reject the current voice candidate.",
        "parameters": {
            "type": "object",
            "properties": {"voice": _VOICE_PROPERTY},
            "required": ["voice"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_block_voice",
        "description": "Blacklist the current voice.",
        "parameters": {
            "type": "object",
            "properties": {
                "voice": _VOICE_PROPERTY,
                "reason": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["voice", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_unblock_voice",
        "description": "Remove the voice from the blacklist.",
        "parameters": {
            "type": "object",
            "properties": {"voice": _VOICE_PROPERTY},
            "required": ["voice"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_remember_fact",
        "description": "Save a fact stated by the confirmed speaker.",
        "parameters": {
            "type": "object",
            "properties": {
                "voice": _VOICE_PROPERTY,
                "fact": {"type": "string", "minLength": 1, "maxLength": 500},
                "topic": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["voice", "fact"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_recall",
        "description": "Recall private facts for the confirmed speaker.",
        "parameters": {
            "type": "object",
            "properties": {
                "voice": _VOICE_PROPERTY,
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["voice", "query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "speaker_memory_forget",
        "description": "Forget requested private facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "voice": _VOICE_PROPERTY,
                "scope": {"type": "string", "enum": ["fact", "facts"]},
                "fact_id": {"type": "string", "minLength": 3},
            },
            "required": ["voice", "scope"],
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

    try:
        speaker_ref = _resolve_speaker_ref(service, conversation_id, arguments)
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
        elif name == "speaker_memory_enroll":
            session = service.start_enrollment(speaker_ref, conversation_id=conversation_id)
            attribution = service.inspect(speaker_ref, conversation_id=conversation_id)
            return ToolResult(
                output={
                    **_compact_attribution(attribution),
                    "enroll": {
                        "phrases": list(session.phrases),
                        "remaining": session.remaining,
                    },
                },
                create_response=True,
            )
        elif name == "speaker_memory_confirm":
            candidate = service.inspect(speaker_ref, conversation_id=conversation_id).candidate
            if candidate is None:
                raise InvalidPersonCandidate("no candidate was proposed for this voice")
            attribution = service.confirm(
                speaker_ref,
                candidate.person_id,
                conversation_id=conversation_id,
            )
            create_response = False
        elif name == "speaker_memory_reject":
            candidate = service.inspect(speaker_ref, conversation_id=conversation_id).candidate
            if candidate is None:
                raise InvalidPersonCandidate("no candidate was proposed for this voice")
            attribution = service.reject(
                speaker_ref,
                candidate.person_id,
                conversation_id=conversation_id,
            )
            create_response = False
        elif name == "speaker_memory_block_voice":
            before = service.inspect(speaker_ref, conversation_id=conversation_id)
            service.set_voice_blocked(
                speaker_ref,
                blocked=True,
                reason=_required_string(arguments, "reason"),
                conversation_id=conversation_id,
            )
            return ToolResult(
                output=_compact_attribution(before),
                create_response=False,
            )
        elif name == "speaker_memory_unblock_voice":
            before = service.inspect(speaker_ref, conversation_id=conversation_id)
            service.set_voice_blocked(
                speaker_ref,
                blocked=False,
                conversation_id=conversation_id,
            )
            return ToolResult(
                output=_compact_attribution(before),
                create_response=False,
            )
        elif name == "speaker_memory_remember_fact":
            service.remember_fact(
                speaker_ref,
                _required_string(arguments, "fact"),
                topic=arguments.get("topic"),
                conversation_id=conversation_id,
            )
            attribution = service.inspect(speaker_ref, conversation_id=conversation_id)
            return ToolResult(
                output=_compact_attribution(attribution),
                create_response=False,
            )
        elif name == "speaker_memory_recall":
            attribution = service.inspect(speaker_ref, conversation_id=conversation_id)
            facts = service.recall(
                speaker_ref,
                query=_required_string(arguments, "query"),
                limit=arguments.get("limit", 5),
                conversation_id=conversation_id,
            )
            return ToolResult(
                output={
                    **_compact_attribution(attribution),
                    "facts": [
                        {"topic": fact.topic, "fact": fact.fact}
                        for fact in facts
                    ],
                },
                create_response=True,
            )
        elif name == "speaker_memory_forget":
            attribution = service.inspect(speaker_ref, conversation_id=conversation_id)
            deleted = service.forget(
                speaker_ref,
                scope=_required_string(arguments, "scope"),
                fact_id=arguments.get("fact_id"),
                conversation_id=conversation_id,
            )
            return ToolResult(
                output={**_compact_attribution(attribution), "deleted": deleted},
                create_response=False,
            )
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
        output=_compact_attribution(attribution),
        create_response=create_response,
    )


def _compact_attribution(attribution: Any) -> dict[str, str]:
    return {
        "voice": compact_voice_id(attribution.voice_id),
        "name": attribution.candidate.name if attribution.candidate is not None else "unknown",
    }


def _resolve_speaker_ref(
    service: SpeakerMemoryService,
    conversation_id: str,
    arguments: dict[str, Any],
) -> str:
    """Translate the compact public token to a short-lived private reference.

    ``speaker_ref`` remains accepted only as a compatibility path for older
    local clients; it is no longer present in the model-visible schemas.
    """

    voice = arguments.get("voice")
    if isinstance(voice, str) and voice.strip():
        return service.resolve_reference_for_voice(voice, conversation_id=conversation_id)
    legacy_ref = arguments.get("speaker_ref")
    if isinstance(legacy_ref, str) and legacy_ref.strip():
        return legacy_ref
    raise ValueError("voice must be a string")


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value
