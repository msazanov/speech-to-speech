"""Native OpenAI Realtime function tools for speaker identity memory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import SpeakerReferenceError
from .service import SpeakerMemoryService

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
        else:
            raise ValueError(f"unknown speaker memory tool: {name}")
    except SpeakerReferenceError:
        return ToolResult(
            output={"ok": False, "error": "speaker_reference_invalid", "recommendation": "clarify"},
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
