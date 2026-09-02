"""Optional MCP v2 adapter over the native speaker-memory domain service."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .models import compact_voice_id
from .service import InvalidPersonCandidate, SpeakerMemoryService
from .store import SpeakerMemoryStore
from .tools import tool_error_output


def _tool_call(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except Exception as exc:
        output = tool_error_output(exc)
        if output is None:
            raise
        return output


def _resolve_voice(service: SpeakerMemoryService, voice: str, conversation_id: str) -> str:
    """Translate the compact public token to the private current-turn ref."""

    return service.resolve_reference_for_voice(voice, conversation_id=conversation_id)


def _compact_attribution(attribution: Any) -> dict[str, str]:
    return {
        "voice": compact_voice_id(attribution.voice_id),
        "name": attribution.candidate.name if attribution.candidate is not None else "unknown",
    }


def build_mcp_server(
    service: SpeakerMemoryService,
    *,
    conversation_id_provider: Callable[[], str],
) -> Any:
    """Register thin MCP tools; scoring and persistence remain in the service."""

    try:
        from mcp.server import MCPServer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The MCP adapter requires speech-to-speech[speaker-memory-mcp] (mcp>=2,<3)."
        ) from exc

    server = MCPServer("HuggingVoice Speaker Memory")

    @server.tool()
    def speaker_memory_inspect(voice: str) -> dict[str, Any]:
        return _tool_call(
            lambda: _compact_attribution(
                service.inspect(
                    _resolve_voice(service, voice, conversation_id_provider()),
                    conversation_id=conversation_id_provider(),
                )
            )
        )

    @server.tool()
    def speaker_memory_remember_name(voice: str, name: str) -> dict[str, Any]:
        return _tool_call(
            lambda: _compact_attribution(
                service.remember_name(
                    _resolve_voice(service, voice, conversation_id_provider()),
                    name,
                    conversation_id=conversation_id_provider(),
                )
            )
        )

    @server.tool()
    def speaker_memory_confirm(voice: str) -> dict[str, Any]:
        return _tool_call(
            lambda: _decision(service, voice, conversation_id_provider(), confirm=True)
        )

    @server.tool()
    def speaker_memory_reject(voice: str) -> dict[str, Any]:
        return _tool_call(
            lambda: _decision(service, voice, conversation_id_provider(), confirm=False)
        )

    @server.tool()
    def speaker_memory_block_voice(voice: str, reason: str) -> dict[str, Any]:
        return _tool_call(
            lambda: _block_voice(
                service,
                voice,
                conversation_id_provider(),
                blocked=True,
                reason=reason,
            )
        )

    @server.tool()
    def speaker_memory_unblock_voice(voice: str) -> dict[str, Any]:
        return _tool_call(
            lambda: _block_voice(
                service,
                voice,
                conversation_id_provider(),
                blocked=False,
            )
        )

    @server.tool()
    def speaker_memory_remember_fact(
        voice: str,
        fact: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        return _tool_call(
            lambda: _remember_fact(
                service,
                voice,
                fact,
                topic,
                conversation_id_provider(),
            )
        )

    @server.tool()
    def speaker_memory_recall(
        voice: str,
        query: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        return _tool_call(
            lambda: _recall(service, voice, query, limit, conversation_id_provider())
        )

    @server.tool()
    def speaker_memory_forget(
        voice: str,
        scope: str,
        fact_id: str | None = None,
    ) -> dict[str, Any]:
        return _tool_call(
            lambda: _forget(service, voice, scope, fact_id, conversation_id_provider())
        )

    return server


def _decision(
    service: SpeakerMemoryService,
    voice: str,
    conversation_id: str,
    *,
    confirm: bool,
) -> dict[str, str]:
    speaker_ref = _resolve_voice(service, voice, conversation_id)
    candidate = service.inspect(speaker_ref, conversation_id=conversation_id).candidate
    if candidate is None:
        raise InvalidPersonCandidate("no candidate was proposed for this voice")
    attribution = (
        service.confirm(speaker_ref, candidate.person_id, conversation_id=conversation_id)
        if confirm
        else service.reject(speaker_ref, candidate.person_id, conversation_id=conversation_id)
    )
    return _compact_attribution(attribution)


def _block_voice(
    service: SpeakerMemoryService,
    voice: str,
    conversation_id: str,
    *,
    blocked: bool,
    reason: str | None = None,
) -> dict[str, str]:
    speaker_ref = _resolve_voice(service, voice, conversation_id)
    before = service.inspect(speaker_ref, conversation_id=conversation_id)
    service.set_voice_blocked(
        speaker_ref,
        blocked=blocked,
        reason=reason,
        conversation_id=conversation_id,
    )
    return _compact_attribution(before)


def _remember_fact(
    service: SpeakerMemoryService,
    voice: str,
    fact: str,
    topic: str | None,
    conversation_id: str,
) -> dict[str, str]:
    speaker_ref = _resolve_voice(service, voice, conversation_id)
    service.remember_fact(
        speaker_ref,
        fact,
        topic=topic,
        conversation_id=conversation_id,
    )
    return _compact_attribution(service.inspect(speaker_ref, conversation_id=conversation_id))


def _recall(
    service: SpeakerMemoryService,
    voice: str,
    query: str,
    limit: int,
    conversation_id: str,
) -> dict[str, Any]:
    speaker_ref = _resolve_voice(service, voice, conversation_id)
    attribution = service.inspect(speaker_ref, conversation_id=conversation_id)
    facts = service.recall(
        speaker_ref,
        query=query,
        limit=limit,
        conversation_id=conversation_id,
    )
    return {
        **_compact_attribution(attribution),
        "facts": [{"topic": fact.topic, "fact": fact.fact} for fact in facts],
    }


def _forget(
    service: SpeakerMemoryService,
    voice: str,
    scope: str,
    fact_id: str | None,
    conversation_id: str,
) -> dict[str, Any]:
    speaker_ref = _resolve_voice(service, voice, conversation_id)
    attribution = service.inspect(speaker_ref, conversation_id=conversation_id)
    deleted = service.forget(
        speaker_ref,
        scope=scope,
        fact_id=fact_id,
        conversation_id=conversation_id,
    )
    return {**_compact_attribution(attribution), "deleted": deleted}


def _default_database_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return root / "huggingvoice" / "speaker-memory.sqlite3"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HuggingVoice speaker-memory MCP server")
    parser.add_argument(
        "--database",
        default=os.environ.get("HUGGINGVOICE_SPEAKER_MEMORY_DATABASE", str(_default_database_path())),
    )
    parser.add_argument(
        "--conversation-id",
        default=os.environ.get("HUGGINGVOICE_SPEAKER_CONVERSATION_ID"),
        help="Conversation scope that issued the current-turn voice tokens.",
    )
    arguments = parser.parse_args(argv)
    if not arguments.conversation_id:
        parser.error("--conversation-id or HUGGINGVOICE_SPEAKER_CONVERSATION_ID is required")

    store = SpeakerMemoryStore(arguments.database)
    try:
        service = SpeakerMemoryService(store)
        server = build_mcp_server(service, conversation_id_provider=lambda: arguments.conversation_id)
        server.run()
    finally:
        store.close()


if __name__ == "__main__":
    main()
