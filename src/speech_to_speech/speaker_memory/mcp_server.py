"""Optional MCP v2 adapter over the native speaker-memory domain service."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .service import SpeakerMemoryService
from .store import SpeakerMemoryStore


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
    def speaker_memory_inspect(speaker_ref: str) -> dict[str, Any]:
        return service.inspect(
            speaker_ref,
            conversation_id=conversation_id_provider(),
        ).model_dump(mode="json", exclude_none=True)

    @server.tool()
    def speaker_memory_remember_name(speaker_ref: str, name: str) -> dict[str, Any]:
        return service.remember_name(
            speaker_ref,
            name,
            conversation_id=conversation_id_provider(),
        ).model_dump(mode="json", exclude_none=True)

    @server.tool()
    def speaker_memory_confirm(speaker_ref: str, person_id: str) -> dict[str, Any]:
        return service.confirm(
            speaker_ref,
            person_id,
            conversation_id=conversation_id_provider(),
        ).model_dump(mode="json", exclude_none=True)

    @server.tool()
    def speaker_memory_reject(speaker_ref: str, person_id: str) -> dict[str, Any]:
        return service.reject(
            speaker_ref,
            person_id,
            conversation_id=conversation_id_provider(),
        ).model_dump(mode="json", exclude_none=True)

    @server.tool()
    def speaker_memory_remember_fact(
        speaker_ref: str,
        fact: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        return service.remember_fact(
            speaker_ref,
            fact,
            topic=topic,
            conversation_id=conversation_id_provider(),
        ).model_dump(mode="json")

    @server.tool()
    def speaker_memory_recall(speaker_ref: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        facts = service.recall(
            speaker_ref,
            query=query,
            limit=limit,
            conversation_id=conversation_id_provider(),
        )
        return [fact.model_dump(mode="json") for fact in facts]

    @server.tool()
    def speaker_memory_forget(
        speaker_ref: str,
        scope: str,
        fact_id: str | None = None,
    ) -> dict[str, Any]:
        deleted = service.forget(
            speaker_ref,
            scope=scope,
            fact_id=fact_id,
            conversation_id=conversation_id_provider(),
        )
        return {"deleted": deleted}

    return server


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
        help="Conversation scope that issued the short-lived speaker_ref values.",
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
