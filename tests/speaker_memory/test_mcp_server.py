from __future__ import annotations

import asyncio
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

from speech_to_speech.speaker_memory.mcp_server import build_mcp_server


class FakeMCPServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.functions = {}

    def tool(self):
        def register(function):
            self.functions[function.__name__] = function
            return function

        return register

    async def list_tools(self):
        return [SimpleNamespace(name=name) for name in self.functions]


class RecordingService:
    def __init__(self) -> None:
        self.calls = []

    def inspect(self, speaker_ref, *, conversation_id):
        self.calls.append(("inspect", speaker_ref, conversation_id))
        return SimpleNamespace(model_dump=lambda **_kwargs: {"state": "unknown"})

    def remember_name(self, speaker_ref, name, *, conversation_id):
        self.calls.append(("remember_name", speaker_ref, name, conversation_id))
        return SimpleNamespace(model_dump=lambda **_kwargs: {"state": "known"})

    def confirm(self, speaker_ref, person_id, *, conversation_id):
        self.calls.append(("confirm", speaker_ref, person_id, conversation_id))
        return SimpleNamespace(model_dump=lambda **_kwargs: {"state": "known"})

    def reject(self, speaker_ref, person_id, *, conversation_id):
        self.calls.append(("reject", speaker_ref, person_id, conversation_id))
        return SimpleNamespace(model_dump=lambda **_kwargs: {"state": "conflict"})

    def remember_fact(self, speaker_ref, fact, *, topic, conversation_id):
        self.calls.append(("remember_fact", speaker_ref, fact, topic, conversation_id))
        return SimpleNamespace(model_dump=lambda **_kwargs: {"id": "f_1", "fact": fact})

    def recall(self, speaker_ref, *, query, limit, conversation_id):
        self.calls.append(("recall", speaker_ref, query, limit, conversation_id))
        return [SimpleNamespace(model_dump=lambda **_kwargs: {"id": "f_1"})]

    def forget(self, speaker_ref, *, scope, fact_id, conversation_id):
        self.calls.append(("forget", speaker_ref, scope, fact_id, conversation_id))
        return 1


def install_fake_mcp(monkeypatch) -> None:
    mcp_module = ModuleType("mcp")
    server_module = ModuleType("mcp.server")
    server_module.MCPServer = FakeMCPServer
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)


def test_mcp_adapter_registers_same_semantic_tools(monkeypatch) -> None:
    install_fake_mcp(monkeypatch)
    server = build_mcp_server(RecordingService(), conversation_id_provider=lambda: "conv_1")

    names = {tool.name for tool in asyncio.run(server.list_tools())}

    assert names == {
        "speaker_memory_inspect",
        "speaker_memory_remember_name",
        "speaker_memory_confirm",
        "speaker_memory_reject",
        "speaker_memory_block_voice",
        "speaker_memory_unblock_voice",
        "speaker_memory_remember_fact",
        "speaker_memory_recall",
        "speaker_memory_forget",
    }


def test_mcp_adapter_delegates_to_service_with_bound_conversation(monkeypatch) -> None:
    install_fake_mcp(monkeypatch)
    service = RecordingService()
    server = build_mcp_server(service, conversation_id_provider=lambda: "conv_bound")

    result = server.functions["speaker_memory_remember_name"]("sr_1", "Аркадий")

    assert result == {"state": "known"}
    assert service.calls == [("remember_name", "sr_1", "Аркадий", "conv_bound")]


def test_importing_adapter_does_not_require_optional_mcp_sdk() -> None:
    assert callable(build_mcp_server)


def test_mcp_adapter_returns_structured_retryable_database_lock(monkeypatch) -> None:
    install_fake_mcp(monkeypatch)
    service = RecordingService()
    service.inspect = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        sqlite3.OperationalError("database is locked")
    )
    server = build_mcp_server(service, conversation_id_provider=lambda: "conv_1")

    result = server.functions["speaker_memory_inspect"]("sr_1")

    assert result == {"ok": False, "error": "speaker_memory_busy", "retryable": True}
