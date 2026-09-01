from __future__ import annotations

import json

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.fast_tool_router import route_fast_tool

TOOLS = [
    {"type": "function", "name": "web_search"},
    {"type": "function", "name": "speaker_memory_remember_name"},
    {"type": "function", "name": "speaker_memory_remember_fact"},
    {"type": "function", "name": "speaker_memory_recall"},
]


def with_speaker(text: str, *, state: str = "unknown") -> str:
    context = {"speaker_ref": "sr_trusted", "voice_id": "v_1", "state": state}
    return f"<huggingvoice_speaker_context>{json.dumps(context)}</huggingvoice_speaker_context>\n{text}"


def test_routes_name_introduction_using_only_trusted_reference() -> None:
    call = route_fast_tool(with_speaker("Меня зовут Михаил"), TOOLS)

    assert isinstance(call, ResponseFunctionToolCall)
    assert call.name == "speaker_memory_remember_name"
    assert json.loads(call.arguments) == {"speaker_ref": "sr_trusted", "name": "Михаил"}
    long_stt = route_fast_tool(
        with_speaker("меня зовут аркадий это длинная проверочная фраза для распознавания голоса"),
        TOOLS,
    )
    assert long_stt is not None
    assert json.loads(long_stt.arguments) == {"speaker_ref": "sr_trusted", "name": "аркадий"}
    recall = route_fast_tool(with_speaker("Как меня зовут?"), TOOLS)
    assert recall is not None and recall.name == "speaker_memory_recall"
    assert json.loads(recall.arguments) == {"speaker_ref": "sr_trusted", "query": "Как меня зовут?"}
    assert route_fast_tool("Меня зовут Михаил", TOOLS) is None


def test_routes_web_search_without_inventing_speaker_reference() -> None:
    call = route_fast_tool("Поищи в интернете последние новости о GigaAM", TOOLS)

    assert call is not None
    assert call.name == "web_search"
    assert json.loads(call.arguments) == {"query": "последние новости о GigaAM"}
    assert "speaker_ref" not in call.arguments
