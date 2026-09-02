from __future__ import annotations

import json

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.fast_tool_router import route_fast_tool

TOOLS = [
    {"type": "function", "name": "web_search"},
    {"type": "function", "name": "speaker_memory_remember_name"},
    {"type": "function", "name": "speaker_memory_remember_fact"},
    {"type": "function", "name": "speaker_memory_recall"},
    {"type": "function", "name": "speaker_memory_confirm"},
    {"type": "function", "name": "speaker_memory_reject"},
]


def with_speaker(text: str, *, state: str = "unknown") -> str:
    context = {"voice": "deadbeaf", "name": "unknown", "state": state}
    return f"<huggingvoice_speaker_context>{json.dumps(context)}</huggingvoice_speaker_context>\n{text}"


def test_routes_name_introduction_using_only_trusted_reference() -> None:
    call = route_fast_tool(with_speaker("Меня зовут Михаил"), TOOLS)

    assert isinstance(call, ResponseFunctionToolCall)
    assert call.name == "speaker_memory_remember_name"
    assert json.loads(call.arguments) == {"voice": "deadbeaf", "name": "Михаил"}
    long_stt = route_fast_tool(
        with_speaker("меня зовут аркадий это длинная проверочная фраза для распознавания голоса"),
        TOOLS,
    )
    assert long_stt is not None
    assert json.loads(long_stt.arguments) == {"voice": "deadbeaf", "name": "аркадий"}
    compact = route_fast_tool(with_speaker("я Марат"), TOOLS)
    assert compact is not None and compact.name == "speaker_memory_remember_name"
    assert json.loads(compact.arguments) == {"voice": "deadbeaf", "name": "Марат"}
    assert route_fast_tool(with_speaker("я хочу поговорить"), TOOLS) is None
    named = route_fast_tool(with_speaker("моё имя — Тимур"), TOOLS)
    assert named is not None
    assert json.loads(named.arguments)["name"] == "Тимур"
    assert route_fast_tool(with_speaker("Как меня зовут?"), TOOLS) is None
    assert route_fast_tool("Меня зовут Михаил", TOOLS) is None
    confirmed_name = route_fast_tool(
        with_speaker("да, верно"),
        TOOLS,
        previous_assistant_text="Ты Марат, верно?",
    )
    assert confirmed_name is not None and confirmed_name.name == "speaker_memory_remember_name"
    assert json.loads(confirmed_name.arguments) == {"voice": "deadbeaf", "name": "Марат"}

    candidate_context = {
        "speaker_ref": "sr_trusted",
        "voice_id": "v_1",
        "state": "conflict",
        "candidate": {"person_id": "p_1", "name": "Михаил"},
    }
    prefix = f"<huggingvoice_speaker_context>{json.dumps(candidate_context)}</huggingvoice_speaker_context>\n"
    confirmed = route_fast_tool(prefix + "Да", TOOLS)
    assert confirmed is not None and confirmed.name == "speaker_memory_confirm"
    assert json.loads(confirmed.arguments) == {"voice": "00000001"}
    rejected = route_fast_tool(prefix + "Нет, это не я", TOOLS)
    assert rejected is not None and rejected.name == "speaker_memory_reject"
    assert json.loads(rejected.arguments) == {"voice": "00000001"}


def test_routes_web_search_without_inventing_speaker_reference() -> None:
    call = route_fast_tool("Поищи в интернете последние новости о GigaAM", TOOLS)

    assert call is not None
    assert call.name == "web_search"
    assert json.loads(call.arguments) == {"query": "последние новости о GigaAM"}
    assert "speaker_ref" not in call.arguments
