"""Small, deterministic tool router for latency-sensitive voice turns.

Some OpenAI-compatible local models describe the right action in plain text but
do not emit a structured ``tool_calls`` field. This module recognizes only a
few explicit, unambiguous voice commands and turns them into the same Realtime
function-call item used by the provider path. Speaker-memory mutations can only
use the short-lived reference injected by HuggingVoice.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.utils.utils import _generate_id

_CONTEXT_OPEN = "<huggingvoice_speaker_context>"
_CONTEXT_CLOSE = "</huggingvoice_speaker_context>"
_MAX_FIELD_LENGTH = 200


def _tool_names(tools: Iterable[Any] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        value: Any = tool.get("name") if isinstance(tool, Mapping) else getattr(tool, "name", None)
        if isinstance(value, str):
            names.add(value)
    return names


def _speaker_context(text: str) -> tuple[str | None, str]:
    start = text.find(_CONTEXT_OPEN)
    if start < 0:
        return None, text.strip()
    end = text.find(_CONTEXT_CLOSE, start + len(_CONTEXT_OPEN))
    if end < 0:
        return None, text.strip()
    raw = text[start + len(_CONTEXT_OPEN) : end]
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None, text[end + len(_CONTEXT_CLOSE) :].strip()
    reference = payload.get("speaker_ref") if isinstance(payload, dict) else None
    if not isinstance(reference, str) or not reference.strip():
        reference = None
    return reference, text[end + len(_CONTEXT_CLOSE) :].strip()


def _call(name: str, arguments: dict[str, Any]) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        call_id=_generate_id("call"),
        id=_generate_id("fc"),
        status="completed",
    )


def _clean(value: str, *, preserve_terminal_punctuation: bool = False) -> str:
    cleaned = " ".join(value.strip().split())
    if not preserve_terminal_punctuation:
        cleaned = cleaned.strip(" .,!?;:\n\t")
    return cleaned[:_MAX_FIELD_LENGTH]


def route_fast_tool(text: str, tools: Iterable[Any] | None) -> ResponseFunctionToolCall | None:
    """Return one explicit local tool call, or ``None`` for ordinary text."""

    names = _tool_names(tools)
    speaker_ref, utterance = _speaker_context(text)
    if not utterance:
        return None

    if speaker_ref and "speaker_memory_remember_name" in names:
        match = re.match(
            r"^\s*(?:меня\s+зовут|зовут\s+меня|my\s+name\s+is|call\s+me)\s+(.+?)\s*[.!?]*\s*$",
            utterance,
            flags=re.IGNORECASE,
        )
        if match:
            name = _clean(match.group(1))
            if name:
                return _call("speaker_memory_remember_name", {"speaker_ref": speaker_ref, "name": name})

    if speaker_ref and "speaker_memory_remember_fact" in names:
        match = re.match(
            r"^\s*(?:запомни(?:,?\s+что)?|remember\s+that)\s+(.+?)\s*[.!?]*\s*$",
            utterance,
            flags=re.IGNORECASE,
        )
        if match:
            fact = _clean(match.group(1))
            if fact:
                return _call("speaker_memory_remember_fact", {"speaker_ref": speaker_ref, "fact": fact})

    if speaker_ref and "speaker_memory_recall" in names and re.search(
        r"(?:что\s+ты\s+помнишь|что\s+ты\s+знаешь\s+обо\s+мне|как\s+меня\s+зовут|кто\s+я|вспомни|"
        r"what\s+do\s+you\s+remember|what\s+do\s+you\s+know\s+about\s+me|what\s+is\s+my\s+name|who\s+am\s+i|"
        r"remember\s+about\s+me)",
        utterance,
        flags=re.IGNORECASE,
    ):
        return _call(
            "speaker_memory_recall",
            {"speaker_ref": speaker_ref, "query": _clean(utterance, preserve_terminal_punctuation=True)},
        )

    if "web_search" in names:
        match = re.match(
            r"^\s*(?:поищи|найди|проверь|узнай)(?:\s+(?:в\s+интернете|в\s+сети|онлайн))?\s+(.+?)\s*[.!?]*\s*$"
            r"|^\s*(?:search|look\s+up|find)\s+(?:the\s+web\s+for\s+)?(.+?)\s*[.!?]*\s*$",
            utterance,
            flags=re.IGNORECASE,
        )
        if match:
            query = _clean(match.group(1) or match.group(2) or "")
            if query:
                return _call("web_search", {"query": query})
    return None
