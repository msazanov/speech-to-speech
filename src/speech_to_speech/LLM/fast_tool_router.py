"""Small, deterministic tool router for latency-sensitive voice turns.

Some OpenAI-compatible local models describe the right action in plain text but
do not emit a structured ``tool_calls`` field. This module recognizes only a
few explicit, unambiguous voice commands and turns them into the same Realtime
function-call item used by the provider path. Speaker-memory mutations carry
only the compact current-turn voice token; the server resolves its private
short-lived reference before mutation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.speaker_memory.models import SpeakerAttribution, compact_voice_id
from speech_to_speech.utils.utils import _generate_id

_CONTEXT_OPEN = "<huggingvoice_speaker_context>"
_CONTEXT_CLOSE = "</huggingvoice_speaker_context>"
_MAX_FIELD_LENGTH = 200
_MAX_NAME_WORDS = 4
_INTRO_STOP_WORDS = frozenset(
    {
        # Avoid turning ordinary first-person sentences into a name when STT
        # emits the compact form "я хочу..." / "I'm trying...".
        "это",
        "я",
        "хочу",
        "могу",
        "буду",
        "тут",
        "здесь",
        "привет",
        "нужно",
        "надо",
        "давай",
        "тебе",
        "тебя",
        "here",
        "there",
        "trying",
        "going",
        "just",
        "the",
    }
)
_NAME_STOP_WORDS = frozenset(
    {
        # STT often removes punctuation, so stop an introduction before the
        # rest of the sentence instead of sending the whole utterance as the
        # person's name (the service deliberately rejects names over 80 chars).
        "это",
        "я",
        "мне",
        "меня",
        "мой",
        "моя",
        "мое",
        "моё",
        "говорю",
        "сейчас",
        "здесь",
        "для",
        "and",
        "i",
        "my",
        "this",
    }
)


def _tool_names(tools: Iterable[Any] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        value: Any = tool.get("name") if isinstance(tool, Mapping) else getattr(tool, "name", None)
        if isinstance(value, str):
            names.add(value)
    return names


def _speaker_context_data(text: str) -> tuple[dict[str, Any] | None, str]:
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
    if not isinstance(payload, dict):
        payload = None
    return payload, text[end + len(_CONTEXT_CLOSE) :].strip()


def _speaker_context(text: str) -> tuple[str | None, str]:
    payload, utterance = _speaker_context_data(text)
    voice = payload.get("voice") if payload is not None else None
    if not isinstance(voice, str) or not voice.strip():
        legacy_voice = payload.get("voice_id") if payload is not None else None
        voice = legacy_voice if isinstance(legacy_voice, str) and legacy_voice.strip() else None
    return voice, utterance


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


def _intro_name(value: str) -> str:
    """Extract a short name from a punctuation-free STT introduction."""

    words = _clean(value).split()
    if not words or words[0].casefold().strip(".,!?;:") in _INTRO_STOP_WORDS:
        return ""
    selected: list[str] = []
    for word in words:
        normalized = word.casefold().strip(".,!?;:")
        if selected and normalized in _NAME_STOP_WORDS:
            break
        selected.append(word)
        if len(selected) >= _MAX_NAME_WORDS:
            break
    return " ".join(selected)[:80]


def _confirmed_name(previous_assistant_text: str | None, utterance: str) -> str | None:
    """Extract a name from the immediately preceding identity question."""

    if not previous_assistant_text or not re.fullmatch(
        r"\s*(?:да|да\s*,\s*верно|верно|так|угу|ага|yes|yes\s*,\s*(?:right|correct)|correct|that's right|right)[.!?\s]*",
        utterance,
        flags=re.IGNORECASE,
    ):
        return None
    patterns = (
        r"(?:ты|вы)\s+([\w'-]{2,40})\s*,?\s*(?:верно|правильно|right|correct)\s*[?!.]*",
        r"(?:are\s+you|you're)\s+([A-Za-z][A-Za-z'-]{1,39})\s*,?\s*(?:right|correct)\s*[?!.]*",
    )
    for pattern in patterns:
        match = re.search(pattern, previous_assistant_text, flags=re.IGNORECASE)
        if match:
            return _intro_name(match.group(1))
    return None


def route_fast_tool(
    text: str,
    tools: Iterable[Any] | None,
    *,
    speaker: SpeakerAttribution | None = None,
    previous_assistant_text: str | None = None,
) -> ResponseFunctionToolCall | None:
    """Return one explicit local tool call, or ``None`` for ordinary text."""

    names = _tool_names(tools)
    context, utterance = _speaker_context_data(text)
    context_voice = context.get("voice") if context is not None else None
    if not isinstance(context_voice, str) or not context_voice.strip():
        context_voice = context.get("voice_id") if context is not None else None
    voice = (
        compact_voice_id(speaker.voice_id)
        if speaker is not None
        else compact_voice_id(context_voice) if isinstance(context_voice, str) and context_voice.strip() else None
    )
    if not isinstance(voice, str) or not voice.strip() or voice == "unknown":
        voice = None
    state = speaker.state.value if speaker is not None else context.get("state") if context is not None else None
    if not utterance:
        return None

    confirmed_name = _confirmed_name(previous_assistant_text, utterance)
    if confirmed_name and voice and "speaker_memory_remember_name" in names:
        return _call("speaker_memory_remember_name", {"voice": voice, "name": confirmed_name})

    if (
        voice
        and state in {"ambiguous", "conflict"}
        and "speaker_memory_confirm" in names
        and re.match(
            r"^\s*(?:да|угу|верно|правильно|точно|это\s+я|yes|correct|right|that's\s+me)\s*[.!?]*\s*$",
            utterance,
            flags=re.IGNORECASE,
        )
    ):
        return _call(
            "speaker_memory_confirm",
            {"voice": voice},
        )
    if (
        voice
        and state in {"ambiguous", "conflict"}
        and "speaker_memory_reject" in names
        and re.match(
            r"^\s*(?:нет(?:\s*,?\s*(?:не\s+я|это\s+не\s+я))?|не\s+я|это\s+не\s+я|неправильно|ошибка|no|not\s+me|wrong)\s*[.!?]*\s*$",
            utterance,
            flags=re.IGNORECASE,
        )
    ):
        return _call(
            "speaker_memory_reject",
            {"voice": voice},
        )

    if voice and "speaker_memory_remember_name" in names:
        match = re.match(
            r"^\s*(?:меня\s+зовут|зовут\s+меня|мо[её]\s+имя(?:\s*[-—–:]\s*)?|я(?:\s+это)?|my\s+name\s+is|call\s+me|i(?:\s+am|'m))\s+(.+?)\s*[.!?]*\s*$",
            utterance,
            flags=re.IGNORECASE,
        )
        if match:
            name = _intro_name(match.group(1))
            if name:
                return _call("speaker_memory_remember_name", {"voice": voice, "name": name})

    if voice and "speaker_memory_remember_fact" in names:
        match = re.match(
            r"^\s*(?:запомни(?:,?\s+что)?|remember\s+that)\s+(.+?)\s*[.!?]*\s*$",
            utterance,
            flags=re.IGNORECASE,
        )
        if match:
            fact = _clean(match.group(1))
            if fact:
                return _call("speaker_memory_remember_fact", {"voice": voice, "fact": fact})

    if voice and "speaker_memory_recall" in names and re.search(
        r"(?:что\s+ты\s+помнишь|что\s+ты\s+знаешь\s+обо\s+мне|вспомни|"
        r"what\s+do\s+you\s+remember|what\s+do\s+you\s+know\s+about\s+me|"
        r"remember\s+about\s+me)",
        utterance,
        flags=re.IGNORECASE,
    ):
        return _call(
            "speaker_memory_recall",
            {"voice": voice, "query": _clean(utterance, preserve_terminal_punctuation=True)},
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
