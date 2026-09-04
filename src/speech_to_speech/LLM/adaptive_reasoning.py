"""Cheap, deterministic gate for local model reasoning."""

from __future__ import annotations

import re

_COMPLEX_MARKERS = re.compile(
    r"\b(почему|зачем|сравни|сравнение|проанализируй|объясни подробно|пошагово|"
    r"составь план|рассчитай|вычисли|несколько|если|услови|перечисли|таблиц|"
    r"код|скрипт|архитектур|why|compare|analy[sz]e|step[- ]by[- ]step|plan|"
    r"calculate|debug|implement)\b",
    re.IGNORECASE,
)


def reasoning_decision(text: str) -> tuple[bool, str]:
    """Return whether a user message deserves the deeper generation path."""
    normalized = " ".join(text.split())
    if len(normalized) >= 180:
        return True, "long_query"
    if normalized.count("?") >= 2 or normalized.count(";") >= 1:
        return True, "multi_part_query"
    if _COMPLEX_MARKERS.search(normalized):
        return True, "complexity_marker"
    return False, "simple_query"
