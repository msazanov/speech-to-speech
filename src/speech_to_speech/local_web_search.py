"""Small, keyless web-search tool for a local Realtime audio client."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from speech_to_speech.api.openai_realtime.audio_client import ToolResult

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
MAX_QUERY_CHARS = 240
MAX_TITLE_CHARS = 200
MAX_SNIPPET_CHARS = 700
MAX_URL_CHARS = 600

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the public web for current information or facts that need verification. "
            "Returns up to five titles, snippets, and source URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": MAX_QUERY_CHARS,
                    "description": "A concise search query, preferably in the user's language.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
]


def _bounded_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _search(query: str) -> list[dict[str, str]]:
    from ddgs import DDGS

    raw_results = DDGS(timeout=8).text(
        query,
        region="ru-ru",
        safesearch="moderate",
        max_results=MAX_RESULTS,
        backend="auto",
    )
    return [
        {
            "title": _bounded_text(item.get("title"), MAX_TITLE_CHARS),
            "snippet": _bounded_text(item.get("body"), MAX_SNIPPET_CHARS),
            "url": _bounded_text(item.get("href"), MAX_URL_CHARS),
        }
        for item in raw_results[:MAX_RESULTS]
        if isinstance(item, dict)
    ]


async def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute a bounded web search without exposing provider failures to speech."""

    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")
    query = _bounded_text(arguments.get("query"), MAX_QUERY_CHARS + 1)
    if len(query) < 2 or len(query) > MAX_QUERY_CHARS:
        return ToolResult({"ok": False, "error": "invalid_query"})

    logger.info("web_search query=%s max_results=%d", query, MAX_RESULTS)
    try:
        results = await asyncio.wait_for(asyncio.to_thread(_search, query), timeout=10)
    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, type(exc).__name__)
        return ToolResult({"ok": False, "query": query, "error": "search_unavailable"})

    return ToolResult(
        {
            "ok": True,
            "query": query,
            "searched_at": datetime.now(timezone.utc).date().isoformat(),
            "results": results,
        }
    )
