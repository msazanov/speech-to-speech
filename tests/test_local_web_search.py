from __future__ import annotations

from speech_to_speech.local_web_search import MAX_RESULTS, TOOLS, execute_tool


def test_web_search_tool_has_a_bounded_query_schema() -> None:
    tool = TOOLS[0]
    assert tool["name"] == "web_search"
    assert tool["parameters"]["properties"]["query"]["maxLength"] == 240
    assert tool["parameters"]["additionalProperties"] is False


async def test_web_search_normalizes_and_bounds_results(monkeypatch) -> None:
    class FakeDDGS:
        def __init__(self, *, timeout):
            assert timeout == 8

        def text(self, query, **kwargs):
            assert query == "новости GigaAM"
            assert kwargs["region"] == "ru-ru"
            return [
                {"title": "  Заголовок  ", "body": " текст   результата ", "href": "https://example.test/1"}
            ] * (MAX_RESULTS + 2)

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    result = await execute_tool("web_search", {"query": "  новости   GigaAM "})

    assert result.output["ok"] is True
    assert len(result.output["results"]) == MAX_RESULTS
    assert result.output["results"][0] == {
        "title": "Заголовок",
        "snippet": "текст результата",
        "url": "https://example.test/1",
    }


async def test_web_search_handles_invalid_query_and_provider_failure(monkeypatch) -> None:
    invalid = await execute_tool("web_search", {"query": " "})
    assert invalid.output == {"ok": False, "error": "invalid_query"}

    class BrokenDDGS:
        def __init__(self, **kwargs):
            pass

        def text(self, *args, **kwargs):
            raise RuntimeError("provider detail must not be returned")

    monkeypatch.setattr("ddgs.DDGS", BrokenDDGS)
    failed = await execute_tool("web_search", {"query": "погода сейчас"})
    assert failed.output == {
        "ok": False,
        "query": "погода сейчас",
        "error": "search_unavailable",
    }
