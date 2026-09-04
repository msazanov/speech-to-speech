from speech_to_speech.LLM.adaptive_reasoning import reasoning_decision


def test_simple_query_skips_reasoning():
    assert reasoning_decision("ты меня слышишь") == (False, "simple_query")


def test_complex_query_enables_reasoning():
    enabled, reason = reasoning_decision("сравни два варианта и составь пошаговый план")
    assert enabled is True
    assert reason == "complexity_marker"


def test_long_query_enables_reasoning():
    enabled, reason = reasoning_decision("слово " * 50)
    assert enabled is True
    assert reason == "long_query"
