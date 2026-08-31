from __future__ import annotations

import numpy as np
from scripts.synthetic_speaker_memory_smoke import (
    AGENT_PROMPT,
    DEFAULT_MICROPHONE_TIMEOUT_S,
    SUCCESS_REPLY,
    SYNTHETIC_SPEAKERS,
    resolve_confirmation,
    response_for_transcript,
    summarize_similarity,
)


def test_yes_challenge_uses_the_exact_requested_dialogue() -> None:
    assert AGENT_PROMPT == "Для теста ответов скажите ДА"
    assert DEFAULT_MICROPHONE_TIMEOUT_S == 30.0
    assert response_for_transcript("Да!") == "Пизда, тест пройден"
    assert SUCCESS_REPLY == "Пизда, тест пройден"
    assert response_for_transcript("давай продолжим") is None
    assert resolve_confirmation("Да", timed_out=False) == ("Да", "microphone")
    assert resolve_confirmation("", timed_out=False) == (None, "listening")
    assert resolve_confirmation("", timed_out=True) == ("Да", "synthetic_timeout")
    assert SYNTHETIC_SPEAKERS == ("xenia", "baya", "aidar")


def test_similarity_summary_separates_same_and_different_synthetic_voices() -> None:
    labels = ["xenia", "xenia", "aidar", "aidar"]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
        dtype=np.float32,
    )

    summary = summarize_similarity(labels, embeddings)

    assert summary["same_voice_min"] > 0.99
    assert summary["different_voice_max"] < 0.03
    assert summary["separation_margin"] > 0.96
