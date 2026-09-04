"""Guided voice calibration: phrase bank and per-conversation session state."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

# Phonetically diverse Russian-first calibration utterances.  They span
# different vowels, consonant clusters, and intonations so a few repetitions
# give the speaker embedder a much richer voiceprint than casual short turns.
ENROLL_PHRASES: tuple[str, ...] = (
    "Включи свет",
    "Какая сейчас погода?",
    "Какой курс биткоина?",
    "Сколько у паука ног?",
    "Поставь будильник на семь утра",
    "Напомни купить молоко",
    "Который час?",
    "Открой окно, здесь душно",
    "Завтрак готов, иди к столу",
    "Расскажи анекдот про программиста",
    "Закрой дверь, пожалуйста",
    "Спасибо, ты большая умница",
)

ENROLL_SAMPLE_TARGET = 3
ENROLL_SESSION_TTL_S = 180.0


@dataclass
class EnrollmentSession:
    """One guided calibration run anchored to a canonical voice cluster."""

    voice_id: str
    phrases: tuple[str, ...]
    remaining: int
    expires_at: float

    @property
    def active(self) -> bool:
        return self.remaining > 0 and time.monotonic() < self.expires_at


def sample_phrases(count: int = ENROLL_SAMPLE_TARGET) -> tuple[str, ...]:
    if count < 1:
        raise ValueError("enrollment phrase count must be positive")
    if count > len(ENROLL_PHRASES):
        raise ValueError("enrollment phrase count exceeds the phrase bank")
    return tuple(random.sample(ENROLL_PHRASES, count))
