"""Short spoken acknowledgements used while a provider is reasoning.

The pool is deliberately kept in source control instead of in the system prompt:
these words are local UX, not instructions for the model. Russian is the primary
pool; a small English pool keeps the same behavior for explicit English turns.
Profane variants are opt-in because the same profile can be used around other
people.
"""

from __future__ import annotations

import random
from threading import Lock

THINKING_PHRASES_RU: tuple[str, ...] = (
    "Пу-пу-пуууу... секунду.",
    "Ща, тут надо подумать...",
    "Погоди чутка, не мандражуй.",
    "Сейчас осмыслю.",
    "Тааак, ща подумаю.",
    "Хм, интересный поворот...",
    "Дай соображу.",
    "Секундочку, собираю мысли.",
    "Ща разложу всё по полочкам.",
    "Так, проверяю логику.",
    "Минуточку, считаю внимательно.",
    "Сейчас найду самый точный ответ.",
    "Хоп, включаю режим размышлений.",
    "Не спеши, я сверяю детали.",
    "Секунду, формулирую мысль.",
    "Так-так, дай подумать.",
    "Проверю себя и отвечу.",
    "Ща докопаюсь до сути.",
    "Хм... сейчас аккуратно разберусь.",
    "Дай мозгам секундочку.",
    "Сейчас свяжу все кусочки.",
    "Так, не теряю нить.",
    "Погоди, сверяю варианты.",
    "Секунду, ответ почти готов.",
    "Ща включу голову на полную.",
    "Сейчас разберёмся без суеты.",
    "Хм, тут нужен аккуратный ход.",
    "Дай мне мгновение на расчёт.",
    "Тааак... ищу лучший вариант.",
    "Секундочку, уточняю для себя.",
    "Сейчас соберу всё в ответ.",
    "Не дёргайся, я думаю.",
    "Ща проверю каждый шаг.",
    "Так, мозги прогреваются.",
    "Погоди немного, я вникаю.",
    "Сейчас будет осмысленный ответ.",
    "Хм, дай-ка прикину.",
    "Ща вытащу главное.",
    "Секунду, не хочу ошибиться.",
    "Так, думаю вслух про себя.",
    "Сейчас аккуратно сведу факты.",
    "Пу-пу... почти придумал.",
    "Хм, сейчас, блин, соберу решение.",
    "Ща, блин, соображу.",
    "Так, ё-моё, сейчас разберусь.",
    "Сейчас, чёрт побери, проверю.",
    "Погоди, мать его, ищу ответ.",
    "Ща раскручу эту хрень.",
    "Дай мозгам, блин, включиться.",
    "Не кипишуй, сейчас всё разрулим.",
)

# Keep stronger wording separate so it can be enabled explicitly in a private
# setup without making accidental profanity the default voice UX.
THINKING_PHRASES_RU_PROFANE: tuple[str, ...] = THINKING_PHRASES_RU[-8:]
THINKING_PHRASES_RU_CLEAN: tuple[str, ...] = THINKING_PHRASES_RU[:-8]

THINKING_PHRASES_EN: tuple[str, ...] = (
    "Give me a second to think.",
    "Let me work this out.",
    "One moment, checking the details.",
    "Hmm, interesting. Let me think.",
    "I am putting the pieces together.",
    "Give me a moment, please.",
    "Let me reason this through.",
    "One second, I am checking myself.",
    "I am looking for the clearest answer.",
    "Hold on, I am thinking it through.",
)

_LOCK = Lock()
_RNG = random.Random()
_REMAINING: dict[str, list[str]] = {}
_LAST: dict[str, str] = {}


def _next_from_pool(key: str, phrases: tuple[str, ...]) -> str:
    with _LOCK:
        remaining = _REMAINING.get(key)
        if not remaining:
            remaining = list(phrases)
            _RNG.shuffle(remaining)
            _REMAINING[key] = remaining
        phrase = remaining.pop()
        previous = _LAST.get(key)
        if phrase == previous and remaining:
            replacement = remaining.pop()
            remaining.insert(0, phrase)
            phrase = replacement
        _LAST[key] = phrase
        return phrase


def next_thinking_phrase(language_code: str | None, *, allow_profanity: bool = False) -> str:
    """Return a short phrase without immediate repetition.

    The no-repeat state is process-local and bounded by the pool size. It is not
    persisted and has no effect on the conversation or model context.
    """
    if (language_code or "").casefold().startswith("en"):
        return _next_from_pool("en", THINKING_PHRASES_EN)
    russian_pool = THINKING_PHRASES_RU if allow_profanity else THINKING_PHRASES_RU_CLEAN
    return _next_from_pool("ru:all" if allow_profanity else "ru:clean", russian_pool)


if len(THINKING_PHRASES_RU) != 50:  # pragma: no cover - source-data guard
    raise RuntimeError("Russian thinking phrase pool must contain exactly 50 templates")
