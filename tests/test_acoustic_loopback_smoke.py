import pytest
from scripts.acoustic_loopback_smoke import (
    build_parser,
    character_similarity,
    normalize_text,
    validate_reference,
    word_error_rate,
)


def test_acoustic_loopback_defaults_to_normal_playback_volume() -> None:
    args = build_parser().parse_args([])

    assert args.volume == pytest.approx(0.85)


def test_normalize_text_is_russian_friendly():
    assert normalize_text("  Ёлка, всё ещё здесь?!  ") == "елка все еще здесь"


@pytest.mark.parametrize(
    ("reference", "transcript", "expected"),
    [
        ("раз два три", "раз два три", 0.0),
        ("раз два три", "раз три", 1 / 3),
        ("раз два три", "раз два четыре", 1 / 3),
        ("раз два", "раз очень два", 1 / 2),
    ],
)
def test_word_error_rate_counts_word_edits(reference, transcript, expected):
    assert word_error_rate(reference, transcript) == pytest.approx(expected)


def test_character_similarity_ignores_case_and_punctuation():
    assert character_similarity("Привет, мир!", "привет мир") == 1.0


def test_validate_reference_accepts_at_most_twenty_words():
    text = " ".join(f"слово{index}" for index in range(20))

    assert validate_reference(text) == text


def test_validate_reference_rejects_more_than_twenty_words():
    text = " ".join(f"слово{index}" for index in range(21))

    with pytest.raises(ValueError, match="не более 20 слов"):
        validate_reference(text)


def test_validate_reference_rejects_empty_text():
    with pytest.raises(ValueError, match="не должна быть пустой"):
        validate_reference("  ")
