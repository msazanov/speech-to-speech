from __future__ import annotations

from speech_to_speech import nltk_resources


def test_existing_required_resources_do_not_trigger_network_download(monkeypatch) -> None:
    looked_up: list[str] = []

    def fake_find(resource_path: str):
        looked_up.append(resource_path)
        return object()

    def fail_download(package_name: str):
        raise AssertionError(f"unexpected download of {package_name}")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fail_download)

    nltk_resources.ensure_required_nltk_resources()

    assert looked_up == [
        "tokenizers/punkt_tab",
        "taggers/averaged_perceptron_tagger_eng",
    ]
