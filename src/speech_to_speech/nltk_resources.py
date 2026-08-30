from __future__ import annotations

import nltk

REQUIRED_NLTK_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)


def ensure_required_nltk_resources() -> None:
    for resource_path, package_name in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            nltk.download(package_name)
