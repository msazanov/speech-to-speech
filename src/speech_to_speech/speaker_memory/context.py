"""Bounded trusted metadata formatting for language-model input."""

from __future__ import annotations

import json

from .models import SpeakerAttribution, compact_voice_id

_CONTEXT_OPEN = "<huggingvoice_speaker_context>"
_CONTEXT_CLOSE = "</huggingvoice_speaker_context>"
_MAX_NAME_LENGTH = 80


def public_speaker_metadata(attribution: SpeakerAttribution | None) -> dict[str, object] | None:
    """Return the tiny display identity record sent over the realtime API."""

    if attribution is None:
        return None
    return {
        "voice": compact_voice_id(attribution.voice_id),
        "name": (
            attribution.candidate.name[:_MAX_NAME_LENGTH]
            if attribution.state.value == "known" and attribution.candidate is not None
            else "unknown"
        ),
    }


def format_speaker_context(attribution: SpeakerAttribution) -> str:
    """Serialize only the two compact fields the model needs each turn."""

    payload = public_speaker_metadata(attribution) or {"voice": "unknown", "name": "unknown"}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{_CONTEXT_OPEN}{encoded}{_CONTEXT_CLOSE}"


def add_speaker_context(transcript: str, attribution: SpeakerAttribution | None) -> str:
    """Keep legacy input byte-for-byte when no trusted attribution exists."""

    if attribution is None:
        return transcript
    return f"{format_speaker_context(attribution)}\n{transcript}"
