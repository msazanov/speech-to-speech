"""Bounded trusted metadata formatting for language-model input."""

from __future__ import annotations

import json

from .models import SpeakerAttribution

_CONTEXT_OPEN = "<huggingvoice_speaker_context>"
_CONTEXT_CLOSE = "</huggingvoice_speaker_context>"
_MAX_NAME_LENGTH = 80


def public_speaker_metadata(attribution: SpeakerAttribution | None) -> dict[str, object] | None:
    """Return display-only identity data, never the mutation-capable speaker reference."""

    if attribution is None:
        return None
    payload: dict[str, object] = {"state": attribution.state.value}
    if attribution.voice_id is not None:
        payload["voice_id"] = attribution.voice_id
    if attribution.state.value == "known" and attribution.candidate is not None:
        payload["person"] = {
            "person_id": attribution.candidate.person_id,
            "name": attribution.candidate.name[:_MAX_NAME_LENGTH],
        }
    return payload


def format_speaker_context(attribution: SpeakerAttribution) -> str:
    """Serialize only the compact identity indicator as escaped JSON data."""

    candidate = None
    if attribution.candidate is not None:
        candidate = {
            "person_id": attribution.candidate.person_id,
            "name": attribution.candidate.name[:_MAX_NAME_LENGTH],
        }
    payload = {
        "speaker_ref": attribution.speaker_ref,
        "voice_id": attribution.voice_id,
        "state": attribution.state.value,
        "candidate": candidate,
        "margin": attribution.margin,
        "recommendation": attribution.recommendation,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{_CONTEXT_OPEN}{encoded}{_CONTEXT_CLOSE}"


def add_speaker_context(transcript: str, attribution: SpeakerAttribution | None) -> str:
    """Keep legacy input byte-for-byte when no trusted attribution exists."""

    if attribution is None:
        return transcript
    return f"{format_speaker_context(attribution)}\n{transcript}"
