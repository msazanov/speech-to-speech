"""Typed records and domain errors for speaker memory."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict


class SpeakerState(str, Enum):
    """Identity state visible to the conversation policy."""

    KNOWN = "known"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    MIXED = "mixed"
    BLACKLISTED = "blacklisted"


class SpeakerMemoryError(Exception):
    """Base class for speaker-memory domain errors."""


class InvalidEmbedding(ValueError, SpeakerMemoryError):
    """Raised when an embedding cannot safely be stored or compared."""


class SpeakerReferenceError(SpeakerMemoryError):
    """Base class for invalid short-lived speaker references."""


class UnknownSpeakerReference(SpeakerReferenceError):
    """Raised when a speaker reference does not exist."""


class ExpiredSpeakerReference(SpeakerReferenceError):
    """Raised when a speaker reference is no longer valid."""


class SupersededSpeakerReference(SpeakerReferenceError):
    """Raised when a newer speaker turn replaced this mutation authority."""


class SpeakerReferenceConversationMismatch(SpeakerReferenceError):
    """Raised when a reference is used outside its issuing conversation."""


class VoiceCluster(BaseModel):
    """A stable voice identifier and its adaptive normalized centroid."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    centroid: np.ndarray
    sample_count: int
    quality_weight: float
    created_at: float
    last_seen: float


class SpeakerObservation(BaseModel):
    """One attributed final speech segment."""

    id: str
    voice_id: str
    turn_id: str
    turn_revision: int
    conversation_id: str
    quality: float
    created_at: float


class SpeakerReference(BaseModel):
    """Resolved authority carried by a short-lived tool reference."""

    value: str
    observation_id: str
    voice_id: str
    conversation_id: str
    expires_at: float


class Person(BaseModel):
    """Persistent person record independent from individual voice clusters."""

    id: str
    name: str
    created_at: float
    last_seen: float


class PersonCandidate(BaseModel):
    """Aggregated identity evidence for one person."""

    person_id: str
    name: str
    evidence_score: float


class PersonalFact(BaseModel):
    """One private fact attached to a confirmed person, never to a voice."""

    id: str
    person_id: str
    fact: str
    topic: str | None = None
    created_at: float
    updated_at: float


class SpeakerAttribution(BaseModel):
    """Compact identity state attached to an internal pipeline message."""

    voice_id: str | None = None
    speaker_ref: str | None = None
    state: SpeakerState = SpeakerState.UNKNOWN
    candidate: PersonCandidate | None = None
    margin: float | None = None
    recommendation: Literal["none", "clarify", "do_not_learn"] = "none"
    speaker_ms: float = 0.0
