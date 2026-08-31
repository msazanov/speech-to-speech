"""Tool-facing semantic speaker identity operations."""

from __future__ import annotations

from .models import PersonCandidate, SpeakerAttribution, SpeakerState
from .store import SpeakerMemoryStore


class SpeakerMemoryService:
    """Expose semantic decisions while keeping evidence weights server-owned."""

    _REMEMBER_WEIGHT = 3.0
    _CONFIRM_WEIGHT = 2.0
    _REJECT_WEIGHT = -1.0
    _MAX_NAME_LENGTH = 80

    def __init__(
        self,
        store: SpeakerMemoryStore,
        *,
        identity_threshold: float = 2.0,
        identity_margin: float = 0.75,
    ) -> None:
        self.store = store
        self.identity_threshold = identity_threshold
        self.identity_margin = identity_margin

    def inspect(self, speaker_ref: str, *, conversation_id: str) -> SpeakerAttribution:
        reference = self.store.resolve_reference(speaker_ref, conversation_id=conversation_id)
        state, candidate = self._identity_state(reference.voice_id)
        return SpeakerAttribution(
            voice_id=reference.voice_id,
            speaker_ref=speaker_ref,
            state=state,
            candidate=candidate,
            recommendation="clarify" if state is SpeakerState.CONFLICT else "none",
        )

    def remember_name(self, speaker_ref: str, name: str, *, conversation_id: str) -> SpeakerAttribution:
        normalized_name = " ".join(name.split())
        if not normalized_name or len(normalized_name) > self._MAX_NAME_LENGTH:
            raise ValueError(f"name must contain 1 to {self._MAX_NAME_LENGTH} characters")
        reference = self.store.resolve_reference(speaker_ref, conversation_id=conversation_id)
        person = self.store.create_person(normalized_name)
        self.store.add_identity_evidence(
            reference.voice_id,
            person.id,
            kind="self_introduction",
            weight=self._REMEMBER_WEIGHT,
            observation_id=reference.observation_id,
        )
        return self.inspect(speaker_ref, conversation_id=conversation_id)

    def confirm(self, speaker_ref: str, person_id: str, *, conversation_id: str) -> SpeakerAttribution:
        return self._record_decision(
            speaker_ref,
            person_id,
            conversation_id=conversation_id,
            kind="agent_confirmation",
            weight=self._CONFIRM_WEIGHT,
        )

    def reject(self, speaker_ref: str, person_id: str, *, conversation_id: str) -> SpeakerAttribution:
        return self._record_decision(
            speaker_ref,
            person_id,
            conversation_id=conversation_id,
            kind="agent_rejection",
            weight=self._REJECT_WEIGHT,
        )

    def _record_decision(
        self,
        speaker_ref: str,
        person_id: str,
        *,
        conversation_id: str,
        kind: str,
        weight: float,
    ) -> SpeakerAttribution:
        reference = self.store.resolve_reference(speaker_ref, conversation_id=conversation_id)
        self.store.add_identity_evidence(
            reference.voice_id,
            person_id,
            kind=kind,
            weight=weight,
            observation_id=reference.observation_id,
        )
        return self.inspect(speaker_ref, conversation_id=conversation_id)

    def _identity_state(self, voice_id: str) -> tuple[SpeakerState, PersonCandidate | None]:
        candidates = self.store.resolve_person_candidates(voice_id)
        if not candidates:
            return SpeakerState.UNKNOWN, None
        candidate = candidates[0]
        runner_score = candidates[1].evidence_score if len(candidates) > 1 else float("-inf")
        decisive = (
            candidate.evidence_score >= self.identity_threshold
            and candidate.evidence_score - runner_score >= self.identity_margin
        )
        return (SpeakerState.KNOWN if decisive else SpeakerState.CONFLICT), candidate
