"""Tool-facing semantic speaker identity operations."""

from __future__ import annotations

import logging

from .models import PersonalFact, PersonCandidate, SpeakerAttribution, SpeakerState, compact_voice_id
from .store import SpeakerMemoryStore

logger = logging.getLogger(__name__)


class IdentityNotConfirmed(PermissionError):
    """Private memory is unavailable until identity evidence is decisive."""


class InvalidPersonCandidate(PermissionError):
    """A decision targeted a person not proposed for this speaker reference."""


class SpeakerMemoryService:
    """Expose semantic decisions while keeping evidence weights server-owned."""

    _REMEMBER_WEIGHT = 3.0
    _CONFIRM_WEIGHT = 2.0
    # Rejection is an explicit privacy boundary.  A single negative event
    # must outweigh an earlier introduction/confirmation on this reference.
    _REJECT_WEIGHT = -4.0
    _RECENT_VOICE_LIMIT = 32
    _RECENT_MERGE_THRESHOLD = 0.70
    _MAX_NAME_LENGTH = 80
    _MAX_FACT_LENGTH = 500
    _MAX_TOPIC_LENGTH = 80

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
        canonical_voice_id = self.store.resolve_voice_id(reference.voice_id)
        state, candidate = self._identity_state(canonical_voice_id)
        return SpeakerAttribution(
            voice_id=canonical_voice_id,
            speaker_ref=speaker_ref,
            state=state,
            candidate=candidate,
            recommendation="clarify" if state is SpeakerState.CONFLICT else "none",
        )

    def resolve_reference_for_voice(self, voice: str, *, conversation_id: str) -> str:
        """Resolve a compact public voice token to the current private reference."""

        return self.store.resolve_reference_for_voice(voice, conversation_id=conversation_id).value

    def set_voice_blocked(
        self,
        speaker_ref: str,
        *,
        blocked: bool,
        reason: str | None = None,
        conversation_id: str,
    ) -> str:
        """Change routing only for the voice authorized by this reference."""

        return self.store.set_voice_blocked_by_reference(
            speaker_ref,
            conversation_id=conversation_id,
            blocked=blocked,
            reason=reason,
        )

    def remember_name(self, speaker_ref: str, name: str, *, conversation_id: str) -> SpeakerAttribution:
        normalized_name = " ".join(name.split())
        if not normalized_name or len(normalized_name) > self._MAX_NAME_LENGTH:
            raise ValueError(f"name must contain 1 to {self._MAX_NAME_LENGTH} characters")
        reference = self.store.resolve_reference(speaker_ref, conversation_id=conversation_id)
        candidates = [
            *self.store.resolve_person_candidates(reference.voice_id),
            *self.store.resolve_reference_candidates(speaker_ref),
        ]
        person = next((candidate for candidate in candidates if candidate.name.casefold() == normalized_name.casefold()), None)
        if person is None:
            created = self.store.create_person(normalized_name, reuse=False)
            person_id = created.id
        else:
            person_id = person.person_id
        self.store.clear_voice_person_block(reference.voice_id, person_id)
        self.store.add_identity_evidence(
            reference.voice_id,
            person_id,
            kind="self_introduction",
            weight=self._REMEMBER_WEIGHT,
            observation_id=reference.observation_id,
        )
        # A self-introduction is an explicit identity assertion.  If this is
        # a newly-created acoustic cluster, fold it into the person's mature
        # cluster while retaining the source alias for later rejection/audit.
        self.store.merge_voice_with_person(
            reference.voice_id,
            person_id,
            reason="explicit_name_introduction",
        )
        self._merge_recent_voice_clusters(
            reference.voice_id,
            person_id,
            conversation_id=conversation_id,
        )
        self.store.bind_reference_candidate(speaker_ref, person_id)
        return self.inspect(speaker_ref, conversation_id=conversation_id)

    def _merge_recent_voice_clusters(
        self,
        voice_id: str,
        person_id: str,
        *,
        conversation_id: str,
    ) -> list[str]:
        """Fold recent unassigned duplicate clusters into a named voice.

        Clusters are considered only from the same recent conversation.  A
        cluster carrying positive evidence for another person is never folded
        by acoustic similarity alone; explicit identity evidence remains the
        boundary between people.
        """

        canonical_id = self.store.resolve_voice_id(voice_id)
        merged: list[str] = []
        for recent_id in self.store.recent_voice_ids(
            conversation_id,
            limit=self._RECENT_VOICE_LIMIT,
        ):
            source_id = self.store.resolve_voice_id(recent_id)
            if source_id == canonical_id or self.store.is_voice_blocked(source_id):
                continue
            other_candidates = self.store.resolve_person_candidates(source_id)
            if any(candidate.person_id != person_id and candidate.evidence_score > 0 for candidate in other_candidates):
                continue
            similarity = self.store.voice_similarity(source_id, canonical_id)
            if similarity < self._RECENT_MERGE_THRESHOLD:
                continue
            self.store.merge_voice_clusters(
                source_id,
                canonical_id,
                reason="recent_voice_after_name",
            )
            merged.append(source_id)
            logger.info(
                "Speaker memory merged recent voice source=%s target=%s similarity=%.3f person_id=%s",
                compact_voice_id(source_id),
                compact_voice_id(canonical_id),
                similarity,
                person_id,
            )
        return merged

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

    def remember_fact(
        self,
        speaker_ref: str,
        fact: str,
        *,
        topic: str | None = None,
        conversation_id: str,
    ) -> PersonalFact:
        person_id = self._require_confirmed_person(speaker_ref, conversation_id=conversation_id)
        normalized_fact = " ".join(fact.split())
        normalized_topic = " ".join(topic.split()) if topic is not None else None
        if not normalized_fact or len(normalized_fact) > self._MAX_FACT_LENGTH:
            raise ValueError(f"fact must contain 1 to {self._MAX_FACT_LENGTH} characters")
        if normalized_topic is not None and len(normalized_topic) > self._MAX_TOPIC_LENGTH:
            raise ValueError(f"topic must contain at most {self._MAX_TOPIC_LENGTH} characters")
        return self.store.add_personal_fact(person_id, normalized_fact, topic=normalized_topic or None)

    def recall(
        self,
        speaker_ref: str,
        *,
        query: str,
        limit: int = 5,
        conversation_id: str,
    ) -> list[PersonalFact]:
        person_id = self._require_confirmed_person(speaker_ref, conversation_id=conversation_id)
        if not 1 <= limit <= 20:
            raise ValueError("recall limit must be between 1 and 20")
        return self.store.search_personal_facts(person_id, query, limit=limit)

    def forget(
        self,
        speaker_ref: str,
        *,
        scope: str,
        fact_id: str | None = None,
        conversation_id: str,
    ) -> int:
        person_id = self._require_confirmed_person(speaker_ref, conversation_id=conversation_id)
        if scope == "fact":
            if not fact_id:
                raise ValueError("fact_id is required when scope is 'fact'")
            return self.store.delete_personal_facts(person_id, fact_id=fact_id)
        if scope == "facts":
            if fact_id is not None:
                raise ValueError("fact_id is not accepted when scope is 'facts'")
            return self.store.delete_personal_facts(person_id)
        raise ValueError("scope must be 'fact' or 'facts'")

    def _require_confirmed_person(self, speaker_ref: str, *, conversation_id: str) -> str:
        attribution = self.inspect(speaker_ref, conversation_id=conversation_id)
        if attribution.state is not SpeakerState.KNOWN or attribution.candidate is None:
            raise IdentityNotConfirmed("speaker identity is not confirmed")
        return attribution.candidate.person_id

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
        if not self.store.reference_allows_candidate(speaker_ref, person_id):
            raise InvalidPersonCandidate("person was not proposed for this speaker reference")
        self.store.add_identity_evidence(
            reference.voice_id,
            person_id,
            kind=kind,
            weight=weight,
            observation_id=reference.observation_id,
        )
        if kind == "agent_confirmation":
            self.store.clear_voice_person_block(reference.voice_id, person_id)
            self.store.merge_voice_with_person(
                reference.voice_id,
                person_id,
                reason="explicit_agent_confirmation",
            )
        elif kind == "agent_rejection":
            # Keep this source cluster independent from the previously linked
            # canonical voice, then let the strong negative evidence force a
            # fresh clarification on the next turn.
            self.store.block_voice_person(
                reference.voice_id,
                person_id,
                reason="explicit speaker rejection",
            )
            self.store.detach_voice_alias(reference.voice_id)
        return self.inspect(speaker_ref, conversation_id=conversation_id)

    def _identity_state(self, voice_id: str) -> tuple[SpeakerState, PersonCandidate | None]:
        candidates = self.store.resolve_person_candidates(voice_id)
        if not candidates:
            return SpeakerState.UNKNOWN, None
        candidate = candidates[0]
        if candidate.evidence_score < 0:
            return SpeakerState.UNKNOWN, None
        runner_score = candidates[1].evidence_score if len(candidates) > 1 else float("-inf")
        decisive = (
            candidate.evidence_score >= self.identity_threshold
            and candidate.evidence_score - runner_score >= self.identity_margin
        )
        return (SpeakerState.KNOWN if decisive else SpeakerState.CONFLICT), candidate
