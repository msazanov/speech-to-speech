"""Online voice clustering and semantic identity operations."""

from __future__ import annotations

import time

import numpy as np

from .models import PersonCandidate, SpeakerAttribution, SpeakerState
from .store import SpeakerMemoryStore, normalize_embedding


class SpeakerTracker:
    """Assign stable voice IDs without allowing uncertain samples to cause drift."""

    def __init__(
        self,
        store: SpeakerMemoryStore,
        *,
        match_threshold: float = 0.82,
        candidate_threshold: float = 0.70,
        ambiguity_margin: float = 0.08,
        minimum_quality: float = 0.5,
        reference_ttl_s: float = 300.0,
        identity_threshold: float = 2.0,
        identity_margin: float = 0.75,
    ) -> None:
        if not 0 <= candidate_threshold <= match_threshold <= 1:
            raise ValueError("voice thresholds must satisfy 0 <= candidate <= match <= 1")
        self.store = store
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        self.ambiguity_margin = ambiguity_margin
        self.minimum_quality = minimum_quality
        self.reference_ttl_s = reference_ttl_s
        self.identity_threshold = identity_threshold
        self.identity_margin = identity_margin

    def observe(
        self,
        embedding: np.ndarray,
        *,
        quality: float,
        turn_id: str,
        turn_revision: int,
        conversation_id: str,
    ) -> SpeakerAttribution:
        started = time.perf_counter()
        vector = normalize_embedding(embedding)
        if quality < self.minimum_quality:
            return SpeakerAttribution(
                state=SpeakerState.UNKNOWN,
                recommendation="do_not_learn",
                speaker_ms=(time.perf_counter() - started) * 1000,
            )

        compatible_clusters = [
            cluster for cluster in self.store.get_voice_clusters() if cluster.centroid.size == vector.size
        ]
        scored = sorted(
            ((cluster, float(np.dot(vector, cluster.centroid))) for cluster in compatible_clusters),
            key=lambda item: item[1],
            reverse=True,
        )
        top_score = scored[0][1] if scored else -1.0
        runner_up = scored[1][1] if len(scored) > 1 else -1.0
        voice_margin = top_score - runner_up

        if not scored or top_score < self.candidate_threshold:
            voice = self.store.create_voice_cluster(vector, quality=quality)
            acoustic_state = SpeakerState.UNKNOWN
            reported_margin: float | None = None
        elif top_score < self.match_threshold or voice_margin < self.ambiguity_margin:
            voice = scored[0][0]
            acoustic_state = SpeakerState.AMBIGUOUS
            reported_margin = voice_margin
        else:
            if self.store.is_voice_blocked(scored[0][0].id):
                return SpeakerAttribution(
                    voice_id=scored[0][0].id,
                    state=SpeakerState.BLACKLISTED,
                    recommendation="do_not_learn",
                    margin=voice_margin,
                    speaker_ms=(time.perf_counter() - started) * 1000,
                )
            voice = self.store.update_voice_cluster(scored[0][0].id, vector, quality=quality)
            acoustic_state = SpeakerState.UNKNOWN
            reported_margin = voice_margin

        observation = self.store.create_observation(
            voice.id,
            turn_id=turn_id,
            turn_revision=turn_revision,
            conversation_id=conversation_id,
            quality=quality,
        )
        reference = self.store.issue_reference(
            observation.id,
            conversation_id=conversation_id,
            ttl_s=self.reference_ttl_s,
        )

        if acoustic_state is SpeakerState.AMBIGUOUS:
            state = acoustic_state
            candidate = self._top_candidate(voice.id)
            recommendation = "clarify"
        else:
            state, candidate = self._identity_state(voice.id)
            recommendation = "clarify" if state is SpeakerState.CONFLICT else "none"

        if candidate is not None:
            self.store.bind_reference_candidate(reference, candidate.person_id)

        return SpeakerAttribution(
            voice_id=voice.id,
            speaker_ref=reference,
            state=state,
            candidate=candidate,
            margin=reported_margin,
            recommendation=recommendation,
            speaker_ms=(time.perf_counter() - started) * 1000,
        )

    def _top_candidate(self, voice_id: str) -> PersonCandidate | None:
        candidates = self.store.resolve_person_candidates(voice_id)
        return candidates[0] if candidates else None

    def _identity_state(self, voice_id: str) -> tuple[SpeakerState, PersonCandidate | None]:
        candidates = self.store.resolve_person_candidates(voice_id)
        if not candidates:
            return SpeakerState.UNKNOWN, None
        top = candidates[0]
        runner_score = candidates[1].evidence_score if len(candidates) > 1 else float("-inf")
        if top.evidence_score >= self.identity_threshold and top.evidence_score - runner_score >= self.identity_margin:
            return SpeakerState.KNOWN, top
        return SpeakerState.CONFLICT, top


# Backwards-compatible import for callers that used the early tracker module.
from .service import SpeakerMemoryService  # noqa: E402,F401
