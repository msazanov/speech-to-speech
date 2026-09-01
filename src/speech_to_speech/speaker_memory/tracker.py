"""Online voice clustering and semantic identity operations."""

from __future__ import annotations

import time

import numpy as np

from .models import PersonCandidate, SpeakerAttribution, SpeakerState, VoiceCluster
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
        group_threshold: float = 0.55,
        soft_match_threshold: float = 0.60,
        soft_match_min_samples: int = 3,
        soft_match_weight: float = 0.25,
        automerge_threshold: float = 0.82,
        automerge_min_samples: int = 3,
    ) -> None:
        if not 0 <= candidate_threshold <= match_threshold <= 1:
            raise ValueError("voice thresholds must satisfy 0 <= candidate <= match <= 1")
        if not 0 <= group_threshold <= candidate_threshold:
            raise ValueError("voice group threshold must satisfy 0 <= group <= candidate")
        if not group_threshold <= soft_match_threshold <= candidate_threshold:
            raise ValueError("soft match threshold must satisfy group <= soft <= candidate")
        if soft_match_min_samples < 1:
            raise ValueError("soft match minimum samples must be at least 1")
        if not 0 < soft_match_weight <= 1:
            raise ValueError("soft match weight must be in (0, 1]")
        if not match_threshold <= automerge_threshold <= 1:
            raise ValueError("automerge threshold must satisfy match <= automerge <= 1")
        if automerge_min_samples < 2:
            raise ValueError("automerge minimum samples must be at least 2")
        self.store = store
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        self.ambiguity_margin = ambiguity_margin
        self.minimum_quality = minimum_quality
        self.reference_ttl_s = reference_ttl_s
        self.identity_threshold = identity_threshold
        self.identity_margin = identity_margin
        self.group_threshold = group_threshold
        self.soft_match_threshold = soft_match_threshold
        self.soft_match_min_samples = soft_match_min_samples
        self.soft_match_weight = soft_match_weight
        self.automerge_threshold = automerge_threshold
        self.automerge_min_samples = automerge_min_samples

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
        scored = self._score_clusters(vector, compatible_clusters)
        if self._automerge_unassigned(scored):
            compatible_clusters = [
                cluster for cluster in self.store.get_voice_clusters() if cluster.centroid.size == vector.size
            ]
            scored = self._score_clusters(vector, compatible_clusters)
        top_score = scored[0][1] if scored else -1.0
        runner_up = scored[1][1] if len(scored) > 1 else -1.0
        voice_margin = top_score - runner_up
        if not scored or top_score < self.candidate_threshold:
            top_cluster = scored[0][0] if scored else None
            # A mature, still-unassigned cluster can absorb a weak outlier with
            # a very small update. This prevents one short/noisy turn from
            # creating a new ID on every utterance, while keeping uncertain
            # samples out of confirmed-person memory. Clusters with identity
            # evidence deliberately stay on the conservative candidate path.
            soft_match = (
                top_cluster is not None
                and top_score >= self.soft_match_threshold
                and top_cluster.sample_count >= self.soft_match_min_samples
                and voice_margin >= self.ambiguity_margin
                and not self.store.resolve_person_candidates(top_cluster.id)
            )
            if soft_match:
                if self.store.is_voice_blocked(top_cluster.id):
                    return SpeakerAttribution(
                        voice_id=top_cluster.id,
                        state=SpeakerState.BLACKLISTED,
                        recommendation="do_not_learn",
                        margin=voice_margin,
                        speaker_ms=(time.perf_counter() - started) * 1000,
                    )
                voice = self.store.update_voice_cluster(
                    top_cluster.id,
                    vector,
                    quality=quality * self.soft_match_weight,
                )
                group_candidate = None
                acoustic_state = SpeakerState.UNKNOWN
                reported_margin = voice_margin
            else:
                voice = self.store.create_voice_cluster(vector, quality=quality)
                group_candidate = self._group_candidate(vector, excluded_voice_id=voice.id)
                acoustic_state = SpeakerState.UNKNOWN
                reported_margin = None
        elif top_score < self.match_threshold or voice_margin < self.ambiguity_margin:
            voice = scored[0][0]
            group_candidate = None
            acoustic_state = SpeakerState.AMBIGUOUS
            reported_margin = voice_margin
        else:
            group_candidate = None
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
            if candidate is None and group_candidate is not None:
                # Acoustic grouping is a suggestion only. It never silently
                # merges clusters or grants access to personal facts.
                state, candidate, recommendation = SpeakerState.CONFLICT, group_candidate, "clarify"
            else:
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

    @staticmethod
    def _score_clusters(
        vector: np.ndarray,
        clusters: list[VoiceCluster],
    ) -> list[tuple[VoiceCluster, float]]:
        return sorted(
            ((cluster, float(np.dot(vector, cluster.centroid))) for cluster in clusters),
            key=lambda item: item[1],
            reverse=True,
        )

    def _automerge_unassigned(self, scored: list[tuple[VoiceCluster, float]]) -> bool:
        """Consolidate mature, unassigned duplicates without granting identity."""

        eligible = [
            (cluster, similarity)
            for cluster, similarity in scored
            if similarity >= self.automerge_threshold
            and cluster.sample_count >= self.automerge_min_samples
            and not self.store.resolve_person_candidates(cluster.id)
            and not self.store.is_voice_blocked(cluster.id)
        ]
        if len(eligible) < 2:
            return False
        # Keep the oldest/most-sampled cluster as the canonical ID so a noisy
        # new cluster cannot replace a stable identifier merely by scoring a
        # few hundredths higher on this turn.
        target, _ = max(
            eligible,
            key=lambda item: (item[0].sample_count, item[0].quality_weight, -item[0].created_at),
        )
        merged = False
        for source, _ in eligible:
            if source.id == target.id:
                continue
            self.store.merge_voice_clusters(
                source.id,
                target.id,
                reason="high_similarity_unassigned_clusters",
            )
            target = self.store.get_voice_cluster(target.id)
            merged = True
        return merged

    def _group_candidate(self, vector: np.ndarray, *, excluded_voice_id: str | None) -> PersonCandidate | None:
        """Suggest a person from a nearby cluster without merging voice IDs."""
        best: tuple[float, PersonCandidate] | None = None
        for cluster in self.store.get_voice_clusters():
            if cluster.id == excluded_voice_id or cluster.centroid.size != vector.size:
                continue
            similarity = float(np.dot(vector, cluster.centroid))
            if similarity < self.group_threshold:
                continue
            candidates = self.store.resolve_person_candidates(cluster.id)
            if not candidates:
                continue
            candidate = candidates[0]
            score = similarity * max(0.1, candidate.evidence_score)
            if best is None or score > best[0]:
                best = (score, candidate)
        return best[1] if best is not None else None

    def _identity_state(self, voice_id: str) -> tuple[SpeakerState, PersonCandidate | None]:
        candidates = self.store.resolve_person_candidates(voice_id)
        if not candidates:
            return SpeakerState.UNKNOWN, None
        top = candidates[0]
        if top.evidence_score < 0:
            return SpeakerState.UNKNOWN, None
        runner_score = candidates[1].evidence_score if len(candidates) > 1 else float("-inf")
        if top.evidence_score >= self.identity_threshold and top.evidence_score - runner_score >= self.identity_margin:
            return SpeakerState.KNOWN, top
        return SpeakerState.CONFLICT, top


# Backwards-compatible import for callers that used the early tracker module.
from .service import SpeakerMemoryService  # noqa: E402,F401
