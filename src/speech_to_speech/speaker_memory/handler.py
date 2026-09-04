"""Pass-through pipeline stage that attributes final VAD segments."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.transcript_logging import transcript_for_log

from .extractor import SpeakerEmbeddingExtractor
from .models import SpeakerAttribution, SpeakerState, compact_voice_id
from .tracker import SpeakerTracker

logger = logging.getLogger(__name__)


class SpeakerMemoryHandler(BaseHandler[VADAudio, VADAudio]):
    """Run CPU speaker attribution only for sufficiently long final audio."""

    def setup(
        self,
        *,
        extractor: SpeakerEmbeddingExtractor,
        tracker: SpeakerTracker,
        sample_rate: int = 16_000,
        min_audio_ms: int = 700,
        conversation_id: str | None = None,
        close_store_on_cleanup: bool = False,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("speaker memory sample rate must be positive")
        if min_audio_ms < 0:
            raise ValueError("speaker memory minimum audio duration must be non-negative")
        self.extractor = extractor
        self.tracker = tracker
        self.sample_rate = sample_rate
        self.min_audio_ms = min_audio_ms
        self.conversation_id = conversation_id or self._new_conversation_id()
        self.close_store_on_cleanup = close_store_on_cleanup

    @staticmethod
    def _new_conversation_id() -> str:
        return f"conv_{uuid4().hex}"

    def process(self, vad_audio: VADAudio) -> Iterator[VADAudio]:
        if vad_audio.mode != "final":
            yield vad_audio
            return

        # A new final turn supersedes every old LLM mutation capability even if
        # this segment is later rejected as too short, silent, or invalid.
        try:
            self.tracker.store.invalidate_references(self.conversation_id)
        except Exception as exc:
            logger.warning(
                "Audio rejected reason=speaker_authority_error error_type=%s",
                type(exc).__name__,
            )
            return

        audio = np.asarray(vad_audio.audio, dtype=np.float32).reshape(-1)
        duration_ms = audio.size * 1000 / self.sample_rate
        if duration_ms < self.min_audio_ms:
            logger.info(
                "Speaker attribution skipped reason=short_audio duration_ms=%.1f min_audio_ms=%d",
                duration_ms,
                self.min_audio_ms,
            )
            yield self._unknown(vad_audio)
            return

        quality = self._quality(audio)
        minimum_quality = getattr(self.tracker, "minimum_quality", 0.5)
        if quality < minimum_quality:
            logger.info(
                "Speaker attribution skipped reason=low_signal duration_ms=%.1f quality=%.3f minimum_quality=%.3f",
                duration_ms,
                quality,
                minimum_quality,
            )
            yield self._unknown(vad_audio)
            return

        try:
            started = perf_counter()
            embedding = self.extractor.extract(audio, self.sample_rate)
            turn_id = vad_audio.turn_id or f"turn_{uuid4().hex}"
            turn_revision = vad_audio.turn_revision or 0
            enrollment = self._active_enrollment()
            if enrollment is not None:
                attribution = self.tracker.enroll(
                    embedding,
                    voice_id=enrollment.voice_id,
                    quality=quality,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    conversation_id=self.conversation_id,
                )
                remaining = self.service.note_enrollment_sample(self.conversation_id)
                logger.info(
                    "Speaker enrollment sample voice=%s remaining=%d",
                    compact_voice_id(attribution.voice_id),
                    remaining,
                )
            else:
                attribution = self.tracker.observe(
                    embedding,
                    quality=quality,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    conversation_id=self.conversation_id,
                )
            attribution = attribution.model_copy(update={"speaker_ms": (perf_counter() - started) * 1000})
            logger.info(
                "Speaker attributed voice=%s state=%s person_id=%s person=%s margin=%s speaker_ms=%.1f",
                compact_voice_id(attribution.voice_id),
                attribution.state.value,
                attribution.candidate.person_id if attribution.candidate is not None else "unknown",
                json.dumps(
                    transcript_for_log(attribution.candidate.name),
                    ensure_ascii=False,
                )
                if attribution.candidate is not None
                else "unknown",
                f"{attribution.margin:.3f}" if attribution.margin is not None else "n/a",
                attribution.speaker_ms,
            )
            if attribution.state.value == "blacklisted":
                logger.info("Audio rejected reason=blacklisted_voice voice=%s", attribution.voice_id)
                return
        except Exception as exc:
            logger.warning("Speaker attribution skipped after %s", type(exc).__name__)
            yield self._unknown(vad_audio)
            return

        yield vad_audio.model_copy(update={"speaker": attribution})

    @staticmethod
    def _unknown(vad_audio: VADAudio) -> VADAudio:
        """Keep a metadata-bearing pass-through for short/failed segments."""
        return vad_audio.model_copy(
            update={
                "speaker": SpeakerAttribution(
                    state=SpeakerState.UNKNOWN,
                    recommendation="do_not_learn",
                )
            }
        )

    @staticmethod
    def _quality(audio: np.ndarray) -> float:
        """Conservative signal quality proxy bounded to [0, 1]."""

        if audio.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))
        signal_score = min(1.0, rms / 0.05)
        return max(0.0, signal_score * (1.0 - clipping_fraction))

    def on_session_end(self) -> None:
        service = getattr(self, "service", None)
        if service is not None:
            service.clear_enrollment(self.conversation_id)
        store = getattr(self.tracker, "store", None)
        if store is not None:
            store.prune_expired()
        self.conversation_id = self._new_conversation_id()

    def _active_enrollment(self) -> Any | None:
        service = getattr(self, "service", None)
        if service is None:
            return None
        try:
            return service.active_enrollment(self.conversation_id)
        except Exception as exc:
            logger.warning("Speaker enrollment lookup failed after %s", type(exc).__name__)
            return None

    def cleanup(self) -> None:
        if self.close_store_on_cleanup:
            self.tracker.store.close()
