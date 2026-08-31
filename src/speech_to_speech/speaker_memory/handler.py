"""Pass-through pipeline stage that attributes final VAD segments."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from time import perf_counter
from uuid import uuid4

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import VADAudio

from .extractor import SpeakerEmbeddingExtractor
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

        audio = np.asarray(vad_audio.audio, dtype=np.float32).reshape(-1)
        duration_ms = audio.size * 1000 / self.sample_rate
        if duration_ms < self.min_audio_ms:
            yield vad_audio
            return

        quality = self._quality(audio)
        if quality < getattr(self.tracker, "minimum_quality", 0.5):
            yield vad_audio
            return

        try:
            started = perf_counter()
            embedding = self.extractor.extract(audio, self.sample_rate)
            attribution = self.tracker.observe(
                embedding,
                quality=quality,
                turn_id=vad_audio.turn_id or f"turn_{uuid4().hex}",
                turn_revision=vad_audio.turn_revision or 0,
                conversation_id=self.conversation_id,
            )
            attribution = attribution.model_copy(update={"speaker_ms": (perf_counter() - started) * 1000})
        except Exception as exc:
            logger.warning("Speaker attribution skipped after %s", type(exc).__name__)
            yield vad_audio
            return

        yield vad_audio.model_copy(update={"speaker": attribution})

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
        store = getattr(self.tracker, "store", None)
        if store is not None:
            store.prune_expired()
        self.conversation_id = self._new_conversation_id()

    def cleanup(self) -> None:
        if self.close_store_on_cleanup:
            self.tracker.store.close()
