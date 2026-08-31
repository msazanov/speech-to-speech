"""Optional CPU-only sherpa-onnx speaker embedding adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .store import normalize_embedding


class AudioTooShortForSpeakerEmbedding(RuntimeError):
    """Raised when the native extractor cannot produce an embedding."""


class SpeakerEmbeddingExtractor(Protocol):
    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


class SherpaOnnxSpeakerEmbeddingExtractor:
    """Thin sherpa-onnx adapter that never selects a GPU provider."""

    def __init__(self, model_path: str | Path, *, num_threads: int = 1) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"speaker embedding model not found: {path}")
        if num_threads < 1:
            raise ValueError("speaker embedding threads must be at least 1")
        try:
            import sherpa_onnx
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Speaker memory requires the 'speaker-memory' extra (sherpa-onnx); "
                "install speech-to-speech[speaker-memory]."
            ) from exc

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(path),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config=config)

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=waveform)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise AudioTooShortForSpeakerEmbedding("audio is too short for a speaker embedding")
        return normalize_embedding(np.asarray(self._extractor.compute(stream), dtype=np.float32))
