from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.speaker_memory.extractor import (
    AudioTooShortForSpeakerEmbedding,
    SherpaOnnxSpeakerEmbeddingExtractor,
)


class FakeStream:
    def __init__(self) -> None:
        self.sample_rate = None
        self.waveform = None
        self.finished = False

    def accept_waveform(self, *, sample_rate, waveform) -> None:
        self.sample_rate = sample_rate
        self.waveform = waveform

    def input_finished(self) -> None:
        self.finished = True


class FakeNativeExtractor:
    latest = None

    def __init__(self, *, config) -> None:
        self.config = config
        self.stream = FakeStream()
        self.ready = True
        type(self).latest = self

    def create_stream(self) -> FakeStream:
        return self.stream

    def is_ready(self, stream: FakeStream) -> bool:
        assert stream is self.stream
        return self.ready

    def compute(self, stream: FakeStream):
        assert stream.finished
        return [3.0, 4.0]


def install_fake_sherpa(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        SpeakerEmbeddingExtractor=FakeNativeExtractor,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)


def test_extractor_forces_cpu_one_thread_and_normalizes(monkeypatch, tmp_path: Path) -> None:
    install_fake_sherpa(monkeypatch)
    model_path = tmp_path / "campplus.onnx"
    model_path.touch()

    extractor = SherpaOnnxSpeakerEmbeddingExtractor(model_path, num_threads=1)
    result = extractor.extract(np.ones(16000, dtype=np.float64), sample_rate=16000)

    native = FakeNativeExtractor.latest
    assert native.config.provider == "cpu"
    assert native.config.num_threads == 1
    assert native.config.debug is False
    assert native.stream.waveform.dtype == np.float32
    assert native.stream.sample_rate == 16000
    assert result.dtype == np.float32
    assert result.tolist() == pytest.approx([0.6, 0.8])


def test_extractor_rejects_non_ready_audio(monkeypatch, tmp_path: Path) -> None:
    install_fake_sherpa(monkeypatch)
    model_path = tmp_path / "campplus.onnx"
    model_path.touch()
    extractor = SherpaOnnxSpeakerEmbeddingExtractor(model_path)
    FakeNativeExtractor.latest.ready = False

    with pytest.raises(AudioTooShortForSpeakerEmbedding):
        extractor.extract(np.ones(100, dtype=np.float32), sample_rate=16000)


def test_missing_model_fails_before_sherpa_import(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="speaker embedding model"):
        SherpaOnnxSpeakerEmbeddingExtractor(tmp_path / "missing.onnx")


def test_missing_optional_dependency_has_actionable_message(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "campplus.onnx"
    model_path.touch()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)

    with pytest.raises(RuntimeError, match=r"speaker-memory.*sherpa-onnx"):
        SherpaOnnxSpeakerEmbeddingExtractor(model_path)
