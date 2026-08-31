from __future__ import annotations

from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np

from speech_to_speech.arguments_classes.speaker_memory_arguments import SpeakerMemoryArguments
from speech_to_speech.pipeline.events import TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.s2s_pipeline import _build_speaker_memory_handler, parse_arguments
from speech_to_speech.speaker_memory.models import SpeakerAttribution, SpeakerState
from speech_to_speech.STT.gigaam_onnx_handler import GigaAMONNXSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def attribution() -> SpeakerAttribution:
    return SpeakerAttribution(
        voice_id="v_test",
        speaker_ref="sr_test",
        state=SpeakerState.UNKNOWN,
    )


def test_gigaam_preserves_speaker_attribution() -> None:
    handler = object.__new__(GigaAMONNXSTTHandler)
    handler.language = "auto"
    handler.model = SimpleNamespace(recognize=lambda *_args, **_kwargs: "привет")
    speaker = attribution()

    output = list(
        handler.process(
            VADAudio(
                audio=np.ones(1600, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=0,
                speaker=speaker,
            )
        )
    )[0]

    assert output.speaker == speaker


def test_notifier_keeps_speaker_internal_but_wire_dump_stays_standard() -> None:
    event_queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=event_queue)
    speaker = attribution()

    list(
        notifier.process(
            Transcription(
                text="привет",
                language_code="ru",
                turn_id="turn_1",
                turn_revision=0,
                speaker=speaker,
            )
        )
    )
    event = event_queue.get_nowait()

    assert isinstance(event, TranscriptionCompletedEvent)
    assert event.speaker == speaker
    assert "speaker" not in event.model_dump()
    assert event.model_dump() == {
        "type": "transcription_completed",
        "transcript": "привет",
        "language_code": "ru",
        "turn_id": "turn_1",
        "turn_revision": 0,
    }


def test_speaker_memory_is_opt_in_and_parser_carries_cpu_configuration() -> None:
    defaults = parse_arguments([])
    configured = parse_arguments(
        [
            "--speaker_memory_enabled",
            "true",
            "--speaker_memory_model_path",
            "/models/campplus.onnx",
            "--speaker_memory_threads",
            "1",
        ]
    )

    assert defaults.speaker_memory_kwargs.speaker_memory_enabled is False
    assert configured.speaker_memory_kwargs.speaker_memory_enabled is True
    assert configured.speaker_memory_kwargs.speaker_memory_model_path == "/models/campplus.onnx"
    assert configured.speaker_memory_kwargs.speaker_memory_threads == 1


def test_builder_inserts_cpu_speaker_handler_only_when_enabled(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeExtractor:
        def __init__(self, model_path, *, num_threads):
            captured.update(model_path=model_path, num_threads=num_threads)

    monkeypatch.setattr(
        "speech_to_speech.speaker_memory.extractor.SherpaOnnxSpeakerEmbeddingExtractor",
        FakeExtractor,
    )
    queue_in = Queue()
    queue_out = Queue()
    disabled = _build_speaker_memory_handler(
        stop_event=Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        arguments=SpeakerMemoryArguments(),
        sample_rate=16000,
    )
    enabled = _build_speaker_memory_handler(
        stop_event=Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        arguments=SpeakerMemoryArguments(
            speaker_memory_enabled=True,
            speaker_memory_model_path=str(tmp_path / "campplus.onnx"),
            speaker_memory_database_path=str(tmp_path / "memory.sqlite3"),
            speaker_memory_threads=1,
        ),
        sample_rate=16000,
    )

    assert disabled is None
    assert enabled is not None
    assert enabled.queue_in is queue_in
    assert enabled.queue_out is queue_out
    assert captured == {"model_path": str(tmp_path / "campplus.onnx"), "num_threads": 1}
