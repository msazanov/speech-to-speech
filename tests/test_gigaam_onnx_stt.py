from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio


def handler_module():
    from speech_to_speech.STT import gigaam_onnx_handler

    return gigaam_onnx_handler


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("проверяем русский текст", "ru-auto"),
        ("check english speech", "en-auto"),
        ("проверяем OpenAI speech", "ru-auto"),
        ("12345", "ru-auto"),
    ],
)
def test_language_detection_keeps_russian_dominant(text, expected):
    assert handler_module().detect_ru_en_language(text) == expected


def test_setup_loads_int8_model_on_cpu_with_bounded_threads(monkeypatch):
    captured = {}

    def load_model(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return SimpleNamespace(recognize=lambda *_args, **_kwargs: "")

    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_model=load_model))
    handler = object.__new__(handler_module().GigaAMONNXSTTHandler)

    handler.setup(
        model_name="gigaam-multilingual-ctc",
        quantization="int8",
        provider="CPUExecutionProvider",
        threads=6,
        language="auto",
    )

    assert captured["model_name"] == "gigaam-multilingual-ctc"
    assert captured["quantization"] == "int8"
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["sess_options"].intra_op_num_threads == 6
    assert handler.language == "auto"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"provider": "CUDAExecutionProvider"}, "CPUExecutionProvider"),
        ({"threads": 0}, "threads"),
    ],
)
def test_setup_rejects_non_cpu_or_unbounded_configuration(kwargs, message):
    handler = object.__new__(handler_module().GigaAMONNXSTTHandler)

    with pytest.raises(ValueError, match=message):
        handler.setup(**kwargs)


def make_handler(text: str, *, language: str = "auto"):
    handler = object.__new__(handler_module().GigaAMONNXSTTHandler)
    handler.language = language
    handler.model = SimpleNamespace(recognize=lambda audio, *, sample_rate: text)
    return handler


def test_final_transcription_preserves_turn_and_infers_russian():
    handler = make_handler("проверяем OpenAI speech")

    outputs = list(
        handler.process(
            VADAudio(
                audio=np.zeros(1600, dtype=np.float32),
                mode="final",
                turn_id="turn-1",
                turn_revision=2,
                created_at_s=42.0,
            )
        )
    )

    assert outputs == [
        Transcription(
            text="проверяем OpenAI speech",
            language_code="ru-auto",
            turn_id="turn-1",
            turn_revision=2,
            speech_stopped_at_s=42.0,
        )
    ]


def test_progressive_transcription_has_no_language_field():
    handler = make_handler("check english speech")

    outputs = list(
        handler.process(
            VADAudio(
                audio=np.zeros(1600, dtype=np.float32),
                mode="progressive",
                turn_id="turn-1",
                turn_revision=1,
            )
        )
    )

    assert outputs == [PartialTranscription(text="check english speech", turn_id="turn-1", turn_revision=1)]


def test_fixed_language_overrides_text_detection():
    handler = make_handler("check english speech", language="ru")

    outputs = list(handler.process(VADAudio(audio=np.zeros(1600, dtype=np.float32))))

    assert outputs[0].language_code == "ru"


def test_empty_transcription_is_skipped():
    handler = make_handler("   ")

    assert list(handler.process(VADAudio(audio=np.zeros(1600, dtype=np.float32)))) == []
