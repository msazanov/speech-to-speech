"""Contract tests for the compact Vosk backend carried by the device fork."""

from __future__ import annotations

import sys
import types
from queue import Queue
from threading import Event

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.speaker_memory.models import SpeakerAttribution


def test_vosk_keeps_all_segments_and_speaker_metadata(monkeypatch, tmp_path) -> None:
    class Model:
        def __init__(self, _path: str) -> None:
            pass

    class KaldiRecognizer:
        def __init__(self, _model, _sample_rate: int) -> None:
            self.calls = 0

        def SetWords(self, _value: bool) -> None:
            pass

        def SetMaxAlternatives(self, _value: int) -> None:
            pass

        def AcceptWaveform(self, _pcm: bytes) -> bool:
            self.calls += 1
            return self.calls == 1

        def Result(self) -> str:
            return '{"text": "первая фраза"}'

        def FinalResult(self) -> str:
            return '{"text": "вторая фраза"}'

    monkeypatch.setitem(
        sys.modules,
        "vosk",
        types.SimpleNamespace(Model=Model, KaldiRecognizer=KaldiRecognizer, SetLogLevel=lambda _level: None),
    )

    from speech_to_speech.STT.vosk_handler import VoskSTTHandler

    VoskSTTHandler._models.clear()
    handler = VoskSTTHandler(Event(), Queue(), Queue(), setup_kwargs={"model_path": str(tmp_path)})
    speaker = SpeakerAttribution(voice_id="v_test", speaker_ref="sr_test")
    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(8_000, dtype=np.float32),
                mode="final",
                turn_id="turn-1",
                turn_revision=2,
                created_at_s=3.5,
                speaker=speaker,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "первая фраза вторая фраза"
    assert result[0].speaker is speaker
    assert result[0].turn_id == "turn-1"
    assert result[0].speech_stopped_at_s == 3.5


def test_vosk_pcm_clips_normalized_audio() -> None:
    from speech_to_speech.STT.vosk_handler import VoskSTTHandler

    assert VoskSTTHandler._pcm16(np.array([-2.0, 0.0, 2.0], dtype=np.float32)) == b"\x01\x80\x00\x00\xff\x7f"


def test_vosk_isolated_decoder_restarts_after_timeout() -> None:
    from speech_to_speech.STT.vosk_handler import VoskSTTHandler

    class AliveProcess:
        @staticmethod
        def is_alive() -> bool:
            return True

    handler = object.__new__(VoskSTTHandler)
    handler.timeout_s = 0.001
    handler._request_id = 0
    handler._worker_process = AliveProcess()
    handler._worker_requests = Queue(maxsize=1)
    handler._worker_results = Queue(maxsize=1)
    restarts: list[str] = []
    handler._stop_worker = lambda: restarts.append("stop")
    handler._start_worker = lambda: restarts.append("start")

    with pytest.raises(TimeoutError, match="Vosk decoding exceeded"):
        handler._decode_isolated(b"\x00\x00")

    assert restarts == ["stop", "start"]
