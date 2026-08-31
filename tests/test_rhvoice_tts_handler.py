from __future__ import annotations

import io
import subprocess
import wave
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.local_tts_handler import LocalTTSHandler, parse_local_voice
from speech_to_speech.TTS.rhvoice_tts_handler import RHVoiceTTSHandler


def _wav_bytes(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def _runtime_voice(voice: str) -> RuntimeConfig:
    config = RuntimeConfig()
    assert config.session.audio is not None and config.session.audio.output is not None
    config.session.audio.output.voice = voice
    return config


def test_rhvoice_invokes_cli_for_selected_voice_and_emits_16k_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def communicate(self, input=None):
            calls[0][1]["input"] = input
            return _wav_bytes(np.array([0, 32767, -32768, 1000], dtype=np.int16)), b""

        def terminate(self):
            raise AssertionError("successful synthesis must not be terminated")

        def kill(self):
            raise AssertionError("successful synthesis must not be killed")

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    handler = RHVoiceTTSHandler.__new__(RHVoiceTTSHandler)
    handler.setup(
        Event(),
        executable="/opt/rhvoice/RHVoice-test",
        data_path="/opt/rhvoice/share/RHVoice",
        library_path="/opt/rhvoice/lib",
        voice="Aleksandr",
        blocksize=4,
    )
    monkeypatch.setattr(
        "speech_to_speech.TTS.rhvoice_tts_handler.scipy.signal.resample_poly",
        lambda *_args: np.array([0, 32767, -32768], dtype=np.int16),
    )

    chunks = list(
        handler.process(
            TTSInput(
                text="Привет.",
                language_code="ru",
                runtime_config=_runtime_voice("rhvoice:Mikhail"),
            )
        )
    )

    assert calls[0][0] == [
        "/opt/rhvoice/RHVoice-test",
        "-p",
        "Mikhail",
        "-r",
        "100",
        "-t",
        "100",
        "-v",
        "100",
        "-o",
        "-",
    ]
    assert calls[0][1]["input"] == "Привет.".encode()
    assert calls[0][1]["env"]["RHVOICE_DATA_PATH"] == "/opt/rhvoice/share/RHVoice"
    assert calls[0][1]["env"]["LD_LIBRARY_PATH"].startswith("/opt/rhvoice/lib")
    np.testing.assert_array_equal(chunks[0], np.array([0, 32767, -32768, 0], dtype=np.int16))


def test_rhvoice_rejects_non_rhvoice_session_selection() -> None:
    handler = RHVoiceTTSHandler.__new__(RHVoiceTTSHandler)
    handler.setup(Event(), executable="RHVoice-test", voice="Aleksandr")

    with pytest.raises(ValueError, match="not an RHVoice selection"):
        list(
            handler.process(
                TTSInput(text="Привет.", runtime_config=_runtime_voice("silero:xenia"))
            )
        )


@pytest.mark.parametrize(
    ("raw", "fallback_backend", "fallback_voice", "expected"),
    [
        ("silero:aidar", "silero", "xenia", ("silero", "aidar")),
        ("rhvoice:Mikhail", "silero", "xenia", ("rhvoice", "Mikhail")),
        ("xenia", "silero", "aidar", ("silero", "xenia")),
        ("Aiden", "silero", "xenia", ("silero", "xenia")),
        (None, "rhvoice", "Pavel", ("rhvoice", "Pavel")),
    ],
)
def test_parse_local_voice(raw, fallback_backend, fallback_voice, expected) -> None:
    assert parse_local_voice(raw, fallback_backend, fallback_voice) == expected


@pytest.mark.parametrize("raw", ["unknown:xenia", "rhvoice:", "bogus"])
def test_parse_local_voice_rejects_invalid_explicit_selection(raw: str) -> None:
    with pytest.raises(ValueError, match="TTS selection"):
        parse_local_voice(raw, "silero", "xenia")


def test_legacy_aiden_always_migrates_to_silero_xenia() -> None:
    assert parse_local_voice("Aiden", "rhvoice", "Pavel") == ("silero", "xenia")


def test_rhvoice_terminates_subprocess_when_barge_in_cancels_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicating = Event()
    terminated = Event()

    class FakeProcess:
        returncode = None

        def __init__(self, *_args, **_kwargs):
            pass

        def communicate(self, input=None):
            communicating.set()
            terminated.wait(1.0)
            self.returncode = -15
            return b"", b""

        def terminate(self):
            terminated.set()

        def kill(self):
            terminated.set()

    class CancellingScope:
        generation = 1

        def is_stale(self, _generation):
            return communicating.is_set()

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    handler = RHVoiceTTSHandler.__new__(RHVoiceTTSHandler)
    handler.setup(Event(), cancel_scope=CancellingScope())

    assert list(handler.process(TTSInput(text="Останови меня."))) == []
    assert terminated.is_set()


def test_local_router_dispatches_session_backend_and_voice() -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeBackend:
        def __init__(self, name: str) -> None:
            self.name = name
            self.speaker = "xenia"
            self.voice = "Aleksandr"

        def process(self, item):
            selected = self.speaker if self.name == "silero" else self.voice
            calls.append((self.name, selected, type(item).__name__))
            yield np.zeros(4, dtype=np.int16)

    handler = LocalTTSHandler.__new__(LocalTTSHandler)
    handler.default_backend = "silero"
    handler.default_silero_voice = "xenia"
    handler.default_rhvoice_voice = "Aleksandr"
    handler.silero = FakeBackend("silero")
    handler.rhvoice = FakeBackend("rhvoice")

    list(handler.process(TTSInput(text="Раз.", runtime_config=_runtime_voice("silero:aidar"))))
    list(handler.process(TTSInput(text="Два.", runtime_config=_runtime_voice("rhvoice:Mikhail"))))

    assert calls == [
        ("silero", "aidar", "TTSInput"),
        ("rhvoice", "Mikhail", "TTSInput"),
    ]


def test_local_router_preserves_terminal_event() -> None:
    backend = SimpleNamespace(process=lambda _item: iter([AUDIO_RESPONSE_DONE]))
    handler = LocalTTSHandler.__new__(LocalTTSHandler)
    handler.default_backend = "silero"
    handler.default_silero_voice = "xenia"
    handler.default_rhvoice_voice = "Aleksandr"
    handler.silero = backend
    handler.rhvoice = backend

    assert list(handler.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]
