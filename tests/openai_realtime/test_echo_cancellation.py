from __future__ import annotations

import ctypes

import pytest

from speech_to_speech.api.openai_realtime.echo_cancellation import SpeexEchoCanceller


class _NativeCall:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class FakeSpeexDSP:
    def __init__(self):
        self.events: list[object] = []
        self.speex_echo_state_init = _NativeCall(self._state_init)
        self.speex_echo_state_destroy = _NativeCall(lambda state: self.events.append(("destroy", state)))
        self.speex_echo_ctl = _NativeCall(self._ctl)
        self.speex_echo_playback = _NativeCall(self._playback)
        self.speex_echo_capture = _NativeCall(self._capture)

    def _state_init(self, frame_size, filter_length):
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.events.append(("init", frame_size, filter_length))
        return 1234

    def _ctl(self, state, request, value):
        self.events.append(("ctl", state, request, ctypes.cast(value, ctypes.POINTER(ctypes.c_int))[0]))
        return 0

    def _playback(self, _state, play):
        frame = ctypes.string_at(play, self.frame_size * 2)
        self.events.append(("playback", frame))

    def _capture(self, _state, recorded, output):
        frame = ctypes.string_at(recorded, self.frame_size * 2)
        self.events.append(("capture", frame))
        ctypes.memmove(output, recorded, self.frame_size * 2)


def test_speex_aec_reblocks_callback_audio_into_realtime_frames():
    native = FakeSpeexDSP()
    canceller = SpeexEchoCanceller(
        sample_rate=16000,
        callback_frames=1024,
        frame_ms=16,
        filter_ms=300,
        library=native,
    )
    playback = b"".join(bytes([index, index]) * 256 for index in range(4))
    capture = b"".join(bytes([index + 10, index + 10]) * 256 for index in range(4))

    cleaned = canceller.process_duplex(playback, capture)
    canceller.close()
    canceller.close()

    assert cleaned == capture
    assert native.events[:2] == [("init", 256, 4800), ("ctl", 1234, 24, 16000)]
    assert [event[0] for event in native.events].count("playback") == 4
    assert [event[0] for event in native.events].count("capture") == 4
    assert [event[0] for event in native.events[2:-1]] == ["playback", "capture"] * 4
    assert [event[0] for event in native.events].count("destroy") == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"callback_frames": 1000}, "multiple"),
        ({"frame_ms": 25}, "frame_ms"),
        ({"filter_ms": 50}, "filter_ms"),
    ],
)
def test_speex_aec_rejects_unsafe_frame_configuration(kwargs, message):
    config = {
        "sample_rate": 16000,
        "callback_frames": 1024,
        "frame_ms": 16,
        "filter_ms": 300,
        "library": FakeSpeexDSP(),
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SpeexEchoCanceller(**config)


def test_speex_aec_rejects_partial_pcm_frames():
    canceller = SpeexEchoCanceller(
        sample_rate=16000,
        callback_frames=1024,
        frame_ms=16,
        filter_ms=300,
        library=FakeSpeexDSP(),
    )

    with pytest.raises(ValueError, match="2048 bytes"):
        canceller.process_duplex(b"\x00" * 2048, b"\x00" * 512)

    canceller.close()


def test_speex_aec_requires_equal_sized_playback_and_capture_callbacks():
    canceller = SpeexEchoCanceller(
        sample_rate=16000,
        callback_frames=1024,
        library=FakeSpeexDSP(),
    )

    with pytest.raises(ValueError, match="2048 bytes"):
        canceller.process_duplex(b"\x00" * 512, b"\x00" * 2048)

    canceller.close()


def test_speex_aec_drain_flushes_delayed_playback_without_new_microphone_audio():
    native = FakeSpeexDSP()
    canceller = SpeexEchoCanceller(
        sample_rate=16000,
        callback_frames=1024,
        library=native,
    )
    canceller.process_duplex(b"\x01\x00" * 1024, b"\x02\x00" * 1024)

    canceller.drain()

    capture_frames = [event[1] for event in native.events if event[0] == "capture"]
    assert capture_frames[-4:] == [b"\x00" * 512] * 4
    canceller.close()


def test_speex_aec_fails_clearly_when_native_library_is_missing(monkeypatch):
    monkeypatch.setattr(
        "speech_to_speech.api.openai_realtime.echo_cancellation._load_speexdsp",
        lambda: (_ for _ in ()).throw(OSError("not found")),
    )

    with pytest.raises(RuntimeError, match="libspeexdsp"):
        SpeexEchoCanceller(sample_rate=16000, callback_frames=1024)
