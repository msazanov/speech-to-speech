"""Small CPU-only SpeexDSP acoustic echo cancellation adapter."""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any

_SPEEX_ECHO_SET_SAMPLING_RATE = 24
_MIN_FRAME_MS = 10
_MAX_FRAME_MS = 20
_MIN_FILTER_MS = 100
_MAX_FILTER_MS = 500


def _load_speexdsp() -> Any:
    candidates = [ctypes.util.find_library("speexdsp"), "libspeexdsp.so.1", "libspeexdsp.so"]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(str(exc))
    detail = f": {'; '.join(errors)}" if errors else ""
    raise OSError(f"unable to load libspeexdsp{detail}")


def _configure_library(library: Any) -> None:
    int16_pointer = ctypes.POINTER(ctypes.c_int16)
    library.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    library.speex_echo_state_init.restype = ctypes.c_void_p
    library.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
    library.speex_echo_state_destroy.restype = None
    library.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    library.speex_echo_ctl.restype = ctypes.c_int
    library.speex_echo_playback.argtypes = [ctypes.c_void_p, int16_pointer]
    library.speex_echo_playback.restype = None
    library.speex_echo_capture.argtypes = [ctypes.c_void_p, int16_pointer, int16_pointer]
    library.speex_echo_capture.restype = None


class SpeexEchoCanceller:
    """Remove local playback from mono PCM16 microphone callbacks.

    The sounddevice callback may be larger than the frame SpeexDSP expects. The
    public methods therefore split one callback into fixed 10--20 ms frames and
    keep all native calls serialized across input/output callback threads.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        callback_frames: int,
        frame_ms: int = 16,
        filter_ms: int = 300,
        library: Any | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not _MIN_FRAME_MS <= frame_ms <= _MAX_FRAME_MS:
            raise ValueError(f"frame_ms must be between {_MIN_FRAME_MS} and {_MAX_FRAME_MS}")
        if not _MIN_FILTER_MS <= filter_ms <= _MAX_FILTER_MS:
            raise ValueError(f"filter_ms must be between {_MIN_FILTER_MS} and {_MAX_FILTER_MS}")
        frame_product = sample_rate * frame_ms
        if frame_product % 1000:
            raise ValueError("sample_rate * frame_ms must produce a whole number of samples")
        frame_samples = frame_product // 1000
        if callback_frames <= 0 or callback_frames % frame_samples:
            raise ValueError(f"callback_frames must be a positive multiple of {frame_samples} AEC samples")

        self.sample_rate = sample_rate
        self.callback_frames = callback_frames
        self.frame_ms = frame_ms
        self.filter_ms = filter_ms
        self.frame_samples = frame_samples
        self.frame_bytes = frame_samples * ctypes.sizeof(ctypes.c_int16)
        self.callback_bytes = callback_frames * ctypes.sizeof(ctypes.c_int16)
        self._library = library
        if self._library is None:
            try:
                self._library = _load_speexdsp()
            except OSError as exc:
                raise RuntimeError(
                    "Acoustic echo cancellation requires the system library libspeexdsp"
                ) from exc
        _configure_library(self._library)

        filter_samples = sample_rate * filter_ms // 1000
        self._state = self._library.speex_echo_state_init(frame_samples, filter_samples)
        if not self._state:
            raise RuntimeError("libspeexdsp failed to create an acoustic echo canceller")
        rate = ctypes.c_int(sample_rate)
        if self._library.speex_echo_ctl(
            self._state,
            _SPEEX_ECHO_SET_SAMPLING_RATE,
            ctypes.byref(rate),
        ) != 0:
            self.close()
            raise RuntimeError("libspeexdsp rejected the acoustic echo canceller sample rate")

        buffer_type = ctypes.c_int16 * frame_samples
        self._play_buffer = buffer_type()
        self._recorded_buffer = buffer_type()
        self._output_buffer = buffer_type()
        self._silence_callback = bytes(self.callback_bytes)

    def _validate_callback(self, pcm: bytes | bytearray | memoryview) -> memoryview:
        view = memoryview(pcm).cast("B")
        if len(view) != self.callback_bytes:
            raise ValueError(f"AEC callbacks must contain exactly {self.callback_bytes} bytes of mono PCM16")
        return view

    def process_duplex(
        self,
        playback_pcm: bytes | bytearray | memoryview,
        capture_pcm: bytes | bytearray | memoryview,
    ) -> bytes:
        """Clean one synchronized full-duplex PCM16 callback.

        This method is intentionally owned by one non-PortAudio worker. The
        paired blocks come from a single full-duplex device callback, avoiding
        clock drift between independent input and output streams.
        """

        playback_view = self._validate_callback(playback_pcm)
        capture_view = self._validate_callback(capture_pcm)
        if self._state is None:
            raise RuntimeError("acoustic echo canceller is closed")
        output = bytearray(self.callback_bytes)
        for offset in range(0, self.callback_bytes, self.frame_bytes):
            ctypes.memmove(
                self._play_buffer,
                bytes(playback_view[offset : offset + self.frame_bytes]),
                self.frame_bytes,
            )
            ctypes.memmove(
                self._recorded_buffer,
                bytes(capture_view[offset : offset + self.frame_bytes]),
                self.frame_bytes,
            )
            self._library.speex_echo_playback(self._state, self._play_buffer)
            self._library.speex_echo_capture(self._state, self._recorded_buffer, self._output_buffer)
            output[offset : offset + self.frame_bytes] = bytes(self._output_buffer)
        return bytes(output)

    def drain(self) -> None:
        """Flush SpeexDSP's delayed playback frames between TTS utterances."""

        self.process_duplex(self._silence_callback, self._silence_callback)

    def close(self) -> None:
        if self._state is None:
            return
        state = self._state
        self._state = None
        self._library.speex_echo_state_destroy(state)
