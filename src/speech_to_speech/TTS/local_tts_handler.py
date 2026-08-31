from __future__ import annotations

import logging
from threading import Event
from typing import Iterator

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.rhvoice_tts_handler import RHVOICE_RUSSIAN_VOICES, RHVoiceTTSHandler, _requested_voice
from speech_to_speech.TTS.silero_tts_handler import SILERO_RUSSIAN_SPEAKERS, SileroTTSHandler

logger = logging.getLogger(__name__)


def parse_local_voice(
    requested: str | None,
    fallback_backend: str,
    fallback_voice: str,
) -> tuple[str, str]:
    if requested and ":" in requested:
        backend, voice = requested.split(":", 1)
        if backend not in {"silero", "rhvoice"} or not voice:
            raise ValueError(f"Invalid local TTS selection {requested!r}")
        return backend, voice
    if requested in SILERO_RUSSIAN_SPEAKERS:
        return "silero", requested
    if requested in RHVOICE_RUSSIAN_VOICES:
        return "rhvoice", requested
    if requested == "Aiden":
        return "silero", "xenia"
    if requested:
        raise ValueError(f"Invalid local TTS selection {requested!r}")
    return fallback_backend, fallback_voice


class LocalTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Route each realtime session to a concrete CPU TTS backend and voice."""

    def setup(
        self,
        should_listen: Event,
        default_backend: str = "silero",
        silero_voice: str = "xenia",
        silero_sample_rate: int = 24000,
        silero_threads: int = 6,
        silero_english_fallback: bool = True,
        silero_english_voice: str = "M1",
        rhvoice_executable: str = "RHVoice-test",
        rhvoice_data_path: str = "",
        rhvoice_library_path: str = "",
        rhvoice_voice: str = "Aleksandr",
        rhvoice_rate: int = 100,
        rhvoice_pitch: int = 100,
        rhvoice_volume: int = 100,
        blocksize: int = 512,
        rhvoice_timeout: float = 15.0,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: object,
    ) -> None:
        if default_backend not in {"silero", "rhvoice"}:
            raise ValueError("default_backend must be 'silero' or 'rhvoice'")
        self.default_backend = default_backend
        self.default_silero_voice = silero_voice
        self.default_rhvoice_voice = rhvoice_voice

        self.silero = SileroTTSHandler.__new__(SileroTTSHandler)
        self.silero.setup(
            should_listen,
            speaker=silero_voice,
            sample_rate=silero_sample_rate,
            threads=silero_threads,
            blocksize=blocksize,
            english_fallback=silero_english_fallback,
            english_voice=silero_english_voice,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
        )
        self.rhvoice = RHVoiceTTSHandler.__new__(RHVoiceTTSHandler)
        self.rhvoice.setup(
            should_listen,
            executable=rhvoice_executable,
            data_path=rhvoice_data_path,
            library_path=rhvoice_library_path,
            voice=rhvoice_voice,
            rate=rhvoice_rate,
            pitch=rhvoice_pitch,
            volume=rhvoice_volume,
            blocksize=blocksize,
            timeout=rhvoice_timeout,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
        )

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        requested = None if isinstance(tts_input, EndOfResponse) else _requested_voice(tts_input)
        fallback_voice = (
            self.default_rhvoice_voice if self.default_backend == "rhvoice" else self.default_silero_voice
        )
        backend, voice = parse_local_voice(requested, self.default_backend, fallback_voice)
        if backend == "silero":
            if voice not in SILERO_RUSSIAN_SPEAKERS:
                raise ValueError(f"Unsupported Silero Russian voice {voice!r}")
            self.silero.speaker = voice
            yield from self.silero.process(tts_input)
            return
        if voice not in RHVOICE_RUSSIAN_VOICES:
            raise ValueError(f"Unsupported RHVoice Russian voice {voice!r}")
        self.rhvoice.voice = voice
        yield from self.rhvoice.process(tts_input)
