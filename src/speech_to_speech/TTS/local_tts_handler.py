from __future__ import annotations

import logging
from threading import Event
from typing import Iterator

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.glados_tts_handler import GLADOS_STYLES, GladosTTSHandler
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
        if backend not in {"silero", "rhvoice", "glados"} or not voice:
            raise ValueError(f"Invalid local TTS selection {requested!r}")
        return backend, voice
    if requested in SILERO_RUSSIAN_SPEAKERS:
        return "silero", requested
    if requested in RHVOICE_RUSSIAN_VOICES:
        return "rhvoice", requested
    if requested in GLADOS_STYLES:
        return "glados", requested
    if requested in {"Aiden", "alloy"}:
        return fallback_backend, fallback_voice
    if requested:
        raise ValueError(f"Invalid local TTS selection {requested!r}")
    return fallback_backend, fallback_voice


class LocalTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Route each realtime session to a concrete CPU TTS backend and voice."""

    def setup(
        self,
        should_listen: Event,
        default_backend: str = "glados",
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
        glados_python: str = "/home/random/glados-tts/.venv/bin/python",
        glados_workdir: str = "/home/random/orange-RAG",
        glados_espeak_ng: str = "/home/random/glados-tts/espeak-ng/bin/espeak-ng",
        glados_profile: str = "v3-1000",
        glados_style: str = "Neutral",
        glados_timeout: float = 90.0,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: object,
    ) -> None:
        if default_backend not in {"silero", "rhvoice", "glados"}:
            raise ValueError("default_backend must be 'silero', 'rhvoice', or 'glados'")
        self.default_backend = default_backend
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.default_silero_voice = silero_voice
        self.default_rhvoice_voice = rhvoice_voice
        self.default_glados_style = glados_style

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
        self.glados = GladosTTSHandler.__new__(GladosTTSHandler)
        self.glados.setup(
            should_listen,
            python=glados_python,
            workdir=glados_workdir,
            espeak_ng=glados_espeak_ng,
            profile=glados_profile,
            style=glados_style,
            timeout=glados_timeout,
            blocksize=blocksize,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
        )
        if default_backend == "glados":
            self.glados.warmup()

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if (
            not isinstance(tts_input, EndOfResponse)
            and self.cancel_scope is not None
            and tts_input.cancel_generation is not None
            and self.cancel_scope.is_stale(tts_input.cancel_generation)
        ):
            return
        requested = None if isinstance(tts_input, EndOfResponse) else _requested_voice(tts_input)
        fallback_voice = {
            "rhvoice": self.default_rhvoice_voice,
            "silero": self.default_silero_voice,
            "glados": self.default_glados_style,
        }[self.default_backend]
        backend, voice = parse_local_voice(requested, self.default_backend, fallback_voice)
        if backend == "silero":
            if voice not in SILERO_RUSSIAN_SPEAKERS:
                raise ValueError(f"Unsupported Silero Russian voice {voice!r}")
            self.silero.speaker = voice
            yield from self.silero.process(tts_input)
            return
        if backend == "glados":
            if voice not in GLADOS_STYLES:
                raise ValueError(f"Unsupported GLaDOS style {voice!r}")
            self.glados.style = voice
            try:
                yield from self.glados.process(tts_input)
            except RuntimeError as exc:
                # Keep the voice call usable if the RU-only frontend rejects an
                # unexpected token. Avoid logging transcript-bearing exceptions.
                if (
                    self.cancel_scope is not None
                    and tts_input.cancel_generation is not None
                    and self.cancel_scope.is_stale(tts_input.cancel_generation)
                ):
                    return
                logger.warning("GLaDOS TTS failed (%s); using Silero fallback", type(exc).__name__)
                self.silero.speaker = self.default_silero_voice
                yield from self.silero.process(tts_input)
            return
        if voice not in RHVOICE_RUSSIAN_VOICES:
            raise ValueError(f"Unsupported RHVoice Russian voice {voice!r}")
        self.rhvoice.voice = voice
        yield from self.rhvoice.process(tts_input)

    def on_session_end(self) -> None:
        self.silero.speaker = self.default_silero_voice
        self.rhvoice.voice = self.default_rhvoice_voice
        self.glados.style = self.default_glados_style

    def cleanup(self) -> None:
        self.glados.cleanup()
