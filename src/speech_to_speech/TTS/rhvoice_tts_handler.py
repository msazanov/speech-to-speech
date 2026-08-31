from __future__ import annotations

import io
import logging
import os
import subprocess
import wave
from math import gcd
from threading import Event, Thread
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import scipy.signal
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import transcript_for_log

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SAMPLE_RATE = 16000
RHVOICE_RUSSIAN_VOICES = frozenset({"Aleksandr", "Mikhail", "Evgeniy-Rus", "Pavel"})


def _requested_voice(tts_input: Any) -> str | None:
    response = getattr(tts_input, "response", None)
    response_output = getattr(getattr(getattr(response, "audio", None), "output", None), "voice", None)
    if response_output:
        return str(response_output)
    runtime_config = getattr(tts_input, "runtime_config", None)
    audio = getattr(getattr(runtime_config, "session", None), "audio", None)
    output = getattr(audio, "output", None)
    voice = getattr(output, "voice", None)
    return str(voice) if voice else None


class RHVoiceTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def setup(
        self,
        should_listen: Event,
        executable: str = "RHVoice-test",
        data_path: str = "",
        library_path: str = "",
        voice: str = "Aleksandr",
        rate: int = 100,
        pitch: int = 100,
        volume: int = 100,
        blocksize: int = 512,
        timeout: float = 15.0,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: object,
    ) -> None:
        if voice not in RHVOICE_RUSSIAN_VOICES:
            raise ValueError(f"Unsupported RHVoice Russian voice {voice!r}")
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self.should_listen = should_listen
        self.executable = executable
        self.data_path = data_path
        self.library_path = library_path
        self.voice = voice
        self.rate = int(rate)
        self.pitch = int(pitch)
        self.volume = int(volume)
        self.blocksize = blocksize
        self.timeout = timeout
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    def _resolve_voice(self, tts_input: Any) -> str:
        requested = _requested_voice(tts_input)
        if requested and ":" in requested:
            backend, voice = requested.split(":", 1)
            if backend != "rhvoice":
                raise ValueError(f"Voice {requested!r} is not an RHVoice selection")
            requested = voice
        if requested in RHVOICE_RUSSIAN_VOICES:
            return requested
        if requested and requested not in {"Aiden", "alloy"}:
            raise ValueError(f"Unsupported RHVoice Russian voice {requested!r}")
        return self.voice

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.data_path:
            env["RHVOICE_DATA_PATH"] = self.data_path
        if self.library_path:
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = self.library_path + (os.pathsep + current if current else "")
        return env

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes], done: Event) -> None:
        process.terminate()
        if not done.wait(0.25):
            process.kill()
            done.wait(1.0)

    def _synthesize_wav(self, command: list[str], text: str, generation: int | None) -> bytes | None:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._environment(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"RHVoice executable not found: {self.executable}") from exc

        done = Event()
        result: dict[str, Any] = {}

        def communicate() -> None:
            try:
                result["output"] = process.communicate(input=text.encode())
            except BaseException as exc:  # relay worker failures to the handler thread
                result["error"] = exc
            finally:
                done.set()

        Thread(target=communicate, name="rhvoice-communicate", daemon=True).start()
        deadline = perf_counter() + self.timeout
        while not done.wait(0.02):
            if self._is_cancelled(generation):
                self._stop_process(process, done)
                logger.info("RHVoice TTS subprocess terminated (interruption)")
                return None
            if perf_counter() >= deadline:
                self._stop_process(process, done)
                raise RuntimeError(f"RHVoice synthesis exceeded {self.timeout:.1f}s")

        if "error" in result:
            raise RuntimeError("RHVoice subprocess communication failed") from result["error"]
        stdout, stderr = result["output"]
        if process.returncode:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"RHVoice synthesis failed: {detail or process.returncode}")
        return stdout

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id, tts_input.turn_revision
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id, tts_input.turn_revision
        ):
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)
        if not tts_input.text.strip():
            return

        voice = self._resolve_voice(tts_input)
        generation = self.cancel_scope.generation if self.cancel_scope is not None else None
        command = [
            self.executable,
            "-p",
            voice,
            "-r",
            str(self.rate),
            "-t",
            str(self.pitch),
            "-v",
            str(self.volume),
            "-o",
            "-",
        ]
        console.print(f"[green]ASSISTANT: {tts_input.text}")
        started = perf_counter()
        wav_bytes = self._synthesize_wav(command, tts_input.text, generation)
        if wav_bytes is None or self._is_cancelled(generation):
            return
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise RuntimeError("RHVoice returned unsupported WAV format; expected mono PCM16")
            sample_rate = wav.getframerate()
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")

        if sample_rate != PIPELINE_SAMPLE_RATE:
            divisor = gcd(sample_rate, PIPELINE_SAMPLE_RATE)
            audio = scipy.signal.resample_poly(
                audio.astype(np.float32),
                PIPELINE_SAMPLE_RATE // divisor,
                sample_rate // divisor,
            )
        audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)
        synth_ms = (perf_counter() - started) * 1000
        audio_ms = len(audio_int16) * 1000 / PIPELINE_SAMPLE_RATE
        logger.info(
            "TTS completed backend=rhvoice voice=%s synth_ms=%.1f audio_ms=%.1f rtf=%.3f text=%s",
            voice,
            synth_ms,
            audio_ms,
            synth_ms / audio_ms if audio_ms else 0.0,
            transcript_for_log(tts_input.text),
        )
        for offset in range(0, len(audio_int16), self.blocksize):
            if self._is_cancelled(generation):
                return
            chunk = audio_int16[offset : offset + self.blocksize]
            if len(chunk) < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield chunk
