"""CPU renderer for the hash-verified native Russian GLaDOS profile.

The model lives in its dedicated Style-Bert-VITS2 environment so its NumPy and
PyTorch constraints do not affect the realtime server environment.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import wave
from math import gcd
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import scipy.signal

from speech_to_speech.TTS.rhvoice_tts_handler import _requested_voice
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import transcript_for_log

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
GLADOS_STYLES = frozenset({"Neutral", "Standard", "Deep", "Light", "Standard_02"})
_LATIN_GLADOS = re.compile(r"\bglados\b", re.IGNORECASE)


class GladosTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Render Russian text with the trained v3 GLaDOS profile in a worker process."""

    def setup(
        self,
        should_listen: Event,
        python: str = "/home/random/glados-tts/.venv/bin/python",
        workdir: str = "/home/random/orange-RAG",
        espeak_ng: str = "/home/random/glados-tts/espeak-ng/bin/espeak-ng",
        profile: str = "v3-1000",
        style: str = "Neutral",
        timeout: float = 90.0,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: object,
    ) -> None:
        if style not in GLADOS_STYLES:
            raise ValueError(f"Unsupported GLaDOS style {style!r}")
        self.python, self.workdir, self.espeak_ng = python, workdir, espeak_ng
        self.profile, self.style, self.timeout, self.blocksize = profile, style, timeout, blocksize
        self.cancel_scope, self.speculative_turns = cancel_scope, speculative_turns

    def _resolve_style(self, tts_input: Any) -> str:
        requested = _requested_voice(tts_input)
        if requested and ":" in requested:
            backend, requested = requested.split(":", 1)
            if backend != "glados":
                raise ValueError(f"Voice {backend!r} is not a GLaDOS selection")
        if requested in GLADOS_STYLES:
            return requested
        if requested and requested not in {"Aiden", "alloy"}:
            raise ValueError(f"Unsupported GLaDOS style {requested!r}")
        return self.style

    def _cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    def _render(self, text: str, style: str, generation: int | None) -> bytes | None:
        with tempfile.TemporaryDirectory(prefix="huggingvoice-glados-") as directory:
            output = Path(directory) / "speech.wav"
            env = os.environ.copy()
            env["GLADOS_ESPEAK_NG"] = self.espeak_ng
            env["CUDA_VISIBLE_DEVICES"] = ""
            # The native profile is Russian-only. Its frontend deliberately rejects
            # Latin tokens; rewrite the one assistant name it commonly emits.
            text = _LATIN_GLADOS.sub("Глэдос", text)
            command = [self.python, "-m", "glados_ru.run_verified_russian_glados", "--profile", self.profile,
                       "--text", text, "--style", style, "--output", str(output)]
            process = subprocess.Popen(command, cwd=self.workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            done, result = Event(), {}

            def wait() -> None:
                result["output"] = process.communicate()
                done.set()

            Thread(target=wait, daemon=True).start()
            deadline = perf_counter() + self.timeout
            while not done.wait(0.02):
                if self._cancelled(generation) or perf_counter() >= deadline:
                    process.terminate()
                    if not done.wait(1):
                        process.kill()
                    if self._cancelled(generation):
                        return None
                    raise RuntimeError(f"GLaDOS synthesis exceeded {self.timeout:.1f}s")
            _stdout, stderr = result["output"]
            if process.returncode:
                detail = stderr.decode(errors="replace").strip()
                logger.error("GLaDOS worker failed: %s", detail)
                raise RuntimeError(f"GLaDOS synthesis failed: {detail}")
            return output.read_bytes()

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return
        if not tts_input.text.strip():
            return
        generation = self.cancel_scope.generation if self.cancel_scope else None
        style, started = self._resolve_style(tts_input), perf_counter()
        wav_bytes = self._render(tts_input.text, style, generation)
        if wav_bytes is None or self._cancelled(generation):
            return
        with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise RuntimeError("GLaDOS returned unsupported WAV format")
            sample_rate, audio = wav.getframerate(), np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if sample_rate != PIPELINE_SAMPLE_RATE:
            divisor = gcd(sample_rate, PIPELINE_SAMPLE_RATE)
            audio = scipy.signal.resample_poly(audio.astype(np.float32), PIPELINE_SAMPLE_RATE // divisor, sample_rate // divisor)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        logger.info("TTS completed backend=glados style=%s synth_ms=%.1f text=%s", style, (perf_counter()-started)*1000, transcript_for_log(tts_input.text))
        for offset in range(0, len(audio), self.blocksize):
            if self._cancelled(generation):
                return
            chunk = audio[offset:offset + self.blocksize]
            yield np.pad(chunk, (0, self.blocksize-len(chunk))) if len(chunk) < self.blocksize else chunk
