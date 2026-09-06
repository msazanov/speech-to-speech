"""Native Russian GLaDOS in a warm, isolated CPU worker process."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import wave
from math import gcd
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import scipy.signal

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.rhvoice_tts_handler import _requested_voice

logger = logging.getLogger(__name__)
PIPELINE_SAMPLE_RATE = 16000
GLADOS_STYLES = frozenset({"Neutral", "Standard", "Deep", "Light", "Standard_02"})
_LATIN_GLADOS = re.compile(r"\bglados\b", re.IGNORECASE)
WORKER_PATH = Path(__file__).with_name("glados_worker.py")


class GladosTTSHandler(BaseHandler[TTSIn, TTSOut]):
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
        if timeout <= 0 or blocksize <= 0:
            raise ValueError("GLaDOS timeout and blocksize must be positive")
        self.should_listen = should_listen
        self.python, self.workdir, self.espeak_ng = python, workdir, espeak_ng
        self.profile, self.style, self.timeout, self.blocksize = profile, style, timeout, blocksize
        self.cancel_scope, self.speculative_turns = cancel_scope, speculative_turns
        self._worker = None
        self._replies = Queue()
        self._reader = None
        self._ready = False
        self._request_id = 0
        self._pending = None

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

    def _start_worker(self):
        env = os.environ.copy()
        # The server's ONNX shared libraries must not override the worker venv's.
        for name in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        env.update(GLADOS_ESPEAK_NG=self.espeak_ng, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6")
        try:
            self._worker = subprocess.Popen(
                [self.python, "-u", str(WORKER_PATH), "--profile", self.profile],
                cwd=self.workdir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
            )
        except OSError as exc:
            raise RuntimeError("Cannot start native GLaDOS worker") from exc
        self._replies = Queue()
        worker, replies = self._worker, self._replies

        def read():
            try:
                for line in worker.stdout:
                    replies.put(json.loads(line))
            except Exception:
                replies.put({"error": "Invalid worker protocol"})
            finally:
                replies.put({"error": "Worker exited"})

        self._reader = Thread(target=read, name="glados-replies", daemon=True)
        self._reader.start()

    def _receive(self, deadline, generation):
        while not self._cancelled(generation):
            remaining = deadline - perf_counter()
            if remaining <= 0:
                self.cleanup()
                raise RuntimeError(f"GLaDOS synthesis exceeded {self.timeout:.1f}s")
            try:
                return self._replies.get(timeout=min(0.02, remaining))
            except Empty:
                pass
        return None

    def _ensure_ready(self, deadline, generation):
        if self._worker is None or self._worker.poll() is not None:
            self.cleanup()
            self._start_worker()
        if not self._ready:
            reply = self._receive(deadline, generation)
            if reply is None:
                return False
            if not reply.get("ready"):
                self.cleanup()
                raise RuntimeError(f"GLaDOS startup failed: {reply.get('error', 'invalid reply')}")
            self._ready = True
            logger.info(
                "Native GLaDOS worker ready profile=%s device=cpu runtime=%s threads=%s accent_mode=%s startup_seconds=%s",
                self.profile,
                reply.get("runtime"),
                reply.get("threads"),
                reply.get("accent_mode"),
                reply.get("startup_seconds"),
            )
        return True

    def warmup(self):
        self._ensure_ready(perf_counter() + self.timeout, None)

    def _render(self, text: str, style: str, generation: int | None) -> bytes | None:
        if self._cancelled(generation):
            return None
        deadline = perf_counter() + self.timeout
        if not self._ensure_ready(deadline, generation):
            return None
        # A cancelled sentence may still be finishing on the CPU. Drain exactly
        # its reply before accepting more work, preserving the warm model.
        if self._pending is not None:
            if self._receive(deadline, generation) is None:
                return None
            self._pending = None
        if self._cancelled(generation):
            return None
        self._request_id += 1
        self._pending = self._request_id
        request = {"id": self._pending, "text": _LATIN_GLADOS.sub("Глэдос", text), "style": style}
        try:
            self._worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()
        except (OSError, ValueError) as exc:
            self.cleanup()
            raise RuntimeError("GLaDOS worker disconnected") from exc
        reply = self._receive(deadline, generation)
        if reply is None:
            return None
        expected_id, self._pending = self._pending, None
        if reply.get("id") != expected_id:
            self.cleanup()
            raise RuntimeError("GLaDOS worker returned an unexpected response")
        if "error" in reply:
            raise RuntimeError(f"GLaDOS synthesis failed: {reply['error']}")
        return base64.b64decode(reply["wav"], validate=True)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        tracker = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if tracker and not tracker.is_latest_after_reopen_grace(tts_input.turn_id, tts_input.turn_revision):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return
        generation = tts_input.cancel_generation
        if generation is None and self.cancel_scope is not None:
            generation = self.cancel_scope.generation
        if self._cancelled(generation) or not tts_input.text.strip():
            return
        if tracker and not tracker.commit_if_latest_after_reopen_grace(tts_input.turn_id, tts_input.turn_revision):
            return
        style, started = self._resolve_style(tts_input), perf_counter()
        wav_bytes = self._render(tts_input.text, style, generation)
        if wav_bytes is None or self._cancelled(generation):
            return
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise RuntimeError("GLaDOS returned unsupported WAV format")
            sample_rate = wav.getframerate()
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if sample_rate != PIPELINE_SAMPLE_RATE:
            divisor = gcd(sample_rate, PIPELINE_SAMPLE_RATE)
            audio = scipy.signal.resample_poly(
                audio.astype(np.float32),
                PIPELINE_SAMPLE_RATE // divisor,
                sample_rate // divisor,
            )
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        logger.info(
            "TTS completed backend=glados style=%s synth_ms=%.1f audio_ms=%.1f",
            style,
            (perf_counter() - started) * 1000,
            len(audio) / 16,
        )
        for offset in range(0, len(audio), self.blocksize):
            if self._cancelled(generation):
                return
            chunk = audio[offset : offset + self.blocksize]
            yield np.pad(chunk, (0, self.blocksize - len(chunk))) if len(chunk) < self.blocksize else chunk

    def cleanup(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=1)
            if worker.stdin:
                worker.stdin.close()
            if self._reader is not None:
                self._reader.join(timeout=1)
            if worker.stdout:
                worker.stdout.close()
        self._ready, self._pending = False, None
