from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
from threading import Lock
from typing import Any, Iterator

import numpy as np

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)


def _recognize_pcm(model: Any, sample_rate: int, pcm: bytes) -> str:
    from vosk import KaldiRecognizer

    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(False)
    recognizer.SetMaxAlternatives(0)
    texts: list[str] = []
    chunk_bytes = 4000 * 2
    for offset in range(0, len(pcm), chunk_bytes):
        if recognizer.AcceptWaveform(pcm[offset : offset + chunk_bytes]):
            segment = json.loads(recognizer.Result()).get("text", "")
            if isinstance(segment, str) and segment.strip():
                texts.append(segment.strip())
    final = json.loads(recognizer.FinalResult()).get("text", "")
    if isinstance(final, str) and final.strip():
        texts.append(final.strip())
    return " ".join(texts)


def _vosk_worker(
    model_path: str,
    sample_rate: int,
    request_queue: Any,
    result_queue: Any,
) -> None:
    """Own the native decoder so a stuck call can be terminated safely."""

    try:
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        model = Model(model_path)
    except BaseException as exc:
        result_queue.put(("startup_error", None, type(exc).__name__))
        return
    result_queue.put(("ready", None, None))
    while True:
        request = request_queue.get()
        if request is None:
            return
        request_id, pcm = request
        try:
            result_queue.put(("result", request_id, _recognize_pcm(model, sample_rate, pcm)))
        except BaseException as exc:
            result_queue.put(("error", request_id, type(exc).__name__))


class VoskSTTHandler(BaseSTTHandler):
    """Compact local Vosk STT for complete 16 kHz mono VAD segments."""

    _models: dict[Path, Any] = {}
    _model_lock = Lock()

    def setup(
        self,
        model_path: str = "/home/orangepi/hugging-voice-device/models/vosk-model-small-ru-0.22",
        sample_rate: int = 16000,
        language: str = "ru",
        timeout_s: float = 0.0,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del gen_kwargs
        if sample_rate <= 0:
            raise ValueError(f"Vosk sample_rate must be positive, got {sample_rate}")
        if timeout_s < 0:
            raise ValueError(f"Vosk timeout_s must be non-negative, got {timeout_s}")

        try:
            from vosk import Model, SetLogLevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Vosk STT requires the optional `vosk` package. "
                'Install it with `pip install "speech-to-speech[vosk]"`.'
            ) from exc

        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"Vosk model directory does not exist: {resolved_path}")

        self.sample_rate = sample_rate
        self.language = language
        self.timeout_s = float(timeout_s)
        self._model_path = resolved_path
        self._worker_process: Any | None = None
        self._worker_requests: Any | None = None
        self._worker_results: Any | None = None
        self._request_id = 0
        SetLogLevel(-1)
        if self.timeout_s > 0:
            self._start_worker()
            return
        with self._model_lock:
            model = self._models.get(resolved_path)
            if model is None:
                logger.info("Loading Vosk STT model from %s", resolved_path)
                model = Model(str(resolved_path))
                self._models[resolved_path] = model
        self.model = model

    @staticmethod
    def _pcm16(audio: np.ndarray | Any) -> bytes:
        samples = np.asarray(audio)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        if np.issubdtype(samples.dtype, np.integer):
            clipped = np.clip(samples, -32768, 32767).astype("<i2", copy=False)
            return np.ascontiguousarray(clipped).tobytes()

        normalized = np.nan_to_num(
            samples.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        pcm = np.rint(np.clip(normalized, -1.0, 1.0) * 32767.0).astype("<i2")
        return np.ascontiguousarray(pcm).tobytes()

    def _decode(self, audio: np.ndarray | Any) -> str:
        pcm = self._pcm16(audio)
        if self.timeout_s > 0:
            return self._decode_isolated(pcm)
        return _recognize_pcm(self.model, self.sample_rate, pcm)

    def _start_worker(self) -> None:
        context = mp.get_context("spawn")
        self._worker_requests = context.Queue(maxsize=1)
        self._worker_results = context.Queue(maxsize=2)
        self._worker_process = context.Process(
            target=_vosk_worker,
            args=(str(self._model_path), self.sample_rate, self._worker_requests, self._worker_results),
            name="vosk-decoder",
            daemon=True,
        )
        self._worker_process.start()
        try:
            kind, _request_id, detail = self._worker_results.get(timeout=30.0)
        except Empty as exc:
            self._stop_worker()
            raise TimeoutError("Vosk worker did not become ready within 30 seconds") from exc
        if kind != "ready":
            self._stop_worker()
            raise RuntimeError(f"Vosk worker failed to start ({detail or kind})")
        logger.info("Vosk decoder worker ready timeout_s=%.1f", self.timeout_s)

    def _stop_worker(self) -> None:
        process = self._worker_process
        requests = self._worker_requests
        results = self._worker_results
        if process is not None and process.is_alive() and requests is not None:
            try:
                requests.put_nowait(None)
            except Full:
                pass
            process.join(timeout=1.0)
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        for queue in (requests, results):
            if queue is not None:
                queue.close()
                queue.join_thread()
        self._worker_process = None
        self._worker_requests = None
        self._worker_results = None

    def _decode_isolated(self, pcm: bytes) -> str:
        if self._worker_process is None or not self._worker_process.is_alive():
            self._stop_worker()
            self._start_worker()
        self._request_id += 1
        request_id = self._request_id
        assert self._worker_requests is not None
        assert self._worker_results is not None
        self._worker_requests.put((request_id, pcm), timeout=1.0)
        try:
            kind, result_id, payload = self._worker_results.get(timeout=self.timeout_s)
        except Empty as exc:
            logger.error("Vosk decoder timed out after %.1fs; restarting worker", self.timeout_s)
            self._stop_worker()
            self._start_worker()
            raise TimeoutError(f"Vosk decoding exceeded {self.timeout_s:.1f} seconds") from exc
        if result_id != request_id:
            raise RuntimeError("Vosk worker returned an unexpected request id")
        if kind == "error":
            raise RuntimeError(f"Vosk worker decoding failed ({payload})")
        if kind != "result" or not isinstance(payload, str):
            raise RuntimeError("Vosk worker returned an invalid result")
        return payload

    def cleanup(self) -> None:
        self._stop_worker()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        pred_text = self._decode(vad_audio.audio)
        if not pred_text:
            logger.debug("Vosk produced no text for turn=%s rev=%s", vad_audio.turn_id, vad_audio.turn_revision)
            return

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        yield Transcription(
            text=pred_text,
            language_code=self.language,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
            speaker=vad_audio.speaker,
        )
