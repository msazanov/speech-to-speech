from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import onnxruntime as ort

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.pipeline.transcript_logging import transcript_for_log
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16_000
SUPPORTED_LANGUAGES = ["ru", "en"]


def detect_ru_en_language(text: str) -> str:
    """Infer Russian or English while keeping Russian dominant for mixed text."""
    has_cyrillic = any("а" <= char.casefold() <= "я" or char.casefold() == "ё" for char in text)
    if has_cyrillic:
        return "ru-auto"
    has_latin = any("a" <= char.casefold() <= "z" for char in text)
    return "en-auto" if has_latin else "ru-auto"


class GigaAMONNXSTTHandler(BaseSTTHandler):
    """CPU-only GigaAM CTC speech recognition through ONNX Runtime."""

    def setup(
        self,
        model_name: str = "gigaam-multilingual-ctc",
        quantization: str = "int8",
        provider: str = "CPUExecutionProvider",
        threads: int = 6,
        language: str = "auto",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del gen_kwargs
        if provider != "CPUExecutionProvider":
            raise ValueError("GigaAM ONNX is CPU-only; provider must be CPUExecutionProvider")
        if threads < 1:
            raise ValueError("GigaAM ONNX threads must be at least 1")

        import onnx_asr

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.language = language.strip().lower()
        self.model = onnx_asr.load_model(
            model_name,
            quantization=quantization,
            providers=[provider],
            sess_options=session_options,
        )
        logger.info(
            "Loaded GigaAM ONNX model %s (quantization=%s, provider=%s, threads=%d)",
            model_name,
            quantization,
            provider,
            threads,
        )

    def _language_code(self, text: str) -> str:
        if self.language == "auto":
            return detect_ru_en_language(text)
        return self.language

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        audio = np.asarray(vad_audio.audio, dtype=np.float32).reshape(-1)
        pred_text = str(self.model.recognize(audio, sample_rate=PIPELINE_SAMPLE_RATE)).strip()
        if not pred_text:
            logger.debug("GigaAM ONNX returned an empty transcription")
            return

        logger.debug("GigaAM ONNX transcription: %s", transcript_for_log(pred_text))
        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        yield Transcription(
            text=pred_text,
            language_code=self._language_code(pred_text),
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
            speaker=vad_audio.speaker,
        )

    def cleanup(self) -> None:
        logger.info("Stopping GigaAMONNXSTTHandler")
        del self.model
