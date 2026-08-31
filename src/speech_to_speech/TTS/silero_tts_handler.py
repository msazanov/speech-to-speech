from __future__ import annotations

import logging
import re
from math import gcd
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Iterator

import numpy as np
import scipy.signal
import torch
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
SILERO_REPOSITORY = "snakers4/silero-models:d9355348e2781dc8fa25a135d1602c530afae24c"
SILERO_RUSSIAN_SPEAKERS = frozenset({"aidar", "baya", "kseniya", "xenia", "eugene"})
SILERO_SAMPLE_RATES = frozenset({8000, 24000, 48000})
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _resolve_silero_repository(torch_hub_dir: str | Path) -> str:
    cached_repository = Path(torch_hub_dir) / "snakers4_silero-models_master"
    if cached_repository.is_dir():
        return str(cached_repository)
    return SILERO_REPOSITORY


def _detect_tts_language(text: str, inherited: str | None) -> str:
    """Prefer the generated script while keeping mixed technical RU text stable."""

    inherited_code = (inherited or "").strip().lower().split("-", 1)[0]
    cyrillic = len(_CYRILLIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if latin and not cyrillic:
        return "en"
    if cyrillic and not latin:
        return "ru"
    if cyrillic and latin:
        if latin >= cyrillic * 4:
            return "en"
        if cyrillic >= latin * 2:
            return "ru"
    return inherited_code or "ru"


class SileroTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def _is_cancelled(self, generation: int | None) -> bool:
        cancel_scope = getattr(self, "cancel_scope", None)
        return generation is not None and cancel_scope is not None and cancel_scope.is_stale(generation)

    def setup(
        self,
        should_listen: Event,
        speaker: str = "xenia",
        sample_rate: int = 24000,
        threads: int = 6,
        blocksize: int = 512,
        english_fallback: bool = True,
        english_voice: str = "M1",
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: object,
    ) -> None:
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}")
        if sample_rate not in SILERO_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {sorted(SILERO_SAMPLE_RATES)}, got {sample_rate}")
        if speaker not in SILERO_RUSSIAN_SPEAKERS:
            raise ValueError(
                f"Unsupported Silero Russian speaker {speaker!r}; "
                f"choose one of {', '.join(sorted(SILERO_RUSSIAN_SPEAKERS))}"
            )
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")

        self.should_listen = should_listen
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.threads = threads
        self.blocksize = blocksize
        self.english_fallback = english_fallback
        self.english_voice = english_voice
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self._english_handler = None
        torch.set_num_threads(threads)
        repository = _resolve_silero_repository(torch.hub.get_dir())
        loaded = torch.hub.load(
            repo_or_dir=repository,
            model="silero_tts",
            language="ru",
            speaker="v5_5_ru",
            trust_repo=True,
            source="local" if Path(repository).is_dir() else "github",
        )
        self.model = loaded[0]
        self.model.apply_tts(
            text="Раз, два.",
            speaker=self.speaker,
            sample_rate=self.sample_rate,
            put_accent=True,
            put_yo=True,
        )

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
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
            logger.debug(
                "Dropping stale Silero TTS input for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        language_code = _detect_tts_language(tts_input.text, tts_input.language_code)
        if language_code == "en" and getattr(self, "english_fallback", False):
            english_handler = getattr(self, "_english_handler", None)
            if english_handler is None:
                english_handler = self._load_english_handler()
                self._english_handler = english_handler
            inherited_code = (tts_input.language_code or "").strip().lower().split("-", 1)[0]
            english_input = (
                tts_input
                if inherited_code == "en"
                else tts_input.model_copy(update={"language_code": "en"})
            )
            yield from english_handler.process(english_input)
            return

        cancel_scope = getattr(self, "cancel_scope", None)
        cancel_generation = cancel_scope.generation if cancel_scope is not None else None
        text = tts_input.text
        if not text.strip():
            return

        console.print(f"[green]ASSISTANT: {text}")
        started = perf_counter()
        wav = self.model.apply_tts(
            text=text,
            speaker=self.speaker,
            sample_rate=self.sample_rate,
            put_accent=True,
            put_yo=True,
        )
        if self._is_cancelled(cancel_generation):
            logger.info("Silero TTS output cancelled (interruption)")
            return
        audio = np.asarray(wav, dtype=np.float32).squeeze()
        divisor = gcd(self.sample_rate, PIPELINE_SAMPLE_RATE)
        audio_16k = scipy.signal.resample_poly(
            audio,
            PIPELINE_SAMPLE_RATE // divisor,
            self.sample_rate // divisor,
        )
        audio_int16 = np.clip(audio_16k * 32768, -32768, 32767).astype(np.int16)
        synth_ms = (perf_counter() - started) * 1000
        audio_ms = len(audio_int16) * 1000 / PIPELINE_SAMPLE_RATE
        logger.info(
            "TTS completed backend=silero voice=%s synth_ms=%.1f audio_ms=%.1f rtf=%.3f text=%s",
            self.speaker,
            synth_ms,
            audio_ms,
            synth_ms / audio_ms if audio_ms else 0.0,
            transcript_for_log(text),
        )

        full_samples = (len(audio_int16) // self.blocksize) * self.blocksize
        for offset in range(0, full_samples, self.blocksize):
            if self._is_cancelled(cancel_generation):
                return
            yield audio_int16[offset : offset + self.blocksize]
        if full_samples < len(audio_int16):
            if self._is_cancelled(cancel_generation):
                return
            tail = audio_int16[full_samples:]
            yield np.pad(tail, (0, self.blocksize - len(tail)))

    def _load_english_handler(self):
        from speech_to_speech.TTS.supertonic_tts_handler import SupertonicTTSHandler

        handler = SupertonicTTSHandler.__new__(SupertonicTTSHandler)
        handler.setup(
            self.should_listen,
            voice=self.english_voice,
            lang="en",
            blocksize=self.blocksize,
            cancel_scope=getattr(self, "cancel_scope", None),
            speculative_turns=getattr(self, "speculative_turns", None),
        )
        return handler
