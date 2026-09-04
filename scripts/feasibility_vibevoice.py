"""Feasibility probe: VibeVoice-ASR-Streaming-1.5B on CPU vs synthetic Russian speech.

Loads the model with explicit classes (AutoConfig does not register the
checkpoint's ``vibevoice`` model_type), transcribes Silero-TTS Russian phrases,
and reports wall time / RTF per utterance.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 24_000

RUSSIAN_PHRASES = [
    "Привет, как дела?",
    "Сегодня хорошая погода для прогулки по парку.",
    "Поставь будильник на семь утра, пожалуйста.",
    "Сколько будет семнадцать умножить на восемь?",
]


def synthesize(model: object, text: str, speaker: str) -> np.ndarray:
    from scipy.signal import resample_poly

    audio = np.asarray(
        model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=TTS_SAMPLE_RATE,
            put_accent=True,
            put_yo=True,
        ),
        dtype=np.float32,
    ).reshape(-1)
    divisor = math.gcd(TTS_SAMPLE_RATE, SAMPLE_RATE)
    return np.ascontiguousarray(
        resample_poly(audio, SAMPLE_RATE // divisor, TTS_SAMPLE_RATE // divisor),
        dtype=np.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="microsoft/VibeVoice-ASR-Streaming-1.5B",
    )
    parser.add_argument("--speaker", default="xenia")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    from transformers.models.vibevoice_asr import (
        VibeVoiceAsrForConditionalGeneration,
        VibeVoiceAsrProcessor,
    )

    print(f"loading processor from {args.model_path} ...")
    processor = VibeVoiceAsrProcessor.from_pretrained(args.model_path)

    print(f"loading model on CPU ({args.dtype}, threads={args.threads}) ...")
    load_start = time.perf_counter()
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()
    print(f"model loaded in {time.perf_counter() - load_start:.1f}s")

    from speech_to_speech.TTS.silero_tts_handler import _resolve_silero_repository

    torch.set_num_threads(min(6, args.threads))
    repository = _resolve_silero_repository(torch.hub.get_dir())
    tts, *_ = torch.hub.load(
        repo_or_dir=repository,
        model="silero_tts",
        language="ru",
        speaker="v5_5_ru",
        trust_repo=True,
        source="local",
    )

    for phrase in RUSSIAN_PHRASES:
        audio = synthesize(tts, phrase, args.speaker)
        inputs = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        elapsed = time.perf_counter() - start
        transcript = processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]
        duration_s = audio.size / SAMPLE_RATE
        print(f"--- audio={duration_s:.2f}s  wall={elapsed:.2f}s  RTF={elapsed / duration_s:.2f}")
        print(f"    expected: {phrase}")
        print(f"    actual:   {transcript}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
