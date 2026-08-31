#!/usr/bin/env python3
"""Interactive microphone-first speaker attribution smoke with synthetic fallback."""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

AGENT_PROMPT = "Для теста ответов скажите ДА"
SUCCESS_REPLY = "Пизда, тест пройден"
SYNTHETIC_CONFIRMATION = "Да"
DEFAULT_MICROPHONE_TIMEOUT_S = 30.0
SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 24_000
MICROPHONE_BLOCKSIZE = 1_024
SYNTHETIC_SPEAKERS = ("xenia", "baya", "aidar")
SPEAKER_TEXTS = (
    "Сегодня мы проверяем устойчивость памяти голоса.",
    "Вторая реплика нужна для точного сравнения диктора.",
)


def normalize_text(text: str) -> str:
    normalized = text.casefold().replace("ё", "е")
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def response_for_transcript(transcript: str) -> str | None:
    return SUCCESS_REPLY if "да" in normalize_text(transcript).split() else None


def resolve_confirmation(transcript: str, *, timed_out: bool) -> tuple[str | None, str]:
    if response_for_transcript(transcript) is not None:
        return transcript, "microphone"
    if timed_out:
        return SYNTHETIC_CONFIRMATION, "synthetic_timeout"
    return None, "listening"


def summarize_similarity(labels: list[str], embeddings: np.ndarray) -> dict[str, float]:
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    matrix = vectors @ vectors.T
    same: list[float] = []
    different: list[float] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            target = same if labels[left] == labels[right] else different
            target.append(float(matrix[left, right]))
    same_min = min(same) if same else 0.0
    different_max = max(different) if different else 0.0
    return {
        "same_voice_min": same_min,
        "same_voice_mean": float(np.mean(same)) if same else 0.0,
        "different_voice_max": different_max,
        "different_voice_mean": float(np.mean(different)) if different else 0.0,
        "separation_margin": same_min - different_max,
    }


def _load_silero() -> Any:
    import torch

    from speech_to_speech.TTS.silero_tts_handler import _resolve_silero_repository

    torch.set_num_threads(6)
    repository = _resolve_silero_repository(torch.hub.get_dir())
    model, *_rest = torch.hub.load(
        repo_or_dir=repository,
        model="silero_tts",
        language="ru",
        speaker="v5_5_ru",
        trust_repo=True,
        source="local" if Path(repository).is_dir() else "github",
    )
    return model


def _synthesize(model: Any, text: str, speaker: str) -> np.ndarray:
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


def _write_wav(path: Path, audio: np.ndarray) -> None:
    from scipy.io import wavfile

    pcm = np.clip(np.asarray(audio) * 32768, -32768, 32767).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, pcm)


def _play(audio: np.ndarray, *, output_device: int | None, volume: float) -> None:
    import sounddevice as sd

    sd.play(np.asarray(audio, dtype=np.float32) * volume, SAMPLE_RATE, device=output_device)
    sd.wait()


def _load_gigaam() -> Any:
    from speech_to_speech.STT.gigaam_onnx_handler import GigaAMONNXSTTHandler

    handler = object.__new__(GigaAMONNXSTTHandler)
    handler.setup(threads=6)
    return handler.model


def _recognize(model: Any, audio: np.ndarray) -> str:
    return str(model.recognize(np.asarray(audio, dtype=np.float32), sample_rate=SAMPLE_RATE)).strip()


def listen_for_confirmation(
    model: Any,
    *,
    timeout_s: float,
    input_device: int | None,
) -> tuple[str, np.ndarray | None, list[str]]:
    """Listen until a segmented utterance contains the standalone Russian word «да»."""

    import sounddevice as sd

    chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
    heard: list[str] = []
    pre_roll: list[np.ndarray] = []
    utterance: list[np.ndarray] = []
    speaking = False
    silent_blocks = 0
    required_silence_blocks = math.ceil(0.7 * SAMPLE_RATE / MICROPHONE_BLOCKSIZE)

    def callback(indata: Any, _frames: int, _time_info: Any, _status: Any) -> None:
        try:
            chunks.put_nowait(np.frombuffer(bytes(indata), dtype="<i2").astype(np.float32) / 32768.0)
        except queue.Full:
            pass

    deadline = time.monotonic() + timeout_s
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=MICROPHONE_BLOCKSIZE,
        callback=callback,
        device=input_device,
    ):
        while time.monotonic() < deadline:
            try:
                chunk = chunks.get(timeout=min(0.1, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            voiced = rms >= 0.008
            if not speaking:
                pre_roll.append(chunk)
                pre_roll = pre_roll[-4:]
                if not voiced:
                    continue
                speaking = True
                utterance = list(pre_roll)
                silent_blocks = 0
                continue
            utterance.append(chunk)
            silent_blocks = 0 if voiced else silent_blocks + 1
            if silent_blocks < required_silence_blocks:
                continue
            audio = np.concatenate(utterance)
            transcript = _recognize(model, audio)
            if transcript:
                heard.append(transcript)
            if response_for_transcript(transcript) is not None:
                return transcript, audio, heard
            speaking = False
            pre_roll = []
            utterance = []

    return "", None, heard


def run_speaker_probe(model: Any, output_dir: Path, speaker_model_path: Path) -> dict[str, Any]:
    from speech_to_speech.speaker_memory.extractor import SherpaOnnxSpeakerEmbeddingExtractor
    from speech_to_speech.speaker_memory.store import SpeakerMemoryStore
    from speech_to_speech.speaker_memory.tracker import SpeakerTracker

    extractor = SherpaOnnxSpeakerEmbeddingExtractor(speaker_model_path, num_threads=1)
    labels: list[str] = []
    embeddings: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    database = output_dir / "speaker-memory.sqlite3"
    store = SpeakerMemoryStore(database)
    tracker = SpeakerTracker(store)
    expected_voice: dict[str, str] = {}
    latencies: list[float] = []
    try:
        for speaker in SYNTHETIC_SPEAKERS:
            for phrase_index, text in enumerate(SPEAKER_TEXTS, start=1):
                audio = _synthesize(model, text, speaker)
                path = output_dir / f"speaker-{speaker}-{phrase_index}.wav"
                _write_wav(path, audio)
                started = time.perf_counter()
                embedding = extractor.extract(audio, SAMPLE_RATE)
                attribution = tracker.observe(
                    embedding,
                    quality=1.0,
                    turn_id=f"synthetic_{speaker}_{phrase_index}",
                    turn_revision=0,
                    conversation_id="synthetic-smoke",
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                latencies.append(elapsed_ms)
                expected_voice.setdefault(speaker, str(attribution.voice_id))
                labels.append(speaker)
                embeddings.append(embedding)
                rows.append(
                    {
                        "speaker": speaker,
                        "phrase": phrase_index,
                        "path": str(path),
                        "voice_id": attribution.voice_id,
                        "speaker_ms": round(elapsed_ms, 3),
                        "same_cluster": attribution.voice_id == expected_voice[speaker],
                    }
                )
    finally:
        store.close()

    matrix = np.asarray(embeddings, dtype=np.float32)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    unique_voice_ids = {speaker: expected_voice[speaker] for speaker in SYNTHETIC_SPEAKERS}
    assignment_passed = all(row["same_cluster"] for row in rows) and len(set(unique_voice_ids.values())) == len(
        SYNTHETIC_SPEAKERS
    )
    return {
        "ok": assignment_passed,
        "rows": rows,
        "voice_ids": unique_voice_ids,
        "similarity_labels": labels,
        "similarity_matrix": np.round(similarity, 4).tolist(),
        "similarity_summary": summarize_similarity(labels, matrix),
        "speaker_ms_max": round(max(latencies), 3),
        "speaker_ms_mean": round(float(np.mean(latencies)), 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=DEFAULT_MICROPHONE_TIMEOUT_S)
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--volume", type=float, default=0.75)
    parser.add_argument("--agent-voice", default="xenia", choices=SYNTHETIC_SPEAKERS)
    parser.add_argument("--fallback-voice", default="aidar", choices=SYNTHETIC_SPEAKERS)
    parser.add_argument(
        "--speaker-model",
        default="models/speaker-memory/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true")
    return parser


def run(arguments: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if arguments.timeout <= 0:
        raise ValueError("timeout must be positive")
    if not 0 <= arguments.volume <= 1:
        raise ValueError("volume must be between 0 and 1")
    output_dir.mkdir(parents=True, exist_ok=True)

    silero = _load_silero()
    gigaam = _load_gigaam()
    prompt_audio = _synthesize(silero, AGENT_PROMPT, arguments.agent_voice)
    _write_wav(output_dir / "agent-prompt.wav", prompt_audio)
    print(f"AGENT: {AGENT_PROMPT}", flush=True)
    _play(prompt_audio, output_device=arguments.output_device, volume=arguments.volume)

    print(f"MICROPHONE: listening for {arguments.timeout:.1f}s", flush=True)
    transcript, microphone_audio, heard = listen_for_confirmation(
        gigaam,
        timeout_s=arguments.timeout,
        input_device=arguments.input_device,
    )
    resolved, source = resolve_confirmation(transcript, timed_out=not bool(transcript))
    if source == "synthetic_timeout":
        synthetic_yes = _synthesize(silero, SYNTHETIC_CONFIRMATION, arguments.fallback_voice)
        _write_wav(output_dir / "synthetic-timeout-yes.wav", synthetic_yes)
        transcript = _recognize(gigaam, synthetic_yes)
        resolved = transcript if response_for_transcript(transcript) is not None else None
    elif microphone_audio is not None:
        _write_wav(output_dir / "microphone-confirmation.wav", microphone_audio)

    reply = response_for_transcript(resolved or "")
    if reply is not None:
        reply_audio = _synthesize(silero, reply, arguments.agent_voice)
        _write_wav(output_dir / "agent-success.wav", reply_audio)
        print(f"AGENT: {reply}", flush=True)
        _play(reply_audio, output_device=arguments.output_device, volume=arguments.volume)

    speaker_probe = run_speaker_probe(
        silero,
        output_dir,
        Path(arguments.speaker_model).expanduser(),
    )
    return {
        "ok": reply == SUCCESS_REPLY and speaker_probe["ok"],
        "challenge": {
            "prompt": AGENT_PROMPT,
            "microphone_timeout_s": arguments.timeout,
            "source": source,
            "heard_transcripts": heard,
            "confirmation_transcript": transcript,
            "reply": reply,
            "passed": reply == SUCCESS_REPLY,
        },
        "speaker_probe": speaker_probe,
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.output_dir:
        output_dir = Path(arguments.output_dir).expanduser()
        report = run(arguments, output_dir)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="huggingvoice-synthetic-speakers-"))
        report = run(arguments, output_dir)
    if arguments.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
