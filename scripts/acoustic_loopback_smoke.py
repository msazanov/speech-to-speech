#!/usr/bin/env python3
"""Run one synthetic phrase through the real speaker -> microphone voice path."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from queue import Empty, Full, Queue
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from speech_to_speech.api.openai_realtime.audio_client import (
    RealtimeAudioClientConfig,
    build_session_update,
    normalize_realtime_url,
)

DEFAULT_TEXT = "Проверяем скорость и точность распознавания русской речи в голосовом помощнике."
SAMPLE_RATE = 16_000
CHUNK_SIZE = 1_024


def normalize_text(text: str) -> str:
    normalized = text.casefold().replace("ё", "е")
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def validate_reference(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Тестовая фраза не должна быть пустой")
    if len(normalized.split()) > 20:
        raise ValueError("Тестовая фраза должна содержать не более 20 слов")
    return text.strip()


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + (ref_token != hyp_token)
            deletion = previous[hyp_index] + 1
            insertion = current[hyp_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, transcript: str) -> float:
    reference_words = normalize_text(reference).split()
    if not reference_words:
        raise ValueError("Эталон для WER не должен быть пустым")
    transcript_words = normalize_text(transcript).split()
    return _edit_distance(reference_words, transcript_words) / len(reference_words)


def character_similarity(reference: str, transcript: str) -> float:
    normalized_reference = normalize_text(reference)
    normalized_transcript = normalize_text(transcript)
    return SequenceMatcher(None, normalized_reference, normalized_transcript).ratio()


@dataclass
class EventTimes:
    playback_started: float | None = None
    speech_started: float | None = None
    speech_stopped: float | None = None
    transcription_completed: float | None = None
    response_created: float | None = None
    first_audio: float | None = None
    response_done: float | None = None


@dataclass
class RealtimeResult:
    transcript: str = ""
    assistant_transcript: str = ""
    response_status: str = "unknown"
    times: EventTimes = field(default_factory=EventTimes)


def _seconds_between(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return round(later - earlier, 3)


def _event_error(event: Any) -> str:
    error = getattr(event, "error", None)
    message = getattr(error, "message", None)
    return str(message or error or event)


async def _receive_until_done(
    conn: Any,
    result: RealtimeResult,
    session_ready: asyncio.Event,
) -> None:
    assistant_parts: list[str] = []
    while True:
        event = await conn.recv()
        event_type = getattr(event, "type", "")
        now = time.monotonic()

        if event_type == "session.updated":
            session_ready.set()
        elif event_type == "input_audio_buffer.speech_started":
            if result.times.playback_started is not None and result.times.speech_started is None:
                result.times.speech_started = now
        elif event_type == "input_audio_buffer.speech_stopped":
            if result.times.playback_started is not None and result.times.speech_stopped is None:
                result.times.speech_stopped = now
        elif event_type == "conversation.item.input_audio_transcription.completed":
            result.transcript = (getattr(event, "transcript", "") or "").strip()
            result.times.transcription_completed = now
        elif event_type == "response.created":
            result.times.response_created = now
        elif event_type == "response.output_audio_transcript.delta":
            assistant_parts.append(getattr(event, "delta", "") or "")
        elif event_type == "response.output_audio_transcript.done":
            transcript = getattr(event, "transcript", "") or ""
            result.assistant_transcript = transcript.strip() or "".join(assistant_parts).strip()
        elif event_type == "response.output_audio.delta" and result.times.first_audio is None:
            result.times.first_audio = now
        elif event_type == "response.done":
            response = getattr(event, "response", None)
            result.response_status = str(getattr(response, "status", "unknown"))
            result.times.response_done = now
            if not result.assistant_transcript:
                result.assistant_transcript = "".join(assistant_parts).strip()
            return
        elif event_type == "error":
            raise RuntimeError(f"Realtime error: {_event_error(event)}")


def synthesize_reference(text: str, voice: str) -> tuple[np.ndarray, int, float]:
    from supertonic import TTS

    tts = TTS(auto_download=True)
    voice_style = tts.get_voice_style(voice_name=voice)
    started = time.monotonic()
    waveform, _duration = tts.synthesize(
        text=text,
        lang="ru",
        voice_style=voice_style,
        speed=1.0,
    )
    elapsed = time.monotonic() - started
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    return np.clip(audio, -1.0, 1.0), int(tts.sample_rate), elapsed


def _play_audio(audio: np.ndarray, sample_rate: int, output_device: int | None, volume: float) -> None:
    import sounddevice as sd

    playback = (audio * volume).astype(np.float32).reshape(-1, 1)
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=output_device,
        blocksize=CHUNK_SIZE,
    ) as stream:
        stream.write(playback)


async def run_acoustic_loopback(
    *,
    text: str,
    url: str,
    input_device: int | None,
    output_device: int | None,
    timeout_s: float,
    settle_ms: int,
    voice: str,
    volume: float,
) -> dict[str, Any]:
    import sounddevice as sd

    reference = validate_reference(text)
    audio, playback_rate, synthesis_s = await asyncio.to_thread(synthesize_reference, reference, voice)

    base_url, websocket_base_url = normalize_realtime_url(url)
    client = AsyncOpenAI(api_key="local", base_url=base_url, websocket_base_url=websocket_base_url)
    config = RealtimeAudioClientConfig(url=url, input_device=input_device, output_device=output_device)
    microphone_queue: Queue[bytes] = Queue(maxsize=128)
    stop_sending = asyncio.Event()
    session_ready = asyncio.Event()
    result = RealtimeResult()
    microphone_peak = 0

    def microphone_callback(indata: Any, _frames: int, _time_info: Any, _status: Any) -> None:
        nonlocal microphone_peak
        chunk = bytes(indata)
        if chunk:
            samples = np.frombuffer(chunk, dtype="<i2")
            if samples.size:
                microphone_peak = max(microphone_peak, int(np.abs(samples.astype(np.int32)).max()))
        try:
            microphone_queue.put_nowait(chunk)
        except Full:
            pass

    async def send_microphone(conn: Any) -> None:
        while not stop_sending.is_set():
            try:
                chunk = await asyncio.to_thread(microphone_queue.get, True, 0.1)
            except Empty:
                continue
            await conn.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )

    started = time.monotonic()
    try:
        async with client.realtime.connect(model="local") as conn:
            await conn.send(build_session_update(config))  # type: ignore[arg-type]
            receiver = asyncio.create_task(_receive_until_done(conn, result, session_ready))
            await asyncio.wait_for(session_ready.wait(), timeout=min(5.0, timeout_s))

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=microphone_callback,
                device=input_device,
            ):
                sender = asyncio.create_task(send_microphone(conn))
                try:
                    await asyncio.sleep(settle_ms / 1000)
                    result.times.playback_started = time.monotonic()
                    await asyncio.to_thread(_play_audio, audio, playback_rate, output_device, volume)
                    await asyncio.wait_for(receiver, timeout=timeout_s)
                finally:
                    stop_sending.set()
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
                    if not receiver.done():
                        receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)
    finally:
        await client.close()

    wer = word_error_rate(reference, result.transcript)
    return {
        "reference": reference,
        "transcript": result.transcript,
        "assistant": result.assistant_transcript,
        "response_status": result.response_status,
        "reference_words": len(normalize_text(reference).split()),
        "wer": round(wer, 4),
        "word_accuracy": round(max(0.0, 1.0 - wer), 4),
        "character_similarity": round(character_similarity(reference, result.transcript), 4),
        "microphone_peak": microphone_peak,
        "synthesis_s": round(synthesis_s, 3),
        "audio_duration_s": round(len(audio) / playback_rate, 3),
        "play_to_speech_start_s": _seconds_between(result.times.speech_started, result.times.playback_started),
        "speech_stop_to_transcript_s": _seconds_between(
            result.times.transcription_completed,
            result.times.speech_stopped,
        ),
        "transcript_to_first_audio_s": _seconds_between(
            result.times.first_audio,
            result.times.transcription_completed,
        ),
        "transcript_to_response_done_s": _seconds_between(
            result.times.response_done,
            result.times.transcription_completed,
        ),
        "play_to_response_done_s": _seconds_between(result.times.response_done, result.times.playback_started),
        "total_script_s": round(time.monotonic() - started, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Русская тестовая фраза, не более 20 слов.")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--settle-ms", type=int, default=750)
    parser.add_argument("--voice", default="M1")
    parser.add_argument("--volume", type=float, default=0.85)
    parser.add_argument("--json", action="store_true", help="Вывести только JSON.")
    return parser


def _print_human(result: dict[str, Any]) -> None:
    print(f"Эталон:       {result['reference']}")
    print(f"Распознано:   {result['transcript'] or '<пусто>'}")
    print(f"WER:          {result['wer']:.1%}")
    print(f"Точность слов:{result['word_accuracy']:.1%}")
    print(f"Сходство:     {result['character_similarity']:.1%}")
    print(f"Пик микрофона:{result['microphone_peak']}")
    print(f"STT после речи: {result['speech_stop_to_transcript_s']} с")
    print(f"До первого аудио ответа: {result['transcript_to_first_audio_s']} с")
    print(f"Полный цикл от воспроизведения: {result['play_to_response_done_s']} с")
    print(f"Ответ:        {result['assistant'] or '<нет>'}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_reference(args.text)
        if args.timeout <= 0:
            raise ValueError("--timeout должен быть положительным")
        if args.settle_ms < 0:
            raise ValueError("--settle-ms не может быть отрицательным")
        if not 0 < args.volume <= 1:
            raise ValueError("--volume должен быть в диапазоне (0, 1]")
        result = asyncio.run(
            run_acoustic_loopback(
                text=args.text,
                url=args.url,
                input_device=args.input_device,
                output_device=args.output_device,
                timeout_s=args.timeout,
                settle_ms=args.settle_ms,
                voice=args.voice,
                volume=args.volume,
            )
        )
    except Exception as exc:
        print(f"Ошибка акустического теста: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
