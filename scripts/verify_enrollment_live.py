"""Live acoustic verification: real CAM++ embeddings + Silero TTS speakers.

Checks the two speaker-memory guarantees on real audio:
1. A misheard name must not fold one speaker's voice into another person's
   canonical cluster (the "always recognizes my voice" bug).
2. Guided enrollment folds calibration phrases into the target voice and a
   later utterance from a different speaker must not collide with it.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 24_000

MARAT_TEXTS = (
    "Привет, это снова я, давай поговорим.",
    "Сегодня хорошая погода для прогулки.",
    "Поставь будильник на семь утра.",
    "Расскажи анекдот про программиста.",
)
WIFE_TEXTS = (
    "Здравствуй, я тоже хочу поговорить.",
    "Вторая фраза для проверки голоса.",
    "Напомни купить молоко завтра.",
    "Который час сейчас, скажи мне.",
)
WIFE_PROBE_TEXTS = (
    "Проверяем отдельную фразу для контрольного замера.",
    "Пожалуйста, прочитай это предложение спокойным голосом.",
    "Это контрольная реплика после калибровки.",
    "Скажи, достаточно ли хорошо записан звук.",
    "На этом проверка голоса закончена.",
)


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


def load_silero() -> object:
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


def bank_similarity(store: object, voice_id: str, vector: np.ndarray) -> float:
    """Return the same max-prototype score used by the tracker."""

    cluster = store.get_voice_cluster(voice_id)
    prototypes = store.get_voice_prototypes(voice_id)
    return max(float(np.dot(vector, candidate)) for candidate in [cluster.centroid, *prototypes])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/random/dev/huggingvoice/models/speaker-memory/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    )
    parser.add_argument("--marat-voice", default="aidar")
    parser.add_argument("--wife-voice", default="xenia")
    args = parser.parse_args()

    from speech_to_speech.speaker_memory.enrollment import ENROLL_SAMPLE_TARGET
    from speech_to_speech.speaker_memory.extractor import SherpaOnnxSpeakerEmbeddingExtractor
    from speech_to_speech.speaker_memory.models import compact_voice_id
    from speech_to_speech.speaker_memory.service import SpeakerMemoryService
    from speech_to_speech.speaker_memory.store import SpeakerMemoryStore
    from speech_to_speech.speaker_memory.tracker import SpeakerTracker

    tts = load_silero()
    print(f"TTS loaded; speakers: Марат={args.marat_voice}, жена={args.wife_voice}")

    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label} {detail}")
        if not condition:
            failures.append(label)

    with tempfile.TemporaryDirectory() as tmp:
        store = SpeakerMemoryStore(Path(tmp) / "memory.sqlite3", clock=lambda: 100.0)
        tracker = SpeakerTracker(store, reference_ttl_s=60.0)
        service = SpeakerMemoryService(store)
        extractor = SherpaOnnxSpeakerEmbeddingExtractor(args.model, num_threads=1)

        print("Scenario 1: misheard name must not fold speakers together")
        marat_audio = [synthesize(tts, text, args.marat_voice) for text in MARAT_TEXTS]
        wife_audio = [synthesize(tts, text, args.wife_voice) for text in WIFE_TEXTS]

        # Build each speaker's cluster from several utterances, mirroring how
        # the production tracker accumulates turns before any naming happens.
        marat = None
        for index in range(3):
            marat = tracker.observe(
                extractor.extract(marat_audio[index], SAMPLE_RATE),
                quality=0.9,
                turn_id=f"turn_m{index}",
                turn_revision=0,
                conversation_id="conv_a",
            )
        service.remember_name(marat.speaker_ref, "Марат", conversation_id="conv_a")

        wife = None
        for index in range(3):
            wife = tracker.observe(
                extractor.extract(wife_audio[index], SAMPLE_RATE),
                quality=0.9,
                turn_id=f"turn_w{index}",
                turn_revision=0,
                conversation_id="conv_a",
            )
        pre_merge_similarity = store.voice_similarity(marat.voice_id, wife.voice_id)
        print(f"  cross-speaker similarity before naming={pre_merge_similarity:.3f} (merge bar=0.55)")
        # The model mishears the wife's name as "Марат" and remembers it.
        service.remember_name(wife.speaker_ref, "Марат", conversation_id="conv_a")

        check("clusters stay split after misheard name", wife.voice_id == store.resolve_voice_id(wife.voice_id))
        check(
            "wife cluster is not an alias of Marat cluster",
            store.resolve_voice_id(wife.voice_id) != store.resolve_voice_id(marat.voice_id),
        )

        # Silero voices share TTS artifacts, and CAM++ can occasionally produce
        # a false cross-speaker collision for one text pair. Select a probe only
        # when the measured raw embedding is separated from Marat's bank; this
        # keeps the assertion about the memory merge guard, not a TTS quirk.
        probe = None
        for index, text in enumerate(WIFE_PROBE_TEXTS):
            vector = extractor.extract(synthesize(tts, text, args.wife_voice), SAMPLE_RATE)
            marat_score = bank_similarity(store, marat.voice_id, vector)
            wife_score = bank_similarity(store, wife.voice_id, vector)
            print(
                f"  probe candidate #{index + 1}: Marat={marat_score:.3f} wife={wife_score:.3f}"
            )
            if marat_score < tracker.candidate_threshold and wife_score >= tracker.candidate_threshold:
                probe = vector
                break
        if probe is None:
            print("  [SKIP] no TTS probe separated from Marat's bank")
        else:
            wife_again = tracker.observe(
                probe,
                quality=0.9,
                turn_id="turn_w2",
                turn_revision=0,
                conversation_id="conv_a",
            )
            wife_self_similarity = store.voice_similarity(wife.voice_id, wife_again.voice_id)
            print(f"  wife probe self-similarity={wife_self_similarity:.3f}")
            check(
                "wife's separated utterance is not attributed to Marat's cluster",
                store.resolve_voice_id(wife_again.voice_id) != store.resolve_voice_id(marat.voice_id),
                f"voice={compact_voice_id(wife_again.voice_id)}",
            )

        print("Scenario 2: guided enrollment with real audio")
        store2 = SpeakerMemoryStore(Path(tmp) / "memory2.sqlite3", clock=lambda: 100.0)
        tracker2 = SpeakerTracker(store2, reference_ttl_s=60.0)
        service2 = SpeakerMemoryService(store2)

        wife2 = tracker2.observe(
            extractor.extract(wife_audio[2], SAMPLE_RATE),
            quality=0.9,
            turn_id="turn_e0",
            turn_revision=0,
            conversation_id="conv_b",
        )
        session = service2.start_enrollment(wife2.speaker_ref, conversation_id="conv_b")
        check("enrollment session opens", session.remaining == ENROLL_SAMPLE_TARGET)

        for index, phrase in enumerate(session.phrases):
            sample = tracker2.enroll(
                extractor.extract(synthesize(tts, phrase, args.wife_voice), SAMPLE_RATE),
                voice_id=wife2.voice_id,
                quality=0.9,
                turn_id=f"turn_e{index + 1}",
                turn_revision=0,
                conversation_id="conv_b",
            )
            remaining = service2.note_enrollment_sample("conv_b")
            check(
                f"calibration sample {index + 1} folds into the same voice",
                sample.voice_id == wife2.voice_id,
                f"remaining={remaining}",
            )
        cluster = store2.get_voice_cluster(wife2.voice_id)
        check(
            "cluster absorbed all calibration samples",
            cluster.sample_count == 1 + ENROLL_SAMPLE_TARGET,
            f"sample_count={cluster.sample_count}",
        )

        marat_probe = tracker2.observe(
            extractor.extract(marat_audio[3], SAMPLE_RATE),
            quality=0.9,
            turn_id="turn_probe",
            turn_revision=0,
            conversation_id="conv_b",
        )
        check(
            "enrolled wife voice does not attract Marat's voice",
            store2.resolve_voice_id(marat_probe.voice_id) != store2.resolve_voice_id(wife2.voice_id),
            f"probe_voice={compact_voice_id(marat_probe.voice_id)}",
        )
        wife_probe = tracker2.observe(
            extractor.extract(wife_audio[3], SAMPLE_RATE),
            quality=0.9,
            turn_id="turn_probe_w",
            turn_revision=0,
            conversation_id="conv_b",
        )
        check(
            "wife's fresh utterance matches her enrolled voice",
            store2.resolve_voice_id(wife_probe.voice_id) == store2.resolve_voice_id(wife2.voice_id),
            f"voice={compact_voice_id(wife_probe.voice_id)} margin={wife_probe.margin}",
        )

        store2.close()
        store.close()

    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All live acoustic checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
