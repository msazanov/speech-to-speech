#!/usr/bin/env python3
"""Run a bounded offline speaker-memory smoke over labelled WAV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local sherpa-onnx speaker embedding model")
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        metavar="NAME=PATH.wav",
        help="Labelled 16 kHz WAV; repeat twice or more per expected person",
    )
    parser.add_argument("--database", help="Optional SQLite output; defaults to an isolated temporary file")
    parser.add_argument("--match-threshold", type=float, default=0.82)
    parser.add_argument("--candidate-threshold", type=float, default=0.70)
    parser.add_argument("--ambiguity-margin", type=float, default=0.08)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    return parser


def parse_sample(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError(f"sample must use NAME=PATH.wav syntax: {value!r}")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label.strip(), path


def load_mono_16k(path: Path):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sample_rate, audio = wavfile.read(path)
    array = np.asarray(audio)
    if array.ndim == 2:
        array = array.astype(np.float32).mean(axis=1)
    if np.issubdtype(array.dtype, np.integer):
        maximum = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / float(maximum)
    else:
        array = array.astype(np.float32)
    if sample_rate != 16_000:
        divisor = math.gcd(sample_rate, 16_000)
        array = resample_poly(array, 16_000 // divisor, sample_rate // divisor).astype(np.float32)
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float32)


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def validate_samples(samples: list[tuple[str, Path]]) -> None:
    counts = Counter(label for label, _path in samples)
    if len(counts) < 2:
        raise ValueError("speaker smoke requires at least two distinct people")
    insufficient = sorted(label for label, count in counts.items() if count < 2)
    if insufficient:
        raise ValueError(f"speaker smoke requires at least two recordings for: {', '.join(insufficient)}")
    hashes_by_label: dict[str, set[bytes]] = {}
    for label, path in samples:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        hashes_by_label.setdefault(label, set()).add(digest)
    duplicates = sorted(label for label, hashes in hashes_by_label.items() if len(hashes) < 2)
    if duplicates:
        raise ValueError(f"speaker smoke requires two distinct recordings for: {', '.join(duplicates)}")


def run_smoke(arguments: argparse.Namespace, database_path: Path) -> dict[str, object]:
    samples = [parse_sample(raw_sample) for raw_sample in arguments.sample]
    validate_samples(samples)

    import numpy as np

    from speech_to_speech.speaker_memory.extractor import SherpaOnnxSpeakerEmbeddingExtractor
    from speech_to_speech.speaker_memory.models import SpeakerState
    from speech_to_speech.speaker_memory.service import SpeakerMemoryService
    from speech_to_speech.speaker_memory.store import SpeakerMemoryStore
    from speech_to_speech.speaker_memory.tracker import SpeakerTracker

    extractor = SherpaOnnxSpeakerEmbeddingExtractor(arguments.model, num_threads=1)
    store = SpeakerMemoryStore(database_path)
    tracker = SpeakerTracker(
        store,
        match_threshold=arguments.match_threshold,
        candidate_threshold=arguments.candidate_threshold,
        ambiguity_margin=arguments.ambiguity_margin,
    )
    service = SpeakerMemoryService(store)
    expected_voice_by_label: dict[str, str] = {}
    representative_embedding_by_label: dict[str, np.ndarray] = {}
    last_reference_by_label: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    try:
        for index, (label, path) in enumerate(samples, start=1):
            audio = load_mono_16k(path)
            started = time.perf_counter()
            embedding = extractor.extract(audio, 16_000)
            representative_embedding_by_label.setdefault(label, embedding)
            attribution = tracker.observe(
                embedding,
                quality=1.0,
                turn_id=f"smoke_{index}",
                turn_revision=0,
                conversation_id="smoke",
            )
            speaker_ms = (time.perf_counter() - started) * 1000
            latencies.append(speaker_ms)
            expected_voice = expected_voice_by_label.get(label)
            if expected_voice is None:
                expected_voice_by_label[label] = attribution.voice_id
                service.remember_name(attribution.speaker_ref, label, conversation_id="smoke")
                expected_voice = attribution.voice_id
            last_reference_by_label[label] = attribution.speaker_ref
            other_voices = {voice for known_label, voice in expected_voice_by_label.items() if known_label != label}
            correct = attribution.voice_id == expected_voice and attribution.voice_id not in other_voices
            inspected = service.inspect(attribution.speaker_ref, conversation_id="smoke")
            rows.append(
                {
                    "label": label,
                    "path": str(path),
                    "voice_id": attribution.voice_id,
                    "state": inspected.state.value,
                    "candidate": inspected.candidate.name if inspected.candidate else None,
                    "speaker_ms": round(speaker_ms, 3),
                    "expected_voice_id": expected_voice,
                    "decision_correct": correct,
                }
            )

        probe_label = next(iter(expected_voice_by_label))
        probe_voice = expected_voice_by_label[probe_label]
        probe_reference = last_reference_by_label[probe_label]
        before_confirmation = store.resolve_person_candidates(probe_voice)[0]
        service.confirm(probe_reference, before_confirmation.person_id, conversation_id="smoke")
        after_confirmation = store.resolve_person_candidates(probe_voice)[0]
        service.reject(probe_reference, before_confirmation.person_id, conversation_id="smoke")
        after_rejection = store.resolve_person_candidates(probe_voice)[0]
        semantic_probe = {
            "person_id": before_confirmation.person_id,
            "before": before_confirmation.evidence_score,
            "after_confirmation": after_confirmation.evidence_score,
            "after_rejection": after_rejection.evidence_score,
            "confirmation_increased": after_confirmation.evidence_score > before_confirmation.evidence_score,
            "rejection_decreased": after_rejection.evidence_score < after_confirmation.evidence_score,
        }

        first_label, second_label = list(expected_voice_by_label)[:2]
        first_embedding = np.asarray(representative_embedding_by_label[first_label], dtype=np.float32)
        second_embedding = np.asarray(representative_embedding_by_label[second_label], dtype=np.float32)
        mixed_embedding = first_embedding + second_embedding
        centroids_before = {cluster.id: cluster.centroid.copy() for cluster in store.get_voice_clusters()}
        mixed_norm = float(np.linalg.norm(mixed_embedding))
        if mixed_norm <= 1e-6:
            ambiguity_probe = {
                "state": "invalid_probe",
                "centroids_unchanged": True,
                "passed": False,
            }
        else:
            mixed_embedding /= mixed_norm
            mixed = tracker.observe(
                mixed_embedding,
                quality=1.0,
                turn_id="smoke_ambiguous_probe",
                turn_revision=0,
                conversation_id="smoke",
            )
            centroids_after = {cluster.id: cluster.centroid.copy() for cluster in store.get_voice_clusters()}
            unchanged = centroids_before.keys() == centroids_after.keys() and all(
                np.array_equal(centroid, centroids_after[voice_id])
                for voice_id, centroid in centroids_before.items()
            )
            ambiguity_probe = {
                "state": mixed.state.value,
                "centroids_unchanged": unchanged,
                "passed": mixed.state is SpeakerState.AMBIGUOUS and unchanged,
            }
    finally:
        store.close()
    warm_p95 = percentile_95(latencies[1:]) if len(latencies) > 1 else float("inf")
    decisions_correct = all(bool(row["decision_correct"]) for row in rows)
    semantic_passed = bool(
        semantic_probe["confirmation_increased"] and semantic_probe["rejection_decreased"]
    )
    return {
        "ok": decisions_correct and semantic_passed and ambiguity_probe["passed"] and warm_p95 <= 100.0,
        "samples": rows,
        "minimum_samples_passed": True,
        "same_different_passed": decisions_correct,
        "semantic_probe": semantic_probe,
        "ambiguity_probe": ambiguity_probe,
        "warm_speaker_ms_p95": round(warm_p95, 3),
        "warm_speaker_ms_budget": 100.0,
        "latency_passed": warm_p95 <= 100.0,
        "thresholds": {
            "match": arguments.match_threshold,
            "candidate": arguments.candidate_threshold,
            "margin": arguments.ambiguity_margin,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.database:
        report = run_smoke(arguments, Path(arguments.database).expanduser())
    else:
        with tempfile.TemporaryDirectory(prefix="huggingvoice-speaker-smoke-") as directory:
            report = run_smoke(arguments, Path(directory) / "memory.sqlite3")
    if arguments.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in report["samples"]:
            print(
                f"{row['label']}: voice={row['voice_id']} state={row['state']} "
                f"speaker_ms={row['speaker_ms']} correct={row['decision_correct']}"
            )
        print(f"warm speaker p95: {report['warm_speaker_ms_p95']} ms; ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
