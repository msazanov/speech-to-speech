"""Transactional SQLite persistence for local speaker identity memory."""

from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .models import (
    ExpiredSpeakerReference,
    InvalidEmbedding,
    Person,
    PersonCandidate,
    SpeakerObservation,
    SpeakerReference,
    SpeakerReferenceConversationMismatch,
    UnknownSpeakerReference,
    VoiceCluster,
)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a contiguous normalized float32 vector or reject unsafe input."""

    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise InvalidEmbedding("speaker embedding must be a finite, non-empty vector")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise InvalidEmbedding("speaker embedding must have a finite non-zero norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


class SpeakerMemoryStore:
    """Small synchronous store intended for one in-process pipeline."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        timeout_s: float = 2.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=timeout_s,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")
            self._migrate()

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS schema_meta (
                version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS voice_clusters (
                id TEXT PRIMARY KEY,
                centroid BLOB NOT NULL,
                dimension INTEGER NOT NULL CHECK (dimension > 0),
                sample_count INTEGER NOT NULL CHECK (sample_count > 0),
                quality_weight REAL NOT NULL CHECK (quality_weight >= 0),
                created_at REAL NOT NULL,
                last_seen REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                last_seen REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS speaker_observations (
                id TEXT PRIMARY KEY,
                voice_id TEXT NOT NULL REFERENCES voice_clusters(id),
                turn_id TEXT NOT NULL,
                turn_revision INTEGER NOT NULL,
                conversation_id TEXT NOT NULL,
                quality REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS speaker_observations_voice_idx
                ON speaker_observations(voice_id, created_at);
            CREATE TABLE IF NOT EXISTS speaker_references (
                ref TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL REFERENCES speaker_observations(id),
                conversation_id TEXT NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS voice_person_evidence (
                id TEXT PRIMARY KEY,
                voice_id TEXT NOT NULL REFERENCES voice_clusters(id),
                person_id TEXT NOT NULL REFERENCES persons(id),
                kind TEXT NOT NULL,
                weight REAL NOT NULL,
                observation_id TEXT REFERENCES speaker_observations(id),
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS voice_person_evidence_lookup_idx
                ON voice_person_evidence(voice_id, person_id, created_at);
            COMMIT;
            """
        )
        row = self._connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if row is None:
            self._connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (self._SCHEMA_VERSION,))
        elif row["version"] != self._SCHEMA_VERSION:
            raise RuntimeError(f"unsupported speaker-memory schema version: {row['version']}")

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{secrets.token_hex(8)}"

    @staticmethod
    def _cluster_from_row(row: sqlite3.Row) -> VoiceCluster:
        centroid = np.frombuffer(row["centroid"], dtype=np.float32, count=row["dimension"]).copy()
        return VoiceCluster(
            id=row["id"],
            centroid=centroid,
            sample_count=row["sample_count"],
            quality_weight=row["quality_weight"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
        )

    def create_voice_cluster(self, embedding: np.ndarray, *, quality: float) -> VoiceCluster:
        vector = normalize_embedding(embedding)
        if not np.isfinite(quality) or quality < 0:
            raise ValueError("quality must be finite and non-negative")
        cluster_id = self._id("v")
        now = self.clock()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO voice_clusters
                   (id, centroid, dimension, sample_count, quality_weight, created_at, last_seen)
                   VALUES (?, ?, ?, 1, ?, ?, ?)""",
                (cluster_id, vector.tobytes(), vector.size, float(quality), now, now),
            )
        return VoiceCluster(
            id=cluster_id,
            centroid=vector.copy(),
            sample_count=1,
            quality_weight=float(quality),
            created_at=now,
            last_seen=now,
        )

    def get_voice_cluster(self, voice_id: str) -> VoiceCluster:
        with self._lock:
            row = self._connection.execute("SELECT * FROM voice_clusters WHERE id = ?", (voice_id,)).fetchone()
        if row is None:
            raise KeyError(voice_id)
        return self._cluster_from_row(row)

    def get_voice_clusters(self) -> list[VoiceCluster]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM voice_clusters ORDER BY created_at, id").fetchall()
        return [self._cluster_from_row(row) for row in rows]

    def create_observation(
        self,
        voice_id: str,
        *,
        turn_id: str,
        turn_revision: int,
        conversation_id: str,
        quality: float,
    ) -> SpeakerObservation:
        observation_id = self._id("o")
        now = self.clock()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO speaker_observations
                   (id, voice_id, turn_id, turn_revision, conversation_id, quality, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (observation_id, voice_id, turn_id, turn_revision, conversation_id, float(quality), now),
            )
        return SpeakerObservation(
            id=observation_id,
            voice_id=voice_id,
            turn_id=turn_id,
            turn_revision=turn_revision,
            conversation_id=conversation_id,
            quality=float(quality),
            created_at=now,
        )

    def issue_reference(self, observation_id: str, *, conversation_id: str, ttl_s: float) -> str:
        if ttl_s <= 0:
            raise ValueError("reference TTL must be positive")
        reference = self._id("sr")
        expires_at = self.clock() + ttl_s
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT conversation_id FROM speaker_observations WHERE id = ?", (observation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(observation_id)
            if row["conversation_id"] != conversation_id:
                raise SpeakerReferenceConversationMismatch("observation belongs to another conversation")
            self._connection.execute(
                "INSERT INTO speaker_references(ref, observation_id, conversation_id, expires_at) VALUES (?, ?, ?, ?)",
                (reference, observation_id, conversation_id, expires_at),
            )
        return reference

    def resolve_reference(self, reference: str, *, conversation_id: str) -> SpeakerReference:
        with self._lock:
            row = self._connection.execute(
                """SELECT r.ref, r.observation_id, r.conversation_id, r.expires_at, o.voice_id
                   FROM speaker_references AS r
                   JOIN speaker_observations AS o ON o.id = r.observation_id
                   WHERE r.ref = ?""",
                (reference,),
            ).fetchone()
        if row is None:
            raise UnknownSpeakerReference("speaker reference does not exist")
        if row["conversation_id"] != conversation_id:
            raise SpeakerReferenceConversationMismatch("speaker reference belongs to another conversation")
        if self.clock() > row["expires_at"]:
            raise ExpiredSpeakerReference("speaker reference has expired")
        return SpeakerReference(
            value=row["ref"],
            observation_id=row["observation_id"],
            voice_id=row["voice_id"],
            conversation_id=row["conversation_id"],
            expires_at=row["expires_at"],
        )

    def create_person(self, name: str) -> Person:
        display_name = " ".join(name.split())
        if not display_name:
            raise ValueError("person name must not be empty")
        normalized_name = display_name.casefold()
        now = self.clock()
        person_id = self._id("p")
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO persons(id, display_name, normalized_name, created_at, last_seen)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(normalized_name) DO UPDATE SET last_seen = excluded.last_seen""",
                (person_id, display_name, normalized_name, now, now),
            )
            row = self._connection.execute(
                "SELECT * FROM persons WHERE normalized_name = ?", (normalized_name,)
            ).fetchone()
        assert row is not None
        return Person(
            id=row["id"],
            name=row["display_name"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
        )

    def add_identity_evidence(
        self,
        voice_id: str,
        person_id: str,
        *,
        kind: str,
        weight: float,
        observation_id: str | None = None,
    ) -> None:
        if not kind.strip():
            raise ValueError("evidence kind must not be empty")
        if not np.isfinite(weight):
            raise ValueError("evidence weight must be finite")
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO voice_person_evidence
                   (id, voice_id, person_id, kind, weight, observation_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self._id("e"), voice_id, person_id, kind, float(weight), observation_id, self.clock()),
            )

    def resolve_person_candidates(self, voice_id: str) -> list[PersonCandidate]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT p.id AS person_id, p.display_name,
                          SUM(e.weight) AS evidence_score
                   FROM voice_person_evidence AS e
                   JOIN persons AS p ON p.id = e.person_id
                   WHERE e.voice_id = ?
                   GROUP BY p.id, p.display_name
                   ORDER BY evidence_score DESC, p.id""",
                (voice_id,),
            ).fetchall()
        return [
            PersonCandidate(
                person_id=row["person_id"],
                name=row["display_name"],
                evidence_score=float(row["evidence_score"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
