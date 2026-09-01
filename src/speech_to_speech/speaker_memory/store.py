"""Transactional SQLite persistence for local speaker identity memory."""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .models import (
    ExpiredSpeakerReference,
    InvalidEmbedding,
    Person,
    PersonalFact,
    PersonCandidate,
    SpeakerObservation,
    SpeakerReference,
    SpeakerReferenceConversationMismatch,
    SupersededSpeakerReference,
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

    _SCHEMA_VERSION = 6
    _MAX_CENTROID_WEIGHT = 20.0
    _MAX_EVIDENCE_SCORE = 10.0

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        timeout_s: float = 2.0,
        observation_retention_days: int = 30,
    ) -> None:
        self.path = Path(path)
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        self.path.chmod(0o600)
        self.clock = clock
        if observation_retention_days < 1:
            raise ValueError("observation retention must be at least one day")
        self.observation_retention_days = observation_retention_days
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
            self._fts_enabled = self._initialize_fact_search()
            self._restrict_sidecar_permissions()
        self.prune_expired()

    def _restrict_sidecar_permissions(self) -> None:
        for suffix in ("-wal", "-shm"):
            sidecar = self.path.with_name(self.path.name + suffix)
            if sidecar.exists():
                sidecar.chmod(0o600)

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
            CREATE TABLE IF NOT EXISTS voice_aliases (
                source_voice_id TEXT PRIMARY KEY REFERENCES voice_clusters(id) ON DELETE CASCADE,
                canonical_voice_id TEXT NOT NULL REFERENCES voice_clusters(id) ON DELETE CASCADE,
                reason TEXT NOT NULL,
                created_at REAL NOT NULL,
                CHECK (source_voice_id <> canonical_voice_id)
            );
            CREATE INDEX IF NOT EXISTS voice_aliases_canonical_idx
                ON voice_aliases(canonical_voice_id);
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
            CREATE INDEX IF NOT EXISTS speaker_observations_created_idx
                ON speaker_observations(created_at);
            CREATE TABLE IF NOT EXISTS speaker_references (
                ref TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL REFERENCES speaker_observations(id),
                conversation_id TEXT NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS speaker_references_expiry_idx
                ON speaker_references(expires_at);
            CREATE TABLE IF NOT EXISTS speaker_reference_candidates (
                ref TEXT NOT NULL REFERENCES speaker_references(ref) ON DELETE CASCADE,
                person_id TEXT NOT NULL REFERENCES persons(id),
                created_at REAL NOT NULL,
                PRIMARY KEY (ref, person_id)
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
            CREATE TABLE IF NOT EXISTS voice_person_blocks (
                voice_id TEXT NOT NULL REFERENCES voice_clusters(id) ON DELETE CASCADE,
                person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                reason TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (voice_id, person_id)
            );
            CREATE INDEX IF NOT EXISTS voice_person_blocks_person_idx
                ON voice_person_blocks(person_id, voice_id);
            DELETE FROM voice_person_evidence
                WHERE observation_id IS NOT NULL
                  AND rowid NOT IN (
                      SELECT MIN(rowid)
                      FROM voice_person_evidence
                      WHERE observation_id IS NOT NULL
                      GROUP BY observation_id, person_id, kind
                  );
            CREATE UNIQUE INDEX IF NOT EXISTS voice_person_evidence_semantic_once_idx
                ON voice_person_evidence(observation_id, person_id, kind)
                WHERE observation_id IS NOT NULL;
            CREATE TABLE IF NOT EXISTS personal_facts (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                topic TEXT,
                fact TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS personal_facts_person_idx
                ON personal_facts(person_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS voice_policies (
                voice_id TEXT PRIMARY KEY REFERENCES voice_clusters(id) ON DELETE CASCADE,
                blocked INTEGER NOT NULL CHECK (blocked IN (0, 1)),
                reason TEXT,
                updated_at REAL NOT NULL
            );
            COMMIT;
            """
        )
        row = self._connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if row is None:
            self._connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (self._SCHEMA_VERSION,))
        elif row["version"] in {1, 2, 3, 4, 5}:
            self._connection.execute("UPDATE schema_meta SET version = ?", (self._SCHEMA_VERSION,))
        elif row["version"] != self._SCHEMA_VERSION:
            raise RuntimeError(f"unsupported speaker-memory schema version: {row['version']}")

    def _initialize_fact_search(self) -> bool:
        try:
            self._connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS personal_facts_fts USING fts5(
                       fact_id UNINDEXED,
                       person_id UNINDEXED,
                       topic,
                       fact,
                       tokenize='unicode61'
                   )"""
            )
        except sqlite3.OperationalError:
            return False
        return True

    @contextmanager
    def _transaction(self):
        """Serialize writes across SQLite connections with a bounded immediate lock."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

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
        with self._transaction():
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

    def resolve_voice_id(self, voice_id: str) -> str:
        """Resolve a historical voice ID to its current canonical cluster."""

        current = voice_id
        seen: set[str] = set()
        with self._lock:
            while current not in seen:
                seen.add(current)
                row = self._connection.execute(
                    "SELECT canonical_voice_id FROM voice_aliases WHERE source_voice_id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    return current
                current = str(row["canonical_voice_id"])
        raise RuntimeError(f"voice alias cycle detected for {voice_id}")

    def get_voice_clusters(self) -> list[VoiceCluster]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT c.* FROM voice_clusters AS c
                   WHERE NOT EXISTS (
                       SELECT 1 FROM voice_aliases AS a WHERE a.source_voice_id = c.id
                   )
                   ORDER BY c.created_at, c.id"""
            ).fetchall()
        return [self._cluster_from_row(row) for row in rows]

    def merge_voice_clusters(self, source_voice_id: str, target_voice_id: str, *, reason: str) -> str:
        """Merge one cluster into another while retaining a reversible alias audit."""

        if not reason.strip():
            raise ValueError("voice merge reason must not be empty")
        with self._transaction():
            source_id = self.resolve_voice_id(source_voice_id)
            target_id = self.resolve_voice_id(target_voice_id)
            if source_id == target_id:
                return target_id
            source_row = self._connection.execute(
                "SELECT * FROM voice_clusters WHERE id = ?", (source_id,)
            ).fetchone()
            target_row = self._connection.execute(
                "SELECT * FROM voice_clusters WHERE id = ?", (target_id,)
            ).fetchone()
            if source_row is None or target_row is None:
                raise KeyError(source_id if source_row is None else target_id)
            source = self._cluster_from_row(source_row)
            target = self._cluster_from_row(target_row)
            if source.centroid.size != target.centroid.size:
                raise InvalidEmbedding("speaker embedding dimensions do not match for merge")
            source_weight = min(source.quality_weight, self._MAX_CENTROID_WEIGHT)
            target_weight = min(target.quality_weight, self._MAX_CENTROID_WEIGHT)
            total_weight = min(source_weight + target_weight, self._MAX_CENTROID_WEIGHT)
            centroid = normalize_embedding(
                source.centroid * source_weight + target.centroid * target_weight
            )
            now = self.clock()
            self._connection.execute(
                """UPDATE voice_clusters
                   SET centroid = ?, sample_count = ?, quality_weight = ?, last_seen = ?
                   WHERE id = ?""",
                (
                    centroid.tobytes(),
                    source.sample_count + target.sample_count,
                    total_weight,
                    max(source.last_seen, target.last_seen, now),
                    target_id,
                ),
            )
            # Flatten any aliases that pointed at either cluster, then record
            # the source alias itself. Evidence remains attached to its source
            # row for audit and is aggregated through this alias.
            self._connection.execute(
                "UPDATE voice_aliases SET canonical_voice_id = ? WHERE canonical_voice_id IN (?, ?)",
                (target_id, source_id, target_id),
            )
            self._connection.execute(
                """INSERT INTO voice_aliases
                   (source_voice_id, canonical_voice_id, reason, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(source_voice_id) DO UPDATE SET
                       canonical_voice_id = excluded.canonical_voice_id,
                       reason = excluded.reason,
                       created_at = excluded.created_at""",
                (source_id, target_id, reason.strip()[:120], now),
            )
        return target_id

    def detach_voice_alias(self, voice_id: str) -> bool:
        """Detach a source cluster from its canonical voice after an explicit rejection."""

        with self._transaction():
            cursor = self._connection.execute(
                "DELETE FROM voice_aliases WHERE source_voice_id = ?", (voice_id,)
            )
        return bool(cursor.rowcount)

    def merge_voice_with_person(self, voice_id: str, person_id: str, *, reason: str) -> str:
        """Safely merge a newly confirmed voice into that person's strongest cluster."""

        current_id = self.resolve_voice_id(voice_id)
        with self._lock:
            rows = self._connection.execute(
                """SELECT DISTINCT e.voice_id
                   FROM voice_person_evidence AS e
                   WHERE e.person_id = ?""",
                (person_id,),
            ).fetchall()
        candidates: list[VoiceCluster] = []
        for row in rows:
            candidate_id = self.resolve_voice_id(str(row["voice_id"]))
            if candidate_id == current_id:
                continue
            try:
                candidate_cluster = self.get_voice_cluster(candidate_id)
            except KeyError:
                continue
            person_score = next(
                (
                    candidate.evidence_score
                    for candidate in self.resolve_person_candidates(candidate_id)
                    if candidate.person_id == person_id
                ),
                0.0,
            )
            if person_score > 0:
                candidates.append(candidate_cluster)
        if not candidates:
            return current_id
        target = max(candidates, key=lambda cluster: (cluster.sample_count, cluster.quality_weight, -cluster.created_at))
        return self.merge_voice_clusters(current_id, target.id, reason=reason)

    def set_voice_blocked(self, voice_id: str, *, blocked: bool, reason: str | None = None) -> None:
        """Persist a voice-routing decision without exposing its embedding."""

        normalized_reason = " ".join(reason.split()) if reason else None
        if normalized_reason is not None and len(normalized_reason) > 80:
            raise ValueError("voice block reason must be at most 80 characters")
        with self._transaction():
            exists = self._connection.execute(
                "SELECT 1 FROM voice_clusters WHERE id = ?", (voice_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(voice_id)
            self._connection.execute(
                """INSERT INTO voice_policies(voice_id, blocked, reason, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(voice_id) DO UPDATE SET
                       blocked = excluded.blocked,
                       reason = excluded.reason,
                       updated_at = excluded.updated_at""",
                (voice_id, int(blocked), normalized_reason if blocked else None, self.clock()),
            )

    def set_voice_blocked_by_reference(
        self,
        reference: str,
        *,
        conversation_id: str,
        blocked: bool,
        reason: str | None = None,
    ) -> str:
        """Atomically mutate only the voice from the latest turn in a conversation."""

        normalized_reason = " ".join(reason.split()) if reason else None
        if normalized_reason is not None and len(normalized_reason) > 80:
            raise ValueError("voice block reason must be at most 80 characters")
        with self._transaction():
            row = self._connection.execute(
                """SELECT r.conversation_id, r.expires_at, o.voice_id, o.rowid AS observation_rowid
                   FROM speaker_references AS r
                   JOIN speaker_observations AS o ON o.id = r.observation_id
                   WHERE r.ref = ?""",
                (reference,),
            ).fetchone()
            if row is None:
                raise UnknownSpeakerReference("speaker reference does not exist")
            if row["conversation_id"] != conversation_id:
                raise SpeakerReferenceConversationMismatch(
                    "speaker reference belongs to another conversation"
                )
            if self.clock() >= row["expires_at"]:
                raise ExpiredSpeakerReference("speaker reference has expired")
            latest = self._connection.execute(
                """SELECT rowid FROM speaker_observations
                   WHERE conversation_id = ?
                   ORDER BY created_at DESC, rowid DESC
                   LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            if latest is None or latest["rowid"] != row["observation_rowid"]:
                raise SupersededSpeakerReference("speaker reference was superseded by a newer turn")
            self._connection.execute(
                """INSERT INTO voice_policies(voice_id, blocked, reason, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(voice_id) DO UPDATE SET
                       blocked = excluded.blocked,
                       reason = excluded.reason,
                       updated_at = excluded.updated_at""",
                (
                    row["voice_id"],
                    int(blocked),
                    normalized_reason if blocked else None,
                    self.clock(),
                ),
            )
        return str(row["voice_id"])

    def is_voice_blocked(self, voice_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT blocked FROM voice_policies WHERE voice_id = ?", (voice_id,)
            ).fetchone()
        return bool(row["blocked"]) if row is not None else False

    def update_voice_cluster(self, voice_id: str, embedding: np.ndarray, *, quality: float) -> VoiceCluster:
        """Add one trusted sample to a centroid using a bounded quality-weighted mean."""

        vector = normalize_embedding(embedding)
        if not np.isfinite(quality) or quality <= 0:
            raise ValueError("quality must be finite and positive")
        with self._transaction():
            row = self._connection.execute("SELECT * FROM voice_clusters WHERE id = ?", (voice_id,)).fetchone()
            if row is None:
                raise KeyError(voice_id)
            current = self._cluster_from_row(row)
            if current.centroid.size != vector.size:
                raise InvalidEmbedding("speaker embedding dimension does not match the stored cluster")
            bounded_weight = min(float(quality), 1.0)
            retained_weight = min(current.quality_weight, self._MAX_CENTROID_WEIGHT)
            combined_weight = retained_weight + bounded_weight
            total_weight = min(combined_weight, self._MAX_CENTROID_WEIGHT)
            centroid = normalize_embedding(
                current.centroid * retained_weight + vector * bounded_weight
            )
            now = self.clock()
            self._connection.execute(
                """UPDATE voice_clusters
                   SET centroid = ?, sample_count = ?, quality_weight = ?, last_seen = ?
                   WHERE id = ?""",
                (
                    centroid.tobytes(),
                    current.sample_count + 1,
                    total_weight,
                    now,
                    voice_id,
                ),
            )
        return VoiceCluster(
            id=voice_id,
            centroid=centroid,
            sample_count=current.sample_count + 1,
            quality_weight=total_weight,
            created_at=current.created_at,
            last_seen=now,
        )

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
        with self._transaction():
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
        with self._transaction():
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
        if self.clock() >= row["expires_at"]:
            raise ExpiredSpeakerReference("speaker reference has expired")
        return SpeakerReference(
            value=row["ref"],
            observation_id=row["observation_id"],
            # Keep the raw ID here: reject can detach the exact source cluster.
            voice_id=row["voice_id"],
            conversation_id=row["conversation_id"],
            expires_at=row["expires_at"],
        )

    def invalidate_references(self, conversation_id: str) -> int:
        """Revoke prior turn capabilities before routing a new final segment."""

        with self._transaction():
            cursor = self._connection.execute(
                "DELETE FROM speaker_references WHERE conversation_id = ?",
                (conversation_id,),
            )
        return max(cursor.rowcount, 0)

    def create_person(self, name: str, *, reuse: bool = True) -> Person:
        display_name = " ".join(name.split())
        if not display_name:
            raise ValueError("person name must not be empty")
        normalized_name = display_name.casefold()
        now = self.clock()
        person_id = self._id("p")
        storage_name = normalized_name if reuse else f"{normalized_name}\x1f{person_id}"
        with self._transaction():
            if reuse:
                self._connection.execute(
                    """INSERT INTO persons(id, display_name, normalized_name, created_at, last_seen)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(normalized_name) DO UPDATE SET last_seen = excluded.last_seen""",
                    (person_id, display_name, storage_name, now, now),
                )
                row = self._connection.execute(
                    "SELECT * FROM persons WHERE normalized_name = ?", (storage_name,)
                ).fetchone()
            else:
                self._connection.execute(
                    """INSERT INTO persons(id, display_name, normalized_name, created_at, last_seen)
                       VALUES (?, ?, ?, ?, ?)""",
                    (person_id, display_name, storage_name, now, now),
                )
                row = self._connection.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
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
        with self._transaction():
            self._connection.execute(
                """INSERT INTO voice_person_evidence
                   (id, voice_id, person_id, kind, weight, observation_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(observation_id, person_id, kind)
                   WHERE observation_id IS NOT NULL DO NOTHING""",
                (self._id("e"), voice_id, person_id, kind, float(weight), observation_id, self.clock()),
            )

    def block_voice_person(self, voice_id: str, person_id: str, *, reason: str) -> None:
        """Persist a fail-closed rejection for this voice/person relation."""

        if not reason.strip():
            raise ValueError("voice/person block reason must not be empty")
        with self._transaction():
            self._connection.execute(
                """INSERT INTO voice_person_blocks(voice_id, person_id, reason, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(voice_id, person_id) DO UPDATE SET
                       reason = excluded.reason,
                       created_at = excluded.created_at""",
                (voice_id, person_id, reason.strip()[:120], self.clock()),
            )

    def clear_voice_person_block(self, voice_id: str, person_id: str) -> None:
        with self._transaction():
            self._connection.execute(
                "DELETE FROM voice_person_blocks WHERE voice_id = ? AND person_id = ?",
                (voice_id, person_id),
            )

    def resolve_person_candidates(self, voice_id: str) -> list[PersonCandidate]:
        canonical_id = self.resolve_voice_id(voice_id)
        with self._lock:
            rows = self._connection.execute(
                """SELECT p.id AS person_id, p.display_name,
                          MAX(?, MIN(?, SUM(e.weight))) AS evidence_score
                   FROM voice_person_evidence AS e
                   JOIN persons AS p ON p.id = e.person_id
                   WHERE (e.voice_id = ?
                      OR e.voice_id IN (
                          SELECT source_voice_id FROM voice_aliases
                          WHERE canonical_voice_id = ?
                      ))
                     AND NOT EXISTS (
                         SELECT 1 FROM voice_person_blocks AS b
                         WHERE b.person_id = e.person_id
                           AND (b.voice_id = ? OR b.voice_id IN (
                               SELECT source_voice_id FROM voice_aliases
                               WHERE canonical_voice_id = ?
                           ))
                     )
                   GROUP BY p.id, p.display_name
                   ORDER BY evidence_score DESC, p.id""",
                (
                    -self._MAX_EVIDENCE_SCORE,
                    self._MAX_EVIDENCE_SCORE,
                    canonical_id,
                    canonical_id,
                    canonical_id,
                    canonical_id,
                ),
            ).fetchall()
        return [
            PersonCandidate(
                person_id=row["person_id"],
                name=row["display_name"],
                evidence_score=float(row["evidence_score"]),
            )
            for row in rows
        ]

    @staticmethod
    def _fact_from_row(row: sqlite3.Row) -> PersonalFact:
        return PersonalFact(
            id=row["id"],
            person_id=row["person_id"],
            topic=row["topic"],
            fact=row["fact"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def add_personal_fact(self, person_id: str, fact: str, *, topic: str | None = None) -> PersonalFact:
        fact_id = self._id("f")
        now = self.clock()
        with self._transaction():
            self._connection.execute(
                """INSERT INTO personal_facts(id, person_id, topic, fact, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (fact_id, person_id, topic, fact, now, now),
            )
            if self._fts_enabled:
                self._connection.execute(
                    "INSERT INTO personal_facts_fts(fact_id, person_id, topic, fact) VALUES (?, ?, ?, ?)",
                    (fact_id, person_id, topic or "", fact),
                )
        return PersonalFact(
            id=fact_id,
            person_id=person_id,
            topic=topic,
            fact=fact,
            created_at=now,
            updated_at=now,
        )

    def search_personal_facts(self, person_id: str, query: str, *, limit: int = 5) -> list[PersonalFact]:
        tokens = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
        if not tokens:
            return []
        with self._lock:
            if self._fts_enabled:
                expression = " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
                rows = self._connection.execute(
                    """SELECT f.*
                       FROM personal_facts_fts AS search
                       JOIN personal_facts AS f ON f.id = search.fact_id
                       WHERE personal_facts_fts MATCH ? AND search.person_id = ?
                       ORDER BY bm25(personal_facts_fts), f.updated_at DESC, f.id
                       LIMIT ?""",
                    (expression, person_id, limit),
                ).fetchall()
            else:
                clauses = " AND ".join("(LOWER(fact) LIKE ? ESCAPE '\\' OR LOWER(COALESCE(topic, '')) LIKE ? ESCAPE '\\')" for _ in tokens)
                parameters: list[object] = []
                for token in tokens:
                    escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    parameters.extend((f"%{escaped}%", f"%{escaped}%"))
                parameters.extend((person_id, limit))
                rows = self._connection.execute(
                    f"""SELECT * FROM personal_facts
                        WHERE {clauses} AND person_id = ?
                        ORDER BY updated_at DESC, id
                        LIMIT ?""",
                    parameters,
                ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def delete_personal_facts(self, person_id: str, *, fact_id: str | None = None) -> int:
        with self._transaction():
            if fact_id is None:
                rows = self._connection.execute(
                    "SELECT id FROM personal_facts WHERE person_id = ?", (person_id,)
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT id FROM personal_facts WHERE person_id = ? AND id = ?", (person_id, fact_id)
                ).fetchall()
            ids = [row["id"] for row in rows]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            if self._fts_enabled:
                self._connection.execute(
                    f"DELETE FROM personal_facts_fts WHERE fact_id IN ({placeholders})",
                    ids,
                )
            self._connection.execute(
                f"DELETE FROM personal_facts WHERE id IN ({placeholders})",
                ids,
            )
        return len(ids)

    def bind_reference_candidate(self, reference: str, person_id: str) -> None:
        with self._transaction():
            self._connection.execute(
                """INSERT INTO speaker_reference_candidates(ref, person_id, created_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(ref, person_id) DO NOTHING""",
                (reference, person_id, self.clock()),
            )

    def reference_allows_candidate(self, reference: str, person_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM speaker_reference_candidates WHERE ref = ? AND person_id = ?",
                (reference, person_id),
            ).fetchone()
        return row is not None

    def resolve_reference_candidates(self, reference: str) -> list[PersonCandidate]:
        """Return candidates explicitly attached to a short-lived reference."""
        with self._lock:
            rows = self._connection.execute(
                """SELECT p.id AS person_id, p.display_name
                   FROM speaker_reference_candidates AS c
                   JOIN persons AS p ON p.id = c.person_id
                   WHERE c.ref = ?
                   ORDER BY c.created_at, c.person_id""",
                (reference,),
            ).fetchall()
        return [PersonCandidate(person_id=row["person_id"], name=row["display_name"], evidence_score=0.0) for row in rows]

    def prune_expired(self) -> dict[str, int]:
        now = self.clock()
        cutoff = now - self.observation_retention_days * 86400
        with self._transaction():
            references = self._connection.execute(
                "DELETE FROM speaker_references WHERE expires_at <= ?",
                (now,),
            ).rowcount
            self._connection.execute(
                """UPDATE voice_person_evidence
                   SET observation_id = NULL
                   WHERE observation_id IN (
                       SELECT o.id
                       FROM speaker_observations AS o
                       WHERE o.created_at < ?
                         AND NOT EXISTS (
                             SELECT 1 FROM speaker_references AS r
                             WHERE r.observation_id = o.id
                         )
                   )""",
                (cutoff,),
            )
            observations = self._connection.execute(
                """DELETE FROM speaker_observations
                   WHERE created_at < ?
                     AND NOT EXISTS (
                         SELECT 1 FROM speaker_references AS r
                         WHERE r.observation_id = speaker_observations.id
                     )""",
                (cutoff,),
            ).rowcount
        return {"references": max(references, 0), "observations": max(observations, 0)}

    def close(self) -> None:
        with self._lock:
            self._connection.close()
