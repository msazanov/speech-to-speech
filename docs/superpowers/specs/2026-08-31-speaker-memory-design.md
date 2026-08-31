# Native Speaker Memory Design

## Objective

Add a CPU-first, native speaker-memory layer to HuggingVoice. Every usable final
VAD segment receives a stable short `voice_id`; observations update or create
voice clusters without renaming those IDs. Voice clusters accumulate bounded
positive and negative evidence linking them to persistent `person_id` records.
The realtime agent sees a compact identity state and can call standard
HuggingVoice function tools to remember, confirm, reject, inspect, or forget an
identity and to store or retrieve personal facts.

The implementation must stay independent of the LLM provider, keep the current
OpenAI Realtime transcript payload unchanged, run speaker inference on CPU, and
avoid a new always-on service in the realtime critical path.

## Decisions

### Native pipeline stage

`SpeakerMemoryHandler` is a normal `BaseHandler` inserted between VAD and STT:

```text
VAD -> SpeakerMemoryHandler -> STT -> TranscriptionNotifier -> RealtimeService -> LLM -> TTS
```

The stage is disabled by default upstream and enabled in the local Russian-first
profile. Progressive VAD revisions pass through unchanged. Only final segments
that meet duration and signal-quality gates produce an embedding.

The initial implementation is deliberately sequential. Its latency is measured
as `speaker_ms`; parallel STT/embedding execution is considered only if the
measured warm p95 exceeds 100 ms on the target CPU.

### Embeddings and online clustering

The production extractor uses `sherpa-onnx` with an explicit local ONNX model,
`provider="cpu"`, and one inference thread. The initial candidate model is
`3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` (about 30 MB). The adapter
normalizes every embedding before returning it.

The tracker compares one observation to stored L2-normalized centroids with
cosine similarity:

- `known`: one cluster clears the match threshold and the runner-up margin;
- `ambiguous`: at least one plausible cluster exists but the margin is too small;
- `unknown`: no plausible cluster exists; a new cluster is created only when the
  observation passes quality gates;
- `conflict`: identity evidence for the selected voice has no decisive person;
- `mixed`: reserved for an explicit overlap detector; it is never inferred from
  embedding ambiguity alone.

Only high-confidence, non-mixed matches update a centroid. Updates use a
quality-weighted bounded running mean followed by L2 normalization. Ambiguous,
short, or low-quality observations cannot move an existing cluster, preventing
identity drift.

Thresholds are configuration, not hard-coded model claims. They are calibrated
with Russian speech from at least two people before the local profile is enabled.

### Three identity levels

1. `SpeakerObservation`: one final utterance, its embedding, quality, timestamps,
   turn identity, attribution result, and a short tool reference.
2. `VoiceCluster`: immutable `voice_id`, current centroid, sample count, quality
   weight, creation time, and last-seen time.
3. `Person`: immutable `person_id`, mutable display name, and optional aliases.

One person may have several voice clusters (different microphones, illness, or
acoustic conditions). A voice cluster may temporarily have evidence for several
people. Hard one-to-one assignments are not stored.

### Evidence graph

`VoicePersonEvidence` records immutable events instead of overwriting one score:

- `self_introduction`: strong positive evidence;
- `agent_confirmation`: strong positive evidence;
- `agent_rejection`: strong negative evidence;
- `manual_link` and `manual_unlink`: operator-strength evidence;
- `passive_match`: weak positive evidence and never sufficient to reveal private
  memory by itself.

The store aggregates bounded weights per `(voice_id, person_id)`. A decisive top
score and sufficient margin yields `known`; competing scores yield `conflict`.
Tool calls use semantic operations and cannot submit arbitrary numeric weights.

### Tool-safe speaker references

The model never supplies raw database IDs as authority. Each attributed final
turn gets a short random `speaker_ref` mapped in SQLite to the observation,
voice cluster, conversation, and expiry. Native tools and MCP tools require this
reference. An expired, unknown, or conversation-mismatched reference fails
without mutating data.

This makes a copied reference useful for the current conversational decision but
prevents hallucinated identifiers from modifying unrelated profiles.

### Agent indicator and conversational policy

Trusted speaker metadata is added only to the internal LLM user message; the
standard `conversation.item.input_audio_transcription.completed.transcript`
remains the raw STT text. The metadata contains:

```json
{
  "speaker_ref": "sr_...",
  "voice_id": "v_...",
  "state": "ambiguous",
  "candidate": {"person_id": "p_...", "name": "Аркадий", "confidence": 0.68},
  "margin": 0.07,
  "recommendation": "clarify"
}
```

The policy tells the agent:

- use known names naturally but do not repeat a name every turn;
- for `ambiguous` or `conflict`, ask one short natural clarification;
- after an affirmative answer, call the confirm tool;
- after a rejection, call reject and optionally remember the corrected name;
- never claim speaker recognition as certainty;
- never retrieve personal facts until identity is confirmed;
- never update identity from `mixed`, low-quality, or expired observations.

### Native tools

The packaged client uses the existing `session.update.tools` and
`function_call_output` lifecycle through an importable tool module. Tools are:

- `speaker_memory_inspect(speaker_ref)`;
- `speaker_memory_remember_name(speaker_ref, name)`;
- `speaker_memory_confirm(speaker_ref, person_id)`;
- `speaker_memory_reject(speaker_ref, person_id)`;
- `speaker_memory_remember_fact(speaker_ref, fact, topic=None)`;
- `speaker_memory_recall(speaker_ref, query, limit=5)`;
- `speaker_memory_forget(speaker_ref, scope, fact_id=None)`.

Mutation tools return `create_response=False`; the agent should speak the natural
acknowledgement before calling them. Inspect and recall return
`create_response=True` so the standard client requests a follow-up response.

Tool arguments are JSON-schema validated by the existing client and validated
again by the domain service. Names are normalized, length-bounded, and treated as
data rather than instructions.

### Personal memory

Personal facts are linked only to `person_id`, never directly to `voice_id`.
SQLite FTS5 provides the first local retrieval implementation; this is sufficient
for a small private memory and adds no embedding model. The repository exposes a
`PersonalMemorySearch` interface so semantic retrieval can replace FTS later
without changing tools.

Recall requires a confirmed identity relation. `unknown`, `ambiguous`, and
`conflict` states receive `identity_not_confirmed` and no facts. Forget operations
are explicit and auditable.

### Persistence and privacy

Default database location follows XDG:
`$XDG_DATA_HOME/huggingvoice/speaker-memory.sqlite3`, falling back to
`~/.local/share/huggingvoice/speaker-memory.sqlite3`.

SQLite uses WAL, foreign keys, bounded busy timeout, and transactions. Raw audio
is never persisted. Embeddings, centroids, names, evidence, and personal facts
are sensitive biometric/personal data and are omitted from logs. The database
path can be overridden for tests and deployments.

Observations and speaker references have configurable retention. Rare clusters
are not deleted in the first release; later cold archiving can be added without
changing IDs.

### MCP adapter and skill

The MCP server is a thin optional adapter over the same `SpeakerMemoryService`.
It contains no scoring or persistence logic. It uses the official Python MCP v2
`MCPServer` API and defaults to stdio. The optional dependency is isolated in a
`speaker-memory-mcp` extra so ordinary HuggingVoice installs remain unchanged.

`skills/speaker-memory/SKILL.md` documents when an external agent should inspect,
clarify, confirm, reject, remember, recall, and forget. The native voice policy
and the skill share the same concise behavioral rules.

### OpenAI compatibility

No non-standard fields are added to standard Realtime SDK models. Speaker data
travels internally on pipeline Pydantic models and in the LLM chat envelope.
Optional clients may later receive a vendor extension event, but that is outside
this implementation.

### Error behavior

- Missing optional speaker dependencies: startup fails with an actionable extra
  install message only when speaker memory is enabled.
- Missing/invalid model path: startup fails before accepting sessions.
- Too-short or low-quality utterance: pass through to STT with no attribution.
- Embedding failure: log exception type without audio or personal data, pass
  through to STT, and keep the session usable.
- SQLite lock beyond the bounded timeout: tool returns a structured retryable
  error; the realtime handler does not loop forever.
- Invalid or expired `speaker_ref`: no mutation and a structured error.

## Configuration

The speaker-memory argument group includes:

- `speaker_memory_enabled` (default `False`);
- `speaker_memory_model_path`;
- `speaker_memory_database_path`;
- `speaker_memory_threads` (default `1`, CPU only);
- `speaker_memory_min_audio_ms` (default `700`);
- `speaker_memory_match_threshold`;
- `speaker_memory_candidate_threshold`;
- `speaker_memory_ambiguity_margin`;
- `speaker_memory_reference_ttl_s`;
- `speaker_memory_observation_retention_days`.

The local Russian-first JSON enables the feature only after a measured model and
threshold calibration are committed.

## Test strategy

- Pure unit tests use deterministic vectors and temporary SQLite files.
- Store tests cover schema migration, atomic evidence, expiry, privacy gates,
  forgetting, and concurrent connections.
- Tracker tests cover known/unknown/ambiguous/conflict states and prove ambiguous
  observations do not move centroids.
- Extractor tests use a fake sherpa module; a separate opt-in smoke uses the real
  ONNX model.
- Pipeline tests prove metadata survives VAD -> STT -> notifier and raw protocol
  transcripts remain unchanged.
- Tool tests exercise the real existing local ToolCoordinator lifecycle.
- MCP tests call the domain functions through a constructed MCP server without a
  network listener.
- A final two-person Russian acoustic test measures identity accuracy and
  `speaker_ms` without running the entire upstream suite.

## Non-goals

- Separating simultaneous overlapping speakers in one mono segment.
- Treating voice identity as authentication.
- Cloud identity storage or cross-machine synchronization.
- Automatically exposing private memories from a merely similar voice.
- Replacing GigaAM, Silero TTS, Smart Turn, OmniRoute, or FreeToken.
