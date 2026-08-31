# Native Speaker Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in CPU speaker-memory stage that assigns stable voice IDs, maintains evidence-based people and personal memory, and exposes native Realtime tools plus an MCP adapter.

**Architecture:** A sherpa-onnx extractor annotates final `VADAudio` objects before STT. A pure tracker and SQLite store own clustering, identity evidence, references, and facts. Realtime keeps standard transcript events unchanged while adding trusted identity context to the internal LLM message. Native tools and MCP call the same domain service.

**Tech Stack:** Python 3.10+, Pydantic 2, NumPy, SQLite/FTS5, sherpa-onnx CPU, existing OpenAI Realtime tool lifecycle, MCP Python SDK v2, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-speaker-memory-design.md`

## Global Constraints

- Speaker inference is CPU-only with `num_threads=1` by default and must not allocate VRAM.
- The feature is opt-in upstream and cannot change behavior when disabled.
- Raw audio, transcripts, embeddings, names, and facts must not appear in normal logs.
- Standard OpenAI Realtime transcript models and payload text remain unchanged.
- Tools accept expiring `speaker_ref` values; model-supplied database IDs are not mutation authority.
- Personal recall requires a confirmed identity and fails closed for unknown, ambiguous, or conflict states.
- No changes to `/home/random/dev/qwen/freetoken` or any FreeToken live service.
- Run focused tests only during development; do not run the 1400+ upstream suite unless a concrete failure demands it.

---

### Task 1: Typed identity records and transactional SQLite store

**Files:**
- Create: `src/speech_to_speech/speaker_memory/__init__.py`
- Create: `src/speech_to_speech/speaker_memory/models.py`
- Create: `src/speech_to_speech/speaker_memory/store.py`
- Test: `tests/speaker_memory/test_store.py`

**Interfaces:**
- Produces: `SpeakerState`, `SpeakerAttribution`, `VoiceCluster`, `PersonCandidate`, `SpeakerMemoryStore`.
- Produces store methods: `create_observation`, `create_voice_cluster`, `get_voice_clusters`, `add_identity_evidence`, `resolve_person_candidates`, `issue_reference`, `resolve_reference`, and `forget`.

- [ ] **Step 1: Write failing store tests**

```python
def test_reference_resolves_only_before_expiry(tmp_path):
    store = SpeakerMemoryStore(tmp_path / "memory.sqlite3", clock=lambda: 100.0)
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    observation = store.create_observation(voice.id, "turn_1", 0, quality=0.9)
    ref = store.issue_reference(observation.id, conversation_id="conv_1", ttl_s=30)
    assert store.resolve_reference(ref, conversation_id="conv_1").voice_id == voice.id
    store.clock = lambda: 131.0
    with pytest.raises(ExpiredSpeakerReference):
        store.resolve_reference(ref, conversation_id="conv_1")
```

Also test WAL/foreign keys, vector dimension checks, immutable evidence rows, short random IDs, conversation mismatch, and that ordinary log capture contains no names or embeddings.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_store.py`

Expected: collection fails because `speech_to_speech.speaker_memory` does not exist.

- [ ] **Step 3: Implement models and schema**

```python
class SpeakerState(StrEnum):
    KNOWN = "known"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    MIXED = "mixed"

class SpeakerAttribution(BaseModel):
    voice_id: str | None = None
    speaker_ref: str | None = None
    state: SpeakerState = SpeakerState.UNKNOWN
    candidate: PersonCandidate | None = None
    margin: float | None = None
    recommendation: Literal["none", "clarify", "do_not_learn"] = "none"
    speaker_ms: float = 0.0
```

Use explicit migrations in a `schema_meta` table, `sqlite3.connect(..., timeout=2.0)`, `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, and float32 BLOB encoding with dimension metadata.

- [ ] **Step 4: Verify GREEN and lint**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_store.py && /home/random/dev/huggingvoice/.venv/bin/ruff check src/speech_to_speech/speaker_memory tests/speaker_memory/test_store.py`

Expected: all store tests pass and Ruff exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech/speaker_memory tests/speaker_memory/test_store.py
git commit -m "feat: add transactional speaker identity store"
```

### Task 2: Online voice clustering and identity evidence scoring

**Files:**
- Create: `src/speech_to_speech/speaker_memory/tracker.py`
- Test: `tests/speaker_memory/test_tracker.py`

**Interfaces:**
- Consumes: `SpeakerMemoryStore`, normalized `numpy.ndarray` embeddings.
- Produces: `SpeakerTracker.observe(embedding, *, quality, turn_id, turn_revision, conversation_id) -> SpeakerAttribution`.
- Produces: `SpeakerMemoryService.confirm`, `reject`, `remember_name`, and `inspect` semantic operations with bounded evidence weights.

- [ ] **Step 1: Write failing clustering tests**

```python
def test_ambiguous_observation_does_not_move_a_centroid(store):
    first = store.create_voice_cluster(unit([1.0, 0.0]), quality=1.0)
    store.create_voice_cluster(unit([0.98, 0.2]), quality=1.0)
    before = first.centroid.copy()
    result = tracker(store, ambiguity_margin=0.08).observe(
        unit([0.995, 0.1]), quality=1.0, turn_id="turn_3", turn_revision=0, conversation_id="conv"
    )
    assert result.state is SpeakerState.AMBIGUOUS
    assert np.array_equal(store.get_voice_cluster(first.id).centroid, before)
```

Cover new-cluster creation, high-confidence centroid updates, low quality bypass, decisive person, competing person conflict, confirm/reject bounds, and self-introduction creating a person.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_tracker.py`

Expected: fails because `SpeakerTracker` is missing.

- [ ] **Step 3: Implement tracker and service**

```python
similarities = [(cluster, float(np.dot(embedding, cluster.centroid))) for cluster in clusters]
similarities.sort(key=lambda item: item[1], reverse=True)
top_score = similarities[0][1] if similarities else -1.0
runner_up = similarities[1][1] if len(similarities) > 1 else -1.0
margin = top_score - runner_up
```

Normalize inputs, reject zero/non-finite vectors, update only decisive matches, and map semantic operations to fixed evidence weights owned by the service.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_store.py tests/speaker_memory/test_tracker.py`

Expected: all identity tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech/speaker_memory/tracker.py tests/speaker_memory/test_tracker.py
git commit -m "feat: track adaptive voice and person evidence"
```

### Task 3: CPU sherpa-onnx embedding adapter and pass-through handler

**Files:**
- Create: `src/speech_to_speech/speaker_memory/extractor.py`
- Create: `src/speech_to_speech/speaker_memory/handler.py`
- Create: `src/speech_to_speech/arguments_classes/speaker_memory_arguments.py`
- Modify: `pyproject.toml`
- Test: `tests/speaker_memory/test_extractor.py`
- Test: `tests/speaker_memory/test_handler.py`

**Interfaces:**
- Produces: `SpeakerEmbeddingExtractor.extract(audio, sample_rate) -> np.ndarray` protocol.
- Produces: `SherpaOnnxSpeakerEmbeddingExtractor(model_path, num_threads=1)`.
- Produces: `SpeakerMemoryHandler`, consuming and yielding `VADAudio`.

- [ ] **Step 1: Write failing extractor and handler tests**

```python
def test_final_audio_is_attributed_but_progressive_audio_is_not(fake_extractor, tracker):
    handler = bare_handler(fake_extractor, tracker, min_audio_ms=700)
    progressive = VADAudio(audio=np.ones(16000), mode="progressive", turn_id="t", turn_revision=0)
    final = progressive.model_copy(update={"mode": "final"})
    assert list(handler.process(progressive))[0].speaker is None
    attributed = list(handler.process(final))[0]
    assert attributed.speaker is not None
    assert fake_extractor.calls == 1
```

Fake the `sherpa_onnx` module to assert CPU provider, one thread, ready checks, float32 input, normalization, and actionable missing-extra errors.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_extractor.py tests/speaker_memory/test_handler.py`

Expected: fails because extractor and handler modules are missing.

- [ ] **Step 3: Implement optional adapter and handler**

```python
config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=str(model_path), num_threads=num_threads, debug=False, provider="cpu"
)
stream = extractor.create_stream()
stream.accept_waveform(sample_rate=sample_rate, waveform=np.asarray(audio, dtype=np.float32))
stream.input_finished()
if not extractor.is_ready(stream):
    raise AudioTooShortForSpeakerEmbedding()
```

Add optional dependency `speaker-memory = ["sherpa-onnx>=1.13.6,<2"]`. Handler failures must pass the original `VADAudio` through without identity mutation.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_extractor.py tests/speaker_memory/test_handler.py`

Expected: all tests pass without installing sherpa-onnx.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/speech_to_speech/speaker_memory src/speech_to_speech/arguments_classes/speaker_memory_arguments.py tests/speaker_memory
git commit -m "feat: add CPU speaker embedding handler"
```

### Task 4: Carry speaker metadata through STT and pipeline construction

**Files:**
- Modify: `src/speech_to_speech/pipeline/messages.py`
- Modify: `src/speech_to_speech/pipeline/events.py`
- Modify: `src/speech_to_speech/pipeline/handler_types.py`
- Modify: `src/speech_to_speech/pipeline/queue_types.py`
- Modify: `src/speech_to_speech/STT/gigaam_onnx_handler.py`
- Modify: other STT handlers that construct final `Transcription`
- Modify: `src/speech_to_speech/STT/transcription_notifier.py`
- Modify: `src/speech_to_speech/s2s_pipeline.py`
- Test: `tests/speaker_memory/test_pipeline_metadata.py`
- Test: `tests/test_gigaam_onnx_stt.py`

**Interfaces:**
- `VADAudio.speaker: SpeakerAttribution | None`.
- `Transcription.speaker: SpeakerAttribution | None`.
- `TranscriptionCompletedEvent.speaker: SpeakerAttribution | None`, excluded from standard wire serialization.
- `ParsedArguments.speaker_memory_kwargs: SpeakerMemoryArguments`.

- [ ] **Step 1: Write failing metadata propagation tests**

```python
def test_gigaam_preserves_speaker_attribution(fake_gigaam):
    speaker = SpeakerAttribution(voice_id="v_1", speaker_ref="sr_1", state="known")
    output = list(fake_gigaam.process(VADAudio(audio=np.ones(1600), mode="final", speaker=speaker)))[0]
    assert output.speaker == speaker
```

Assert `TranscriptionNotifier` keeps the metadata internally and the OpenAI completed event still serializes only its standard fields.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_pipeline_metadata.py tests/test_gigaam_onnx_stt.py`

Expected: model construction rejects the unknown `speaker` field or output loses it.

- [ ] **Step 3: Implement typed propagation and handler chain**

Construct queues as `VAD -> speaker_queue -> STT` only when enabled; otherwise retain the exact old queue path. Copy speaker attribution in every STT final/partial output constructor where applicable.

- [ ] **Step 4: Verify GREEN and disabled compatibility**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_pipeline_metadata.py tests/test_gigaam_onnx_stt.py tests/test_whisper_progressive_transcription.py tests/test_paraformer_transcription_events.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech tests/speaker_memory/test_pipeline_metadata.py tests/test_gigaam_onnx_stt.py
git commit -m "feat: propagate speaker identity through speech turns"
```

### Task 5: Trusted LLM context and native identity tools

**Files:**
- Create: `src/speech_to_speech/speaker_memory/context.py`
- Create: `src/speech_to_speech/speaker_memory/tools.py`
- Create: `src/speech_to_speech/speaker_memory/service.py`
- Modify: `src/speech_to_speech/api/openai_realtime/service.py`
- Modify: `src/speech_to_speech/api/openai_realtime/audio_client.py`
- Test: `tests/speaker_memory/test_context.py`
- Test: `tests/speaker_memory/test_tools.py`

**Interfaces:**
- Produces: `format_speaker_context(attribution) -> str` with JSON escaping and bounded names.
- Produces importable tool module `TOOLS`, `CREATE_RESPONSE`, and `async execute_tool(name, arguments)`.
- Native service operations accept a `speaker_ref`; no mutation accepts `voice_id` directly.

- [ ] **Step 1: Write failing context and tool tests**

```python
async def test_rejection_tool_adds_negative_evidence_without_followup(memory_service, issued_ref):
    configure_tool_service(memory_service)
    result = await execute_tool(
        "speaker_memory_reject", {"speaker_ref": issued_ref, "person_id": "p_arkady"}
    )
    assert result.create_response is False
    assert memory_service.inspect(issued_ref).candidate is None
```

Test malicious names are JSON-escaped, standard transcript output stays raw, expired references fail closed, and the real `_ToolCoordinator` delivers outputs in order.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_context.py tests/speaker_memory/test_tools.py`

Expected: imports fail because context and tools are missing.

- [ ] **Step 3: Implement context and tool module**

```python
TOOLS = [{
    "type": "function",
    "name": "speaker_memory_remember_name",
    "description": "Remember the name explicitly given by the current speaker.",
    "parameters": {
        "type": "object",
        "properties": {
            "speaker_ref": {"type": "string"},
            "name": {"type": "string", "minLength": 1, "maxLength": 80},
        },
        "required": ["speaker_ref", "name"],
        "additionalProperties": False,
    },
}]
```

Add trusted metadata only to `Chat.add_item(make_user_message(...))`; preserve `event.transcript` for client events. Expose configuration that lets the packaged local client load this module through the existing loader.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_context.py tests/speaker_memory/test_tools.py tests/openai_realtime/test_audio_client.py -k 'tool or speaker'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech tests/speaker_memory
git commit -m "feat: expose speaker memory to realtime agents"
```

### Task 6: Personal facts, privacy gate, and policy skill

**Files:**
- Modify: `src/speech_to_speech/speaker_memory/store.py`
- Modify: `src/speech_to_speech/speaker_memory/service.py`
- Modify: `src/speech_to_speech/speaker_memory/tools.py`
- Create: `src/speech_to_speech/speaker_memory/policy.py`
- Create: `skills/speaker-memory/SKILL.md`
- Test: `tests/speaker_memory/test_personal_memory.py`

**Interfaces:**
- Produces: `remember_fact(speaker_ref, fact, topic=None)` and `recall(speaker_ref, query, limit)`.
- Produces: `forget(speaker_ref, scope, fact_id=None)`.
- Produces: `SPEAKER_MEMORY_POLICY` for native prompt composition.

- [ ] **Step 1: Write failing privacy and retrieval tests**

```python
def test_ambiguous_voice_cannot_recall_private_facts(service, ambiguous_ref):
    with pytest.raises(IdentityNotConfirmed):
        service.recall(ambiguous_ref, query="любимый напиток", limit=5)
```

Test FTS ranking, per-person isolation, explicit forgetting, no fact text in logs, and that the skill contains every tool name and conflict behavior.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_personal_memory.py`

Expected: fails because personal-memory methods are absent.

- [ ] **Step 3: Implement FTS-backed facts and shared policy**

Use an FTS5 external-content table when available and a deterministic escaped `LIKE` fallback when SQLite lacks FTS5. Return fact IDs and excerpts, never raw embeddings or evidence rows.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_personal_memory.py tests/speaker_memory/test_tools.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech/speaker_memory skills/speaker-memory tests/speaker_memory
git commit -m "feat: add privacy-gated personal speaker memory"
```

### Task 7: Thin MCP v2 adapter

**Files:**
- Create: `src/speech_to_speech/speaker_memory/mcp_server.py`
- Modify: `pyproject.toml`
- Test: `tests/speaker_memory/test_mcp_server.py`
- Modify: `skills/speaker-memory/SKILL.md`

**Interfaces:**
- Produces: `build_mcp_server(service: SpeakerMemoryService) -> MCPServer`.
- Produces console entry point `huggingvoice-speaker-memory-mcp` using stdio by default.
- Consumes only public `SpeakerMemoryService` methods.

- [ ] **Step 1: Write failing MCP registration test**

```python
def test_mcp_adapter_registers_the_same_semantic_tools(service):
    server = build_mcp_server(service)
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {"speaker_memory_inspect", "speaker_memory_confirm", "speaker_memory_reject"} <= names
```

- [ ] **Step 2: Verify RED with the optional SDK**

Run: `uv run --with 'mcp>=2,<3' --with-editable . pytest -q tests/speaker_memory/test_mcp_server.py`

Expected: fails because `mcp_server.py` is missing.

- [ ] **Step 3: Implement MCPServer adapter**

```python
from mcp.server import MCPServer

def build_mcp_server(service: SpeakerMemoryService) -> MCPServer:
    server = MCPServer("HuggingVoice Speaker Memory")

    @server.tool()
    def speaker_memory_inspect(speaker_ref: str) -> dict[str, object]:
        return service.inspect(speaker_ref).model_dump(mode="json")

    return server
```

Keep imports lazy so ordinary installs without the MCP extra can import the speaker-memory package.

- [ ] **Step 4: Verify GREEN**

Run: `uv run --with 'mcp>=2,<3' --with-editable . pytest -q tests/speaker_memory/test_mcp_server.py`

Expected: all MCP adapter tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/speech_to_speech/speaker_memory/mcp_server.py tests/speaker_memory/test_mcp_server.py skills/speaker-memory/SKILL.md
git commit -m "feat: expose speaker memory over MCP"
```

### Task 8: Local profile, model acquisition, focused integration and acoustic proof

**Files:**
- Create: `scripts/fetch-speaker-memory-model.sh`
- Modify: `config/omniroute-ru-en.json`
- Modify: `docs/local-omniroute-ru-en.md`
- Modify: `scripts/run-omniroute-ru-en.sh` only if an environment path is required
- Create: `tests/speaker_memory/test_local_profile.py`
- Create: `scripts/speaker_memory_smoke.py`

**Interfaces:**
- Fetches `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` and verifies SHA-256 `357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b`.
- Produces a JSON smoke report with per-turn `voice_id`, state, candidate, `speaker_ms`, and expected same/different-person decisions.

- [ ] **Step 1: Write failing local-profile tests**

```python
def test_russian_profile_enables_cpu_speaker_memory(profile):
    assert profile["speaker_memory_enabled"] is True
    assert profile["speaker_memory_threads"] == 1
    assert profile["speaker_memory_model_path"].endswith("3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx")
```

Test the fetch script URL/checksum statically and shell syntax with `bash -n`.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory/test_local_profile.py`

Expected: fails because the profile has no speaker-memory keys.

- [ ] **Step 3: Implement profile, fetch script, smoke, and docs**

The fetch script downloads to a temporary file, validates SHA-256, then atomically renames into `models/speaker-memory/`. It must not overwrite an existing verified model and must fail without replacing an existing file on checksum mismatch.

- [ ] **Step 4: Run focused source verification**

Run:

```bash
PYTHONPATH=$PWD/src /home/random/dev/huggingvoice/.venv/bin/python -m pytest -q tests/speaker_memory tests/test_gigaam_onnx_stt.py tests/test_chat_completions_backend.py
/home/random/dev/huggingvoice/.venv/bin/ruff check src/speech_to_speech/speaker_memory tests/speaker_memory
bash -n scripts/fetch-speaker-memory-model.sh scripts/run-omniroute-ru-en.sh
git diff --check
```

Expected: all focused tests and static checks pass.

- [ ] **Step 5: Run real CPU and two-person Russian smoke**

Run the fetch script, install the optional dependency in the isolated environment, and execute `scripts/speaker_memory_smoke.py` on at least two enrollment/test recordings per person. Acceptance:

- same-person utterances converge to one voice cluster after enrollment;
- different-person utterances do not update that centroid;
- a confirmation raises the intended person relation;
- a rejection lowers it;
- ambiguous observations do not move any centroid;
- warm `speaker_ms` p95 is reported and no higher than 100 ms, or the feature remains disabled in the live profile with the measured reason documented.

- [ ] **Step 6: Commit**

```bash
git add config docs scripts tests/speaker_memory
git commit -m "feat: enable speaker memory in Russian voice profile"
```

- [ ] **Step 7: Review, push, and activation gate**

Run the requesting-code-review and verification-before-completion workflows. Push `feature/speaker-memory` normally. Do not merge and do not restart `huggingvoice.service` until the user explicitly authorizes activation of the completed branch.
