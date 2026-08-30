# HuggingVoice + OmniRoute RU/EN Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and verify a reproducible, loopback-only HuggingVoice profile that sends LLM traffic through local OmniRoute and defaults to Russian while retaining English fallback.

**Architecture:** Keep the pinned upstream source unchanged and add a small deployment overlay: one JSON runtime profile, two tested shell entrypoints, and an operator guide. Use local Parakeet TDT and Qwen3-TTS on CPU for the first safe profile, with streaming Chat Completions sent to OmniRoute `auto/chat`.

**Tech Stack:** Python 3.12, uv, pytest, Bash, HuggingVoice `speech-to-speech`, OpenAI-compatible Chat Completions, OmniRoute.

**Spec:** `docs/superpowers/specs/2026-08-30-omniroute-ru-en-bootstrap-design.md`

## Global Constraints

- Base all changes on upstream commit `3986f453012a131632eee4731995474046846794`.
- Keep both HuggingVoice and OmniRoute listeners on loopback addresses.
- Store no API token, provider credential, or `.env` secret in Git.
- Russian is the default response language; clearly English turns receive English responses.
- Do not claim fully local LLM inference while `auto/chat` routes to `openrouter/free`.
- Do not claim a voice-path smoke while NVIDIA or desktop audio is unavailable.

---

### Task 1: Establish a reproducible Python baseline

**Files:**
- Generate: `uv.lock`
- Generate (ignored): `.venv/`

**Interfaces:**
- Consumes: upstream `pyproject.toml` at the pinned commit and the installed `uv` executable.
- Produces: `.venv/bin/python`, `.venv/bin/pytest`, and `.venv/bin/speech-to-speech` for later tasks.

- [ ] **Step 1: Confirm isolation and source identity**

Run:

```bash
git branch --show-current
git merge-base --is-ancestor 3986f453012a131632eee4731995474046846794 HEAD
git status --short
```

Expected: branch `local/huggingvoice-bootstrap`, ancestor check exits 0, and only the committed design/plan state is present.

- [ ] **Step 2: Resolve and install the upstream environment**

Run:

```bash
uv sync --python 3.12
```

Expected: `.venv` uses CPython 3.12, `uv.lock` is generated, and the editable `speech-to-speech` package is installed.

- [ ] **Step 3: Download upstream NLTK runtime data explicitly**

Run:

```bash
uv run python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

Expected: both resources report installed or already up to date.

- [ ] **Step 4: Verify the upstream CLI and focused baseline tests**

Run:

```bash
uv run speech-to-speech --help
uv run pytest tests/test_cli_defaults.py tests/test_language_prompt.py -q
```

Expected: CLI help exits 0 and the existing focused tests pass.

- [ ] **Step 5: Commit the lock file**

```bash
git add uv.lock
git commit -m "build: lock local bootstrap dependencies"
```

### Task 2: Add the RU/EN OmniRoute runtime profile with TDD

**Files:**
- Create: `tests/test_local_bootstrap_config.py`
- Create: `config/omniroute-ru-en.json`

**Interfaces:**
- Consumes: `speech_to_speech.s2s_pipeline.parse_arguments` and the upstream backend registry.
- Produces: a JSON profile accepted directly by `speech-to-speech serve <profile>`.

- [ ] **Step 1: Write the failing parser-level profile tests**

Create `tests/test_local_bootstrap_config.py` with tests that call the real parser and assert:

```python
from pathlib import Path

from speech_to_speech.s2s_pipeline import parse_arguments


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "omniroute-ru-en.json"


def parsed_profile():
    return parse_arguments([str(PROFILE)], command="serve")


def test_profile_routes_chat_completions_to_loopback_omniroute():
    parsed = parsed_profile()
    assert parsed.module_kwargs.llm_backend == "chat-completions"
    assert parsed.llm_backend.config["model_name"] == "auto/chat"
    assert parsed.llm_backend.config["base_url"] == "http://127.0.0.1:20128/v1"
    assert parsed.llm_backend.config["api_key"] is None
    assert parsed.realtime_server_kwargs.host == "127.0.0.1"
    assert parsed.realtime_server_kwargs.port == 8765


def test_profile_uses_local_ru_en_speech_backends():
    parsed = parsed_profile()
    assert parsed.module_kwargs.stt == "parakeet-tdt"
    assert parsed.stt_backend.config["device"] == "cpu"
    assert parsed.stt_backend.config["compute_type"] == "float32"
    assert parsed.stt_backend.config["language"] == "auto"
    assert parsed.llm_backend.config["enable_lang_prompt"] is True
    assert parsed.module_kwargs.tts == "qwen3"
    assert parsed.tts_backend.config["device"] == "cpu"
    assert parsed.tts_backend.config["backend"] == "ggml"
    assert parsed.tts_backend.config["language"] == "auto"


def test_profile_prompt_prefers_russian_without_forcing_english_terms():
    prompt = parsed_profile().llm_backend.config["init_chat_prompt"]
    assert "По умолчанию отвечай по-русски" in prompt
    assert "явно говорит по-английски" in prompt
    assert "термины" in prompt
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_local_bootstrap_config.py -q
```

Expected: FAIL with `FileNotFoundError` for `config/omniroute-ru-en.json`.

- [ ] **Step 3: Add the minimal JSON profile**

Create `config/omniroute-ru-en.json` with these effective values:

```json
{
  "stt": "parakeet-tdt",
  "llm_backend": "chat-completions",
  "tts": "qwen3",
  "host": "127.0.0.1",
  "port": 8765,
  "parakeet_tdt_device": "cpu",
  "parakeet_tdt_compute_type": "float32",
  "parakeet_tdt_language": "auto",
  "model_name": "auto/chat",
  "responses_api_base_url": "http://127.0.0.1:20128/v1",
  "responses_api_stream": true,
  "responses_api_disable_thinking": true,
  "responses_api_reasoning_effort": null,
  "init_chat_role": "system",
  "init_chat_prompt": "Ты голосовой помощник. По умолчанию отвечай по-русски. Отвечай по-английски, когда пользователь явно говорит по-английски. Не переключайся на английский только из-за кода, названий продуктов или английских технических терминов. Отвечай естественно, кратко и пригодно для озвучивания.",
  "enable_lang_prompt": true,
  "compact_history": false,
  "stream_batch_sentences": 1,
  "qwen3_tts_device": "cpu",
  "qwen3_tts_backend": "ggml",
  "qwen3_tts_ggml_quantization": "Q4_K_M",
  "qwen3_tts_language": "auto"
}
```

- [ ] **Step 4: Verify GREEN and the existing CLI tests**

Run:

```bash
uv run pytest tests/test_local_bootstrap_config.py tests/test_cli_defaults.py tests/test_language_prompt.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the profile contract**

```bash
git add config/omniroute-ru-en.json tests/test_local_bootstrap_config.py
git commit -m "feat: add Russian-first OmniRoute profile"
```

### Task 3: Add tested bootstrap and launch entrypoints

**Files:**
- Modify: `tests/test_local_bootstrap_config.py`
- Create: `scripts/bootstrap-local.sh`
- Create: `scripts/run-omniroute-ru-en.sh`

**Interfaces:**
- Consumes: `UV_BIN` and `HUGGINGVOICE_BIN` optional executable overrides.
- Produces: an idempotent environment bootstrap and an exact loopback profile launcher.

- [ ] **Step 1: Add failing executable-behavior tests**

Extend `tests/test_local_bootstrap_config.py` with subprocess tests that:

1. Set `UV_BIN` to a temporary executable which records each argv; run
   `scripts/bootstrap-local.sh`; expect calls `sync --python 3.12` and
   `run python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng`.
2. Set `HUGGINGVOICE_BIN` to a temporary executable which prints argv; run
   `scripts/run-omniroute-ru-en.sh`; expect exactly `serve` and the absolute
   profile path.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_local_bootstrap_config.py -q
```

Expected: FAIL because both scripts are absent.

- [ ] **Step 3: Implement the minimal scripts**

`scripts/bootstrap-local.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
uv_bin="${UV_BIN:-uv}"
cd "$repo_root"
"$uv_bin" sync --python 3.12
"$uv_bin" run python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

`scripts/run-omniroute-ru-en.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
speech_to_speech_bin="${HUGGINGVOICE_BIN:-${repo_root}/.venv/bin/speech-to-speech}"
exec "$speech_to_speech_bin" serve "${repo_root}/config/omniroute-ru-en.json"
```

Make both files executable.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_local_bootstrap_config.py -q
bash -n scripts/bootstrap-local.sh scripts/run-omniroute-ru-en.sh
```

Expected: tests and shell syntax checks pass.

- [ ] **Step 5: Commit the entrypoints**

```bash
git add scripts/bootstrap-local.sh scripts/run-omniroute-ru-en.sh tests/test_local_bootstrap_config.py
git commit -m "feat: add local bootstrap launchers"
```

### Task 4: Document and smoke-test the deployment

**Files:**
- Create: `docs/local-omniroute-ru-en.md`

**Interfaces:**
- Consumes: the committed bootstrap/profile/launcher and live OmniRoute.
- Produces: exact operator commands plus verified limitations and recovery guidance.

- [ ] **Step 1: Verify OmniRoute health and both language policies**

Send two non-streaming `/v1/chat/completions` requests to `auto/chat`: one Russian
turn and one clearly English turn. Assert that both responses exist; manually
confirm their language and record the routed model names without recording any
credential.

- [ ] **Step 2: Run focused and upstream test gates**

Run:

```bash
uv run pytest tests/test_local_bootstrap_config.py tests/test_cli_defaults.py tests/test_language_prompt.py -q
uv run pytest -q
```

Expected: the focused suite and full upstream suite pass, or any pre-existing
failure is preserved with its exact test name and error.

- [ ] **Step 3: Run a bounded server startup smoke**

Run the launcher with a bounded timeout while capturing output. Success requires
the process to reach the HuggingVoice startup path and bind loopback; model
download, CUDA, or audio failures must be classified accurately. Do not claim an
interactive voice smoke without a visible desktop audio session.

- [ ] **Step 4: Write the operator guide**

Document:

- `./scripts/bootstrap-local.sh`
- OmniRoute health check at `/api/monitoring/health`
- `./scripts/run-omniroute-ru-en.sh`
- `uv run speech-to-speech talk --url ws://127.0.0.1:8765/v1/realtime`
- expected model downloads and cache location
- the current external `auto/chat` route
- NVIDIA and desktop-audio blockers
- how to switch to a verified offline OmniRoute route later by changing only
  `model_name`

- [ ] **Step 5: Verify docs and commit**

Run:

```bash
git diff --check
git status --short
```

Then:

```bash
git add docs/local-omniroute-ru-en.md
git commit -m "docs: add local HuggingVoice runbook"
```

### Task 5: Final verification

**Files:**
- Verify all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: committed local branch state.
- Produces: evidence-backed handoff with exact successes and blockers.

- [ ] **Step 1: Run final gates from a clean shell**

```bash
./scripts/bootstrap-local.sh
uv run pytest tests/test_local_bootstrap_config.py tests/test_cli_defaults.py tests/test_language_prompt.py -q
bash -n scripts/bootstrap-local.sh scripts/run-omniroute-ru-en.sh
git diff --check
git status --short --branch
```

- [ ] **Step 2: Inspect commit history and source pin**

```bash
git log --oneline --decorate -5
git merge-base --is-ancestor 3986f453012a131632eee4731995474046846794 HEAD
```

Expected: all overlay commits sit on the pinned upstream commit.

- [ ] **Step 3: Hand off exact runtime status**

Report separately: source/environment installed, configuration tests, OmniRoute
bilingual responses, server startup/model readiness, NVIDIA status, and desktop
audio status. Never collapse partial readiness into an end-to-end success claim.
