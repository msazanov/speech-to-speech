# HuggingVoice + OmniRoute RU/EN Bootstrap Design

## Goal

Create a reproducible local HuggingVoice deployment skeleton that uses the
already-running OmniRoute gateway for LLM inference, treats Russian as the
default conversation language, and still follows clearly English user turns in
English.

## Pinned inputs

- Upstream repository: `https://github.com/huggingface/speech-to-speech.git`
- Upstream commit: `3986f453012a131632eee4731995474046846794`
- Local checkout: `/home/random/dev/huggingvoice`
- Python: CPython 3.12 managed by `uv`
- OmniRoute base URL: `http://127.0.0.1:20128/v1`
- OmniRoute model route: `auto/chat`
- HuggingVoice LLM backend: `chat-completions`

The Chat Completions backend is selected because the deployed OmniRoute gateway
has been verified with `/v1/chat/completions`, while this bootstrap does not
assume that every OmniRoute route implements the Responses API event protocol.

## Runtime architecture

The initial pipeline remains loopback-only:

1. Silero VAD detects speech boundaries.
2. Parakeet TDT transcribes Russian or English speech locally.
3. HuggingVoice sends the text conversation to OmniRoute at
   `127.0.0.1:20128` using streaming Chat Completions.
4. Qwen3-TTS synthesizes the response locally.
5. The OpenAI Realtime-compatible server listens on
   `127.0.0.1:8765`; it is not exposed to the LAN.

The first reproducible profile is CPU-safe because the host's RTX 2070 Mobile
is present but `nvidia-smi` cannot currently communicate with the driver. GPU
enablement is a later profile change after the CUDA runtime is healthy; it is
not hidden inside the bootstrap.

## Language policy

Parakeet TDT runs with automatic language detection and supports both `ru` and
`en`. `enable_lang_prompt` is enabled so a confidently detected language is
propagated to the LLM. The session prompt applies these rules:

- answer in Russian by default;
- answer in English when the user's current turn is clearly English;
- do not switch to English merely because the Russian turn contains code,
  product names, or English technical terms;
- keep spoken answers concise and natural.

Qwen3-TTS uses automatic target-language selection so the synthesized voice
follows the LLM output language.

## Files

- `config/omniroute-ru-en.json`: non-secret HuggingVoice runtime configuration.
- `scripts/bootstrap-local.sh`: idempotent Python 3.12 environment bootstrap.
- `scripts/run-omniroute-ru-en.sh`: loopback launcher using the committed JSON
  configuration.
- `tests/test_local_bootstrap_config.py`: static and parser-level contract tests
  for endpoint, backend, language policy, loopback binding, and secret hygiene.
- `docs/local-omniroute-ru-en.md`: operator guide, verification commands, model
  download expectations, and known GPU/audio limitations.

## Secrets and network boundaries

No OmniRoute or upstream provider credential is written to this repository.
The loopback OmniRoute endpoint currently accepts local requests without a
credential, and HuggingVoice supplies its harmless local placeholder internally.
If OmniRoute authentication is enabled later, the launcher must read the key
from the process environment or a protected service credential, not from JSON.

The Realtime server and OmniRoute connection both use loopback addresses. LAN
exposure, authentication, TLS termination, and rate limiting are explicitly out
of scope for this bootstrap.

## Current OmniRoute limitation

The gateway health endpoint is healthy and a test request to `auto/chat`
returned a Russian response. That request was routed to `openrouter/free`.
Therefore this design guarantees a locally hosted gateway, not fully offline
LLM inference.

The installed `ornith-q6.service`, which would provide a fully local model, is
currently restart-looping because the NVIDIA runtime is unhealthy. A later
change may switch `model_name` from `auto/chat` to a verified offline route once
that service passes its own health and Russian-language smoke tests.

## Installation and verification

The bootstrap creates a `.venv` with Python 3.12 and runs the upstream `uv sync`
workflow. It does not edit system Python. The verification sequence is:

1. Run the focused bootstrap configuration tests.
2. Confirm `speech-to-speech serve --help` loads from the local environment.
3. Confirm OmniRoute health and a short Russian Chat Completions request.
4. Start the HuggingVoice server with the committed JSON configuration.
5. Confirm the Realtime endpoint binds only to `127.0.0.1:8765`.
6. Run a text-path Russian prompt and an English fallback prompt before any
   microphone test.
7. Run microphone/speaker smoke only when the desktop audio session is visible.

Model downloads and first warm-up can be several gigabytes and take longer than
the package installation. Any failure must retain the exact command and error;
the launcher must not silently switch to a remote STT or TTS provider.

## Acceptance criteria

- The checkout remains based on the pinned upstream commit.
- No credential or `.env` secret is committed.
- Configuration parsing selects Parakeet TDT, OmniRoute Chat Completions, and
  Qwen3-TTS with automatic RU/EN handling.
- The server binds to loopback only.
- A real OmniRoute completion answers a Russian prompt in Russian and a clearly
  English prompt in English.
- CLI and focused tests pass in the Python 3.12 environment.
- GPU, model-download, and desktop-audio blockers are reported accurately rather
  than being presented as a successful end-to-end voice run.
