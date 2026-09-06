# Native Russian GLaDOS runtime

The local profile uses the trained `v3-1000` model from
[Random118/GLaDOS_TTS](https://huggingface.co/Random118/GLaDOS_TTS/tree/dc9aa6dda71e49b6d2555beda6216c6488efedea/Models/Russian_StyleBert_Pilot/v3-1000).
Checkpoint SHA256: `dd8c8987ddd003236589863b3d483d0f50f27e9ea17c92ed0bf839a4818a53ae`.

`tts=local` routes `glados:Neutral`, `glados:Standard`, `glados:Deep`,
`glados:Light`, and `glados:Standard_02` to native Russian synthesis.
The browser settings select backend and style. Explicit saved selections are
preserved; fresh settings default to GLaDOS Neutral.

The handler launches `glados_worker.py` in the separately installed GLaDOS
Python environment. `local_tts_glados_workdir` must contain the `glados_ru`
package and its verified runtime/profile configuration; see that package's
`HUGGINGVOICE_RU.md` for installation and model assets. The current workstation
paths are configured in `config/omniroute-ru-en.json`. Installing HuggingVoice
alone does not install this external runtime or the trained weights. The optimized
worker requires the `glados_ru` runtime from orange-RAG commit `14362bc` or a
compatible version providing `NativeStyleBert.from_russian_paths` and
`RuAccentizer(tiny_mode=True)`.

The worker keeps only the hash-validated Russian model and tiny-mode RuAccent
(`turbo3.1`, not the lower-quality `tiny2.1` model) in RAM. It does not load the
unused English acoustic model. It uses six Torch/OMP/MKL CPU threads and one
Torch inter-op thread, and warms up before the local server
becomes ready when GLaDOS is the configured default. It survives conversation
restarts and style changes. Stale queued text is discarded before synthesis;
a cancelled in-flight result is drained without being played. Service shutdown
terminates the child process. A worker failure can be retried with a new worker.

A Russian frontend rejection falls back to Silero for that utterance. This
profile does not support arbitrary Latin text; the assistant name GLaDOS has a
Russian pronunciation alias. The original waveform is converted from 44.1 kHz
PCM16 to the pipeline's 16 kHz before the transport resamples it for the client.

On the 2026-09-06 deployment preflight, the optimized worker rendered
«Система готова к работе.» in 0.63–0.72 seconds across all five styles, with
1.53–1.64 seconds of non-silent audio. Cold initialization plus warmup took
11.83 seconds; peak RSS was 1,953,640 KiB (1.86 GiB), with no worker swap.
The previous dual-model/two-thread worker measured 1.35–1.43 seconds per phrase
and 27.45 seconds to ready. These are individual TTS-only measurements under
different system loads, not conversational latency or a controlled performance
guarantee. Model loading, LLM prefill, language processing, memory pressure and
text length affect overall latency. Weights, seed and style vectors are unchanged.

In the actual service after restart, worker initialization was 14.08 seconds;
the complete speech stack took about 99 seconds to become ready, mostly waiting
for the separate Gemma backend. Browser calls measured 1.60 seconds for Neutral
over WebSocket and 1.06 seconds for Deep over WebRTC. Both completed with native
GLaDOS audio and a nonzero playback signal. The live worker later had some swap
under system memory pressure, so the standalone no-swap result is not a promise
about the whole running stack.

The readiness log reports `runtime=ru-only threads=6 accent_mode=tiny` and
worker startup time. The integration preserves the serial JSONL protocol,
cancellation handling and per-utterance Silero fallback. To roll back the runtime
change, revert the optimized-worker integration commit and restart only the
`huggingvoice` user service; the earlier working implementation is in commit
`638f7d5`. No checkpoint rollback or model download is necessary.

The SDK session must use the selected LLM model from its initial connection.
The previous placeholder `s2s-local` caused the server to reject the initial
session update, including voice and output sample rate. Both WebSocket (24 kHz)
and WebRTC playback were verified with nonzero browser audio-analyser readings,
completed responses, no protocol errors, and native GLaDOS synthesis in the logs.

Regression checks: `python -m pytest -q tests/test_glados_worker.py tests/test_glados_tts.py tests/test_demo_tts_options.py tests/test_demo_user_audio.py`.
UI verification must open the served HTTP page and actually click Settings;
`node --check` alone cannot detect missing imported names. Runtime verification
must confirm nonzero decoded/playback audio, not merely HTTP 200 or a generated
WAV in a standalone process.
