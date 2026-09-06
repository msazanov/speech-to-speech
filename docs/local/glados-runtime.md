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
alone does not install this external runtime or the trained weights.

The worker keeps the hash-validated native model and RuAccent in RAM, uses
CPU only with two Torch/OMP/MKL threads, and warms up before the local server
becomes ready when GLaDOS is the configured default. It survives conversation
restarts and style changes. Stale queued text is discarded before synthesis;
a cancelled in-flight result is drained without being played. Service shutdown
terminates the child process. A worker failure can be retried with a new worker.

A Russian frontend rejection falls back to Silero for that utterance. This
profile does not support arbitrary Latin text; the assistant name GLaDOS has a
Russian pronunciation alias. The original waveform is converted from 44.1 kHz
PCM16 to the pipeline's 16 kHz before the transport resamples it for the client.

Measured on this workstation, the warm worker rendered «Система готова к работе.»
in 1.35–1.43 seconds for Neutral, Deep and Light; full cold initialization plus
warmup took 27.45 seconds. These are TTS-only measurements, not conversational
latency or a general performance guarantee. Model loading, LLM prefill, language
processing and text length affect overall latency.

The SDK session must use the selected LLM model from its initial connection.
The previous placeholder `s2s-local` caused the server to reject the initial
session update, including voice and output sample rate. Both WebSocket (24 kHz)
and WebRTC playback were verified with nonzero browser audio-analyser readings,
completed responses, no protocol errors, and native GLaDOS synthesis in the logs.

Regression checks: `python -m pytest -q tests/test_glados_tts.py tests/test_demo_tts_options.py tests/test_demo_user_audio.py`.
UI verification must open the served HTTP page and actually click Settings;
`node --check` alone cannot detect missing imported names. Runtime verification
must confirm nonzero decoded/playback audio, not merely HTTP 200 or a generated
WAV in a standalone process.
