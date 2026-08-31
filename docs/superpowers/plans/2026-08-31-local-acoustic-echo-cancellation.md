# Local Acoustic Echo Cancellation Implementation Plan

**Goal:** Keep the microphone open during assistant playback while removing the
assistant's own loudspeaker signal before it reaches VAD/STT.

**Architecture:** The local audio client already owns the exact mono PCM16
samples written to the output device. With AEC enabled, one sounddevice
full-duplex callback captures microphone and playback blocks on the same device
clock and puts the pair into a preallocated SPSC ring. A non-PortAudio worker feeds the
pair to an in-process SpeexDSP adapter. The adapter reblocks the existing
1024-sample callbacks into 256-sample (16 ms at 16 kHz) AEC frames. SpeexDSP's
playback/capture API supplies its two-frame sound-card delay and double-talk
handling, so a person can interrupt the assistant without muting the microphone.

**Constraints:** CPU only; mono PCM16; equal microphone and playback sample
rates; no Python/native dependency installation at runtime; fail clearly when
AEC is requested but `libspeexdsp` is unavailable; retain the existing
microphone-mute switch as a fallback, but reject enabling both modes together.

## Tasks

1. Add focused tests for configuration validation, native-library lifecycle,
   1024-to-256 sample reblocking, callback integration, and cleanup.
2. Add a small `ctypes` SpeexDSP adapter with preallocated native buffers,
   single-worker ownership, and no native work or blocking lock in PortAudio.
3. Extend `talk`, `local`, and JSON configuration with `echo_cancel`,
   `echo_cancel_frame_ms`, and `echo_cancel_filter_ms` options.
4. Capture the exact post-buffer output and microphone bytes in one lock-free duplex
   callback. Route raw mic while TTS is idle and AEC-cleaned mic only during
   actual TTS overlap and its acoustic tail. Count every routed or explicitly
   rejected block.
5. Add a persistent voice blacklist controlled only by short-lived
   `speaker_ref` agent/MCP tools; decisively matched blocked voices stop before
   STT/LLM and do not update their centroid.
6. Add opt-in full transcript/prompt/response timings and a journal follower
   with stable per-voice colors. Never log PCM, media payloads, secrets, or
   hidden reasoning.
7. Verify with focused tests, an offline synthetic echo check against the
   installed library, and a short isolated local run. Do not restart production
   or FreeToken services.

## Acceptance criteria

- Assistant playback no longer has to disable microphone capture.
- Near-end speech remains present during playback (barge-in is possible).
- Every mic block is accounted as raw, AEC-cleaned, or rejected with an explicit
  reason; no silent drop path exists.
- Voice blacklist decisions require an explicit user command and a valid
  current speaker reference.
- AEC adds no GPU/VRAM usage and bounded memory independent of conversation
  length.
- Invalid sample rate/frame combinations fail before audio streams start.
- If native AEC is disabled, existing audio behavior is unchanged.
