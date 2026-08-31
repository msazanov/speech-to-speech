# Browser Speaker Memory and Live Speech Design

## Goal

Make the local browser client use the native HuggingVoice speaker-memory service,
show progressive GigaAM transcription while a person is still speaking, and route
English assistant output to the English TTS fallback even when the input turn was
classified as Russian.

## Speaker-memory bridge

The database and all identity checks remain inside the realtime HuggingVoice
process. The realtime app exposes a narrow, session-bound
`POST /v1/speaker-memory/tool` endpoint. It accepts only a current session ID, a
packaged `speaker_memory_*` tool name, and its arguments. The endpoint locates the
speaker handler owned by that exact pipeline session and invokes the existing
`SpeakerMemoryService` executor using the handler's conversation ID. No database
path, embedding, secret, or unrestricted upstream URL is exposed to the browser.

The demo server publishes the packaged tool schemas through `/api/config` and
proxies tool calls to its deployment-pinned `SPEECH_TO_SPEECH_URL`. The browser
learns the direct session ID from `session.created`, declares the memory tools to
the Agents SDK, and sends calls through the same-origin proxy. The existing
short-lived `speaker_ref` remains the mutation authority and preserves the
fail-closed identity policy.

## Progressive transcription

The Russian-first profile enables the existing progressive GigaAM path and asks
VAD for an update every 250 ms. GigaAM's stable-prefix logic may wait for a second
hypothesis before exposing words, while the final segment still replaces all
partials with the authoritative full transcript.

## Speech-language routing and Silero prompt

TTS selects a backend from the assistant text itself when the text is clearly
Latin/English or Cyrillic/Russian. This overrides a stale inherited input-language
code, but keeps mixed Russian technical sentences on Silero. Silero v5.5 RU is
prompted using features it actually supports in this pipeline: punctuation,
natural interjections, automatic stress and homograph handling. Undocumented
emotion, laughter, whisper, or raw SSML tags are not invented; laughter and
hesitation are written as speakable words such as `ха-ха`, `хм`, and `эм`.

## Deployment and secrets

Serper's key lives only in a mode-0600 user environment file. A source-controlled
user-systemd unit starts the demo against the pinned local realtime URL. The key
is never committed, logged, or returned by `/api/config`.
