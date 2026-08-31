# Browser Memory and Live Speech Implementation Plan

1. Add failing focused tests for the session-bound speaker-memory HTTP bridge,
   direct-session ID propagation, demo tool exposure/proxying, and browser tool
   dispatch.
2. Implement the backend endpoint and same-origin browser bridge by reusing the
   existing packaged tool schemas and `SpeakerMemoryService` executor.
3. Add failing focused tests for assistant-text language routing and the local
   RU/EN profile, then enable 250 ms progressive STT, speaker memory, and accurate
   Silero-friendly voice instructions.
4. Add a source-controlled user-systemd web unit and keep Serper credentials in
   a private environment file outside the repository.
5. Run only the affected Python and browser tests, commit and push the feature
   branch, deploy it to the confirmed local checkout, restart HuggingVoice/web,
   and run HTTP/log smoke checks.
