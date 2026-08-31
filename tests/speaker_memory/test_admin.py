from __future__ import annotations

import numpy as np

from speech_to_speech.speaker_memory.admin import main
from speech_to_speech.speaker_memory.store import SpeakerMemoryStore


def test_admin_can_recover_a_falsely_blocked_voice_without_an_llm_reference(tmp_path, capsys) -> None:
    database = tmp_path / "memory.sqlite3"
    store = SpeakerMemoryStore(database)
    voice = store.create_voice_cluster(np.array([1.0, 0.0], dtype=np.float32), quality=1.0)
    store.set_voice_blocked(voice.id, blocked=True, reason="false_positive")
    store.close()

    assert main(["--database", str(database), "unblock", voice.id]) == 0

    reopened = SpeakerMemoryStore(database)
    assert reopened.is_voice_blocked(voice.id) is False
    reopened.close()
    assert voice.id in capsys.readouterr().out
