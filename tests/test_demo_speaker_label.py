import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_browser_speaker_labels_use_identity_and_stable_distinct_colors() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo UI tests")
    repo = Path(__file__).resolve().parents[1]
    script = """
const { speakerDisplayLabel, speakerDisplayColor } = await import('./demo/ui/chat.js');
const actual = {
  unknown: speakerDisplayLabel({ voice_id: 'v_aaa', state: 'unknown' }),
  known: speakerDisplayLabel({
    voice_id: 'v_aaa', state: 'known', person: { person_id: 'p_1', name: 'Марат' },
  }),
  pending: speakerDisplayLabel(null),
  stable: speakerDisplayColor('v_aaa') === speakerDisplayColor('v_aaa'),
  distinct: speakerDisplayColor('v_aaa') !== speakerDisplayColor('v_bbb'),
};
console.log(JSON.stringify(actual));
"""

    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "unknown": "v_aaa",
        "known": "Марат",
        "pending": "Voice…",
        "stable": True,
        "distinct": True,
    }
