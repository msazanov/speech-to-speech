import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tts_backend_options_encode_real_server_selection() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            """
import assert from "node:assert/strict";
import {
  TTS_BACKENDS,
  decodeTtsSelection,
  encodeTtsSelection,
  voicesForTtsBackend,
} from "./demo/ui/tts-options.js";

assert.deepEqual(TTS_BACKENDS.map((item) => item.id), ["glados", "silero", "rhvoice"]);
assert.equal(encodeTtsSelection("silero", "xenia"), "silero:xenia");
assert.equal(encodeTtsSelection("rhvoice", "Mikhail"), "rhvoice:Mikhail");
assert.deepEqual(decodeTtsSelection("rhvoice:Pavel"), { backend: "rhvoice", voice: "Pavel" });
assert.deepEqual(decodeTtsSelection(null), { backend: "glados", voice: "Neutral" });
assert.deepEqual(decodeTtsSelection("Aiden"), { backend: "glados", voice: "Neutral" });
for (const style of ["Neutral", "Standard", "Deep", "Light", "Standard_02"]) {
  assert.equal(encodeTtsSelection("glados", style), `glados:${style}`);
  assert.deepEqual(decodeTtsSelection(`glados:${style}`), { backend: "glados", voice: style });
}
assert.ok(voicesForTtsBackend("silero").some((voice) => voice.id === "aidar"));
assert.ok(voicesForTtsBackend("rhvoice").some((voice) => voice.id === "Aleksandr"));
assert.ok(!voicesForTtsBackend("rhvoice").some((voice) => /nikol/i.test(voice.id)));
""",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
