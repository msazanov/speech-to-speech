import base64
import io
import json
import sys
import wave
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.TTS import glados_tts_handler, glados_worker


def test_worker_uses_verified_ru_only_tiny_runtime_and_preserves_styles(monkeypatch, capsys):
    calls = []
    profile = SimpleNamespace(checkpoint="ru-weights", config="config", styles="styles", audit="audit")
    runtime = SimpleNamespace(
        infer=lambda text, **kwargs: calls.append(("infer", text, kwargs)) or (44100, np.ones(64) * 0.1)
    )

    def loader(name):
        def load(*args, **kwargs):
            calls.append((name, args, kwargs))
            return runtime

        return load

    modules = {
        "torch": dict(
            set_num_threads=lambda n: calls.append(("threads", n)),
            set_num_interop_threads=lambda n: calls.append(("interop", n)),
        ),
        "glados_ru": {},
        "glados_ru.native_stylebert": dict(
            NativeStyleBert=SimpleNamespace(
                from_paths=loader("dual"),
                from_russian_paths=loader("ru-only"),
            )
        ),
        "glados_ru.run_verified_russian_glados": dict(
            ORIGINAL="unused-en-weights",
            UPSTREAM="upstream",
            _profile=lambda name: profile,
        ),
        "glados_ru.russian_frontend": dict(RuAccentizer=lambda **kwargs: kwargs),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    styles = sorted(glados_tts_handler.GLADOS_STYLES)
    monkeypatch.setattr(sys, "argv", ["glados_worker", "--profile", "v3-1000"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            "".join(json.dumps(dict(id=i, text="Проверка.", style=style)) + "\n" for i, style in enumerate(styles))
        ),
    )
    glados_worker.main()
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert ("threads", 6) in calls
    assert ("interop", 1) in calls
    loaded = [call for call in calls if call[0] in {"dual", "ru-only"}]
    assert loaded == [
        (
            "ru-only",
            ("ru-weights", "config", "styles", "upstream"),
            {"adapted_audit": "audit", "accentizer": {"tiny_mode": True}},
        )
    ]
    assert replies[0]["ready"] is True
    assert replies[0]["runtime"] == "ru-only"
    assert replies[0]["threads"] == 6
    assert replies[0]["accent_mode"] == "tiny"
    assert [reply["id"] for reply in replies[1:]] == list(range(5))
    for reply in replies[1:]:
        with wave.open(io.BytesIO(base64.b64decode(reply["wav"])), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (44100, 1, 2)
            assert audio.getnframes() == 64
    inferred = [call for call in calls if call[0] == "infer"]
    assert len(inferred) == 6  # One warmup, then all five styles in the same worker.
    assert [call[2]["style"] for call in inferred[1:]] == styles
    assert all(call[2]["language"] == "RU" and call[2]["seed"] == 12345 for call in inferred)


def test_worker_spawn_is_cpu_isolated_and_uses_six_threads(monkeypatch):
    captured = {}

    def popen(*args, **kwargs):
        captured.update(kwargs["env"])
        raise OSError("test spawn")

    monkeypatch.setattr(glados_tts_handler.subprocess, "Popen", popen)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/server/onnx")
    monkeypatch.setenv("PYTHONPATH", "/server/python")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    handler = glados_tts_handler.GladosTTSHandler.__new__(glados_tts_handler.GladosTTSHandler)
    handler.setup(None)
    with pytest.raises(RuntimeError, match="Cannot start"):
        handler._start_worker()
    assert captured["OMP_NUM_THREADS"] == captured["MKL_NUM_THREADS"] == "6"
    assert captured["CUDA_VISIBLE_DEVICES"] == ""
    assert captured["HF_HUB_OFFLINE"] == "1"
    assert "LD_LIBRARY_PATH" not in captured
    assert "PYTHONPATH" not in captured
