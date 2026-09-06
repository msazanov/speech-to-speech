"""Serial JSON-lines worker, launched in the dedicated native GLaDOS environment.

Keep the verified RU-only model and tiny-mode RuAccent warm. stdout is
exclusively protocol data; model imports and progress belong on stderr.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from time import perf_counter


def serve(runtime, source, sink):
    import soundfile as sf

    for line in source:
        request = json.loads(line)
        result = {"id": request["id"]}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                sr, audio = runtime.infer(
                    request["text"],
                    language="RU",
                    style=request["style"],
                    seed=12345,
                )
            output = io.BytesIO()
            sf.write(output, audio, sr, format="WAV", subtype="PCM_16")
            result["wav"] = base64.b64encode(output.getvalue()).decode("ascii")
        except Exception as exc:
            # Frontend exception messages can contain the user's spoken text.
            result["error"] = type(exc).__name__
        sink.write(json.dumps(result) + "\n")
        sink.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="v3-1000")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path.cwd()))
    sink = sys.stdout
    started = perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            torch.set_num_threads(6)
            torch.set_num_interop_threads(1)
            from glados_ru.native_stylebert import NativeStyleBert
            from glados_ru.run_verified_russian_glados import UPSTREAM, _profile
            from glados_ru.russian_frontend import RuAccentizer

            profile = _profile(args.profile)
            runtime = NativeStyleBert.from_russian_paths(
                profile.checkpoint,
                profile.config,
                profile.styles,
                UPSTREAM,
                adapted_audit=profile.audit,
                accentizer=RuAccentizer(tiny_mode=True),
            )
            runtime.infer("Система готова к работе.", language="RU", style="Neutral", seed=12345)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}), file=sink, flush=True)
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "ready": True,
                "runtime": "ru-only",
                "threads": 6,
                "accent_mode": "tiny",
                "startup_seconds": perf_counter() - started,
            }
        ),
        file=sink,
        flush=True,
    )
    serve(runtime, sys.stdin, sink)


if __name__ == "__main__":
    main()
