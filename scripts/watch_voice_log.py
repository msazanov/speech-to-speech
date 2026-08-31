#!/usr/bin/env python3
"""Follow HuggingVoice journald output with readable stages and stable voice colors."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterable

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
VOICE_COLORS = (39, 45, 51, 75, 81, 111, 117, 141, 147, 177, 183, 208, 214, 220, 156, 120)
VOICE_RE = re.compile(r"\bvoice=(v_[A-Za-z0-9]+)")
PERSON_ID_RE = re.compile(r"\bperson_id=(?:p_[A-Za-z0-9]+|unknown)")
PERSON_RE = re.compile(r"\bperson=(?:'[^']*'|\"[^\"]*\"|[^\s]+)")

STAGES = (
    (re.compile(r"\brejected\b|\berror\b|failed", re.IGNORECASE), "DROP", 196),
    (re.compile(r"Speaker attributed|blacklisted_voice", re.IGNORECASE), "VOICE", 75),
    (re.compile(r"GigaAM|transcription|\bSTT\b", re.IGNORECASE), "STT", 45),
    (re.compile(r"\bLLM\b|LanguageModel|ChatCompletions", re.IGNORECASE), "LLM", 177),
    (re.compile(r"\bTTS\b|Silero|first audio", re.IGNORECASE), "TTS", 214),
    (re.compile(r"Audio route|Acoustic echo|\bAEC\b", re.IGNORECASE), "AUDIO", 82),
)


def _preferred_color_index(voice_id: str, size: int) -> int:
    digest = hashlib.blake2s(voice_id.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "big") % size


class VoicePalette:
    """Keep simultaneous speakers visually distinct while preserving stable colors."""

    def __init__(self, colors: tuple[int, ...] = VOICE_COLORS) -> None:
        self.colors = colors
        self._by_voice: dict[str, int] = {}
        self._owners: dict[int, str] = {}

    def color(self, voice_id: str) -> int:
        existing = self._by_voice.get(voice_id)
        if existing is not None:
            return existing
        preferred = _preferred_color_index(voice_id, len(self.colors))
        for offset in range(len(self.colors)):
            color = self.colors[(preferred + offset) % len(self.colors)]
            if color not in self._owners:
                self._owners[color] = voice_id
                self._by_voice[voice_id] = color
                return color
        color = self.colors[preferred]
        self._by_voice[voice_id] = color
        return color


_VOICE_PALETTE = VoicePalette()


def voice_color(voice_id: str) -> int:
    return _VOICE_PALETTE.color(voice_id)


def _stage(line: str) -> tuple[str, int] | None:
    for pattern, label, color in STAGES:
        if pattern.search(line):
            return label, color
    return None


def humanize(line: str) -> str:
    stage = _stage(line)
    return f"[{stage[0]}] {line}" if stage is not None else line


def colorize(line: str, *, color: bool) -> str:
    rendered = humanize(line.rstrip("\n"))
    if not color:
        return rendered
    stage = _stage(line)
    base = f"\x1b[38;5;{stage[1]}m" if stage is not None else ""
    voice_match = VOICE_RE.search(rendered)
    if voice_match is not None:
        voice_escape = f"{BOLD}\x1b[38;5;{voice_color(voice_match.group(1))}m"
        rendered = VOICE_RE.sub(lambda match: f"{voice_escape}{match.group(0)}{RESET}{base}", rendered)
        rendered = PERSON_ID_RE.sub(lambda match: f"{voice_escape}{match.group(0)}{RESET}{base}", rendered)
        rendered = PERSON_RE.sub(lambda match: f"{voice_escape}{match.group(0)}{RESET}{base}", rendered)
    return f"{base}{rendered}{RESET}"


def follow(lines: Iterable[str], *, color: bool) -> None:
    for line in lines:
        print(colorize(line, color=color), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="huggingvoice.service")
    parser.add_argument("--history", type=int, default=0, help="Show this many existing journal lines first.")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    command = [
        "journalctl",
        "--user",
        "-u",
        args.service,
        "-n",
        str(max(0, args.history)),
        "-f",
        "-o",
        "cat",
    ]
    color = not args.no_color and sys.stdout.isatty()
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1) as process:
            assert process.stdout is not None
            follow(process.stdout, color=color)
            return process.wait()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
