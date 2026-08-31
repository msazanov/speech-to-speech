"""Local operator recovery commands that are never exposed to the LLM."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .mcp_server import _default_database_path
from .store import SpeakerMemoryStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(_default_database_path()))
    subcommands = parser.add_subparsers(dest="command", required=True)
    unblock = subcommands.add_parser("unblock", help="Recover a falsely blocked voice ID.")
    unblock.add_argument("voice_id")
    arguments = parser.parse_args(argv)

    store = SpeakerMemoryStore(arguments.database)
    try:
        if arguments.command == "unblock":
            store.set_voice_blocked(arguments.voice_id, blocked=False)
            print(f"Unblocked voice={arguments.voice_id}")
            return 0
    finally:
        store.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
