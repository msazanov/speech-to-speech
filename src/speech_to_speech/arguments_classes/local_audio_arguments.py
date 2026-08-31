from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocalAudioArguments:
    local_audio_tool_module: Optional[str] = field(
        default=None,
        metadata={
            "help": "Importable module defining TOOLS and async execute_tool(name, arguments).",
            "aliases": ["--tool-module"],
        },
    )
    local_audio_input_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice input device index used by the local command."},
    )
    local_audio_output_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice output device index used by the local command."},
    )
    local_audio_chunk_size: int = field(
        default=1024,
        metadata={"help": "Microphone and speaker callback block size in samples. Default is 1024."},
    )
    local_audio_playback_buffer_ms: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Audio to buffer before local playback starts, in milliseconds. "
                "Defaults to 196 for OpenAI-compatible TTS and 0 otherwise."
            ),
            "aliases": ["--playback-buffer-ms"],
        },
    )
    local_audio_block_mic_during_playback: bool = field(
        default=False,
        metadata={
            "help": "Pause local microphone capture while audio is playing. Disabled by default so barge-in works."
        },
    )
    local_audio_echo_cancel: bool = field(
        default=False,
        metadata={
            "help": "Remove local speaker playback from microphone audio with CPU-only SpeexDSP AEC.",
            "aliases": ["--echo-cancel"],
        },
    )
    local_audio_echo_cancel_frame_ms: int = field(
        default=16,
        metadata={"help": "SpeexDSP AEC frame duration in milliseconds (10-20)."},
    )
    local_audio_echo_cancel_filter_ms: int = field(
        default=300,
        metadata={"help": "SpeexDSP acoustic echo tail in milliseconds (100-500)."},
    )
    local_audio_print_json: bool = field(
        default=False,
        metadata={"help": "Print raw Realtime events received by the packaged local audio client."},
    )
