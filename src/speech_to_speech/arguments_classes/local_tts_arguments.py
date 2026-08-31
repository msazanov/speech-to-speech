from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LocalTTSHandlerArguments:
    """CPU-local TTS router configuration for per-session backend selection."""

    local_tts_default_backend: Literal["silero", "rhvoice"] = field(
        default="silero",
        metadata={"help": "Default local TTS backend."},
    )
    local_tts_silero_voice: str = field(default="xenia", metadata={"help": "Default Silero RU voice."})
    local_tts_silero_sample_rate: int = field(default=24000)
    local_tts_silero_threads: int = field(default=6)
    local_tts_silero_english_fallback: bool = field(default=True)
    local_tts_silero_english_voice: str = field(default="M1")
    local_tts_rhvoice_executable: str = field(default="RHVoice-test")
    local_tts_rhvoice_data_path: str = field(default="")
    local_tts_rhvoice_library_path: str = field(default="")
    local_tts_rhvoice_voice: str = field(default="Aleksandr")
    local_tts_rhvoice_rate: int = field(default=100)
    local_tts_rhvoice_pitch: int = field(default=100)
    local_tts_rhvoice_volume: int = field(default=100)
    local_tts_blocksize: int = field(default=512)
    local_tts_rhvoice_timeout: float = field(default=15.0)
