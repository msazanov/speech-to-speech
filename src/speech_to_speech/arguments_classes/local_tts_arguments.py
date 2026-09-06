from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LocalTTSHandlerArguments:
    """CPU-local TTS router configuration for per-session backend selection."""

    local_tts_default_backend: Literal["silero", "rhvoice", "glados"] = field(
        default="glados",
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
    local_tts_glados_python: str = field(default="/home/random/glados-tts/.venv/bin/python")
    local_tts_glados_workdir: str = field(default="/home/random/orange-RAG")
    local_tts_glados_espeak_ng: str = field(default="/home/random/glados-tts/espeak-ng/bin/espeak-ng")
    local_tts_glados_profile: str = field(default="v3-1000")
    local_tts_glados_style: str = field(default="Neutral")
    local_tts_glados_timeout: float = field(default=90.0)
