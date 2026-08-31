from dataclasses import dataclass, field


@dataclass
class RHVoiceTTSHandlerArguments:
    rhvoice_tts_executable: str = field(
        default="RHVoice-test",
        metadata={"help": "Path to the RHVoice-test executable."},
    )
    rhvoice_tts_data_path: str = field(
        default="",
        metadata={"help": "Optional RHVOICE_DATA_PATH for a user-local installation."},
    )
    rhvoice_tts_library_path: str = field(
        default="",
        metadata={"help": "Optional library directory prepended to LD_LIBRARY_PATH."},
    )
    rhvoice_tts_voice: str = field(
        default="Aleksandr",
        metadata={"help": "Default installed RHVoice profile."},
    )
    rhvoice_tts_rate: int = field(default=100, metadata={"help": "RHVoice speech rate in percent."})
    rhvoice_tts_pitch: int = field(default=100, metadata={"help": "RHVoice pitch in percent."})
    rhvoice_tts_volume: int = field(default=100, metadata={"help": "RHVoice volume in percent."})
    rhvoice_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in 16 kHz samples."},
    )
    rhvoice_tts_timeout: float = field(
        default=15.0,
        metadata={"help": "Maximum seconds for one RHVoice CLI synthesis."},
    )
