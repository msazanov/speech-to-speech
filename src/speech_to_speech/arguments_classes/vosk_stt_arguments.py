from dataclasses import dataclass, field


@dataclass
class VoskSTTHandlerArguments:
    """Configuration for the compact local Vosk speech-to-text backend."""

    vosk_stt_model_path: str = field(
        default="/home/orangepi/hugging-voice-device/models/vosk-model-small-ru-0.22",
        metadata={"help": "Directory containing a Vosk speech recognition model."},
    )
    vosk_stt_sample_rate: int = field(
        default=16000,
        metadata={"help": "Vosk decoder sample rate in Hz (default: 16000)."},
    )
    vosk_stt_language: str = field(
        default="ru",
        metadata={"help": "Language code attached to final transcriptions (default: ru)."},
    )
    vosk_stt_timeout_s: float = field(
        default=0.0,
        metadata={
            "help": "Optional per-utterance decoder timeout in seconds. "
            "When positive, Vosk runs in a restartable worker process."
        },
    )
