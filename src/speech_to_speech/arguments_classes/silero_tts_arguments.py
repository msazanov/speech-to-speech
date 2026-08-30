from dataclasses import dataclass, field


@dataclass
class SileroTTSHandlerArguments:
    silero_tts_speaker: str = field(
        default="xenia",
        metadata={"help": "Russian voice in Silero v5.5 RU. Default is xenia."},
    )
    silero_tts_sample_rate: int = field(
        default=24000,
        metadata={"help": "Silero synthesis sample rate before conversion to pipeline 16 kHz. Default is 24000."},
    )
    silero_tts_threads: int = field(
        default=6,
        metadata={"help": "CPU thread count used by Silero. Default is 6."},
    )
    silero_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in 16 kHz samples. Default is 512."},
    )
    silero_tts_english_fallback: bool = field(
        default=True,
        metadata={"help": "Lazily load Supertonic for explicitly English text. Default is true."},
    )
    silero_tts_english_voice: str = field(
        default="M1",
        metadata={"help": "Supertonic voice used by the English fallback. Default is M1."},
    )
