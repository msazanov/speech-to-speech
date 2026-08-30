from dataclasses import dataclass, field


@dataclass
class GigaAMONNXSTTHandlerArguments:
    gigaam_onnx_stt_model_name: str = field(
        default="gigaam-multilingual-ctc",
        metadata={"help": "The onnx-asr GigaAM model name. Default is gigaam-multilingual-ctc."},
    )
    gigaam_onnx_stt_quantization: str = field(
        default="int8",
        metadata={"help": "ONNX model quantization. Default is int8."},
    )
    gigaam_onnx_stt_provider: str = field(
        default="CPUExecutionProvider",
        metadata={"help": "ONNX Runtime provider. This backend only permits CPUExecutionProvider."},
    )
    gigaam_onnx_stt_threads: int = field(
        default=6,
        metadata={"help": "Maximum ONNX Runtime intra-op CPU threads. Default is 6."},
    )
    gigaam_onnx_stt_language: str = field(
        default="auto",
        metadata={
            "help": (
                "Language reported downstream. Use auto for Russian-first Cyrillic/Latin detection, "
                "or a fixed language code. Default is auto."
            )
        },
    )
