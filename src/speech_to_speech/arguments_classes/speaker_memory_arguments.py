from dataclasses import dataclass, field


@dataclass
class SpeakerMemoryArguments:
    speaker_memory_enabled: bool = field(
        default=False,
        metadata={"help": "Enable local CPU speaker identification and memory. Default is disabled."},
    )
    speaker_memory_model_path: str | None = field(
        default=None,
        metadata={"help": "Path to the local sherpa-onnx speaker embedding model."},
    )
    speaker_memory_database_path: str | None = field(
        default=None,
        metadata={"help": "SQLite identity database path. Defaults to the XDG HuggingVoice data directory."},
    )
    speaker_memory_threads: int = field(
        default=1,
        metadata={"help": "CPU inference threads for speaker embeddings. Default is 1."},
    )
    speaker_memory_min_audio_ms: int = field(
        default=700,
        metadata={"help": "Minimum final speech duration eligible for speaker embedding. Default is 700 ms."},
    )
    speaker_memory_match_threshold: float = field(
        default=0.82,
        metadata={"help": "Cosine threshold for a decisive voice-cluster match."},
    )
    speaker_memory_candidate_threshold: float = field(
        default=0.70,
        metadata={"help": "Cosine threshold below which a new voice cluster may be created."},
    )
    speaker_memory_group_threshold: float = field(
        default=0.55,
        metadata={"help": "Cosine threshold for suggesting a person from a nearby voice cluster."},
    )
    speaker_memory_soft_match_threshold: float = field(
        default=0.60,
        metadata={"help": "Lower bound for reusing a mature unassigned cluster with a small centroid update."},
    )
    speaker_memory_soft_match_min_samples: int = field(
        default=3,
        metadata={"help": "Minimum samples before a weak voice match may reuse a cluster."},
    )
    speaker_memory_soft_match_weight: float = field(
        default=0.25,
        metadata={"help": "Centroid update weight for a weak reusable match. Default is conservative."},
    )
    speaker_memory_automerge_threshold: float = field(
        default=0.82,
        metadata={"help": "Cosine threshold for merging mature unassigned voice clusters."},
    )
    speaker_memory_automerge_min_samples: int = field(
        default=3,
        metadata={"help": "Minimum samples in each unassigned cluster before automatic merging."},
    )
    speaker_memory_ambiguity_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum cosine margin over the runner-up voice cluster."},
    )
    speaker_memory_reference_ttl_s: float = field(
        default=300.0,
        metadata={"help": "Conversation-bound speaker tool reference lifetime in seconds."},
    )
    speaker_memory_observation_retention_days: int = field(
        default=30,
        metadata={"help": "Retention target for speaker observations pruned at startup and session end."},
    )
