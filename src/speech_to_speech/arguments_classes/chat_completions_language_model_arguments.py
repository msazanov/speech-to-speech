from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)


@dataclass
class ChatCompletionsLanguageModelHandlerArguments(ResponsesApiLanguageModelHandlerArguments):
    """Arguments for the ``chat-completions`` LLM backend.

    Inherits the OpenAI-compatible connection fields from the Responses-API
    arguments (``responses_api_base_url`` / ``responses_api_api_key`` /
    ``responses_api_stream`` / ``responses_api_disable_thinking``) so the same
    CLI flags and launcher env vars drive both backends. Chat Completions keeps
    the reasoning-effort default unset so provider-specific disable-thinking
    behavior remains unchanged.
    """

    responses_api_reasoning_effort: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provider-specific reasoning level sent as extra_body={'reasoning_effort': <value>} on the "
            "Chat Completions request. Use to disable reasoning on providers where "
            "chat_template_kwargs.enable_thinking is ignored (e.g. 'none' / 'low'). When unset, falls back to "
            "the disable_thinking behaviour (chat_template_kwargs.enable_thinking=false and "
            "thinking_mode='disabled'). Default is None."
        },
    )
    responses_api_gen_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum tokens for text Chat Completions requests. Default is provider-defined."},
    )
    responses_api_gen_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Sampling temperature for text Chat Completions requests. Default is provider-defined."},
    )
    responses_api_external_agent: bool = field(
        default=False,
        metadata={
            "help": "Treat the Chat Completions endpoint as the sole stateful agent. Sends only the current "
            "committed user transcript plus voice_turn_id, with no HuggingVoice prompts, history, tools, "
            "warmup, prefill, compaction, speculative prefetch, or SDK retries. Default is disabled."
        },
    )
