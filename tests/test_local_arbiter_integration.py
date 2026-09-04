from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest
from openai.types.realtime.session_update_event import SessionUpdateEvent

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk


class DelayedChatEndpoint:
    def __init__(self, generation_delay_s: float) -> None:
        self.generation_delay_s = generation_delay_s
        self.requests: list[dict] = []

        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length))
                endpoint.requests.append(body)
                if len(endpoint.requests) > 1:
                    time.sleep(endpoint.generation_delay_s)
                response = {
                    "id": "chatcmpl-local-arbiter",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Готово."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                if body.get("stream"):
                    chunks = [
                        {
                            **response,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "Готово."},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            **response,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        },
                    ]
                    payload = b"".join(
                        f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks
                    ) + b"data: [DONE]\n\n"
                    content_type = "text/event-stream"
                else:
                    response["object"] = "chat.completion"
                    payload = json.dumps(response).encode()
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass

            def log_message(self, _format: str, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> DelayedChatEndpoint:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


def make_handler(endpoint: DelayedChatEndpoint, *, timeout_s: float) -> ChatCompletionsApiModelHandler:
    return ChatCompletionsApiModelHandler(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_kwargs={
            "model_name": "gemma-4-e2b",
            "base_url": endpoint.base_url,
            "api_key": "local",
            "stream": True,
            "disable_thinking": True,
            "request_timeout_s": timeout_s,
            "max_retries": 0,
        },
    )


def generate_once(handler: ChatCompletionsApiModelHandler, *, model: str | None = None) -> list[object]:
    chat = Chat(4)
    chat.add_item(make_user_message("Ответь кратко"))
    runtime_config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(
            type="realtime", instructions="Отвечай по-русски.", model=model,
        ),
    )
    request = GenerateResponseRequest(runtime_config=runtime_config, language_code="ru")
    return list(handler.process(request))


def test_delayed_arbiter_response_completes_within_configured_timeout() -> None:
    with DelayedChatEndpoint(generation_delay_s=0.12) as endpoint:
        handler = make_handler(endpoint, timeout_s=0.5)

        outputs = generate_once(handler)

    assert [request["model"] for request in endpoint.requests] == ["gemma-4-e2b", "gemma-4-e2b"]
    assert any(isinstance(output, LLMResponseChunk) and output.text == "Готово." for output in outputs)
    assert any(isinstance(output, EndOfResponse) and output.error is None for output in outputs)


def test_arbiter_timeout_does_not_retry_request() -> None:
    with DelayedChatEndpoint(generation_delay_s=0.2) as endpoint:
        handler = make_handler(endpoint, timeout_s=0.05)
        started = time.monotonic()

        outputs = generate_once(handler)
        elapsed = time.monotonic() - started

    assert len(endpoint.requests) == 2  # one warmup plus one generation; no requeue/retry
    assert elapsed < 0.3
    assert any(isinstance(output, EndOfResponse) and output.error is not None for output in outputs)


def test_adaptive_thinking_is_forwarded_without_forcing_reasoning() -> None:
    with DelayedChatEndpoint(generation_delay_s=0.0) as endpoint:
        handler = ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs={
                "model_name": "gemma-4-e2b",
                "base_url": endpoint.base_url,
                "api_key": "local",
                "stream": True,
                "disable_thinking": False,
                "thinking_mode": "auto",
                "request_timeout_s": 0.5,
                "max_retries": 0,
            },
        )
        generate_once(handler)

    generation_request = endpoint.requests[-1]
    assert generation_request["chat_template_kwargs"] == {"thinking_mode": "adaptive"}


def test_selected_runtime_model_is_sent_to_generation_but_not_warmup() -> None:
    with DelayedChatEndpoint(generation_delay_s=0.0) as endpoint:
        handler = make_handler(endpoint, timeout_s=0.5)
        outputs = generate_once(handler, model="LFM2.5-2.6B")

    assert [request["model"] for request in endpoint.requests] == ["gemma-4-e2b", "LFM2.5-2.6B"]
    assert any(isinstance(output, EndOfResponse) and output.error is None for output in outputs)


def test_realtime_service_accepts_only_configured_llm_models() -> None:
    service = RealtimeService(allowed_llm_models={"gemma-4-e2b", "LFM2.5-2.6B"})
    conn_id = service.register()

    accepted = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(type="session.update", session=RealtimeSessionCreateRequest(type="realtime", model="LFM2.5-2.6B")),
    )
    rejected = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(type="session.update", session=RealtimeSessionCreateRequest(type="realtime", model="unknown")),
    )

    assert accepted is None
    assert rejected is not None
    assert rejected.error.type == "invalid_model"
    assert service._state(conn_id).runtime_config.session.model == "LFM2.5-2.6B"
