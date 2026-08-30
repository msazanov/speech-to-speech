# Локальный HuggingVoice: OmniRoute, русский и английский

Этот профиль запускает весь речевой тракт на CPU и оставляет GPU для локальной
LLM:

- VAD: Silero + Smart Turn, CPU;
- STT: multilingual Faster Whisper Tiny, CPU/INT8, автоопределение `ru`/`en`;
- LLM: локальный шлюз OmniRoute по OpenAI Chat Completions;
- TTS: Supertonic 3, CPU ONNX, русский и английский;
- Realtime API: только `127.0.0.1:8765`.

## Установка

```bash
cd /home/random/dev/huggingvoice
./scripts/bootstrap-local.sh
```

Скрипт использует CPython 3.12, точный `uv.lock` и устанавливает extras
`faster-whisper` и `supertonic`. Системный Python не изменяется.

Первый старт скачивает активные модели:

- Faster Whisper Tiny: около 75 МБ в
  `~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny`;
- Supertonic 3: около 386 МБ в `~/.cache/supertonic3`;
- Smart Turn и Silero VAD — в стандартные кэши Hugging Face и Torch.

Окружение `.venv` занимает около 6 ГБ на диске, поскольку базовый upstream
пакет включает Torch и Qwen-зависимости, даже если облегчённый профиль не
загружает их в RAM.

## Проверка OmniRoute

```bash
curl -fsSL http://127.0.0.1:20128/api/monitoring/health | jq '{status, version}'
```

Ожидается `"status": "healthy"`. Профиль использует модель-маршрут
`auto/chat`. На момент проверки OmniRoute выбирал для него `openrouter/free`,
поэтому шлюз локальный, но LLM-инференс пока не гарантированно офлайн. После
восстановления локального LLM-маршрута достаточно изменить только `model_name`
в `config/omniroute-ru-en.json` и повторить русский/английский smoke-тест.

Параметр `responses_api_disable_thinking: true` обязателен для текущего
маршрута: он не даёт служебному reasoning-тексту попасть в голосовой ответ.

## Запуск

Терминал 1:

```bash
cd /home/random/dev/huggingvoice
./scripts/run-omniroute-ru-en.sh
```

Готовность подтверждают строки:

```text
OpenAI Realtime API starting on ws://127.0.0.1:8765/v1/realtime
Uvicorn running on http://127.0.0.1:8765
```

Проверка порта:

```bash
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8765/docs
ss -ltnp '( sport = :8765 )'
```

Терминал 2, клиент микрофона и динамиков:

```bash
cd /home/random/dev/huggingvoice
uv run speech-to-speech talk \
  --url ws://127.0.0.1:8765/v1/realtime \
  --block-mic-during-playback
```

## Языковая политика

Системный prompt отвечает по-русски по умолчанию. Если текущая реплика явно
английская, ответ и синтез переключаются на английский. Код, названия продуктов
и отдельные английские технические термины внутри русской реплики не должны
переключать язык ответа.

Проверенный текстовый smoke через OmniRoute вернул:

- `Я сейчас отвечаю на русском языке.` для русской реплики;
- `I am using English.` для английской реплики.

Отдельный CPU smoke Supertonic успешно синтезировал конечные аудиомассивы для
`ru` и `en`.

## Ресурсы и известные ограничения

На этом хосте прогретый процесс HuggingVoice занимал около 1,63 ГБ RSS. Это
примерно на 72% меньше начального профиля Parakeet 0.6B + Qwen3-TTS
(около 5,9 ГБ RSS). OmniRoute работает отдельным процессом и в измерение
HuggingVoice не входит.

GPU профиль намеренно не используется. `nvidia-smi` не связывается с драйвером,
а локальный `ornith-q6.service` перезапускается. Полный PyTorch Qwen3-TTS также
не подходит под бюджет RAM; готовое CPU GGML-колесо падало на CPU Intel
i7-8750H после сообщения об отсутствии AMX.

Интерактивный микрофон/динамики не проверялись из этой терминальной сессии,
поскольку desktop PipeWire здесь недоступен. Серверный старт, loopback-порт,
OmniRoute, русско-английская политика и CPU-синтез проверены отдельно.

Во время исследования были загружены, но больше не используются, кэши
Parakeet (около 2,4 ГБ) и Qwen GGUF (около 1,4 ГБ). Их можно удалить вручную,
если нужно освободить диск; текущий профиль их не читает.

## Проверка исходников

```bash
cd /home/random/dev/huggingvoice
./scripts/bootstrap-local.sh
uv run pytest -q tests/test_local_bootstrap_config.py \
  tests/test_cli_defaults.py tests/test_language_prompt.py
NO_PROXY="${NO_PROXY:+${NO_PROXY},}10.255.255.1" uv run pytest -q
bash -n scripts/bootstrap-local.sh scripts/run-omniroute-ru-en.sh
git diff --check
```

Адрес `10.255.255.1` добавлен в `NO_PROXY` только для upstream-теста
недоступного сервера: глобальный HTTP proxy иначе меняет ожидаемый код ответа
из `502` в `503`.
