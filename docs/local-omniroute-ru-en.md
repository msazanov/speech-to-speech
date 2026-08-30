# Локальный HuggingVoice: OmniRoute, русский и английский

Этот профиль запускает весь речевой тракт на CPU и оставляет GPU для локальной
LLM:

- VAD: Silero + Smart Turn, CPU;
- STT: Faster Whisper Small, дообученный на русском и квантованный в INT8,
  автоопределение `ru`/`en`;
- LLM: тестовый маршрут `kmc/k3-256k` через локальный шлюз OmniRoute;
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

- `ammaraldirawi/faster-whisper-small-ru-int8`: около 254 МБ в
  `~/.cache/huggingface/hub/models--ammaraldirawi--faster-whisper-small-ru-int8`;
- Supertonic 3: около 386 МБ в `~/.cache/supertonic3`;
- Smart Turn и Silero VAD — в стандартные кэши Hugging Face и Torch.

Окружение `.venv` занимает около 6 ГБ на диске, поскольку базовый upstream
пакет включает Torch и Qwen-зависимости, даже если облегчённый профиль не
загружает их в RAM.

## Проверка OmniRoute

```bash
curl -fsSL http://127.0.0.1:20128/api/monitoring/health | jq '{status, version}'
```

Ожидается `"status": "healthy"`. Тестовый профиль закреплён за маршрутом
`kmc/k3-256k`, чтобы не тратить 6–12 секунд на автоматический подбор провайдера
для `auto/chat`. Шлюз работает локально; место выполнения LLM-инференса
определяет конфигурация этого маршрута в OmniRoute. Для смены модели достаточно
изменить `model_name` в `config/omniroute-ru-en.json` и повторить RU/EN
smoke-тест.

Параметр `responses_api_disable_thinking: true` не даёт служебному
reasoning-тексту попасть в голосовой ответ.

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

## Замкнутый акустический тест

Скрипт заранее синтезирует одну русскую фразу через Supertonic, открывает
системный микрофон, проигрывает фразу через системные динамики и ждёт результат
от Realtime API. Аудио ответа ассистента намеренно не проигрывается, чтобы оно
не попало в микрофон вторым пользовательским ходом.

```bash
cd /home/random/dev/huggingvoice
uv run python scripts/acoustic_loopback_smoke.py
```

Другой эталон (не более 20 слов) и явные устройства:

```bash
uv run python scripts/acoustic_loopback_smoke.py \
  --text "Ты меня хорошо слышишь и правильно понимаешь?" \
  --input-device 7 \
  --output-device 7
```

Отчёт содержит распознанный текст, WER, пословную точность, символьное
сходство, пик микрофона и задержки STT/ответа. Первый одинаковый A/B-прогон на
этом хосте показал эффект настройки VAD:

- `min_silence_ms=64`: STT 10,306 с, WER 60%, полный цикл 21,308 с;
- `min_silence_ms=500`: STT 4,527 с, WER 40%, полный цикл 17,608 с.

Во втором прогоне одна фраза прошла одним STT-вызовом. После завершения речи
оставшиеся измеренные узкие места: 4,527 с Faster Whisper, 3,581 с
`kmc/k3-256k` и примерно 2,26 с от готового текста LLM до первого аудиоблока.
Результат зависит от акустики помещения, громкости, размещения микрофона и
текущей задержки провайдера OmniRoute.

## Ресурсы и известные ограничения

Русская Large v3 Turbo оказалась слишком медленной на CPU этого хоста:
12–17 секунд только на распознавание короткой реплики. Поэтому активный профиль
использует русскую Small INT8. Live-transcription отключён: с Faster Whisper он
повторно распознавал одну реплику сначала частично, затем целиком и добавлял
около 3,8 секунды CPU-задержки. `min_silence_ms` увеличен с 64 до 500 мс:
короткие естественные паузы больше не разрезают одну фразу на несколько ревизий
с повторным запуском Whisper. Весь речевой тракт остаётся на CPU и не занимает
VRAM; OmniRoute работает отдельным процессом и в измерение HuggingVoice не
входит.

GPU профиль намеренно не используется. `nvidia-smi` не связывается с драйвером,
а локальный `ornith-q6.service` перезапускается. Полный PyTorch Qwen3-TTS также
не подходит под бюджет RAM; готовое CPU GGML-колесо падало на CPU Intel
i7-8750H после сообщения об отсутствии AMX.

Интерактивный клиент проверен через desktop PipeWire с устройством №20.
Конкретный номер устройства может измениться после перезапуска аудиосессии;
без явного номера клиент использует системное устройство по умолчанию.

Во время исследования были загружены, но больше не используются, кэши
Parakeet (около 2,4 ГБ) и Qwen GGUF (около 1,4 ГБ). Их можно удалить вручную,
если нужно освободить диск; текущий профиль их не читает.

## Проверка исходников

```bash
cd /home/random/dev/huggingvoice
./scripts/bootstrap-local.sh
uv run pytest -q tests/test_local_bootstrap_config.py \
  tests/test_acoustic_loopback_smoke.py \
  tests/test_cli_defaults.py tests/test_language_prompt.py
NO_PROXY="${NO_PROXY:+${NO_PROXY},}10.255.255.1" uv run pytest -q
bash -n scripts/bootstrap-local.sh scripts/run-omniroute-ru-en.sh
git diff --check
```

Адрес `10.255.255.1` добавлен в `NO_PROXY` только для upstream-теста
недоступного сервера: глобальный HTTP proxy иначе меняет ожидаемый код ответа
из `502` в `503`.
