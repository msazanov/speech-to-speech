# Локальный HuggingVoice: быстрый русский голосовой профиль

Рабочий профиль держит речевые модели на CPU, а LLM получает через стабильный
OpenAI-compatible endpoint FreeToken:

- VAD: Silero VAD + Smart Turn, CPU;
- STT: GigaAM Multilingual CTC ONNX INT8, CPU, 6 потоков;
- Speaker memory: подготовленный CAM++/3D-Speaker ONNX, CPU, 1 поток, локальная SQLite;
- LLM: FreeToken arbiter `http://127.0.0.1:1919/v1`, модель `gemma-4-e2b`;
- TTS: выбираемые из браузера Silero `v5_5_ru` и RHVoice, CPU; Supertonic загружается лениво только для английского Silero fallback;
- Realtime API: `127.0.0.1:8765`.

Русский — язык по умолчанию. Явно английская реплика получает английский ответ и
английский TTS fallback. Код и отдельные английские термины внутри русской фразы
не переключают язык.

## Установка

```bash
cd /home/random/dev/huggingvoice
./scripts/bootstrap-local.sh
```

Скрипт создаёт Python 3.12 окружение из `uv.lock` и устанавливает extras
`gigaam`, `silero`, `supertonic` и `speaker-memory`. Затем отдельно загрузите
проверенную ONNX-модель:

```bash
./scripts/fetch-speaker-memory-model.sh
./scripts/fetch-rhvoice-local.sh
```

Скрипт проверяет SHA-256 до атомарной установки и не перезаписывает существующий
файл с неверной контрольной суммой. Активный речевой тракт не использует Faster Whisper.

## Зависимость от FreeToken arbiter

HuggingVoice не запускает и не останавливает LLM. FreeToken владеет портом
`1919`, очередью и фактической загрузкой моделей. Сейчас интерфейс намеренно
ограничен моделью `gemma-4-e2b`; список допустимых id задаётся
`S2S_LLM_MODELS` и должен совпадать с `GET /v1/models`.

Во время разговора интерфейс показывает этап подключения, ожидания модели,
первого токена и синтеза. После ответа он показывает клиентские `TTFT` и
полное время; токены отображаются, если FreeToken передал стандартный
`response.usage`.

Source-controlled unit HuggingVoice находится в `deploy/systemd/`. После
отдельного разрешения установите и запустите только речевой сервис:

```bash
cd /home/random/dev/huggingvoice
./scripts/activate-voice-stack.sh
```

Не выполняйте эти команды, пока владелец FreeToken arbiter не подтвердит
готовность endpoint и не будет дан отдельный сигнал на установку/перезапуск.

При каждом `session.update` Chat Completions backend ставит в очередь один
непотоковый prefill-запрос с `max_tokens=1` и тайм-аутом 5 секунд. Он дебаунсится (75 мс), не меняет
историю диалога и ограничен общим provider-worker, поэтому не создаёт бесконечных
повторов и прогревает prompt/tools prefix до первой реплики.

### Размышление по необходимости

Профиль передаёт `responses_api_thinking_mode=auto`: совместимый провайдер получает
`chat_template_kwargs.thinking_mode=adaptive` и сам выбирает, открывать ли канал
размышления в конкретном запросе. HuggingVoice никогда не отправляет содержимое
этого канала в TTS или историю. Если провайдер действительно начал reasoning,
пользователь один раз слышит короткое «Сейчас надо подумать» (для английской
реплики — `Give me a moment to think.`), затем обычный ответ.

Фразы берутся из локальной библиотеки из 50 русских шаблонов (плюс английские
варианты) и не повторяются подряд. Восемь русских вариантов содержат лёгкую
брань и выключены по умолчанию; для личного запуска их можно включить через
`responses_api_thinking_ack_allow_profanity=true`.

Текущий FreeToken GPU-процесс запущен с `--reasoning-parser off`, поэтому его
`adaptive` сейчас фактически означает быстрый chat-режим; это безопасно, но не
включает скрытое reasoning. Для настоящего выбора Gemma на стороне модели нужно
включить reasoning parser/auto в FreeToken, не меняя HuggingVoice. При случайном
возврате сырых Gemma thought-маркеров клиентский splitter удалит их и не даст
озвучить внутренний текст.

Проверка состояния:

```bash
systemctl --user status huggingvoice.service
curl -fsS http://127.0.0.1:1919/v1/models | jq
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8765/docs
```

Речевой сервис работает с `HF_HUB_OFFLINE=1`: все активные модели должны быть
скачаны при bootstrap/первичной проверке. Это исключает сетевые проверки HF и
Torch cache во время старта. Проверка NLTK ищет POS-tagger в правильном каталоге
`taggers/`, поэтому уже установленный ресурс тоже не вызывает сеть.

## Разговор

```bash
uv run speech-to-speech talk \
  --url ws://127.0.0.1:8765/v1/realtime \
  --echo-cancel
```

AEC использует CPU только во время фактического воспроизведения TTS. Между
репликами исходный микрофонный поток идёт напрямую; при overlap синхронная пара
`playback + mic` очищается вне PortAudio callback. Микрофон не выключается, так
что перебивание не теряется.

Если устройство по умолчанию выбрано неверно, выполните `uv run python -m
sounddevice` и передайте номера через `--input-device` и `--output-device`.

## Живой журнал

Обычный journal без оформления:

```bash
journalctl --user -u huggingvoice.service -n 0 -f -o cat
```

Человекочитаемый просмотрщик с цветами этапов и стабильным отдельным цветом для
каждого `voice_id`/подтверждённого имени:

```bash
python scripts/watch_voice_log.py --service huggingvoice.service --history 50
```

По умолчанию журнал скрывает содержание разговора. Для диагностического запуска
добавьте явный `--log-transcripts`, например
`speech-to-speech serve config/omniroute-ru-en.json --log-transcripts`: тогда journal содержит STT-текст, полный
текстовый prompt к LLM и видимый ответ, а также STT/LLM/TTS/AEC timing, токены и
решения маршрутизатора. Аудио PCM, изображения, API-ключи и скрытое reasoning не
записываются. Такой journal следует считать приватным.

Если голос ошибочно попал в blacklist и его краткоживущая ссылка уже истекла,
оператор может восстановить его напрямую локальной командой (этот путь не
экспонируется LLM):

```bash
huggingvoice-speaker-memory-admin \
  --database ~/.local/share/huggingvoice/speaker-memory.sqlite3 \
  unblock v_ID_ИЗ_ЦВЕТНОГО_ЛОГА
```

## Замкнутый акустический тест

```bash
uv run python scripts/acoustic_loopback_smoke.py \
  --text "Ты меня хорошо слышишь и правильно понимаешь?" \
  --input-device 15 \
  --output-device 15 \
  --json
```

По умолчанию тест воспроизводит эталон на 85% громкости. Для ночных проверок
передайте `--volume 0.2` явно.

Проверенный прогон 30 августа 2026 года:

- распознано: `ты меня хорошо слышишь и правильно понимаешь`;
- WER: 0%, символьное сходство: 100%;
- конец речи → транскрипция: 0.359 с;
- транскрипция → первый аудиоблок: 0.659 с;
- ответ: `Да, я вас хорошо слышу и понимаю. Чем я могу вам помочь?`;
- полный ответ завершён через 0.899 с после транскрипции.

Прямой запрос `Ты меня слышишь?` к Gemma вернул `Да, я вас слышу.` за 0.176 с.
Результаты акустического теста зависят от громкости, помещения и выбранных
PipeWire-устройств.

## Ключевые параметры производительности

- Chat Completions: `max_tokens=64`, `temperature=0.2`, thinking policy `auto`
  (adaptive, если её поддерживает провайдер), timeout 120 с, SDK retries 0;
  prefill — 1 токен;
- GigaAM: `CPUExecutionProvider`, INT8, 6 потоков;
- Silero/RHVoice: 24 кГц синтез с преобразованием в 16 кГц блоками по 512 samples;
- VAD: `min_silence_ms=500`, live transcription включена с интервалом 250 мс.

HuggingVoice держит STT/VAD/TTS на CPU и не управляет размещением LLM в RAM/VRAM.
Speaker embedding также жёстко использует CPU и один поток. Сырые аудиозаписи не
сохраняются; SQLite содержит биометрические центроиды, имена и личные факты и
должна считаться приватной. Факты доступны только при состоянии `known`.

В source-controlled профиле `speaker_memory_enabled=true`: каждый пригодный
финальный сегмент получает стабильный внутренний `voice_id`, а близкий новый
кластер может получить кандидата на уже известного человека (`conflict/clarify`),
без автоматического доступа к фактам. После явного имени или подтверждения
несколько акустических кластеров связываются с одним `person_id`. После явного
имени последние 32 кластера той же беседы дополнительно сравниваются по
максимальному cosine между центроидом и сохранёнными прототипами; при сходстве
от 0.70 незаписанный кластер объединяется с названным. Кластер с положительной
уликой на другого человека не объединяется автоматически. Короткие, тихие и
ошибочные сегменты проходят дальше с явной метаинформацией
`state=unknown, recommendation=do_not_learn`.

В LLM уходит только компактный доверенный контекст (имя показывается только
при состоянии `known`, чтобы неоднозначный кандидат не выглядел подтверждённым):
`<huggingvoice_speaker_context>{"voice":"3138d446","name":"Марат"}</huggingvoice_speaker_context>`.
Публичный `voice` — первые 8 hex-символов (4 байта) внутреннего ID; SQLite
хранит прежние полные IDs и короткоживущую привязку текущей беседы. Все memory
tools принимают только `voice`; backend сам разрешает его в приватный
`speaker_ref`, а model-supplied `person_id`/raw IDs отбрасываются. Мутации
возвращают `{voice,name}` и не запускают второй LLM-turn; `recall` добавляет
только запрошенные факты. В браузере вместо `You` показывается имя или короткий
`voice`, цвет закрепляется за этим идентификатором.

## Проверка памяти голосов

Интерактивный тест сначала произносит «Для теста ответов скажите ДА» и 30 секунд
слушает реальный микрофон. При распознанном отдельном слове «да» он отвечает
«Пизда, тест пройден». Если подтверждения нет, только после таймаута подаётся
синтетическое «Да», затем CAMPPlus сравнивает по две реплики трёх разделимых
голосов Silero (`xenia`, `baya`, `aidar`):

```bash
PYTHONPATH=src python scripts/synthetic_speaker_memory_smoke.py \
  --input-device 20 \
  --output-device 20 \
  --timeout 30 \
  --output-dir /tmp/huggingvoice-synthetic-speakers \
  --json
```

JSON явно указывает `challenge.source=microphone` или
`challenge.source=synthetic_timeout`, матрицу cosine similarity, назначенные
`voice_id` и CPU-время атрибуции. WAV-файлы остаются в `--output-dir`.

Подготовьте минимум по две отдельные русские записи каждого из двух людей и выполните:

```bash
uv run python scripts/speaker_memory_smoke.py \
  --model models/speaker-memory/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx \
  --sample 'Аркадий=/path/arkady-1.wav' \
  --sample 'Аркадий=/path/arkady-2.wav' \
  --sample 'Андрей=/path/andrey-1.wav' \
  --sample 'Андрей=/path/andrey-2.wav' \
  --json
```

Отчёт завершится успешно только если каждый человек представлен минимум двумя
записями, голоса разделены правильно, подтверждение усиливает связь, отказ её
ослабляет, неоднозначная проба не сдвигает центроиды, а тёплый p95 для
embedding+clustering не превышает 100 мс. Пороги предварительные; включайте
`speaker_memory_enabled` только после успешного отчёта на конкретных микрофонах и
реальных голосах.

## Проверка исходников

```bash
uv run pytest -q tests/test_local_arbiter_integration.py tests/test_acoustic_loopback_smoke.py
uv run ruff check src/speech_to_speech/LLM \
  src/speech_to_speech/arguments_classes/responses_api_language_model_arguments.py \
  tests/test_local_arbiter_integration.py tests/test_acoustic_loopback_smoke.py
bash -n scripts/*.sh
systemd-analyze --user verify deploy/systemd/*.service
git diff --check
```
