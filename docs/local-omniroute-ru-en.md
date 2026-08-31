# Локальный HuggingVoice: быстрый русский голосовой профиль

Рабочий профиль держит речевые модели на CPU, а LLM получает через стабильный
OpenAI-compatible endpoint FreeToken:

- VAD: Silero VAD + Smart Turn, CPU;
- STT: GigaAM Multilingual CTC ONNX INT8, CPU, 6 потоков;
- Speaker memory: подготовленный CAM++/3D-Speaker ONNX, CPU, 1 поток, локальная SQLite;
- LLM: FreeToken arbiter `http://127.0.0.1:1919/v1`, модель `gemma-4-e2b`;
- TTS: Silero `v5_5_ru`, голос `xenia`, CPU; Supertonic загружается лениво только для английского;
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
```

Скрипт проверяет SHA-256 до атомарной установки и не перезаписывает существующий
файл с неверной контрольной суммой. Активный речевой тракт не использует Faster Whisper.

## Зависимость от FreeToken arbiter

HuggingVoice не запускает, не останавливает и не переключает LLM. FreeToken
владеет портом `1919`, очередью и переключением моделей. Профиль отправляет
`model=gemma-4-e2b`, ждёт один запрос до 60 секунд и отключает SDK retries,
чтобы timeout не создавал повторный элемент в FIFO.

Source-controlled unit HuggingVoice находится в `deploy/systemd/`. После
отдельного разрешения установите и запустите только речевой сервис:

```bash
cd /home/random/dev/huggingvoice
./scripts/activate-voice-stack.sh
```

Не выполняйте эти команды, пока владелец FreeToken arbiter не подтвердит
готовность endpoint и не будет дан отдельный сигнал на установку/перезапуск.

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

- Chat Completions: `max_tokens=64`, `temperature=0.2`, thinking отключён через
  `chat_template_kwargs`, timeout 60 с, SDK retries 0;
- GigaAM: `CPUExecutionProvider`, INT8, 6 потоков;
- Silero: 24 кГц синтез с преобразованием в 16 кГц блоками по 512 samples;
- VAD: `min_silence_ms=500`, live transcription отключена.

HuggingVoice держит STT/VAD/TTS на CPU и не управляет размещением LLM в RAM/VRAM.
Speaker embedding также жёстко использует CPU и один поток. Сырые аудиозаписи не
сохраняются; SQLite содержит биометрические центроиды, имена и личные факты и
должна считаться приватной. Факты доступны только при состоянии `known`.

В source-controlled профиле `speaker_memory_enabled=false`. Это намеренный
activation gate: память голосов нельзя включать до успешного теста ниже на двух
реальных людях и целевом компьютере. Остальной голосовой конвейер работает без неё.

## Проверка памяти голосов

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
uv run pytest -q tests/test_local_bootstrap_config.py tests/test_local_arbiter_integration.py
uv run ruff check src/speech_to_speech/LLM \
  src/speech_to_speech/arguments_classes/responses_api_language_model_arguments.py \
  tests/test_local_bootstrap_config.py tests/test_local_arbiter_integration.py
bash -n scripts/*.sh
systemd-analyze --user verify deploy/systemd/*.service
git diff --check
```
