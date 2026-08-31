# Локальный HuggingVoice: быстрый русский голосовой профиль

Рабочий профиль держит речь на CPU и использует GPU только для компактной LLM:

- VAD: Silero VAD + Smart Turn, CPU;
- STT: GigaAM Multilingual CTC ONNX INT8, CPU, 6 потоков;
- LLM: Gemma 4 E2B QAT Q4_0 через `llama.cpp`, контекст 4096, reasoning отключён;
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
`gigaam`, `silero` и `supertonic`. Активный речевой тракт не использует Faster Whisper.

Локальная LLM ожидается здесь:

```text
/home/random/dev/huggingvoice-llm-bench/models/gemma4-e2b/gemma-4-E2B_q4_0-it.gguf
SHA256 fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634
```

## Сервисы и переключение с Ornith

Units находятся в `deploy/systemd/` и устанавливаются symlink-ами в
`~/.config/systemd/user/`:

```bash
mkdir -p ~/.config/systemd/user
ln -sfn /home/random/dev/huggingvoice/deploy/systemd/huggingvoice-gemma.service \
  ~/.config/systemd/user/huggingvoice-gemma.service
ln -sfn /home/random/dev/huggingvoice/deploy/systemd/huggingvoice.service \
  ~/.config/systemd/user/huggingvoice.service
systemctl --user daemon-reload
```

Включить голосовой профиль:

```bash
cd /home/random/dev/huggingvoice
./scripts/activate-voice-stack.sh
```

Команда останавливает и отключает автозапуск большого Ornith, затем запускает
Gemma и HuggingVoice. Исходный unit Ornith сохраняется. Одновременно держать обе
LLM нельзя: RTX 2070 имеет 8 ГБ VRAM.

Вернуть Ornith:

```bash
./scripts/restore-ornith.sh
```

Проверка состояния:

```bash
systemctl --user status huggingvoice-gemma.service huggingvoice.service
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
  --block-mic-during-playback
```

Если устройство по умолчанию выбрано неверно, выполните `uv run python -m
sounddevice` и передайте номера через `--input-device` и `--output-device`.

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

- Gemma: `--ctx-size 4096`, `--gpu-layers all`, `--parallel 1`, Flash Attention,
  `--reasoning off`, `--reasoning-budget 0`;
- Chat Completions: `max_tokens=64`, `temperature=0.2`, thinking отключён через
  `chat_template_kwargs`;
- GigaAM: `CPUExecutionProvider`, INT8, 6 потоков;
- Silero: 24 кГц синтез с преобразованием в 16 кГц блоками по 512 samples;
- VAD: `min_silence_ms=500`, live transcription отключена.

На проверенном хосте Gemma использует около 1.6 ГБ VRAM; HuggingVoice держит
STT/VAD/TTS на CPU и не резервирует VRAM.

## Проверка исходников

```bash
uv run pytest -q
uv run ruff check src tests
bash -n scripts/*.sh
systemd-analyze --user verify deploy/systemd/*.service
git diff --check
```
