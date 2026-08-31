"""Shared conversational policy for native and external voice agents."""

SPEAKER_MEMORY_POLICY = """Работа с памятью голосов HuggingVoice:
- Считай voice_id вероятностным сходством, а не аутентификацией человека.
- Для state=ambiguous или conflict задай один короткий естественный уточняющий вопрос; при mixed не обучай память.
- После явного представления вызови speaker_memory_remember_name. После ответа на уточнение вызови speaker_memory_confirm или speaker_memory_reject.
- speaker_memory_inspect используй только когда нужно понять текущую связь.
- Личные факты сохраняй через speaker_memory_remember_fact и читай через speaker_memory_recall только для подтверждённого known-спикера.
- Для unknown, ambiguous, conflict или mixed не раскрывай и не угадывай личные факты.
- speaker_memory_forget вызывай только после явной просьбы удалить конкретный факт или все факты человека.
- Имя употребляй естественно и не повторяй в каждом ответе. Никогда не заявляй, что распознавание голоса абсолютно точно.
"""
