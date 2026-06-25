# Суммаризация (Notebook Guide)

Этот репозиторий содержит ноутбук `суммаризация.ipynb` с несколькими стратегиями суммаризации текста на базе `GigaChat` и локальных цепочек/инструментов из `src/`.

## Что делает ноутбук

В ноутбуке реализованы сценарии:

1. Двухэтапная суммаризация (`extract -> synthesize`).
2. `Chain-of-Density` суммаризация.
3. Многовекторная суммаризация.
4. Контролируемая абстракция.
5. Суммаризация больших документов через `large_document_summarize` с выбором стратегии и чанкованием.

## Импорты из ноутбука

```python
import os
from pathlib import Path

from dotenv import load_dotenv
from rich import print
from langchain_gigachat import GigaChat

from src.chains import Extract_PROMPT, SYNTHESIZE_PROMPT
from src.tools import (
    chain_of_density_summarize,
    multi_vector_summarize,
    controlled_abstraction_summarize,
    large_document_summarize,
)
```

## Настройка окружения

Ноутбук ожидает переменную окружения:

- `GIGACHAT_API_KEY`


## Конфигурация модели

Используется `GigaChat` со следующими параметрами:

- `model='GigaChat-2-Max'`
- `scope='GIGACHAT_API_CORP'`
- `temperature=0.87`
- `verify_ssl_certs=False`
- `profanity_check=False`
- `max_tokens=25000`
- `timeout=300`

## Данные

Исходный текст читается из файла:

- `text.md`


## Логика по стратегиям

### 1) Two-stage

- Создаются цепочки:
  - `extract_chain = Extract_PROMPT | llm`
  - `synthesize_chain = SYNTHESIZE_PROMPT | llm`
- Выполняется извлечение ключевых фрагментов, затем синтез итогового саммари.

### 2) Chain-of-Density

- Вызывается `chain_of_density_summarize.invoke({"text": text})`.

### 3) Multi-vector

- Вызывается `multi_vector_summarize.invoke({"text": text})`.

### 4) Controlled abstraction

- Вызывается `controlled_abstraction_summarize.invoke({"text": text})`.

### 5) Large document summarize

- Для длинных текстов используется `large_document_summarize.invoke(...)`.
- В ноутбуке протестированы стратегии:
  - `controlled_abstraction`
  - `chain_of_density`
  - `two_stage`
  - `multi_vector`
- Параметры вызова:
  - `chunk_size=10000`
  - `chunk_overlap=200`
  - `max_levels=2`
  - `include_debug=True`

## Быстрый запуск

1. Установить зависимости проекта.
2. Указать `GIGACHAT_API_KEY` в `.env` или окружении.
3. Убедиться, что есть `text.md`.
4. Открыть и запустить `суммаризация.ipynb` по ячейкам.

