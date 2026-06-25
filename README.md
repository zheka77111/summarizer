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

## Автоклассификация и автовыбор стратегии

В `src/tools.py` добавлены инструменты:

- `classify_document_type` — LLM-классификатор типа документа.
- `auto_summarize_by_doc_type` — автоклассификация + автоматический выбор стратегии суммаризации.

Поддерживаемые типы документа:

- `business_report`
- `research_article`
- `technical_documentation`
- `policy_or_regulation`
- `educational_or_book`
- `interview_or_meeting`
- `news_or_article`
- `narrative`
- `mixed_or_other`

Маппинг типа в стратегию:

- `business_report` -> `chain_of_density`
- `research_article` -> `two_stage`
- `technical_documentation` -> `controlled_abstraction`
- `policy_or_regulation` -> `controlled_abstraction`
- `educational_or_book` -> `multi_vector`
- `interview_or_meeting` -> `multi_vector`
- `news_or_article` -> `two_stage`
- `narrative` -> `multi_vector`
- `mixed_or_other` -> `two_stage`

Для больших текстов классификатор автоматически делит текст на чанки, классифицирует чанки и агрегирует итоговый тип.

Пример использования:

```python
from src.tools import auto_summarize_by_doc_type

result = auto_summarize_by_doc_type.invoke({
    "text": text,
    "large_doc_threshold": 14000,
    "classification_chunk_size": 12000,
    "classification_chunk_overlap": 400,
    "classification_max_chunks": 8,
    "chunk_size": 12000,
    "chunk_overlap": 800,
    "max_levels": 4,
    "include_debug": False,
})
print(result)
```
