from typing import List
import json
from collections import Counter

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from src.chains import Extract_PROMPT, SYNTHESIZE_PROMPT
from src.llm import llm
from langchain_core.output_parsers import StrOutputParser


class SummarizeInput(BaseModel):
    text: str = Field(..., description="Полный текст статьи для суммаризации")

@tool("two_stage_summarize", args_schema=SummarizeInput)
def two_stage_summarize(text: str) -> str:
    """Двухэтапная суммаризация: экстракция -> абстрактивный синтез."""
    parser = StrOutputParser()

    extract_chain = Extract_PROMPT | llm | parser
    synth_chain = SYNTHESIZE_PROMPT | llm | parser

    extracted_fragments = extract_chain.invoke({"text": text})
    final_summary = synth_chain.invoke({"extracted_fragments": extracted_fragments})

    return final_summary




# ---------- Prompts ----------
ITER_0_PROMPT = PromptTemplate.from_template("""
Ты эксперт по суммаризации деловых отчетов.

Итерация 0:
1. Прочитай отчет:
{text}
2. Определи:
- основную тему,
- ключевые разделы отчета (кратко, списком).

Выведи:
ТЕМА: ...
РАЗДЕЛЫ:
1) ...
2) ...
""")

ITER_1_PROMPT = PromptTemplate.from_template("""
Ты используешь метод Chain-of-Density.

Вход:
- Тема и разделы:
{outline}
- Исходный отчет:
{text}

Итерация 1:
Создай первичное саммари объемом 300-350 слов.
Требования:
- сохранить структуру ключевых разделов,
- покрыть ключевые пункты каждого раздела,
- без выдуманных фактов.

Выведи только саммари.
""")

ITER_2_PROMPT = PromptTemplate.from_template("""
Ты используешь метод Chain-of-Density.

Итерация 2:
Проанализируй первичное саммари:
{summary_v1}

Сделай внутреннюю проверку на:
- предложения, которые можно сделать более информативными,
- избыточность/повторы,
- места, где пункты можно объединить без потери смысла.

Сгенерируй второе саммари объемом 200-250 слов:
- более плотное по информации,
- без повторов,
- без потери ключевых фактов.

Выведи только саммари.
""")

ITER_3_PROMPT = PromptTemplate.from_template("""
Ты используешь метод Chain-of-Density.

Итерация 3:
Проанализируй второе саммари:
{summary_v2}

Определи и исправь:
- фразы, которые можно сделать более информационно-плотными,
- информацию, которую можно сжать без потери смысла,
- второстепенные детали, которые можно опустить.

Сгенерируй финальное саммари объемом 150-180 слов
с максимальной информационной плотностью и связностью.

Выведи только саммари.
""")

ITER_4_VERIFY_PROMPT = PromptTemplate.from_template("""
Ты используешь Chain-of-Verification для финальной проверки.

Проверь финальное саммари:
{summary_v3}

Относительно исходного отчета:
{text}

Проверка:
- фактическая точность,
- полнота ключевых пунктов,
- информационная ценность,
- отсутствие лишней избыточности.

Если нужны правки, внеси их.
Итог: выведи только финальное саммари (150-180 слов).
""")


# ---------- Tool ----------
class DenseSummarizeInput(BaseModel):
    text: str = Field(..., description="Полный текст делового отчета")


@tool("chain_of_density_summarize", args_schema=DenseSummarizeInput)
def chain_of_density_summarize(text: str) -> str:
    """Итеративная суммаризация Chain-of-Density с устранением избыточности на каждом шаге."""
    parser = StrOutputParser()

    iter0_chain = ITER_0_PROMPT | llm | parser
    iter1_chain = ITER_1_PROMPT | llm | parser
    iter2_chain = ITER_2_PROMPT | llm | parser
    iter3_chain = ITER_3_PROMPT | llm | parser
    iter4_chain = ITER_4_VERIFY_PROMPT | llm | parser

    outline = iter0_chain.invoke({"text": text})
    summary_v1 = iter1_chain.invoke({"outline": outline, "text": text})
    summary_v2 = iter2_chain.invoke({"summary_v1": summary_v1})
    summary_v3 = iter3_chain.invoke({"summary_v2": summary_v2})
    final_summary = iter4_chain.invoke({"summary_v3": summary_v3, "text": text})

    return final_summary



# ---------- Schemas ----------
class VectorsSchema(BaseModel):
    vectors: List[str] = Field(
        ..., description="Ровно 5 ключевых векторов анализа книги"
    )


class VectorAnalysisSchema(BaseModel):
    vector: str
    mini_summary: str = Field(
        ..., description="4-6 предложений только по этому вектору"
    )
    key_theses: List[str] = Field(
        ..., description="3-5 ключевых тезисов по вектору"
    )


class MultiVectorInput(BaseModel):
    text: str = Field(..., description="Полный текст книги")


# ---------- Parsers ----------
vectors_parser = JsonOutputParser(pydantic_object=VectorsSchema)
vector_analysis_parser = JsonOutputParser(pydantic_object=VectorAnalysisSchema)
text_parser = StrOutputParser()


# ---------- Prompts ----------

VECTORS_PROMPT = PromptTemplate.from_template("""
Ты делаешь многовекторную саммаризацию.

Текст:
{text}

Шаг 1. Определи характеристики текста:
- format: жанр/формат (книга, статья, отчет, учебный материал, интервью и т.д.)
- domain: предметная область (бизнес, финансы, маркетинг, IT, медицина и т.д.)
- intent: основная цель текста (обучить, убедить
, проинформировать, развлечь и т.д.)
Определи ровно 5 ключевых векторов анализа.
Формат ответа строго JSON:
{format_instructions}
""")

VECTOR_SUMMARY_PROMPT = PromptTemplate.from_template("""
Ты анализируешь только один вектор книги.

Вектор:
{vector}

Текст:
{text}

Сделай:
1) mini_summary: 4-6 предложений только по этому вектору
2) key_theses: 3-5 тезисов

Формат ответа строго JSON:
{format_instructions}
""")

MAP_AND_FINAL_PROMPT = PromptTemplate.from_template("""
Собери итог многовекторной саммаризации на основе результатов по векторам.

Результаты по векторам:
{vector_results}

Сформируй ответ в Markdown со структурой:

## Многовекторная карта
- Ядро концепции текста (1-2 предложения)
- Радиальные векторы (каждый вектор + его ключевые тезисы)
- Взаимосвязи между тезисами разных векторов (если есть)

## Интегрированное саммари
250-300 слов о том, как векторы взаимодействуют и дополняют друг друга.
""")


# ---------- Tool ----------
@tool("multi_vector_summarize", args_schema=MultiVectorInput)
def multi_vector_summarize(text: str) -> str:
    """Многовекторная саммаризация: 5 векторов, мини-саммари по каждому, карта и интегрированное саммари."""
    # 1) Определяем 5 векторов
    vectors_chain = VECTORS_PROMPT | llm | vectors_parser
    vectors_obj = vectors_chain.invoke({
        "text": text,
        "format_instructions": vectors_parser.get_format_instructions(),
    })
    vectors = vectors_obj["vectors"][:5]

    # 2) Параллельная саммаризация по вектору (batch => параллельные LLM-вызовы)
    vector_chain = VECTOR_SUMMARY_PROMPT | llm | vector_analysis_parser
    batch_payloads = [
        {
            "vector": v,
            "text": text,
            "format_instructions": vector_analysis_parser.get_format_instructions(),
        }
        for v in vectors
    ]
    vector_results = vector_chain.batch(batch_payloads)

    # 3) Карта + интегрированное саммари
    final_chain = MAP_AND_FINAL_PROMPT | llm | text_parser
    final_output = final_chain.invoke({"vector_results": vector_results})

    return final_output

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# ---------- Prompts ----------
ANALYZE_PROMPT = PromptTemplate.from_template("""
Ты делаешь первичный анализ текста для контролируемой абстракции.

Текст:
{text}

Задача:
1) Разбей текст на информационные блоки.
2) Оцени каждый блок по шкале 1-10:
- 10 = критически важно, сохранить почти дословно
- 7-9 = важно, сохранить в сжатой форме
- 4-6 = второстепенно, можно сильно обобщить
- 1-3 = малозначимо, можно опустить

Верни в формате:
БЛОКИ:
1) [оценка: X] ...
2) [оценка: X] ...
...
""")

SUMMARIZE_3_LEVELS_PROMPT = PromptTemplate.from_template("""
На основе анализа создай трехуровневое саммари.

Исходный текст:
{text}

Анализ блоков:
{analysis}

Требования:
- Уровень 1 (максимальное сжатие): 2-3 предложения, только блоки 9-10.
- Уровень 2 (средний): 1 абзац, блоки 7-10.
- Уровень 3 (детализированный): 2-3 абзаца, блоки 4-10.

Верни строго в формате:
УРОВЕНЬ 1:
...

УРОВЕНЬ 2:
...

УРОВЕНЬ 3:
...
""")

DENSITY_EVAL_PROMPT = PromptTemplate.from_template("""
Оцени информационную плотность каждого уровня саммари.

Исходный текст:
{text}

Анализ блоков:
{analysis}

Трехуровневое саммари:
{summaries}

Для каждого уровня оцени, какой процент ключевых фактов/идей оригинала сохранен.
Дай краткое обоснование (1-2 предложения на уровень).

Формат:
ПЛОТНОСТЬ:
- Уровень 1: XX% — ...
- Уровень 2: XX% — ...
- Уровень 3: XX% — ...
""")

FINAL_FORMAT_PROMPT = PromptTemplate.from_template("""
Собери финальный отчет из материалов ниже.

Анализ:
{analysis}

Саммари:
{summaries}

Оценка плотности:
{density}

Оформи ответ структурно в Markdown:
## 1) Первичный анализ
## 2) Трехуровневое саммари
## 3) Информационная плотность
""")


# ---------- Chains ----------
parser = StrOutputParser()

analyze_chain = ANALYZE_PROMPT | llm | parser
summaries_chain = SUMMARIZE_3_LEVELS_PROMPT | llm | parser
density_chain = DENSITY_EVAL_PROMPT | llm | parser
final_chain = FINAL_FORMAT_PROMPT | llm | parser


# ---------- Tool ----------
class AbstractionSummarizeInput(BaseModel):
    text: str = Field(..., description="Текст для саммаризации")


@tool("controlled_abstraction_summarize", args_schema=AbstractionSummarizeInput)
def controlled_abstraction_summarize(text: str) -> str:
    """Контролируемая саммаризация с оценкой блоков, 3 уровнями абстракции и оценкой плотности."""
    analysis = analyze_chain.invoke({"text": text})
    summaries = summaries_chain.invoke({"text": text, "analysis": analysis})
    density = density_chain.invoke({"text": text, "analysis": analysis, "summaries": summaries})
    final = final_chain.invoke({"analysis": analysis, "summaries": summaries, "density": density})
    return final


from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter



MERGE_LEVEL_PROMPT = PromptTemplate.from_template("""
Ты объединяешь частичные саммари одного уровня и сжимаешь их без потери ключевых фактов.

Уровень иерархии: {level}

Частичные саммари:
{chunk_summaries}

Требования:
- Удали дубли между частями.
- Сохрани имена, числа, даты, причинно-следственные связи.
- Сформируй единое связное саммари в markdown.
- Не добавляй фактов, которых нет во входе.
""")


def split_long_text(
    text: str,
    *,
    chunk_size: int = 12000,
    chunk_overlap: int = 800,
) -> list[str]:
    """Разбивает длинный текст на устойчивые чанки с перекрытием."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    return text_splitter.split_text(text)


def _invoke_base_tool(tool_name: str, text: str) -> str:
    tools = {
        'two_stage': two_stage_summarize,
        'chain_of_density': chain_of_density_summarize,
        'multi_vector': multi_vector_summarize,
        'controlled_abstraction': controlled_abstraction_summarize,
    }

    if tool_name not in tools:
        raise ValueError(
            f'Неизвестная стратегия: {tool_name}. Доступно: {", ".join(tools.keys())}'
        )

    result = tools[tool_name].invoke({'text': text})
    return result if isinstance(result, str) else str(result)


def recursive_large_doc_summary(
    text: str,
    *,
    strategy: str = 'two_stage',
    chunk_size: int = 12000,
    chunk_overlap: int = 800,
    max_levels: int = 4,
) -> tuple[str, list[dict]]:
    """Иерархическая суммаризация: summarize chunks -> merge -> recurse."""
    if max_levels < 1:
        raise ValueError('max_levels должен быть >= 1')

    merge_chain = MERGE_LEVEL_PROMPT | llm | StrOutputParser()
    level_stats: list[dict] = []
    current_text = text.strip()

    if not current_text:
        return '', level_stats

    for level in range(1, max_levels + 1):
        if len(current_text) <= chunk_size:
            final_summary = _invoke_base_tool(strategy, current_text)
            level_stats.append(
                {
                    'level': level,
                    'chunks': 1,
                    'input_chars': len(current_text),
                    'output_chars': len(final_summary),
                    'terminal': True,
                }
            )
            return final_summary, level_stats

        chunks = split_long_text(
            current_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunk_summaries = []
        for idx, chunk in enumerate(chunks, start=1):
            summary = _invoke_base_tool(strategy, chunk)
            chunk_summaries.append(
                f'### Фрагмент {idx}\n[chars={len(chunk)}]\n{summary.strip()}'
            )

        merged_input = '\n\n'.join(chunk_summaries)
        merged_summary = merge_chain.invoke(
            {'level': level, 'chunk_summaries': merged_input}
        )

        level_stats.append(
            {
                'level': level,
                'chunks': len(chunks),
                'input_chars': len(current_text),
                'output_chars': len(merged_summary),
                'terminal': False,
            }
        )

        current_text = merged_summary.strip()

    # fail-safe: если лимит уровней достигнут, вернем финальное уплотнение выбранным инструментом
    forced_final = _invoke_base_tool(strategy, current_text)
    level_stats.append(
        {
            'level': max_levels + 1,
            'chunks': 1,
            'input_chars': len(current_text),
            'output_chars': len(forced_final),
            'terminal': True,
            'forced_final': True,
        }
    )
    return forced_final, level_stats


class LargeDocumentSummarizeInput(BaseModel):
    text: str = Field(..., description='Длинный исходный текст для суммаризации')
    strategy: Literal[
        'two_stage',
        'chain_of_density',
        'multi_vector',
        'controlled_abstraction',
    ] = Field(default='two_stage', description='Базовый инструмент суммаризации для каждого чанка')
    chunk_size: int = Field(default=12000, description='Максимальный размер чанка в символах')
    chunk_overlap: int = Field(default=800, description='Перекрытие между чанками в символах')
    max_levels: int = Field(default=4, description='Максимальная глубина рекурсивного merge')
    include_debug: bool = Field(default=False, description='Добавить статистику уровней в результат')




@tool('large_document_summarize', args_schema=LargeDocumentSummarizeInput)
def large_document_summarize(
    text: str,
    strategy: Literal[
        'two_stage',
        'chain_of_density',
        'multi_vector',
        'controlled_abstraction',
    ] = 'two_stage',
    chunk_size: int = 12000,
    chunk_overlap: int = 800,
    max_levels: int = 4,
    include_debug: bool = False,
) -> str:
    """Суммаризация длинных документов с рекурсивным разбиением и объединением результатов."""



    final_summary, stats = recursive_large_doc_summary(
        text=text,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_levels=max_levels,
    )

    if not include_debug:
        return final_summary

    debug_lines = ['## Debug Levels']
    for row in stats:
        debug_lines.append(
            '- level={level}, chunks={chunks}, input_chars={input_chars}, output_chars={output_chars}, terminal={terminal}'.format(**row)
        )

    return final_summary + '\n\n' + '\n'.join(debug_lines)


DocumentTypeLiteral = Literal[
    'business_report',
    'research_article',
    'technical_documentation',
    'policy_or_regulation',
    'educational_or_book',
    'interview_or_meeting',
    'news_or_article',
    'narrative',
    'mixed_or_other',
]


DOC_TYPE_TO_STRATEGY: dict[str, str] = {
    'business_report': 'chain_of_density',
    'research_article': 'two_stage',
    'technical_documentation': 'controlled_abstraction',
    'policy_or_regulation': 'controlled_abstraction',
    'educational_or_book': 'multi_vector',
    'interview_or_meeting': 'multi_vector',
    'news_or_article': 'two_stage',
    'narrative': 'multi_vector',
    'mixed_or_other': 'two_stage',
}


class DocTypeSchema(BaseModel):
    doc_type: DocumentTypeLiteral
    confidence: int = Field(..., ge=0, le=100)
    rationale: str = Field(..., description='Короткое объяснение, почему выбран этот тип')


doc_type_parser = JsonOutputParser(pydantic_object=DocTypeSchema)


DOC_TYPE_PROMPT = PromptTemplate.from_template("""
Ты классифицируешь тип документа для выбора стратегии суммаризации.

Категории:
- business_report: бизнес-отчет, performance-отчет, квартальный обзор, KPI/метрики
- research_article: научная или аналитическая статья с гипотезами/методами/выводами
- technical_documentation: техдок, RFC, API/архитектурная документация, руководство
- policy_or_regulation: политика, регламент, стандарт, комплаенс-требования
- educational_or_book: учебный текст, книга, курс, методичка
- interview_or_meeting: интервью, стенограмма встречи, Q&A
- news_or_article: новость, журналистская статья, обзор события
- narrative: повествовательный/эссеистический текст
- mixed_or_other: смешанный или трудноопределимый тип

Текст:
{text}

Верни строго JSON:
{format_instructions}
""")


DOC_TYPE_AGG_PROMPT = PromptTemplate.from_template("""
Ты агрегируешь классификации чанков длинного документа в один итоговый тип.

Классификации чанков (JSON-массив):
{chunk_classifications}

Правила:
- Если есть явное преобладание одного типа, выбери его.
- Если документ явно смешанный, выбери mixed_or_other.
- confidence = 0..100.
- rationale: 1-2 предложения, с учетом распределения по чанкам.

Верни строго JSON:
{format_instructions}
""")


def _classify_single_text(text: str) -> dict:
    classify_chain = DOC_TYPE_PROMPT | llm | doc_type_parser
    return classify_chain.invoke(
        {
            'text': text,
            'format_instructions': doc_type_parser.get_format_instructions(),
        }
    )


def classify_document_type(
    text: str,
    *,
    chunk_size: int = 12000,
    chunk_overlap: int = 400,
    max_chunks: int = 8,
) -> dict:
    """Классификация типа документа с поддержкой длинных текстов через чанки и агрегацию."""
    clean_text = text.strip()
    if not clean_text:
        return {
            'doc_type': 'mixed_or_other',
            'confidence': 0,
            'rationale': 'Пустой текст',
            'is_large_document': False,
            'analyzed_chunks': 0,
            'strategy': DOC_TYPE_TO_STRATEGY['mixed_or_other'],
        }

    if len(clean_text) <= chunk_size:
        result = _classify_single_text(clean_text)
        result['is_large_document'] = False
        result['analyzed_chunks'] = 1
        result['strategy'] = DOC_TYPE_TO_STRATEGY.get(result['doc_type'], 'two_stage')
        return result

    chunks = split_long_text(clean_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sampled_chunks = chunks[:max_chunks]

    chunk_results: list[dict] = []
    for chunk in sampled_chunks:
        chunk_results.append(_classify_single_text(chunk))

    agg_chain = DOC_TYPE_AGG_PROMPT | llm | doc_type_parser
    try:
        aggregated = agg_chain.invoke(
            {
                'chunk_classifications': json.dumps(chunk_results, ensure_ascii=False),
                'format_instructions': doc_type_parser.get_format_instructions(),
            }
        )
    except Exception:
        votes = [item['doc_type'] for item in chunk_results]
        majority = Counter(votes).most_common(1)[0][0]
        aggregated = {
            'doc_type': majority,
            'confidence': 60,
            'rationale': 'Fallback по мажоритарному голосованию чанков.',
        }

    aggregated['is_large_document'] = True
    aggregated['analyzed_chunks'] = len(sampled_chunks)
    aggregated['total_chunks'] = len(chunks)
    aggregated['strategy'] = DOC_TYPE_TO_STRATEGY.get(aggregated['doc_type'], 'two_stage')
    return aggregated


class DocumentTypeClassifierInput(BaseModel):
    text: str = Field(..., description='Исходный текст для классификации')
    chunk_size: int = Field(default=12000, ge=1000, description='Размер чанка для длинных документов')
    chunk_overlap: int = Field(default=400, ge=0, description='Перекрытие чанков')
    max_chunks: int = Field(default=8, ge=1, le=20, description='Максимум чанков для анализа')


@tool('classify_document_type', args_schema=DocumentTypeClassifierInput)
def classify_document_type_tool(
    text: str,
    chunk_size: int = 12000,
    chunk_overlap: int = 400,
    max_chunks: int = 8,
) -> str:
    """LLM-классификатор типа документа (поддерживает большие документы через чанки)."""
    result = classify_document_type(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
    )
    return (
        '## Классификация документа\n'
        f"- doc_type: {result['doc_type']}\n"
        f"- confidence: {result['confidence']}\n"
        f"- strategy: {result['strategy']}\n"
        f"- is_large_document: {result['is_large_document']}\n"
        f"- analyzed_chunks: {result['analyzed_chunks']}\n"
        f"- rationale: {result['rationale']}\n"
    )


class AutoSummarizeByDocTypeInput(BaseModel):
    text: str = Field(..., description='Текст для автоклассификации и суммаризации')
    large_doc_threshold: int = Field(default=14000, ge=1000, description='Порог длины для large_document_summarize')
    classification_chunk_size: int = Field(default=12000, ge=1000, description='Размер чанка классификатора')
    classification_chunk_overlap: int = Field(default=400, ge=0, description='Перекрытие чанков классификатора')
    classification_max_chunks: int = Field(default=8, ge=1, le=20, description='Максимум чанков классификатора')
    chunk_size: int = Field(default=12000, ge=1000, description='Размер чанка для large_document_summarize')
    chunk_overlap: int = Field(default=800, ge=0, description='Перекрытие для large_document_summarize')
    max_levels: int = Field(default=4, ge=1, le=10, description='Глубина рекурсивного merge')
    include_debug: bool = Field(default=False, description='Добавить debug-статистику уровней')


@tool('auto_summarize_by_doc_type', args_schema=AutoSummarizeByDocTypeInput)
def auto_summarize_by_doc_type(
    text: str,
    large_doc_threshold: int = 14000,
    classification_chunk_size: int = 12000,
    classification_chunk_overlap: int = 400,
    classification_max_chunks: int = 8,
    chunk_size: int = 12000,
    chunk_overlap: int = 800,
    max_levels: int = 4,
    include_debug: bool = False,
) -> str:
    """Автовыбор стратегии суммаризации на основе LLM-классификации типа документа."""
    cls = classify_document_type(
        text=text,
        chunk_size=classification_chunk_size,
        chunk_overlap=classification_chunk_overlap,
        max_chunks=classification_max_chunks,
    )
    strategy = cls['strategy']
    print(f"Auto-selected strategy: {strategy} for doc_type: {cls['doc_type']} (confidence: {cls['confidence']})")
    clean_text = text.strip()
    if len(clean_text) > large_doc_threshold:
        summary = large_document_summarize.invoke(
            {
                'text': clean_text,
                'strategy': strategy,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'max_levels': max_levels,
                'include_debug': include_debug,
            }
        )
    else:
        summary = _invoke_base_tool(strategy, clean_text)

    return (
        '## Auto Strategy Selection\n'
        f"- doc_type: {cls['doc_type']}\n"
        f"- confidence: {cls['confidence']}\n"
        f"- strategy: {strategy}\n"
        f"- rationale: {cls['rationale']}\n\n"
        '## Summary\n'
        f'{summary}'
    )


# Пример:
# result = large_document_summarize.invoke({
#     'text': text,
#     'strategy': 'two_stage',
#     'chunk_size': 10000,
#     'chunk_overlap': 700,
#     'max_levels': 4,
#     'include_debug': True,
# })
# print(result)
