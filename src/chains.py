from src.prompt import extract, synthesize


from langchain_core.prompts import PromptTemplate

Extract_PROMPT = PromptTemplate.from_template(extract)
SYNTHESIZE_PROMPT = PromptTemplate.from_template(synthesize)
