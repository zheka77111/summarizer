from langchain_gigachat import GigaChat
from dotenv import load_dotenv
import os
from rich import print

load_dotenv()
key = os.environ.get('GIGACHAT_API_KEY')

llm = GigaChat(
    credentials=key,
    model='GigaChat-2-Max',
    scope='GIGACHAT_API_CORP',
    temperature=0.87,
    verify_ssl_certs=False,
    profanity_check=False,
    max_tokens=25000,
    timeout=300,
)