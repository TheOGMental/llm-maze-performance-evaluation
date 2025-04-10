import dotenv
import json

from openai import OpenAI
from llamaapi import LlamaAPI

OPENAI_API_KEY = dotenv.get_key("OPENAI_KEY")
DEEPSEEK_API_KEY = dotenv.get_key("DEEPSEEK_KEY")
LLAMA_API_KEY = dotenv.get_key("LLAMA_KEY")

openai_client = OpenAI()
deepseek_client = client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
llama_client = LlamaAPI(LLAMA_API_KEY)

def prompt_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = openai_client.chat.completions.create(
        model=model,
        input=prompt
    )
    return response.output_text

def prompt_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    response = deepseek_client.chat.completions.create(
        model=model,
        input=prompt
    )
    return response.output_text

def prompt_llama(prompt: str, model: str = "llama-7b") -> str:
    return None