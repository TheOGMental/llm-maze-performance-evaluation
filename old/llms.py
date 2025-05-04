import dotenv
import json

from openai import OpenAI

OPENAI_API_KEY = dotenv.get_key("OPENAI_KEY")
DEEPSEEK_API_KEY = dotenv.get_key("DEEPSEEK_KEY")
DEEPINFRA_API_KEY = dotenv.get_key("DEEPINFRA_KEY")

openai_client = OpenAI()
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
deepinfra_client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")

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

def prompt_deepinfra(prompt: str, model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct") -> str:
    response = deepinfra_client.chat.completions.create(
        model=model,
        input=prompt
    )
    return response.output_text