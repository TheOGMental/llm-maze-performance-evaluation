# THIS NEEDS TO BE SMARTER

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=API_KEY)

responses = []
responses_folder = "responses"
os.makedirs(responses_folder, exist_ok=True)

mazes_folder = "prompts"
os.makedirs(responses_folder, exist_ok=True)

for prompt_file in sorted(os.listdir(mazes_folder)):
    prompt_path = os.path.join(mazes_folder, prompt_file)

    with open(prompt_path, "r") as f:
        prompt = f.read().strip()

    response = client.chat.completions.create(
        model="model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates solutions for mazes, going from an origin to a target"},
            {"role": "user", "content": f"Solve the following maze:\n{maze}"},
        ]
    )