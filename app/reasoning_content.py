import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [{"role": "user", "content": "1.11 and 1.5, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

content = response.choices[0].message.content
print(f"reasoning_content:\n{response.choices[0].message.reasoning_content}")
print("\n\n")
print(f"content:\n{content}")
print("\n=======================\n")

messages.append({"role": "assistant", "content": content})
messages.append({"role": "user", "content": "How many Rs are there in the word 'strawberry'?"})
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

content = response.choices[0].message.content
print(f"reasoning_content:\n{response.choices[0].message.reasoning_content}")
print("\n\n")
print(f"content:\n{content}")
