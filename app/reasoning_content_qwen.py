import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

messages = [{"role": "user", "content": "1.11 and 1.5, which is greater?"}]
response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    stream=True,
    extra_body={"enable_thinking": True},
)

started_answering = False

print("\n" + "="*20 + "Reasoning Content" + "="*20 + "\n")
for chunk in response:
    delta = chunk.choices[0].delta
    if not delta.content and not delta.reasoning_content:
        # this seems to be specific to "qwen". based on my test, it would reply a first chunk of this:
        #   ChoiceDelta(content=None, function_call=None, refusal=None, role='assistant', tool_calls=None, reasoning_content='')
        # both "content" and "reasoning_content" are empty, and will have a "role" of "assistant"
        # so I have to handle it first here
        continue

    if delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
        continue

    if not started_answering:
        started_answering = True
        print("\n\n" + "="*20 + "Final Answer" + "="*20 + "\n")
    
    if delta.content:
        print(delta.content, end="", flush=True)
        