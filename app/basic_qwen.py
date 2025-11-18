import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": True},
    streaming=True,
)

# first_response = model.invoke("1.11 and 1.5, which is greater?")
# print(first_response)

for chunk in model.stream("1.11 and 1.5, which is greater?"):
    print(chunk.text, end="", flush=True)
