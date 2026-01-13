import os
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str


qwen = LLMConfig(
    model="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

deepseek = LLMConfig(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.cn/v1"
)
