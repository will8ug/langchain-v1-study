import os
from re import M
from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

load_dotenv()

model="qwen-flash"
api_key=os.getenv("DASHSCOPE_API_KEY")
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# model="deepseek-chat"
# api_key=os.getenv("DEEPSEEK_API_KEY")
# base_url="https://api.deepseek.cn/v1"

async def filter_by_tags():
    joke_model = init_chat_model(
        model=model,
        api_key=api_key,
        base_url=base_url,
        model_provider="openai",
        tags=["joke"]
    )
    poem_model = init_chat_model(
        model=model,
        api_key=api_key,
        base_url=base_url,
        model_provider="openai",
        tags=["poem"]
    )

    class State(TypedDict):
        topic: str
        joke: str
        poem: str

    async def call_model_fn(state: State, config) -> State:
        topic = state["topic"]
        print("\nWriting joke...\n")
        joke_response = await joke_model.ainvoke(
            [{"role": "user", "content": f"Write a joke about {topic}"}],
            config,
        )

        print("\nWriting poem...\n")
        poem_response = await poem_model.ainvoke(
            [{"role": "user", "content": f"Write a short poem about {topic}"}],
            config,
        )
        
        return {
            "joke": joke_response.content,
            "poem": poem_response.content,
        }

    graph = (
        StateGraph(State)
        .add_node("call_model", call_model_fn)
        .add_edge(START, "call_model")
        .add_edge("call_model", END)
        .compile()
    )

    async for msg, metadata in graph.astream(
        {"topic": "cats"},
        stream_mode="messages",
    ):
        if metadata["tags"] == ["poem"]:
            print(msg.content, end="", flush=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(filter_by_tags())