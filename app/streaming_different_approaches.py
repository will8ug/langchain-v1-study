import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END

from app.common.llm_configs import qwen

async def filter_by_tags():
    joke_model = init_chat_model(
        model=qwen.model,
        api_key=qwen.api_key,
        base_url=qwen.base_url,
        model_provider="openai",
        tags=["joke"]
    )
    poem_model = init_chat_model(
        model=qwen.model,
        api_key=qwen.api_key,
        base_url=qwen.base_url,
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

        state["joke"] = joke_response.content
        state["poem"] = poem_response.content
        return state

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

def filter_by_node():
    model = init_chat_model(
        model=qwen.model,
        api_key=qwen.api_key,
        base_url=qwen.base_url,
        model_provider="openai",
    )

    class State(TypedDict):
        topic: str
        joke: str
        poem: str

    def write_joke(state: State) -> State:
        topic = state["topic"]
        response = model.invoke(
            [{"role": "user", "content": f"Write a joke about {topic}"}]
        )
        # not working in concurrent run of multiple nodes
        # state["joke"] = response.content
        # return state
        return {"joke": response.content}
    
    def write_poem(state: State) -> State:
        topic = state["topic"]
        response = model.invoke(
            [{"role": "user", "content": f"Write a short poem about {topic}"}],
        )
        # not working in concurrent run of multiple nodes
        # state["poem"] = response.content
        # return state
        return {"poem": response.content}

    graph = (
        StateGraph(State)
        .add_node(write_joke)
        .add_node(write_poem)
        # write both the joke and the poem concurrently
        .add_edge(START, "write_joke")
        .add_edge(START, "write_poem")
        .compile()
    )

    for msg, metadata in graph.stream(
        {"topic": "cats"},
        stream_mode="messages",
    ):
        if msg.content and metadata["langgraph_node"] == "write_joke":
            print(msg.content, end="", flush=True)

def stream_custom_data():
    class State(TypedDict):
        query: str
        answer: str

    def demo_node(state: State):
        writer = get_stream_writer()
        writer({"custom_key": "Generating custom data inside a node"})
        time.sleep(1)
        writer({"custom_key": "this is a first record"})
        time.sleep(1)
        writer({"custom_key": "this is a second record"})
        return {"answer": "some data"}

    graph = (
        StateGraph(State)
        .add_node(demo_node)
        .add_edge(START, "demo_node")
        .compile()
    )

    inputs = {"query": "example query"}
    for chunk in graph.stream(
        inputs,
        stream_mode="custom",
    ):
        print(chunk["custom_key"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(filter_by_tags())

    print("\n\n" + "=*="*30 + "\n\n")
    filter_by_node()

    print("\n\n" + "=*="*30 + "\n\n")
    stream_custom_data()
