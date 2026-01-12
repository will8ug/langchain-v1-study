import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = init_chat_model(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_provider="openai"
)

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State) -> State:
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State) -> State:
    model_response = model.invoke(
        [{"role": "user", "content": f"Tell me a joke about {state['topic']}"}]
    )
    return {"joke": model_response.content}

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates"
):
    print(chunk)
