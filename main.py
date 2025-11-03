
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.runnables.config import RunnableConfig

load_dotenv()

model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

memory = InMemorySaver()

searchTool = TavilySearch(max_results=2)
agent = create_agent(
    model=model,
    tools=[searchTool],
    checkpointer=memory,
)

config = RunnableConfig(
    configurable={"thread_id": "bob-123"}
)

input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in Guangzhou."
}
for step in agent.stream(
    {"messages": [input_message]}, 
    config, 
    stream_mode="values"
):
    step["messages"][-1].pretty_print()

print("\n\n")

input_message = {
    "role": "user",
    "content": "What's the weather where I live?"
}
for step, metadata in agent.stream(
    {"messages": [input_message]}, 
    config, 
    stream_mode="messages"
):
    if metadata["langgraph_node"] == "model" and (content := step.content):
        print(content, end="", flush=True)
