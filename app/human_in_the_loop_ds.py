from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions. Requires human approval for safety."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as ex:
        return f"Error: {str(ex)}"


@tool
def file_reader(filename: str) -> str:
    """Read the contents of a file. Requires human approval for security."""
    try:
        with open(filename, "r") as f:
            content = f.read()
        return f"File content:\n{content}"
    except Exception as ex:
        return f"Error reading file: {str(ex)}"


@tool
def system_info() -> str:
    """Get basic system information. Requires human approval for privacy."""
    import platform
    import os

    info = f"System: {platform.system()}\n"
    info += f"Python Version: {platform.python_version()}\n"
    info += f"Current Directory: {os.getcwd()}\n"
    return info


model = ChatDeepSeek(model="deepseek-chat")

agent = create_agent(
    model=model,
    tools=[calculator, file_reader, system_info],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "calculator": False,
                "file_reader": {"allowed_decisions": ["approve", "reject"]},
                "system_info": True,
            },
            description_prefix="Tool execution pending approval",
        )
    ],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    print("Human-in-the-Loop Agent Demo")
    print("=" * 40)

    config: RunnableConfig = {"configurable": {"thread_id": "demo-thread"}}

    messages = [
        "Calculate 15 * 8 + 3",
        "Read the README.md file if it exists",
        "Show me system information",
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        print("Agent: ", end="")

        try:
            response = agent.invoke(
                input={"messages": [HumanMessage(content=msg)]},
                config=config
            )
            print(response["messages"][-1].content)

            if "__interrupt__" in response:
                # print(f"__interrupt__: \n{response['__interrupt__']}")

                print("\nContinue with approval...")
                resumed_response = agent.invoke(
                    Command(
                        resume={"decisions": [{"type": "approve"}]}
                    ),
                    config=config
                )
                print(resumed_response["messages"][-1].content)

        except Exception as ex:
            print(f"Error: {str(ex)}")

        print("-" * 40)
