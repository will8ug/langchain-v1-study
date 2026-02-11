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


@tool
def file_editor(filename: str, content: str) -> str:
    """Edit a file with new content. Requires human approval and allows editing."""
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as ex:
        return f"Error writing to file: {str(ex)}"


model = ChatDeepSeek(model="deepseek-chat")

agent = create_agent(
    model=model,
    tools=[calculator, file_reader, system_info, file_editor],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "calculator": False,
                "file_reader": {"allowed_decisions": ["approve", "reject"]},
                "system_info": True,
                "file_editor": {"allowed_decisions": ["approve", "reject", "edit"]},
            },
            description_prefix="Tool execution pending approval",
        )
    ],
    checkpointer=InMemorySaver(),
)


def demo_no_interrupt(config: RunnableConfig):
    """Demo calculator tool with no interrupt required."""
    print("1. No Interrupt Demo (Calculator)")
    print("Agent: ", end="")

    try:
        response = agent.invoke(
            input={"messages": [HumanMessage(content="Calculate 15 * 8 + 3")]},
            config=config,
        )
        print(response["messages"][-1].content)
    except Exception as ex:
        print(f"Error: {str(ex)}")

    print("-" * 40)


def demo_interrupt_with_approve(config: RunnableConfig):
    """Demo file reader with interrupt and approve decision."""
    print("2. Interrupt with Approve Demo (File Reader)")
    print("Agent: ", end="")

    try:
        response = agent.invoke(
            input={
                "messages": [
                    HumanMessage(content="Read the README.md file if it exists")
                ]
            },
            config=config,
        )
        print(response["messages"][-1].content)

        if "__interrupt__" in response:
            print("\nContinue with approval...")
            resumed_response = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}]}), config=config
            )
            print(resumed_response["messages"][-1].content)
    except Exception as ex:
        print(f"Error: {str(ex)}")

    print("-" * 40)


def demo_interrupt_with_reject(config: RunnableConfig):
    """Demo system info with interrupt and reject decision."""
    print("3. Interrupt with Reject Demo (System Info)")
    print("Agent: ", end="")

    try:
        response = agent.invoke(
            input={"messages": [HumanMessage(content="Show me system information")]},
            config=config,
        )
        print(response["messages"][-1].content)

        if "__interrupt__" in response:
            print("\nContinue with rejection...")
            resumed_response = agent.invoke(
                Command(resume={"decisions": [{"type": "reject"}]}), config=config
            )
            print(resumed_response["messages"][-1].content)
    except Exception as ex:
        print(f"Error: {str(ex)}")

    print("-" * 40)


def demo_interrupt_with_edit(config: RunnableConfig):
    """Demo file editor with interrupt and edit decision."""
    print("4. Interrupt with Edit Demo (File Editor)")
    print("Agent: ", end="")

    try:
        response = agent.invoke(
            input={
                "messages": [
                    HumanMessage(
                        content="Create a test.txt file with 'Hello World' content"
                    )
                ]
            },
            config=config,
        )
        print(response["messages"][-1].content)

        if "__interrupt__" in response:
            print("\nContinue with edit (modifying content to 'Hello Edited World')...")
            resumed_response = agent.invoke(
                Command(
                    resume={
                        "decisions": [
                            {
                                "type": "edit",
                                "edited_action": {
                                    "name": "file_editor",
                                    "args": {
                                        "filename": "test.txt",
                                        "content": "Hello Edited World",
                                    },
                                },
                            }
                        ]
                    }
                ),
                config=config,
            )
            print(resumed_response["messages"][-1].content)
    except Exception as ex:
        print(f"Error: {str(ex)}")

    print("-" * 40)


if __name__ == "__main__":
    print("Human-in-the-Loop Agent Demo")
    print("=" * 40)

    config: RunnableConfig = {"configurable": {"thread_id": "demo-thread"}}

    demo_no_interrupt(config)
    demo_interrupt_with_approve(config)
    demo_interrupt_with_reject(config)
    demo_interrupt_with_edit(config)
