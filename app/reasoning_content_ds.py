from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-reasoner",
    streaming=True,
)

started_answering = False

print("\n" + "="*20 + "Reasoning Content" + "="*20 + "\n")
for chunk in model.stream("1.11 and 1.5, which is greater?"):
    # print(chunk.text, end="", flush=True)
    # print(chunk.content_blocks)
    if not chunk.content_blocks:  # ended chunk
        print("\n")
        continue  # it could be "break", but use "continue" for a safe purpose

    block = chunk.content_blocks[0]
    if block["type"] == "reasoning":
        print(block["reasoning"], end="", flush=True)
        continue

    if not started_answering:
        started_answering = True
        print("\n\n" + "="*20 + "Final Answer" + "="*20 + "\n")
    
    if block["type"] == "text":
        print(block["text"], end="", flush=True)
