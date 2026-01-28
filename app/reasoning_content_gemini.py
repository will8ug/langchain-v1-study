from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    include_thoughts=True,
)

started_answering = False
print("\n" + "="*20 + "Reasoning Content" + "="*20 + "\n")
for chunk in model.stream("What color is the sky?"):
    if not chunk.content_blocks:  # ended chunk
        print("\n")
        continue

    block = chunk.content_blocks[0]
    if block["type"] == "reasoning":
        print(block["reasoning"], end="", flush=True)
        continue

    if not started_answering:
        started_answering = True
        print("\n\n" + "="*20 + "Final Answer" + "="*20 + "\n")

    if block["type"] == "text":
        print(block["text"], end="", flush=True)
