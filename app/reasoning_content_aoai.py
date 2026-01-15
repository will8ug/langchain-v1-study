import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.shared_params import Reasoning

load_dotenv()

model = AzureOpenAI(
    azure_endpoint="https://your-resource-name.openai.azure.com",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-04-01-preview",
)

resp = model.responses.create(
    model="gpt-5.1",
    input="This is a test message. Please simply let me know if you are working.",
    stream=True
)
for chunk in resp:
    if chunk.type == "response.output_text.delta":
        print(chunk.delta, end="", flush=True)
print("\n" + "="*80 + "\n")

resp = model.responses.create(
    model="gpt-5.1",
    input="Why do parrots have colorful feathers?",  # larger effort
    # input="1.11 and 1.5, which is greater?",    # smaller effort
    reasoning=Reasoning(
        effort="medium",
        summary="auto"
    ),
    stream=True
)

started_answering = False
print("\n" + "="*20 + "Reasoning Content" + "="*20 + "\n")
for chunk in resp:
    if chunk.type == "response.reasoning_text.delta" or chunk.type == "response.reasoning_summary_text.delta":
        print(chunk.delta, end="", flush=True)
        continue

    if not started_answering and chunk.type == "response.reasoning_summary_text.done":
        started_answering = True
        print("\n" + "="*20 + "Final Answer" + "="*20 + "\n")
        continue

    if chunk.type == "response.output_text.delta":
        print(chunk.delta, end="", flush=True)
