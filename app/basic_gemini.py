from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)
response = model.invoke(
    "This is a test message. Please simply let me know if you are working."
)
print(response.content)
print("="*50)

for chunk in model.stream("What color is the sky?"):
    print(chunk.text, end="", flush=True)
