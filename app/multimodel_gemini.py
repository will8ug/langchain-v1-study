import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def analyze_pdf_document(pdf_path: str, question: str = "What is the document about?"):
    """Analyze a PDF document using Gemini model."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    pdf_bytes = open(pdf_path, "rb").read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "file",
                "source_type": "base64",
                "data": pdf_base64,
                "mime_type": "application/pdf",
            },
        ]
    )

    response = model.invoke([message])
    return response.content


if __name__ == "__main__":
    result = analyze_pdf_document("./resources/sample-001.pdf")
    print(result)
