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


def generate_image():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-image")

    response = model.invoke(
        "Generate a photorealistic image of a cuddly cat wearing a hat."
    )
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    print(image_block)
    image_base64 = image_block["image_url"].get("url").split(",")[-1]

    image_data = base64.b64decode(image_base64)
    output_path = "./generated_cat.png"
    with open(output_path, "wb") as f:
        f.write(image_data)
    print(f"Image saved to {output_path}")

    return output_path


if __name__ == "__main__":
    result = analyze_pdf_document("./resources/sample-001.pdf")
    print(result)

    generate_image()
