from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

load_dotenv()

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The rating of the movie out of 10")

def test_with_structured_output():
    model = ChatDeepSeek(
        model="deepseek-chat"
    )
    model_with_structure = model.with_structured_output(Movie)
    response = model_with_structure.invoke("Provide details about the movie Inception")
    print(response)

def test_create_agent():
    ds_model = ChatDeepSeek(
        model="deepseek-chat"
    )

    agent = create_agent(
        model=ds_model,
        response_format=Movie
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "Provide details about the movie Ne Zha"}]
    })
    print(response["structured_response"])

if __name__ == "__main__":
    test_with_structured_output()

    print("================")

    test_create_agent()
