import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_ = load_dotenv()


def test_connection():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    print("Testing connection to OpenRouter...")
    llm = ChatOpenAI(
        model="openrouter/free",
        openai_api_key=api_key,  # pyright: ignore[reportCallIssue]
        openai_api_base="https://openrouter.ai/api/v1",  # pyright: ignore[reportCallIssue]
    )

    try:
        response = llm.invoke("Say 'System Online' if you can read this.")
        print(f"\nResponse: {response.content}")  # pyright: ignore[reportUnknownMemberType]
    except Exception as e:
        print(f"\nError connecting to OpenRouter: {e}")


if __name__ == "__main__":
    test_connection()
