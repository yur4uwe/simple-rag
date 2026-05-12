import os
from dotenv import load_dotenv
from app import get_chain

_ = load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")


def run_demo(mode: str, question: str, model_name: str = "openrouter/free"):
    # get_chain handles the initialization of embeddings internally if use_rag=True
    # but we will call it for each to show the comparison.
    # The HuggingFaceEmbeddings class will use its internal cache if called multiple times in one process.
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    chain = get_chain(mode, model_name, api_key, use_rag=(mode != "no-rag"))  # pyright: ignore[reportUnknownVariableType]
    print(f"\n{'=' * 20}")
    print(f"MODE: {mode.upper()}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 20}")

    if mode == "no-rag":
        res = chain.invoke({"question": question})  # pyright: ignore[reportUnknownMemberType]
        print(f"\nANSWER:\n{res.content}")  # pyright: ignore[reportUnknownMemberType]
    else:
        res = chain.invoke({"query": question})  # pyright: ignore[reportUnknownMemberType]
        print(f"\nANSWER:\n{res['result']}")  # pyright: ignore[reportIndexIssue]
        sources = set(  # pyright: ignore[reportUnknownVariableType]
            d.metadata.get("source", "Unknown")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            for d in res["source_documents"]  # pyright: ignore[reportUnknownVariableType, reportIndexIssue]
        )
        print(f"\nSOURCES USED: {', '.join(sources)}")  # pyright: ignore[reportUnknownArgumentType]


if __name__ == "__main__":
    # Question 1: Specific Theory (HDA)
    q1 = "What are the core concepts of Hypermedia Driven Applications (HDA) as defined in the htmx essays?"

    # Question 2: Technical Implementation (Active Search)
    q2 = "Provide the HTML for an active search that triggers on input with a 500ms delay and also on the 'Enter' key."

    print("--- DEMO 1: THEORY COMPARISON ---")
    run_demo("no-rag", q1)
    run_demo("default", q1)

    print("\n\n--- DEMO 2: TECHNICAL COMPARISON ---")
    run_demo("no-rag", q2)
    run_demo("coder", q2)
