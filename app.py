import os
import argparse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA  # pyright: ignore[reportAny]
from langchain_core.prompts import PromptTemplate

# Suppress parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_ = load_dotenv()

# Configuration
CHROMA_DIR = "./chroma_db"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Define our specialized prompts
PROMPTS = {
    "default": """Use the following context to answer the question concisely. 
If you don't know, say so. Max 3 sentences.

Context: {context}

Question: {question}

Answer:""",
    "coder": """You are an htmx expert. Use the documentation and examples provided 
to write idiomatic htmx code that follows hypermedia principles.

Context (Documentation & Examples):
{context}

Task: {question}

RELEVANCE GUARD:
- If the task asks for a 'simple', 'basic', or 'common' example, prioritize standard htmx attributes (hx-get, hx-post, hx-target, hx-swap) and ignore complex patterns from research essays.
- Avoid architectural jargon found in the context (like 'MCP', 'DSL', 'CallToolResult', 'structuredContent') UNLESS the task specifically mentions them.
- Always include the 'name' attribute on input elements so the backend can actually receive the data.

Requirements:
1. Provide the HTML code with htmx attributes.
2. Explain briefly how the request/response flow works (e.g. "The backend receives a POST body with key=value").

htmx Solution:""",
}


def get_chain(mode: str, model_name: str, api_key: str, use_rag: bool = True):  # pyright: ignore[reportUnknownParameterType]
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,  # pyright: ignore[reportCallIssue]
        openai_api_base=OPENROUTER_BASE_URL,  # pyright: ignore[reportCallIssue]
        temperature=0.2 if mode == "coder" else 0,
    )

    if not use_rag:
        # Simple prompt | llm chain for no-rag mode
        prompt = PromptTemplate.from_template(
            "Answer this question concisely: {question}"
        )
        return prompt | llm  # pyright: ignore[reportUnknownVariableType]

    # RAG Setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector database not found at {CHROMA_DIR}. Run ingest.py first."
        )

    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    template = PROMPTS.get(mode, PROMPTS["default"])
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    return RetrievalQA.from_chain_type(  # pyright: ignore[reportAny]
        llm,
        retriever=db.as_retriever(search_kwargs={"k": 7 if mode == "coder" else 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )


def main():
    parser = argparse.ArgumentParser(description="Unified htmx RAG System")
    _ = parser.add_argument(
        "--mode",
        choices=["default", "coder", "no-rag"],
        default="default",
        help="Choose between general Q&A, coding assistance, or baseline (no-RAG)",
    )
    _ = parser.add_argument(
        "--model", default="openrouter/free", help="The OpenRouter model ID to use"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    try:
        chain = get_chain(  # pyright: ignore[reportUnknownVariableType]
            args.mode,  # pyright: ignore[reportAny]
            args.model,  # pyright: ignore[reportAny]
            api_key,
            use_rag=(args.mode != "no-rag"),  # pyright: ignore[reportAny]
        )
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print(f"\n--- htmx RAG: {args.mode.upper()} mode ---")  # pyright: ignore[reportAny]
    print(f"Model: {args.model}")  # pyright: ignore[reportAny]
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input(f"\n[{args.mode}] Query: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue

            print("Thinking...")
            if args.mode == "no-rag":
                res = chain.invoke({"question": query})  # pyright: ignore[reportUnknownMemberType]
                print("\nAnswer:", res.content)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            else:
                res = chain.invoke({"query": query})  # pyright: ignore[reportUnknownMemberType]
                print("\nAnswer:", res["result"])  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]
                sources = set(  # pyright: ignore[reportUnknownVariableType]
                    d.metadata.get("source", "Unknown")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    for d in res["source_documents"]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
                )
                print("\nSources used:", ", ".join(sources))  # pyright: ignore[reportUnknownArgumentType]

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during query: {e}")


if __name__ == "__main__":
    main()
