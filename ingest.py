from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Directory paths
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"


def main():
    # 1. Load Documents
    print("Loading documents from data directory...")
    # Using TextLoader and targeting Markdown files. Change to *.txt if needed.
    loader = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print(
            "No documents found. Please add some .md or .txt files to the data/ folder."
        )
        return

    print(f"Loaded {len(documents)} documents.")
    if documents:
        print("Sample ingested files:")
        for doc in documents:
            print(f" - {doc.metadata.get('source')}")  # pyright: ignore[reportUnknownMemberType]

    # 2. Split Text into Chunks
    print("Splitting text into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # Overlap ensures context isn't lost between chunks
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Initialize Embedding Model
    print(
        "Initializing embedding model (this will download model weights on the first run)..."
    )
    # all-MiniLM-L6-v2 is highly efficient and runs well on minimal hardware
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and Save to Vector Database
    print("Creating Chroma vector database...")
    _ = Chroma.from_documents(  # pyright: ignore[reportUnknownMemberType]
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR
    )

    print(f"Success! Vector database saved to {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
