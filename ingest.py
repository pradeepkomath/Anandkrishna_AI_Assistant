import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


DOCS_FOLDER = BASE_DIR / os.getenv("DOCS_FOLDER", "docs")
FAISS_INDEX_PATH = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "vectorstore/faiss_index")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def load_documents():
    documents = []

    if not DOCS_FOLDER.exists():
        raise FileNotFoundError(f"Docs folder not found: {DOCS_FOLDER}")

    for file_path in DOCS_FOLDER.iterdir():
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file_path.name

            documents.extend(loaded_docs)

        elif file_path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file_path.name

            documents.extend(loaded_docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


def create_faiss_index(chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_PATH))


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please add it to .env file.")

    print("Starting document ingestion...")
    print(f"Docs folder: {DOCS_FOLDER}")

    documents = load_documents()
    print(f"Loaded documents/pages: {len(documents)}")

    if not documents:
        raise ValueError("No PDF/TXT/MD documents found in docs folder.")

    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    create_faiss_index(chunks)
    print(f"FAISS index saved at: {FAISS_INDEX_PATH}")
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()