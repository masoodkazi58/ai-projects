import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llms import embedder
from config import index_path_from_file,textfile_exists

def build_embeddings_if_needed():
    text_files = textfile_exists()

    if not text_files:
        print("⚠️ No .txt files found in data/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for file_path in text_files:
        index_path = index_path_from_file(file_path)

        if os.path.exists(index_path):
            print(f"✅ Skipping (already embedded): {file_path}")
            continue

        print(f"⚡ Embedding: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        docs = splitter.create_documents([text])
        vectorstore = FAISS.from_documents(docs, embedding=embedder)
        vectorstore.save_local(index_path)

        print(f"✅ Saved embeddings → {index_path}")


def build_embeddings_for_txt(txt_file_path: str | Path) -> Path:
    """
    Build and save embeddings for a given .txt file path.

    Args:
        txt_file_path (str | Path): Full path to a .txt file

    Returns:
        Path: Path where vectorstore is saved
    """

    txt_file_path = Path(txt_file_path)

    if not txt_file_path.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_file_path}")

    if txt_file_path.suffix.lower() != ".txt":
        raise ValueError("this from embedder Only .txt files are allowed")

    index_path = Path(index_path_from_file(txt_file_path))

    if index_path.exists():
        print(f"✅ Already embedded: {txt_file_path.name}")
        return index_path

    print(f"⚡ Embedding: {txt_file_path.name}")

    text = txt_file_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    vectorstore = FAISS.from_documents(
        docs,
        embedding=embedder
    )

    vectorstore.save_local(index_path)

    print(f"✅ Embeddings saved at → {index_path}")

    return index_path