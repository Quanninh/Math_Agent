"""Ingest plain-text and PDF textbooks into the local Chroma index."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Union

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COLLECTION_NAME = "math_textbooks"


def _load_pdf(path: Path) -> List[Document]:
    """Load a PDF with pypdf if installed, otherwise use pdftotext."""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": str(path), "page": number + 1},
            )
            for number, page in enumerate(reader.pages)
        ]
    except ImportError:
        # pdftotext is available on most macOS/Linux machines and avoids making
        # the whole project depend on a heavyweight PDF stack.
        with tempfile.NamedTemporaryFile(suffix=".txt") as output:
            subprocess.run(
                ["pdftotext", "-layout", str(path), output.name],
                check=True,
                capture_output=True,
            )
            text = Path(output.name).read_text(errors="ignore")
        return [Document(page_content=text, metadata={"source": str(path)})]


def load_documents(docs_path: Union[str, Path] = "docs") -> List[Document]:
    root = Path(docs_path)
    if not root.exists():
        raise FileNotFoundError(f"Textbook directory does not exist: {root}")

    documents: List[Document] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".txt":
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))

    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        raise FileNotFoundError(f"No .txt or .pdf textbooks found in {root}")
    return documents


def split_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=220,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def create_vector_store(
    chunks: List[Document], persist_directory: str = "db/chroma_db", batch_size: int = 64
) -> Chroma:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env before indexing textbooks.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # A whole textbook can exceed the embeddings endpoint's per-request token
    # limit. Add chunks in small batches instead of embedding everything at once.
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectorstore.delete_collection()
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch)
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")
    return vectorstore


def main() -> None:
    documents = load_documents("docs")
    chunks = split_documents(documents)
    create_vector_store(chunks)
    print(f"Indexed {len(documents)} document sections as {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
