# ─────────────────────────────────────────────────────────────────────────────
#  rag.py  –  Retrieval-Augmented Generation (RAG) pipeline
#
#  WHAT IS RAG?
#  LLMs are trained on general internet data — they don't know YOU.
#  RAG lets us "inject" your personal knowledge (resume PDF) into the LLM
#  at query time, so it can answer questions about you accurately.
#
#  HOW IT WORKS (step by step):
#
#   1. LOAD    → Read the PDF and extract raw text
#   2. SPLIT   → Break text into small overlapping chunks
#   3. EMBED   → Convert each chunk into a vector (list of numbers)
#   4. STORE   → Save all vectors in a FAISS index (fast search database)
#   5. RETRIEVE→ When asked a question, find the most similar chunks
#   6. ANSWER  → Pass those chunks + question to the LLM for a grounded answer
#
#   [ PDF ] → [ Chunks ] → [ Vectors ] → [ FAISS Index ]
#                                               ↑
#                              query → embed → search → top chunks → LLM
# ─────────────────────────────────────────────────────────────────────────────

import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader reads a PDF file and returns a list of Document objects,
# one per page. Each Document has `.page_content` (text) and `.metadata`.

from langchain.text_splitter import RecursiveCharacterTextSplitter
# This splitter tries to split on paragraphs → sentences → words,
# in that order, to keep semantically related text together.

from langchain_huggingface import HuggingFaceEmbeddings
# Wraps a sentence-transformers model. Converts text → float vectors.
# The model runs locally — no API calls needed.

from langchain_community.vectorstores import FAISS
# FAISS (Facebook AI Similarity Search) is a library that lets us
# store vectors and find the most similar ones very quickly.
# We use it as our local vector database.

from langchain.schema import Document

import config


# ─── Step 1 & 2: Load PDF and Split into Chunks ──────────────────────────────

def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """
    Reads the PDF at `pdf_path` and splits it into small chunks.

    Args:
        pdf_path: File path to the resume PDF

    Returns:
        A list of Document objects, each holding one text chunk.
    """
    print(f"📄  Loading PDF: {pdf_path}")

    # PyPDFLoader extracts text from every page
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # returns [Document(page_content="...", metadata={page:0}), ...]

    print(f"    → Found {len(pages)} page(s)")

    # RecursiveCharacterTextSplitter splits long pages into smaller chunks.
    # Why do we split?
    #   - Embedding models have a token limit (usually 512 tokens)
    #   - Smaller chunks = more precise retrieval
    #   - chunk_overlap prevents losing context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,        # max characters per chunk
        chunk_overlap=config.CHUNK_OVERLAP,  # characters shared between chunks
        separators=["\n\n", "\n", ". ", " ", ""],
        # tries to split on double newlines first, then single newlines, etc.
    )

    chunks = splitter.split_documents(pages)
    print(f"    → Split into {len(chunks)} chunks")
    return chunks


# ─── Step 3 & 4: Embed and Store ─────────────────────────────────────────────

def build_vector_store(chunks: list[Document]) -> FAISS:
    """
    Converts chunks to vectors and stores them in a FAISS index.

    Args:
        chunks: List of Document chunks from load_and_split_pdf()

    Returns:
        A FAISS vector store ready for similarity search
    """
    print(f"🔢  Loading embedding model: {config.EMBEDDING_MODEL}")
    print("    (First run downloads ~90MB — subsequent runs are instant)")

    # HuggingFaceEmbeddings downloads the model once and caches it locally.
    # all-MiniLM-L6-v2 produces 384-dimensional vectors — fast and accurate.
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
        # normalizing keeps cosine similarity scores between 0 and 1
    )

    print("✨  Building FAISS vector store...")

    # FAISS.from_documents():
    #   1. Calls embeddings.embed_documents() on every chunk's text
    #   2. Stores the resulting vectors + the original text in memory
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so we don't rebuild on every run
    vector_store.save_local(config.VECTOR_STORE_PATH)
    print(f"💾  Saved vector store to ./{config.VECTOR_STORE_PATH}/")

    return vector_store


# ─── Load Existing Store (if already built) ───────────────────────────────────

def load_vector_store() -> Optional[FAISS]:
    """
    Loads a previously saved FAISS index from disk.
    Returns None if no saved index exists yet.
    """
    index_file = os.path.join(config.VECTOR_STORE_PATH, "index.faiss")

    if not os.path.exists(index_file):
        return None  # first run — need to build

    print("📦  Loading existing vector store from disk...")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # allow_dangerous_deserialization=True is required by FAISS when loading
    # from disk (it uses pickle internally). Safe here since we wrote the file.
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("✅  Vector store loaded!")
    return vector_store


# ─── Step 5: Retrieve Relevant Chunks ────────────────────────────────────────

def get_retriever(vector_store: FAISS):
    """
    Wraps the FAISS store in a LangChain Retriever interface.

    A Retriever has a simple interface: given a query string,
    return the top-k most relevant Document chunks.
    """
    return vector_store.as_retriever(
        search_type="similarity",         # cosine similarity search
        search_kwargs={"k": config.TOP_K_RESULTS},  # return top 4 chunks
    )


# ─── High-Level Initializer ───────────────────────────────────────────────────

def initialize_rag(pdf_path: Optional[str] = None) -> FAISS:
    """
    Master function: either loads an existing vector store or builds one from
    the given PDF path.

    Call this once at startup:
        vector_store = initialize_rag("my_resume.pdf")
        retriever    = get_retriever(vector_store)
    """
    # Try loading existing first (faster)
    vector_store = load_vector_store()

    if vector_store is None:
        if pdf_path is None or not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"\n❌  No vector store found and no valid PDF path given.\n"
                f"    Please upload your resume PDF and pass its path.\n"
                f"    Example: initialize_rag('resume.pdf')\n"
            )
        chunks = load_and_split_pdf(pdf_path)
        vector_store = build_vector_store(chunks)

    return vector_store
