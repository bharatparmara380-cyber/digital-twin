# ─────────────────────────────────────────────────────────────────────────────
#  config.py  –  Central configuration for the Digital Twin project
#
#  CONCEPT: Instead of hardcoding API keys and settings everywhere,
#  we collect them in ONE place. This is a real-world best practice.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv  # reads key=value pairs from the .env file

# load_dotenv() looks for a file named `.env` in the current directory
# and loads each line as an environment variable automatically.
load_dotenv()

# ── LLM Settings ────────────────────────────────────────────────────────────

# os.getenv() reads the variable we set in .env
# The second argument is a fallback default if the variable is missing.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Groq model choices (all FREE):
#   "llama-3.3-70b-versatile"  ← best quality, recommended
#   "llama3-8b-8192"           ← fastest, lighter
#   "mixtral-8x7b-32768"       ← great for long contexts
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# ── Embedding Model Settings ─────────────────────────────────────────────────

# This small model runs 100% locally (no internet needed after first download).
# It converts text → numbers (vectors) so we can do similarity search.
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── RAG / Vector Store Settings ──────────────────────────────────────────────

# Where to save the FAISS index so we don't re-build it every run
VECTOR_STORE_PATH: str = "vector_store"

# How many characters each text chunk should be (for splitting the PDF)
CHUNK_SIZE: int = 500

# How many characters overlap between consecutive chunks
# (avoids losing context at chunk boundaries)
CHUNK_OVERLAP: int = 50

# How many top-matching chunks to retrieve when answering a question
TOP_K_RESULTS: int = 4

# ── Personal Identity ─────────────────────────────────────────────────────────

YOUR_NAME: str = os.getenv("YOUR_NAME", "Alex")
YOUR_TAGLINE: str = os.getenv("YOUR_TAGLINE", "Software Engineer | AI Enthusiast")

# ── Validation ────────────────────────────────────────────────────────────────

def validate_config() -> None:
    """Raises an error early if required keys are missing."""
    if not GROQ_API_KEY:
        raise ValueError(
            "\n\n❌  GROQ_API_KEY is missing!\n"
            "    Steps to fix:\n"
            "    1. Go to https://console.groq.com and sign up (free)\n"
            "    2. Create an API key\n"
            "    3. Copy .env.example → .env and paste your key there\n"
        )
