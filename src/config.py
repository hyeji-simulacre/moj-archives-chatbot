import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
BM25_DIR = PROJECT_ROOT / "bm25_index"

# Load .env from PKM vault
VAULT_ENV = Path.home() / "Documents" / "정혜지-AI-Archives-PKM-2026-v1" / "00-system" / "03-config" / ".env"
if VAULT_ENV.exists():
    load_dotenv(VAULT_ENV)

# Streamlit Cloud secrets or env
try:
    import streamlit as st
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 100
WEBSNS_TEXT_LIMIT = 500

SEARCH_TOP_K = 10
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
