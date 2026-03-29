"""Root-level conftest.py — sets required env vars BEFORE any test module is collected.

This ensures backend.config.config can be imported without raising ValueError
for missing API_KEY / GEMINI_* env vars in unit test environments.
"""
import os

# Set required env vars before any module-level import triggers config loading.
# These are test-only values; real values must be in .env for production.
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-pro")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")
os.environ.setdefault("DB_URI", "sqlite:///:memory:")
os.environ.setdefault("FARMER_DATA_ALL", "farmer_data_all")
os.environ.setdefault("CHROMA_DB_PATH", "/tmp/test_chroma_lersha")
