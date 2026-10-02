"""Centralized application configuration, loaded from environment variables."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "DocuChat AI"
    environment: str = "development"

    # Ollama (local LLM)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_seconds: int = 120

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "docuchat_docs"

    # Ingestion
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 4

    # Auth
    api_keys: str = "devkey123"  # comma-separated list of valid keys

    # Rate limiting
    rate_limit: str = "30/minute"

    # Agent
    agent_max_steps: int = 5

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
