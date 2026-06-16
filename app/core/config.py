from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "HealthAI Diagnosis Assistant"
    openai_api_key: str = ""
    base_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-ada-002"
    fine_tuned_model: str = ""          # populated after fine-tuning
    vector_store_path: str = "data/medical_kb"
    temperature: float = 0.1            # low temp for clinical accuracy
    max_tokens: int = 2048
    safety_threshold: float = 0.85     # confidence below this → flag for review
    rlhf_log_path: str = "data/rlhf_feedback.jsonl"
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
