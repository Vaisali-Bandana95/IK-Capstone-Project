import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model_name: str | None
    temperature: float
    api_base_url: str | None
    port: int


def load_settings() -> Settings:
    temperature = float(
        os.getenv("TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0"))
    )
    if not 0 <= temperature <= 2:
        raise ValueError("LLM_TEMPERATURE must be between 0 and 2")

    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    if not 1 <= port <= 65535:
        raise ValueError("GRADIO_SERVER_PORT must be between 1 and 65535")

    return Settings(
        api_key=os.getenv("OPENAI_API_KEY") or None,
        model_name=os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME") or None,
        temperature=temperature,
        api_base_url=os.getenv("OPENAI_API_BASE") or None,
        port=port,
    )


settings = load_settings()
