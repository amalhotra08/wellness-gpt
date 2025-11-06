from pydantic import BaseModel, Field
import os


class Settings(BaseModel):
    """Runtime settings. Use GROQ exclusively for LLM calls.

    Note: GROQ_API_KEY is read from the environment. Do NOT commit secrets.
    """
    GROQ_API_KEY: str = Field(default_factory=lambda: os.getenv('GROQ_API_KEY', ''))
    # add other keys: PUBMED_EMAIL, CROSSREF_MAILTO, etc.