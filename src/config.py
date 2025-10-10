from pydantic import BaseModel, Field
import os

class Settings(BaseModel):
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv('OPENAI_API_KEY',''))
    # add other keys: PUBMED_EMAIL, CROSSREF_MAILTO, etc.