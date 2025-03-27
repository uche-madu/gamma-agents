from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    GROQ_API_KEY: SecretStr = Field(..., alias='GROQ_API_KEY')
    ALPHA_VANTAGE_API_KEY: SecretStr


    model_config = SettingsConfigDict(
        env_file="../.env",
        extra="allow",
        env_file_encoding="utf-8",
    )

settings = Settings() # type: ignore
