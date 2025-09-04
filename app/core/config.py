import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "")
    OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "LLM Travel Agent")
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
    
    # Application Configuration
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"  # Disable demo mode
    
    # Model Configuration
    DEFAULT_MODEL: str = "z-ai/glm-4.5-air:free"
    MAX_TOKENS: int = 300
    TEMPERATURE: float = 0.7
    
    # Conversation Configuration
    MAX_CONVERSATION_HISTORY: int = 15  # Increased from 20 for better performance
    CONTEXT_WINDOW_SIZE: int = 12000   # Increased for better context retention
    
    # External API URLs
    WEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    COUNTRIES_BASE_URL: str = "https://restcountries.com/v3.1"


settings = Settings()
