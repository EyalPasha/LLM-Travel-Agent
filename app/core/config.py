import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Configuration
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_BASE_URL: str = os.getenv("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co")
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
    
    # Application Configuration
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"  # Disable demo mode
    
    # Model Configuration
    DEFAULT_MODEL: str = "Qwen/Qwen2.5-7B-Instruct:together"
    MAX_TOKENS: int = 300
    TEMPERATURE: float = 0.7
    
    # Conversation Configuration
    MAX_CONVERSATION_HISTORY: int = 20
    CONTEXT_WINDOW_SIZE: int = 8000
    
    # External API URLs
    WEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    COUNTRIES_BASE_URL: str = "https://restcountries.com/v3.1"


settings = Settings()
