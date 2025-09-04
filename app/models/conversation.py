from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class ConversationState(str, Enum):
    """States in the conversation flow"""
    GREETING = "greeting"
    DESTINATION_PLANNING = "destination_planning" 
    ACTIVITY_DISCOVERY = "activity_discovery"
    PRACTICAL_PLANNING = "practical_planning"
    CONTEXT_ENRICHMENT = "context_enrichment"
    RECOMMENDATION_REFINEMENT = "recommendation_refinement"


class UserIntent(str, Enum):
    """Detected user intents"""
    DESTINATION_INQUIRY = "destination_inquiry"
    ACTIVITY_REQUEST = "activity_request"
    WEATHER_CHECK = "weather_check"
    CULTURAL_INFO = "cultural_info"
    PRACTICAL_ADVICE = "practical_advice"
    BUDGET_PLANNING = "budget_planning"
    PACKING_HELP = "packing_help"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class ConversationContext(BaseModel):
    """Rich context tracking for personalized responses"""
    session_id: str
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    travel_profile: Dict[str, Any] = Field(default_factory=dict)
    current_destination: Optional[str] = None
    travel_dates: Optional[Dict[str, str]] = None
    budget_range: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    previous_destinations: List[str] = Field(default_factory=list)
    conversation_depth: int = 0
    last_external_data_used: Optional[str] = None
    weather_mentioned_for: Optional[str] = None  # Track destination for which weather was mentioned
    weather_mentioned_at: Optional[datetime] = None  # Track when weather was mentioned


class ConversationSession(BaseModel):
    session_id: str
    state: ConversationState = ConversationState.GREETING
    context: ConversationContext
    messages: List[Message] = Field(default_factory=list)
    detected_intents: List[UserIntent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    state: ConversationState
    suggestions: Optional[List[str]] = None
    external_data_used: bool = False
    reasoning_trace: Optional[str] = None


class WeatherData(BaseModel):
    location: str
    temperature: float
    condition: str
    humidity: int
    wind_speed: float
    description: str


class CountryInfo(BaseModel):
    name: str
    capital: str
    population: int
    currencies: List[str]
    languages: List[str]
    timezone: str
    continent: str
