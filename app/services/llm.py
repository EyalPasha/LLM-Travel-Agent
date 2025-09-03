"""
Hugging Face LLM Integration Service with Router API
"""

import httpx
import json
import re
from typing import Dict, Any, Optional, List
from app.core.config import settings


class ResponseValidator:
    """Validates and enhances LLM responses for quality"""
    
    @staticmethod
    def is_hallucination(response: str, user_query: str) -> bool:
        """Detect potential hallucinations in travel responses"""
        hallucination_indicators = [
            r"I don't have access to current",
            r"I cannot provide real-time",
            r"As an AI, I cannot",
            r"I don't have the ability to",
            r"fictional place",
            r"made-up destination"
        ]
        
        for indicator in hallucination_indicators:
            if re.search(indicator, response, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def is_confused_response(response: str) -> bool:
        """Detect confused or off-topic responses"""
        confusion_indicators = [
            len(response.strip()) < 20,  # Too short
            response.count("?") > 5,  # Too many questions
            "I'm not sure" in response,
            "I don't understand" in response,
            response.strip().endswith("...") and len(response) < 100
        ]
        
        return any(confusion_indicators)
    
    @staticmethod
    def enhance_response_quality(response: str, context: str) -> str:
        """Enhance response quality and fix common issues"""
        # Remove redundant phrases
        response = re.sub(r'^(Hello! |Hi! |Great question! )', '', response)
        
        # Ensure practical value
        if len(response.split('.')) < 2:
            response += " Would you like more specific details about this?"
        
        return response.strip()


class HuggingFaceService:
    """Enhanced LLM service with intelligent error handling"""
    
    def __init__(self):
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.base_url = "https://router.huggingface.co/v1/chat/completions"
        self.model = "Qwen/Qwen2.5-7B-Instruct:together"  # Working model
        self.max_tokens = 1500
        self.temperature = 0.7
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None) -> Optional[str]:
        """Generate response using Hugging Face Router API"""
        
        if not self.api_key:
            return self._fallback_response("I need my Hugging Face API key to be configured to provide intelligent responses.")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": messages,
                "model": self.model,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    raw_response = data["choices"][0]["message"]["content"]
                    
                    # Validate response quality
                    user_query = messages[-1]["content"] if messages else ""
                    
                    # Check for hallucinations
                    if ResponseValidator.is_hallucination(raw_response, user_query):
                        return self._recovery_response("hallucination", user_query)
                    
                    # Check for confusion
                    if ResponseValidator.is_confused_response(raw_response):
                        return self._recovery_response("confusion", user_query)
                    
                    # Enhance response quality
                    enhanced_response = ResponseValidator.enhance_response_quality(raw_response, user_query)
                    
                    return enhanced_response
                else:
                    print(f"Hugging Face API error: {response.status_code} - {response.text}")
                    return self._fallback_response("I'm experiencing technical difficulties. Let me provide a basic response.")
        
        except httpx.TimeoutException:
            return self._fallback_response("The response took too long. Let me give you a quick answer.")
        except Exception as e:
            print(f"Hugging Face service error: {e}")
            return self._fallback_response("I'm having trouble connecting to my AI service right now.")
    
    async def generate_with_reasoning_trace(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], Optional[str]]:
        """Generate response with visible reasoning trace for debugging"""
        
        # Add instruction for reasoning trace
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            enhanced_messages[-1]["content"] += "\n\nPlease show your reasoning process clearly."
        
        try:
            response = await self.generate_response(enhanced_messages, temperature=0.3)
            
            if response:
                # Try to extract reasoning if present
                if "REASONING:" in response or "ANALYSIS:" in response:
                    parts = response.split("FINAL RESPONSE:")
                    if len(parts) == 2:
                        return parts[1].strip(), parts[0].strip()
                
                return response, "Standard response generation"
            
        except Exception as e:
            print(f"Reasoning trace error: {e}")
        
        return None, None
    
    def _recovery_response(self, error_type: str, user_query: str) -> str:
        """Generate intelligent recovery responses for different error types"""
        
        if error_type == "hallucination":
            # Gracefully acknowledge and redirect
            return f"I want to give you accurate information about your travel question. Let me focus on what I know for certain: could you tell me more specifically what aspect of travel planning you'd like help with? I can provide reliable guidance on destinations, activities, or logistics."
        
        elif error_type == "confusion":
            # Ask clarifying question based on query
            if any(word in user_query.lower() for word in ["where", "destination", "place"]):
                return "I'd love to help you find the perfect destination! To give you the best recommendations, could you share what type of experience you're looking for - cultural immersion, outdoor adventure, relaxation, or something else?"
            
            elif any(word in user_query.lower() for word in ["do", "activities", "see"]):
                return "There are so many great activities to consider! Are you interested in historical sites, local cuisine, outdoor adventures, arts and culture, or perhaps a mix of everything?"
            
            else:
                return "I want to make sure I understand exactly what you're looking for. Could you help me by sharing a bit more detail about your travel plans or questions?"
        
        return self._fallback_response(user_query)
    
    def _fallback_response(self, base_message: str) -> str:
        """Provide intelligent fallback responses when API is unavailable"""
        fallback_responses = {
            "destination": "I'd love to help you discover amazing travel destinations! Popular choices really depend on what you're seeking - are you drawn to rich cultural experiences, stunning natural landscapes, adventure activities, or peaceful relaxation? Share what excites you most about travel and I can suggest some incredible places.",
            
            "activities": "For activities, I always recommend mixing the iconic must-sees with hidden local gems. The best trips balance famous attractions with neighborhood wandering, organized tours with spontaneous discoveries, and active sightseeing with moments to simply absorb the atmosphere. What kind of experiences usually captivate you when you travel?",
            
            "practical": "For practical planning, the key essentials are visa requirements, local transportation options, currency and payment methods, and emergency contacts. I also recommend registering with your embassy for international trips and sharing your itinerary with someone at home. What specific logistics are you wondering about?",
            
            "general": "I'm excited to help plan your next adventure! I can assist with destination recommendations, activity suggestions, cultural insights, and practical travel advice. What aspect of your trip would you like to explore first?"
        }
        
        # Simple keyword matching for better fallback
        message_lower = base_message.lower()
        if any(word in message_lower for word in ["destination", "where", "place"]):
            return fallback_responses["destination"]
        elif any(word in message_lower for word in ["activity", "do", "see"]):
            return fallback_responses["activities"]
        elif any(word in message_lower for word in ["practical", "visa", "money", "transport"]):
            return fallback_responses["practical"]
        else:
            return fallback_responses["general"]
    
    def validate_response(self, response: str) -> tuple[bool, str]:
        """Validate LLM response quality and provide improvement suggestions"""
        
        if not response or len(response.strip()) < 50:
            return False, "Response too short or empty"
        
        # Check for obvious bot language
        obvious_bot_phrases = [
            "I don't have access to real-time",
            "I cannot browse the internet",
            "As an AI, I cannot",
            "I don't have the ability to",
            "I'm an AI assistant",
            "As a language model",
            "I'm programmed to",
            "My training data",
            "I'm not able to access"
        ]
        
        if any(phrase in response for phrase in obvious_bot_phrases):
            return False, "Response contains obvious bot language"
        
        # Check for helpful travel content
        travel_quality_indicators = [
            "specific recommendations",
            "practical advice",
            "cultural insights",
            "local tips",
            "personal experience",
            "insider knowledge"
        ]
        
        quality_score = sum(1 for indicator in travel_quality_indicators if indicator in response.lower())
        
        if quality_score >= 2:
            return True, "High quality travel response"
        elif quality_score >= 1:
            return True, "Good travel response"
        else:
            return False, "Response lacks specific travel insights"
    
    async def enhance_response_with_data(self, base_response: str, external_data: Dict[str, Any]) -> str:
        """Enhance response by incorporating external data"""
        
        if not external_data:
            return base_response
        
        enhancement_prompt = f"""
Please enhance this travel response by naturally incorporating the following real-time information:

ORIGINAL RESPONSE:
{base_response}

ADDITIONAL INFORMATION TO WEAVE IN:
{json.dumps(external_data, indent=2)}

Please rewrite the response to seamlessly include relevant details while maintaining the original tone and helpfulness. The additional information should enhance rather than replace the original insights.

ENHANCED RESPONSE:
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert travel consultant who seamlessly integrates real-time information into personalized travel advice."},
                {"role": "user", "content": enhancement_prompt}
            ]
            
            enhanced = await self.generate_response(messages, temperature=0.5)
            return enhanced if enhanced else base_response
            
        except Exception as e:
            print(f"Response enhancement error: {e}")
            return base_response
    
    async def generate_contextual_suggestions(self, response: str, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Generate AI-powered contextual suggestions based on response and conversation history"""
        
        if not self.api_key:
            return [
                "Tell me more about this destination",
                "What's the weather like there?",
                "Any local customs I should know?"
            ]
        
        try:
            # Create a focused prompt for suggestion generation
            suggestion_prompt = f"""
Based on this specific travel conversation, generate 3 short questions that the USER would type as follow-up messages.

RECENT AI RESPONSE: "{response}"

CONVERSATION CONTEXT: {json.dumps(conversation_history[-3:] if len(conversation_history) > 3 else conversation_history)}

Generate 3 follow-up questions that the user would send to Sofia, responding to what she just said:

IMPORTANT: Write from USER perspective, NOT AI perspective!
ALWAYS READ THE ENTIRE REPLAY FROM SOFIA AND INCORPORATE IT INTO YOUR RESPONSES.

Read Sofia's response and generate 3 natural user replies:
- Direct responses to Sofia's questions or suggestions
- User statements or user questions to Sofia
- 5-12 words each
- NO "Do you..." or "Would you..." or "Are you..." phrases

Format as a simple list:
1. [user message]
2. [user message] 
3. [user message]
"""
            
            messages = [
                {"role": "system", "content": "Generate user messages that respond to the AI. NEVER use 'Do you', 'Would you', 'Are you' - these are AI phrases. Always write what the USER would type to the AI."},
                {"role": "user", "content": suggestion_prompt}
            ]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": messages,
                "model": self.model,
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    suggestions_text = data["choices"][0]["message"]["content"]
                    
                    # Parse the numbered list
                    suggestions = []
                    for line in suggestions_text.split('\n'):
                        line = line.strip()
                        if line and (line.startswith(('1.', '2.', '3.', '-', '•'))):
                            # Remove numbering and clean up
                            suggestion = line.split('.', 1)[-1].strip()
                            if suggestion and len(suggestion) > 5:
                                suggestions.append(suggestion)
                    
                    return suggestions[:3] if suggestions else self._fallback_suggestions()
                else:
                    return self._fallback_suggestions()
                    
        except Exception as e:
            print(f"Suggestion generation error: {e}")
            return self._fallback_suggestions()
    
    def _fallback_suggestions(self) -> List[str]:
        """Fallback suggestions when AI generation fails"""
        return [
            "What's the best time to visit?",
            "Tell me about local cuisine",
            "Any cultural tips for travelers?"
        ]
