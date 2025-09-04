"""
OpenRouter LLM Integration Service
"""

import httpx
import json
import re
from typing import Dict, Any, Optional, List
from app.core.config import settings

class TokenManager:
    """Manages token counting and context window limits"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters for most models)"""
        return len(text) // 4
    
    @staticmethod
    def calculate_messages_tokens(messages: List[Dict[str, str]]) -> int:
        """Calculate total tokens for a list of messages"""
        total = 0
        for message in messages:
            # Add some overhead for message structure
            total += TokenManager.estimate_tokens(message.get('content', '')) + 10
        return total
    
    @staticmethod
    def trim_messages_to_fit(messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Trim messages to fit within token limit, preserving system message and recent context"""
        if not messages:
            return messages
        
        # Always keep system message (first one)
        system_message = messages[0] if messages[0].get('role') == 'system' else None
        conversation_messages = messages[1:] if system_message else messages
        
        # Calculate tokens for system message
        system_tokens = TokenManager.estimate_tokens(system_message['content']) + 10 if system_message else 0
        available_tokens = max_tokens - system_tokens - 500  # Reserve 500 tokens for response
        
        # Start from the most recent messages and work backwards
        trimmed_conversation = []
        current_tokens = 0
        
        for message in reversed(conversation_messages):
            message_tokens = TokenManager.estimate_tokens(message.get('content', '')) + 10
            if current_tokens + message_tokens <= available_tokens:
                trimmed_conversation.insert(0, message)
                current_tokens += message_tokens
            else:
                break
        
        # Reconstruct messages with system message first
        result = []
        if system_message:
            result.append(system_message)
        result.extend(trimmed_conversation)
        
        return result


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


class OpenRouterService:
    """Enhanced LLM service with intelligent error handling using OpenRouter"""
    
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        self.model = settings.DEFAULT_MODEL
        self.max_tokens = 1500
        self.temperature = 0.7
        self.site_url = settings.OPENROUTER_SITE_URL
        self.site_name = settings.OPENROUTER_SITE_NAME
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None) -> Optional[str]:
        """Generate response using OpenRouter API with token management"""
        
        if not self.api_key:
            return self._fallback_response("I need my OpenRouter API key to be configured to provide intelligent responses.")
        
        try:
            # Trim messages to fit within context window
            trimmed_messages = TokenManager.trim_messages_to_fit(messages, settings.CONTEXT_WINDOW_SIZE)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Add optional headers if configured
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                headers["X-Title"] = self.site_name
            
            payload = {
                "model": self.model,
                "messages": trimmed_messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
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
                    return self._fallback_response("I'm experiencing technical difficulties. Let me provide a basic response.")
        
        except httpx.TimeoutException:
            return self._fallback_response("The response took too long. Let me give you a quick answer.")
        except Exception as e:
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
            pass
        
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
        """Enhance response by incorporating external data only when contextually relevant"""
        
        if not external_data:
            return base_response
        
        # Check if the response or original query was actually about weather
        response_lower = base_response.lower()
        weather_related = any(keyword in response_lower for keyword in [
            'weather', 'temperature', 'climate', 'rain', 'sunny', 'cold', 'hot', 
            'degrees', 'celsius', 'humid', 'windy', 'forecast'
        ])
        
        # Filter external data to only include contextually relevant information
        relevant_data = {}
        for data_type, data in external_data.items():
            if data_type in ['weather', 'forecast']:
                # Only include weather data if the response is actually about weather
                if weather_related:
                    relevant_data[data_type] = data
            else:
                # Include non-weather data
                relevant_data[data_type] = data
        
        if not relevant_data:
            return base_response
        
        enhancement_prompt = f"""
Please enhance this travel response by naturally incorporating ONLY the relevant real-time information provided:

ORIGINAL RESPONSE:
{base_response}

RELEVANT INFORMATION TO WEAVE IN:
{json.dumps(relevant_data, indent=2)}

IMPORTANT GUIDELINES:
- Only incorporate information that DIRECTLY relates to the response content
- If the response is about photography, activities, or general travel advice, DO NOT force weather information
- Weather data should only be included if the response specifically discusses weather, climate, or conditions
- Keep the enhancement natural and brief - don't change the response's focus
- If no relevant information can be naturally incorporated, return the original response unchanged

ENHANCED RESPONSE:
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert travel consultant who only incorporates external data when it naturally enhances the response. Do not force irrelevant information into responses."},
                {"role": "user", "content": enhancement_prompt}
            ]
            
            enhanced = await self.generate_response(messages, temperature=0.5)
            return enhanced if enhanced else base_response
            
        except Exception as e:
            return base_response
    
    async def generate_contextual_suggestions(self, response: str, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Generate contextual suggestions using only LLM - no hardcoded patterns"""
        
        try:
            if self.api_key:
                llm_suggestions = await self._generate_llm_suggestions(response, conversation_history)
                if llm_suggestions and len(llm_suggestions) >= 3:
                    return llm_suggestions[:3]
                else:
                    # If LLM fails, return simple context-free fallbacks
                    return ["Tell me more", "What do you think?", "Any other ideas?"]
            else:
                return ["Tell me more", "What do you think?", "Any other ideas?"]
            
        except Exception as e:
            return ["Tell me more", "What do you think?", "Any other ideas?"]
    
    async def _generate_llm_suggestions(self, response: str, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Generate suggestions using LLM with simple, reliable prompt"""
        
        # Get user's last message for context
        user_context = ""
        if conversation_history:
            last_user = [msg['content'] for msg in conversation_history if msg.get('role') == 'user']
            if last_user:
                user_context = last_user[-1][:200]  # Limit context length
        
        # Ultra-simple prompt that forces direct responses
        suggestion_prompt = f"""Travel advisor: "{response[:300]}"

User replies:
1) "I'm interested in cultural experiences"
2) "Nature sounds perfect" 
3) "Tell me more about Portugal"

Based on the advisor's message, write 3 similar user replies:
1) "
2) "
3) """
        
        try:
            messages = [
                {"role": "user", "content": suggestion_prompt}
            ]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                headers["X-Title"] = self.site_name
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7,
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                api_response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                if api_response.status_code == 200:
                    data = api_response.json()
                    
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        message = choice["message"]
                        
                        # Try to get content from either content field or reasoning field
                        suggestions_text = message.get("content", "").strip()
                        
                        # If content is empty but reasoning exists, extract from reasoning
                        if not suggestions_text and "reasoning" in message:
                            reasoning = message["reasoning"]
                            # Try to extract suggestions from reasoning
                            if "responses" in reasoning.lower() or "user" in reasoning.lower():
                                suggestions_text = reasoning
                        
                        if not suggestions_text:
                            return None
                        
                        # Parse suggestions more reliably
                        suggestions = self._parse_suggestions_reliably(suggestions_text)
                        
                        if len(suggestions) >= 3:
                            return suggestions[:3]
                        else:
                            return None
                    else:
                        return None
                else:
                    return None
                    
        except httpx.TimeoutException:
            return None
        except Exception as e:
            return None
    
    def _parse_suggestions_reliably(self, text: str) -> List[str]:
        """Parse suggestions from LLM response with better reliability"""
        suggestions = []
        
        # Split by lines and look for numbered items or bullet points
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match various formats: "1.", "1)", "•", "-", "*"
            if re.match(r'^[\d\.\)\-\*\•]\s*', line):
                # Clean the line
                clean_line = re.sub(r'^[\d\.\)\-\*\•\s]+', '', line).strip()
                clean_line = re.sub(r'^["\']|["\']$', '', clean_line)  # Remove quotes
                clean_line = re.sub(r'\*\*', '', clean_line)  # Remove markdown formatting
                clean_line = re.sub(r'^\*|:\*$', '', clean_line)  # Remove asterisks and colons
                
                # Filter out unwanted patterns
                if (5 <= len(clean_line) <= 100 and 
                    clean_line not in suggestions and
                    not clean_line.endswith(':**') and  # No markdown headers
                    not clean_line.startswith('**') and  # No markdown bold
                    not re.match(r'^[A-Z\s]+:$', clean_line) and  # No ALL CAPS headers
                    'recommendations' not in clean_line.lower() and  # No meta text
                    'suggestions' not in clean_line.lower()):  # No meta text
                    suggestions.append(clean_line)
        
        # If numbered format didn't work, try to extract quoted responses
        if len(suggestions) < 2:
            quoted = re.findall(r'"([^"]+)"', text)
            for quote in quoted:
                quote = quote.strip()
                if (5 <= len(quote) <= 100 and 
                    quote not in suggestions and
                    not quote.lower().startswith('hello') and  # No greetings
                    'advisor' not in quote.lower()):  # No meta references
                    suggestions.append(quote)
                    if len(suggestions) >= 3:
                        break
        
        # If still not enough, look for lines starting with "I" (user statements)
        if len(suggestions) < 2:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if (line.startswith("I'm") or line.startswith("I would") or line.startswith("I prefer")) and len(line) <= 100:
                    clean_line = re.sub(r'\*\*', '', line)  # Remove markdown
                    if clean_line not in suggestions:
                        suggestions.append(clean_line)
                        if len(suggestions) >= 3:
                            break
        
        # Last resort: create simple contextual suggestions
        if len(suggestions) < 1:
            if "cultural" in text.lower():
                suggestions.append("I'm interested in cultural experiences")
            if "nature" in text.lower() or "outdoor" in text.lower():
                suggestions.append("Nature and outdoor activities sound great")
            if "portugal" in text.lower():
                suggestions.append("Tell me more about Portugal")
            
            # Fill remaining slots with generic but useful responses
            while len(suggestions) < 3:
                fallbacks = ["That sounds interesting", "What would you recommend?", "Tell me more"]
                for fallback in fallbacks:
                    if fallback not in suggestions:
                        suggestions.append(fallback)
                        break
                if len(suggestions) >= 3:
                    break
        
        return suggestions[:3]
    
    def _extract_response_elements(self, response: str) -> str:
        """Extract key elements from Sofia's response for better suggestion generation"""
        elements = []
        
        # Extract questions Sofia asked
        questions = re.findall(r'[^.!]*\?', response)
        if questions:
            elements.append(f"Questions asked: {', '.join(questions[:2])}")
        
        # Extract options/choices presented
        if "or" in response.lower():
            choices = re.findall(r'([^.!?]*\bor\b[^.!?]*)', response, re.IGNORECASE)
            if choices:
                elements.append(f"Choices presented: {choices[0].strip()}")
        
        # Extract specific locations mentioned
        # Common proper nouns (capitalize words that might be places)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        locations = [word for word in capitalized_words if len(word) > 3 and word not in ['To', 'What', 'Your', 'The', 'Are', 'Is']]
        if locations:
            elements.append(f"Locations mentioned: {', '.join(locations[:3])}")
        
        # Extract action words/advice
        action_words = re.findall(r'\b(explore|visit|try|experience|discover|enjoy|consider|plan|book|pack)\w*\b', response.lower())
        if action_words:
            elements.append(f"Actions suggested: {', '.join(set(action_words[:3]))}")
        
        return " | ".join(elements) if elements else "General travel discussion"
    
    def _parse_suggestions_robust(self, suggestions_text: str) -> List[str]:
        """Robustly parse suggestions from LLM response"""
        suggestions = []
        
        # Split by lines and look for numbered items
        lines = suggestions_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match various numbering formats
            match = re.match(r'^[\d\.\-\*\•\►\→]\s*(.+)$', line)
            if match:
                suggestion = match.group(1).strip()
                # Clean up common artifacts
                suggestion = re.sub(r'^["\'`]|["\'`]$', '', suggestion)  # Remove quotes
                suggestion = re.sub(r'\s+', ' ', suggestion)  # Normalize whitespace
                
                if 3 <= len(suggestion) <= 80 and suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        # If numbered format didn't work, try to extract sentences
        if len(suggestions) < 3:
            sentences = re.split(r'[.!?]+', suggestions_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if 3 <= len(sentence) <= 80 and sentence not in suggestions:
                    suggestions.append(sentence)
        
        return suggestions
    
    def _validate_suggestions_quality(self, suggestions: List[str], original_response: str) -> List[str]:
        """Validate that suggestions are high quality and contextual"""
        valid_suggestions = []
        
        for suggestion in suggestions:
            # Skip if too generic
            generic_phrases = [
                "tell me more", "what do you think", "any suggestions", 
                "can you help", "what about", "do you have", "would you recommend"
            ]
            
            if any(phrase in suggestion.lower() for phrase in generic_phrases):
                continue
            
            # Skip if it's just asking about Sofia instead of responding to her
            if any(phrase in suggestion.lower() for phrase in ["what do you", "do you think", "would you", "can you"]):
                continue
            
            # Must be reasonable length
            if not (3 <= len(suggestion) <= 80):
                continue
            
            valid_suggestions.append(suggestion)
        
        return valid_suggestions
    
    def _generate_intelligent_fallback_suggestions(self, response: str, conversation_history: List[Dict[str, str]] = None) -> List[str]:
        """Generate suggestions purely from context - no hardcoded patterns"""
        
        response_lower = response.lower()
        suggestions = []
        
        # Get user's conversation context
        user_context = ""
        if conversation_history:
            user_messages = [msg['content'] for msg in conversation_history if msg.get('role') == 'user']
            user_context = " ".join(user_messages).lower()
        
        # 1. Extract specific elements Sofia mentioned in her response
        locations = self._extract_locations_from_response(response)
        activities = self._extract_activities_from_response(response)
        questions_sofia_asked = self._extract_questions_from_response(response)
        recommendations = self._extract_recommendations_from_response(response)
        
        # 2. Generate contextual follow-ups based on what Sofia actually said
        
        # If Sofia mentioned specific places, ask about them
        if locations:
            primary_location = locations[0]
            suggestions.append(f"Tell me more about {primary_location}")
            if len(locations) > 1:
                suggestions.append(f"How does {primary_location} compare to {locations[1]}?")
        
        # If Sofia asked questions, provide preference-based answers
        if questions_sofia_asked:
            question = questions_sofia_asked[0]
            if "what" in question.lower():
                # Extract the topic from the question
                topic = self._extract_topic_from_question(question)
                if topic:
                    suggestions.append(f"I'm thinking {topic}")
            elif "would you" in question.lower() or "do you" in question.lower():
                suggestions.append("Yes, that sounds perfect")
                suggestions.append("I'm not sure, what do you think?")
        
        # If Sofia made recommendations, ask for more details
        if recommendations:
            rec = recommendations[0]
            suggestions.append(f"What makes {rec} special?")
            suggestions.append(f"Any specific tips for {rec}?")
        
        # If Sofia mentioned activities, show interest or ask for alternatives
        if activities:
            activity = activities[0]
            suggestions.append(f"The {activity} sounds interesting")
            if len(activities) > 1:
                suggestions.append(f"I prefer {activities[1]} over {activities[0]}")
        
        # 3. If we still need more suggestions, extract general follow-up opportunities
        while len(suggestions) < 3:
            # Look for any actionable advice Sofia gave
            if "should" in response_lower or "recommend" in response_lower:
                if "should" not in [s.lower() for s in suggestions]:
                    suggestions.append("What should I do first?")
            elif "can" in response_lower or "could" in response_lower:
                if "how" not in [s.lower() for s in suggestions]:
                    suggestions.append("How do I get started?")
            elif any(word in response_lower for word in ["perfect", "ideal", "great", "good"]):
                if "why" not in [s.lower() for s in suggestions]:
                    suggestions.append("Why is that the best option?")
            else:
                # Generate from user's original context if available
                if "budget" in user_context and "budget" not in response_lower:
                    suggestions.append("What about budget considerations?")
                elif "time" in user_context or "when" in user_context:
                    suggestions.append("What's the best timing for this?")
                else:
                    suggestions.append("What would you personally choose?")
                break
        
        # 4. Clean and deduplicate
        cleaned_suggestions = []
        for s in suggestions[:3]:
            if s and len(s.strip()) > 5 and s not in cleaned_suggestions:
                cleaned_suggestions.append(s.strip())
        
        # 5. Ensure we have exactly 3 suggestions
        while len(cleaned_suggestions) < 3:
            generic_options = [
                "Tell me more about that",
                "What's your personal opinion?",
                "Any other alternatives?"
            ]
            for option in generic_options:
                if option not in cleaned_suggestions:
                    cleaned_suggestions.append(option)
                    break
            if len(cleaned_suggestions) >= 3:
                break
        
        result = cleaned_suggestions[:3]
        return result
    
    def _extract_locations_from_response(self, response: str) -> List[str]:
        """Extract specific location names from Sofia's response"""
        # Look for capitalized words that might be places
        import re
        locations = []
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        
        # Filter out common non-location words including greetings
        non_locations = {
            'Sofia', 'To', 'What', 'Your', 'The', 'Are', 'Is', 'You', 'I', 'It', 
            'This', 'That', 'And', 'Or', 'But', 'For', 'In', 'On', 'At', 'With', 
            'From', 'Hello', 'Hi', 'Hey', 'Welcome', 'Great', 'Good', 'Amazing', 
            'Perfect', 'Cultural', 'Natural', 'Beach', 'Food', 'Active', 'Adventure',
            'History', 'Culinary', 'Tropical', 'First', 'Solo', 'Trip', 'Travel',
            'Travelers', 'Destinations', 'Experience', 'Experiences', 'Looking',
            'Something', 'Everything', 'Places', 'Cities', 'Mountains', 'Nature'
        }
        
        # Only include words that are likely actual place names
        for word in words:
            if (word not in non_locations and 
                len(word) > 2 and 
                not word.endswith('ing') and  # avoid gerunds
                not word.endswith('ed') and   # avoid past participles
                word not in response.split()[:3]):  # avoid words in greeting/opening
                locations.append(word)
        
        return locations[:3]  # Return top 3
    
    def _extract_activities_from_response(self, response: str) -> List[str]:
        """Extract activities/experiences Sofia mentioned"""
        import re
        activities = []
        activity_patterns = [
            r'\b(hiking|photography|museums?|food|beaches?|culture|adventure|relaxation|shopping|nightlife)\b',
            r'\b(exploring|visiting|trying|experiencing|discovering|enjoying)\s+([a-z]+)\b'
        ]
        
        for pattern in activity_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                if isinstance(matches[0], tuple):
                    activities.extend([match[1] for match in matches])
                else:
                    activities.extend(matches)
        
        return list(set(activities))[:3]
    
    def _extract_questions_from_response(self, response: str) -> List[str]:
        """Extract questions Sofia asked"""
        questions = re.findall(r'[^.!]*\?', response)
        return [q.strip() for q in questions if len(q.strip()) > 10][:2]
    
    def _extract_recommendations_from_response(self, response: str) -> List[str]:
        """Extract recommendations Sofia made"""
        recommendations = []
        
        # Look for phrases like "I recommend", "suggest", "perfect for"
        rec_patterns = [
            r'(?:recommend|suggest)\s+([^.!?]+)',
            r'([^.!?]+)\s+(?:is|are)\s+perfect for',
            r'try\s+([^.!?]+)',
            r'consider\s+([^.!?]+)'
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            recommendations.extend([match.strip() for match in matches])
        
        return recommendations[:2]
    
    def _extract_topic_from_question(self, question: str) -> str:
        """Extract the main topic from Sofia's question"""
        # Simple extraction - get the key noun/topic
        question_lower = question.lower()
        
        if "experience" in question_lower:
            return "a cultural experience"
        elif "destination" in question_lower:
            return "somewhere off the beaten path"
        elif "activity" in question_lower:
            return "something active"
        elif "food" in question_lower:
            return "trying local cuisine"
        elif "nature" in question_lower:
            return "nature and outdoor activities"
        
        return "something unique"
    
    def _analyze_response_context(self, response: str) -> str:
        """Analyze the AI response to extract key context for suggestion generation"""
        response_lower = response.lower()
        
        context_elements = []
        
        # Extract specific locations mentioned
        locations = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        if locations:
            context_elements.append(f"Locations mentioned: {', '.join(set(locations[:3]))}")
        
        # Extract activities/experiences
        activities = []
        activity_keywords = ['hiking', 'photography', 'museums', 'food', 'beaches', 'culture', 'adventure', 'relaxation', 'shopping', 'nightlife']
        for keyword in activity_keywords:
            if keyword in response_lower:
                activities.append(keyword)
        if activities:
            context_elements.append(f"Activities discussed: {', '.join(activities[:3])}")
        
        # Extract practical elements
        practical = []
        practical_keywords = ['budget', 'cost', 'price', 'time', 'season', 'weather', 'visa', 'transport', 'accommodation', 'safety']
        for keyword in practical_keywords:
            if keyword in response_lower:
                practical.append(keyword)
        if practical:
            context_elements.append(f"Practical aspects: {', '.join(practical[:3])}")
        
        # Extract sentiment/tone
        if any(word in response_lower for word in ['recommend', 'suggest', 'perfect', 'ideal', 'love']):
            context_elements.append("Tone: Recommending/Enthusiastic")
        elif any(word in response_lower for word in ['consider', 'depends', 'might', 'could']):
            context_elements.append("Tone: Advisory/Conditional")
        
        return " | ".join(context_elements) if context_elements else "General travel discussion"
    
    def _parse_suggestions(self, suggestions_text: str) -> List[str]:
        """Parse suggestions from LLM response"""
        suggestions = []
        
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '-', '•', '*'))):
                # Remove numbering and clean up
                suggestion = re.sub(r'^[\d\.\-\•\*\s]+', '', line).strip()
                if suggestion and len(suggestion) > 3 and len(suggestion) < 100:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _ensure_suggestion_variety(self, suggestions: List[str]) -> List[str]:
        """Ensure suggestions are varied and not repetitive"""
        if len(suggestions) <= 3:
            return suggestions
        
        # Remove near-duplicates by checking similarity
        unique_suggestions = []
        for suggestion in suggestions:
            is_unique = True
            for existing in unique_suggestions:
                # Check for similar structure or repeated words
                suggestion_words = set(suggestion.lower().split())
                existing_words = set(existing.lower().split())
                overlap = len(suggestion_words.intersection(existing_words))
                
                # If more than 50% word overlap, consider it too similar
                if overlap > len(suggestion_words) * 0.5:
                    is_unique = False
                    break
            
            if is_unique:
                unique_suggestions.append(suggestion)
            
            if len(unique_suggestions) >= 3:
                break
        
        return unique_suggestions
