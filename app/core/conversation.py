import uuid
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from app.models.conversation import (
    ConversationSession, ConversationContext, ConversationState, 
    UserIntent, Message, MessageRole
)
from app.prompts.engineering import PromptChainOrchestrator


class IntentDetector:
    """Advanced intent detection using pattern matching and context analysis"""
    
    INTENT_PATTERNS = {
        UserIntent.DESTINATION_INQUIRY: [
            # Basic destination questions
            r'\b(where|destination|place|country|city|visit|go|travel to|head to|fly to)\b',
            r'\b(recommend|suggest|best|good|top|favorite|amazing|incredible)\b.*\b(place|destination|country|city|spot|location)\b',
            r'\b(thinking about|considering|planning|want to|looking to|hoping to)\b.*\b(trip|travel|visit|vacation|journey|adventure)\b',
            # More natural patterns
            r'\b(should I go|worth visiting|tell me about|know about|heard about)\b',
            r'\b(which country|what country|which city|what city|which place|what place)\b',
            r'\b(where would you|where do you|any ideas for|suggestions for)\b.*\b(travel|trip|vacation)\b',
            # Decision making
            r'\b(choose between|deciding between|torn between|can\'t decide)\b.*\b(destination|place|country)\b',
            r'\b(first time|never been|always wanted)\b.*\b(visit|go|travel)\b'
        ],
        UserIntent.ACTIVITY_REQUEST: [
            # What to do patterns
            r'\b(what.*do|what.*see|what.*visit|what.*experience|what.*try)\b',
            r'\b(activities|things to do|attractions|sightseeing|must see|must do|highlights)\b',
            r'\b(fun|interesting|exciting|cool|awesome|amazing)\b.*\b(do|see|visit|experience|try)\b',
            # Specific activity types
            r'\b(museums|galleries|restaurants|cafes|bars|nightlife|shopping|markets|hiking|beaches|parks)\b',
            r'\b(tours|excursions|day trips|experiences|adventures|cultural|historic|scenic)\b',
            r'\b(activity|activities|entertainment|recreation)\b.*\b(suggestions|recommend|ideas)\b',
            # More natural expressions
            r'\b(keep me busy|things to see|worth doing|can\'t miss|bucket list)\b',
            r'\b(local spots|hidden gems|off beaten path|authentic|traditional)\b',
            r'\b(family friendly|romantic|solo travel|group)\b.*\b(activities|things|spots)\b',
            # Specific interests
            r'\b(food scene|art scene|music scene|outdoor|nature|history|architecture)\b',
            r'\b(photography|instagram|photos|scenic|beautiful|stunning)\b.*\b(spots|places|views)\b'
        ],
        UserIntent.WEATHER_CHECK: [
            # Weather basics
            r'\b(weather|temperature|climate|conditions|forecast)\b',
            r'\b(rain|sunny|cloudy|cold|hot|warm|cool|humid|dry|windy)\b',
            r'\b(degrees|celsius|fahrenheit|°C|°F)\b',
            # Seasonal and timing
            r'\b(best time|when to visit|good time|right time|season|seasonal)\b',
            r'\b(spring|summer|fall|autumn|winter|january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(rainy season|dry season|monsoon|hurricane|typhoon)\b',
            # Clothing and packing related
            r'\b(pack|packing|clothing|clothes|what to wear|dress|outfit)\b',
            r'\b(jacket|coat|shorts|sandals|boots|umbrella|sunscreen)\b',
            # Weather concerns
            r'\b(too hot|too cold|comfortable|pleasant|avoid|unbearable)\b.*\b(weather|temperature|climate)\b'
        ],
        UserIntent.CULTURAL_INFO: [
            # Culture basics
            r'\b(culture|cultural|customs|traditions|etiquette|manners|protocol)\b',
            r'\b(people|locals|society|community|lifestyle|way of life)\b',
            r'\b(history|historical|heritage|ancient|traditional|indigenous)\b',
            # Social aspects
            r'\b(language|speak|dialect|communication|translate|interpreter)\b',
            r'\b(religion|religious|spiritual|beliefs|ceremonies|temples|churches)\b',
            r'\b(festivals|celebrations|holidays|events|carnival|parade)\b',
            # Behavioral guidance
            r'\b(should I know|need to know|be aware|respectful|appropriate|inappropriate)\b',
            r'\b(taboo|forbidden|offensive|rude|polite|courtesy|respect)\b',
            r'\b(dress code|conservative|modest|formal|casual)\b',
            # Local insights
            r'\b(local customs|local way|how locals|local culture|authentic|genuine)\b',
            r'\b(social norms|cultural norms|expectations|behavior)\b'
        ],
        UserIntent.PRACTICAL_ADVICE: [
            # Documentation
            r'\b(visa|passport|documents|paperwork|requirements|permit|authorization)\b',
            r'\b(ID|identification|travel documents|embassy|consulate)\b',
            # Transportation
            r'\b(transportation|transport|getting around|how to get|travel within)\b',
            r'\b(airport|train|bus|metro|subway|taxi|uber|rental car|public transport)\b',
            r'\b(flights|airlines|booking|tickets|connections|layover)\b',
            # Safety and security
            r'\b(safety|safe|secure|dangerous|risky|crime|scam|theft|pickpocket)\b',
            r'\b(emergency|help|police|hospital|insurance|medical)\b',
            r'\b(precautions|careful|aware|avoid|sketchy|unsafe)\b',
            # Financial
            r'\b(money|currency|exchange|ATM|bank|credit card|cash|payment)\b',
            r'\b(tip|tipping|gratuity|service charge|tax)\b',
            # Communication and tech
            r'\b(internet|wifi|phone|sim card|data|roaming|calling)\b',
            r'\b(plug|adapter|voltage|charger|electronics)\b',
            # Accommodation
            r'\b(hotel|hostel|airbnb|accommodation|where to stay|booking|reservation)\b',
            # Health and safety specific
            r'\b(shots|vaccination|immunization|health|medical|water|food safety)\b',
            r'\b(what to watch out for|scams|avoid|careful|sketchy)\b'
        ],
        UserIntent.BUDGET_PLANNING: [
            # Cost basics
            r'\b(cost|costs|price|prices|expensive|cheap|affordable|budget|money)\b',
            r'\b(how much|spend|spending|expense|financial|economical)\b',
            r'\b(dollars|euros|pounds|yen|currency|exchange rate)\b',
            # Budget planning
            r'\b(budget|budgeting|financial planning|cost breakdown|estimate)\b',
            r'\b(daily cost|per day|weekly|monthly|total cost)\b',
            r'\b(save money|cut costs|economical|frugal|backpacker|luxury)\b',
            # Specific cost categories
            r'\b(food cost|meal prices|restaurant prices|street food)\b',
            r'\b(accommodation cost|hotel prices|hostel prices)\b',
            r'\b(transport cost|flight cost|train prices|bus fares)\b',
            r'\b(attraction prices|museum fees|tour costs|activity prices)\b',
            # Value and comparison
            r'\b(worth it|value|bang for buck|overpriced|reasonable|fair price)\b',
            r'\b(compare costs|cheaper alternative|budget option|splurge)\b',
            # Slang and casual expressions
            r'\b(break the bank|damage|pricey|dirt cheap|costs a fortune)\b',
            r'\b(on a dime|shoestring|tight budget|money tight)\b'
        ],
        UserIntent.PACKING_HELP: [
            # Packing basics
            r'\b(pack|packing|bring|take|luggage|suitcase|backpack|bag)\b',
            r'\b(what to pack|packing list|essentials|must bring|necessary)\b',
            # Clothing
            r'\b(clothes|clothing|outfit|dress|wear|wardrobe)\b',
            r'\b(shirts|pants|dress|jacket|coat|shoes|sandals|boots)\b',
            r'\b(underwear|socks|pajamas|swimwear|formal|casual)\b',
            # Items and accessories
            r'\b(toiletries|cosmetics|shampoo|toothbrush|medication)\b',
            r'\b(camera|charger|adapter|electronics|phone|laptop)\b',
            r'\b(documents|passport|tickets|insurance|copies)\b',
            # Packing strategies
            r'\b(light|minimal|overpacking|weight limit|carry on|checked)\b',
            r'\b(forgot|forgotten|leave behind|don\'t need|unnecessary)\b',
            # Climate specific
            r'\b(warm weather|cold weather|tropical|winter gear|summer clothes)\b',
            # Gear and equipment
            r'\b(gear|equipment|essentials|must have|need)\b'
        ]
    }
    
    def detect_intents(self, message: str, context: ConversationContext) -> List[UserIntent]:
        """Detect user intents from message using patterns and context"""
        detected = []
        message_lower = message.lower()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    detected.append(intent)
                    break
        
        # Enhanced context-based intent detection for implicit queries
        if context and context.current_destination:
            # Detect implicit weather queries when destination is established
            implicit_weather_patterns = [
                r'\bweather\b', r'\bhow\'s\b.*\bthere\b', r'\btemperature\b',
                r'\bcold\b', r'\bhot\b', r'\bwarm\b', r'\brain\b', r'\bsunny\b',
                r'\bwhat\'s.*like\b.*\bthere\b', r'\bclimate\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\bweather.*in\b.*\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
            ]
            
            if not detected or UserIntent.WEATHER_CHECK not in detected:
                for pattern in implicit_weather_patterns:
                    if re.search(pattern, message_lower):
                        detected.append(UserIntent.WEATHER_CHECK)
                        break
            
            # Detect implicit activity requests
            implicit_activity_patterns = [
                r'\bthere\b', r'\bwhat.*do\b', r'\bgood\b', r'\bbest\b',
                r'\brecommend\b', r'\bsuggestion\b', r'\bvisit\b'
            ]
            
            if not detected:
                for pattern in implicit_activity_patterns:
                    if re.search(pattern, message_lower):
                        detected.append(UserIntent.ACTIVITY_REQUEST)
                        break
        
        # Context-based intent enhancement
        if context and context.current_destination and not detected:
            # If we know destination but no clear intent, assume activity request
            detected.append(UserIntent.ACTIVITY_REQUEST)
        
        # Default to destination inquiry if no intents detected
        if not detected:
            detected.append(UserIntent.DESTINATION_INQUIRY)
        
        return list(set(detected))  # Remove duplicates


class PsychologicalProfiler:
    """Advanced psychological profiling system for personalized travel recommendations"""
    
    TRAVELER_ARCHETYPES = {
        'Explorer': {
            'keywords': ['adventure', 'explore', 'discover', 'unknown', 'off-beaten', 'remote'],
            'anti_keywords': ['familiar', 'safe', 'predictable', 'crowded']
        },
        'Connoisseur': {
            'keywords': ['authentic', 'local', 'culture', 'history', 'art', 'cuisine', 'wine', 'craft'],
            'anti_keywords': ['tourist', 'mainstream', 'commercial']
        },
        'Connector': {
            'keywords': ['people', 'locals', 'community', 'social', 'friends', 'meet', 'interact'],
            'anti_keywords': ['alone', 'solo', 'quiet']
        },
        'Creator': {
            'keywords': ['photography', 'art', 'workshop', 'learn', 'create', 'inspire', 'document'],
            'anti_keywords': ['passive', 'just looking']
        },
        'Seeker': {
            'keywords': ['spiritual', 'meaning', 'growth', 'transform', 'mindful', 'reflection', 'meditation'],
            'anti_keywords': ['party', 'nightlife', 'busy']
        }
    }
    
    def analyze_user_psychology(self, messages: List, session_id: str) -> Dict:
        """Deep psychological analysis of user based on conversation patterns"""
        
        all_text = ' '.join([msg.content.lower() for msg in messages if msg.role == MessageRole.USER])
        
        profile = {
            'traveler_archetype': self._detect_archetype(all_text),
            'energy_signature': self._detect_energy_pattern(all_text),
            'decision_pattern': self._detect_decision_style(all_text),
            'communication_style': self._analyze_communication(messages),
            'motivation_drivers': ', '.join(self._extract_motivations(all_text)),
            'risk_tolerance': self._assess_risk_tolerance(all_text),
            'life_stage_clues': self._detect_life_stage(all_text),
            'cultural_context': self._infer_cultural_background(all_text)
        }
        
        return profile
    
    def _detect_archetype(self, text: str) -> str:
        """Detect primary traveler archetype"""
        scores = {}
        
        for archetype, patterns in self.TRAVELER_ARCHETYPES.items():
            score = 0
            # Positive keywords
            for keyword in patterns['keywords']:
                score += text.count(keyword) * 2
            # Negative keywords (reduce score)
            for anti_keyword in patterns['anti_keywords']:
                score -= text.count(anti_keyword)
            scores[archetype] = max(0, score)
        
        return max(scores.keys(), key=lambda k: scores[k]) if any(scores.values()) else 'Explorer'
    
    def _detect_energy_pattern(self, text: str) -> str:
        """Detect energy and activity patterns"""
        energy_patterns = {
            'Steady': ['consistent', 'daily', 'routine', 'regular', 'plan', 'schedule'],
            'Burst': ['intense', 'full-day', 'packed', 'maximize', 'everything'],
            'Adaptive': ['flexible', 'spontaneous', 'depends', 'maybe', 'see how'],
            'Low-key': ['relaxed', 'slow', 'casual', 'easy', 'comfortable']
        }
        
        scores = {}
        for pattern, keywords in energy_patterns.items():
            score = sum(text.count(keyword) for keyword in keywords)
            scores[pattern] = score
        
        return max(scores.keys(), key=lambda k: scores[k]) if any(scores.values()) else 'Adaptive'
    
    def _detect_decision_style(self, text: str) -> str:
        """Analyze decision-making patterns"""
        decision_patterns = {
            'Analytical': ['research', 'compare', 'data', 'reviews', 'details', 'facts'],
            'Intuitive': ['feel', 'sense', 'vibe', 'instinct', 'drawn to', 'speaks to'],
            'Collaborative': ['partner', 'group', 'family', 'together', 'we', 'us'],
            'Decisive': ['decided', 'booked', 'confirmed', 'definitely', 'sure']
        }
        
        scores = {}
        for style, keywords in decision_patterns.items():
            score = sum(text.count(keyword) for keyword in keywords)
            scores[style] = score
        
        return max(scores.keys(), key=lambda k: scores[k]) if any(scores.values()) else 'Intuitive'
    
    def _analyze_communication(self, messages: List) -> str:
        """Analyze communication style from message patterns"""
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        
        if not user_messages:
            return 'Direct'
        
        avg_length = sum(len(msg.content) for msg in user_messages) / len(user_messages)
        question_ratio = sum('?' in msg.content for msg in user_messages) / len(user_messages)
        
        if avg_length > 100 and question_ratio > 0.3:
            return 'Detailed-Inquiring'
        elif avg_length < 50:
            return 'Concise-Direct'
        elif question_ratio > 0.5:
            return 'Question-Heavy'
        else:
            return 'Conversational'
    
    def _extract_motivations(self, text: str) -> List[str]:
        """Extract core motivations from text"""
        motivations = []
        
        motivation_patterns = {
            'Adventure': ['adventure', 'thrill', 'excitement', 'adrenaline'],
            'Learning': ['learn', 'understand', 'history', 'culture', 'knowledge'],
            'Relaxation': ['relax', 'unwind', 'peaceful', 'calm', 'spa'],
            'Connection': ['connect', 'meet', 'people', 'social', 'community'],
            'Status': ['luxury', 'exclusive', 'premium', 'prestigious', 'VIP'],
            'Authenticity': ['authentic', 'real', 'local', 'traditional', 'genuine']
        }
        
        for motivation, keywords in motivation_patterns.items():
            if any(keyword in text for keyword in keywords):
                motivations.append(motivation)
        
        return motivations or ['Adventure']
    
    def _assess_risk_tolerance(self, text: str) -> str:
        """Assess risk tolerance level"""
        high_risk_indicators = ['adventure', 'challenge', 'extreme', 'dangerous', 'risky']
        low_risk_indicators = ['safe', 'secure', 'comfortable', 'familiar', 'easy']
        
        high_score = sum(text.count(indicator) for indicator in high_risk_indicators)
        low_score = sum(text.count(indicator) for indicator in low_risk_indicators)
        
        if high_score > low_score * 2:
            return 'High'
        elif low_score > high_score * 2:
            return 'Low'
        else:
            return 'Moderate'
    
    def _detect_life_stage(self, text: str) -> str:
        """Infer life stage from language patterns"""
        indicators = {
            'Young-Professional': ['career', 'work', 'job', 'networking', 'professional'],
            'Family': ['family', 'kids', 'children', 'parents', 'together'],
            'Couple': ['partner', 'boyfriend', 'girlfriend', 'husband', 'wife', 'together'],
            'Solo-Explorer': ['solo', 'alone', 'myself', 'independent'],
            'Retiree': ['retirement', 'golden years', 'finally', 'time']
        }
        
        for stage, keywords in indicators.items():
            if any(keyword in text for keyword in keywords):
                return stage
        
        return 'General'
    
    def _infer_cultural_background(self, text: str) -> str:
        """Infer cultural background from communication patterns"""
        formal_indicators = ['would like', 'could you please', 'I would appreciate']
        casual_indicators = ['gonna', 'wanna', 'yeah', 'cool', 'awesome']
        
        formal_score = sum(text.count(indicator) for indicator in formal_indicators)
        casual_score = sum(text.count(indicator) for indicator in casual_indicators)
        
        if formal_score > casual_score:
            return 'Formal-Traditional'
        elif casual_score > formal_score:
            return 'Casual-Modern'
        else:
            return 'Balanced'


class SmartContextManager:
    """Advanced context management with conversation memory and pattern recognition"""
    
    def __init__(self):
        self.conversation_themes = {}
        self.user_preferences = {}
        self.decision_points = {}
        self.psychological_profiler = PsychologicalProfiler()
        self.conversation_memory = {}  # Enhanced memory system
        self.learning_patterns = {}    # Pattern learning from interactions
    
    def extract_evolving_context(self, messages: List, session_id: str) -> Dict:
        """Extract and track evolving conversation context"""
        
        context = {
            "destinations_mentioned": [],
            "activity_preferences": [],
            "budget_indicators": [],
            "travel_style": None,
            "decision_stage": "exploration",
            "conversation_momentum": "building"
        }
        
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        for msg in recent_messages:
            if msg.role == MessageRole.USER:
                # Track destination evolution
                destinations = self._extract_destinations(msg.content)
                context["destinations_mentioned"].extend(destinations)
                
                # Detect decision progression
                if any(word in msg.content.lower() for word in ["book", "reserve", "confirmed", "decided"]):
                    context["decision_stage"] = "commitment"
                elif any(word in msg.content.lower() for word in ["compare", "vs", "versus", "better"]):
                    context["decision_stage"] = "evaluation"
        
        # Remove duplicates and keep recent
        context["destinations_mentioned"] = list(dict.fromkeys(context["destinations_mentioned"]))[-3:]
        
        return context
    
    def track_conversation_quality(self, session_id: str, user_msg: str, ai_response: str) -> Dict:
        """Track conversation quality metrics for improvement"""
        
        quality_metrics = {
            "user_engagement": self._measure_engagement(user_msg),
            "response_relevance": self._measure_relevance(user_msg, ai_response),
            "conversation_progress": self._measure_progress(session_id),
            "user_satisfaction_indicators": self._detect_satisfaction(user_msg),
            "conversation_depth_quality": self._assess_depth_quality(user_msg),
            "follow_through_score": self._calculate_follow_through(session_id, user_msg)
        }
        
        # Store in memory for learning
        self._update_conversation_memory(session_id, quality_metrics, user_msg, ai_response)
        
        return quality_metrics
    
    def _measure_engagement(self, user_msg: str) -> float:
        """Measure user engagement level"""
        engagement_score = 0.5  # baseline
        
        # Length indicates engagement
        if len(user_msg) > 50:
            engagement_score += 0.2
        
        # Questions indicate interest
        if "?" in user_msg:
            engagement_score += 0.1
        
        # Specific details indicate investment
        if any(pattern in user_msg.lower() for pattern in ["specific", "detail", "exactly", "precisely"]):
            engagement_score += 0.1
        
        # Enthusiasm indicators
        if any(word in user_msg.lower() for word in ["excited", "love", "amazing", "perfect", "exactly"]):
            engagement_score += 0.1
        
        return min(engagement_score, 1.0)
    
    def _measure_relevance(self, user_msg: str, ai_response: str) -> float:
        """Measure how relevant the AI response is to user message"""
        user_keywords = set(user_msg.lower().split())
        response_keywords = set(ai_response.lower().split())
        
        # Basic keyword overlap
        overlap = len(user_keywords.intersection(response_keywords))
        relevance = min(overlap / max(len(user_keywords), 1), 1.0)
        
        return relevance
    
    def _detect_satisfaction(self, user_msg: str) -> List[str]:
        """Detect satisfaction indicators in user messages"""
        indicators = []
        
        positive_signals = ["perfect", "exactly", "great", "helpful", "thank you", "thanks"]
        negative_signals = ["not what", "confused", "don't understand", "not helpful"]
        
        msg_lower = user_msg.lower()
        
        for signal in positive_signals:
            if signal in msg_lower:
                indicators.append(f"positive_{signal.replace(' ', '_')}")
        
        for signal in negative_signals:
            if signal in msg_lower:
                indicators.append(f"negative_{signal.replace(' ', '_')}")
        
        return indicators
    
    def _extract_destinations(self, text: str) -> List[str]:
        """Extract destination mentions from text using intelligent pattern matching"""
        import re
        
        destinations = []
        
        # Enhanced destination patterns for natural language
        destination_patterns = [
            # Direct mentions with action verbs
            r'\b(?:visit|go to|traveling to|going to|planning.*to|flying to|heading to|off to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:trip|travel|vacation|visit|journey|adventure|getaway)\b',
            r'\b(?:in|at|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            
            # Question patterns
            r'\b(?:about|regarding|concerning)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:any good|worth visiting|recommendations|suggestions)\b',
            
            # Comparative patterns
            r'\b(?:than|versus|vs\.?|compared to|like|similar to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            
            # Possessive and descriptive patterns
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[\'s]*\s+(?:weather|climate|food|culture|people|attractions|museums|restaurants)\b',
            r'\b(?:the|beautiful|amazing|incredible|stunning)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        # Extract from enhanced patterns
        for pattern in destination_patterns:
            matches = re.findall(pattern, text)
            destinations.extend(matches)
        
        # Smart filtering to remove common false positives
        false_positives = {
            'the', 'and', 'or', 'but', 'with', 'from', 'like', 'about', 'what', 'where', 'when', 
            'how', 'why', 'who', 'which', 'this', 'that', 'these', 'those', 'there', 'here',
            'good', 'best', 'nice', 'great', 'amazing', 'beautiful', 'wonderful', 'perfect',
            'place', 'places', 'destination', 'country', 'city', 'trip', 'travel', 'vacation',
            'time', 'day', 'week', 'month', 'year', 'season', 'weather', 'climate',
            'people', 'person', 'man', 'woman', 'family', 'friend', 'friends',
            'food', 'restaurant', 'hotel', 'museum', 'beach', 'mountain', 'river', 'lake',
            'first', 'second', 'third', 'last', 'next', 'previous', 'many', 'much', 'some',
            'all', 'every', 'each', 'both', 'either', 'neither', 'other', 'another',
            'same', 'different', 'new', 'old', 'young', 'big', 'small', 'large', 'little',
            'long', 'short', 'high', 'low', 'hot', 'cold', 'warm', 'cool', 'wet', 'dry'
        }
        
        # Enhanced word detection for geographical entities
        # Look for capitalized words that might be places
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out false positives and add potential destinations
        for word in capitalized_words:
            word_clean = word.strip()
            if (word_clean.lower() not in false_positives and 
                len(word_clean) > 2 and 
                not word_clean.lower().startswith(('i\'', 'i ', 'a ', 'an ')) and
                word_clean not in destinations):
                destinations.append(word_clean)
        
        # Additional context-based extraction
        # Look for "in [place]" or "to [place]" patterns more aggressively
        location_indicators = re.findall(r'\b(?:in|to|at|from|near|around|visiting|exploring)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text)
        destinations.extend(location_indicators)
        
        # Remove duplicates and clean up
        unique_destinations = []
        for dest in destinations:
            cleaned = dest.strip()
            if cleaned and cleaned not in unique_destinations and cleaned.lower() not in false_positives:
                unique_destinations.append(cleaned)
        
        return unique_destinations
    
    def _measure_progress(self, session_id: str) -> float:
        """Measure conversation progress towards actionable outcome"""
        if session_id in self.conversation_memory:
            memory = self.conversation_memory[session_id]
            progress_score = len(memory.get('concrete_plans', [])) * 0.2
            progress_score += len(memory.get('decisions_made', [])) * 0.3
            return min(progress_score, 1.0)
        return 0.5
    
    def _assess_depth_quality(self, user_msg: str) -> float:
        """Assess the depth and quality of user engagement"""
        quality_indicators = [
            len(user_msg) > 80,  # Detailed messages
            '?' in user_msg,     # Questions show engagement
            any(word in user_msg.lower() for word in ['why', 'how', 'what about', 'tell me more']),
            any(word in user_msg.lower() for word in ['experience', 'feel', 'think', 'prefer'])
        ]
        return sum(quality_indicators) / len(quality_indicators)
    
    def _calculate_follow_through(self, session_id: str, user_msg: str) -> float:
        """Calculate how well the conversation follows previous threads"""
        if session_id not in self.conversation_memory:
            return 0.5
        
        memory = self.conversation_memory[session_id]
        previous_topics = memory.get('topics_discussed', [])
        
        # Check if user is building on previous topics
        msg_lower = user_msg.lower()
        follow_through_score = 0.0
        
        for topic in previous_topics[-3:]:  # Check last 3 topics
            if any(word in msg_lower for word in topic.lower().split()):
                follow_through_score += 0.3
        
        return min(follow_through_score, 1.0)
    
    def _update_conversation_memory(self, session_id: str, quality_metrics: Dict, 
                                  user_msg: str, ai_response: str):
        """Update conversation memory for learning and improvement"""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = {
                'topics_discussed': [],
                'concrete_plans': [],
                'decisions_made': [],
                'user_preferences': {},
                'conversation_flow': [],
                'quality_trends': []
            }
        
        memory = self.conversation_memory[session_id]
        
        # Extract topics from current exchange
        topics = self._extract_topics(user_msg + " " + ai_response)
        memory['topics_discussed'].extend(topics)
        
        # Track concrete plans and decisions
        if any(word in user_msg.lower() for word in ['decided', 'booked', 'will go', 'planning to']):
            memory['decisions_made'].append({
                'decision': user_msg,
                'timestamp': datetime.now().isoformat()
            })
        
        # Update quality trends
        memory['quality_trends'].append({
            'timestamp': datetime.now().isoformat(),
            'engagement': quality_metrics['user_engagement'],
            'relevance': quality_metrics['response_relevance']
        })
        
        # Keep memory manageable
        if len(memory['topics_discussed']) > 20:
            memory['topics_discussed'] = memory['topics_discussed'][-15:]
        if len(memory['quality_trends']) > 50:
            memory['quality_trends'] = memory['quality_trends'][-30:]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from conversation text using dynamic keyword analysis"""
        import re
        
        travel_topics = []
        text_lower = text.lower()
        
        # Dynamic topic detection with comprehensive keyword sets
        topic_keywords = {
            'destinations': {
                'patterns': [
                    r'\b(city|cities|country|countries|destination|place|location|region|area)\b',
                    r'\b(visit|travel|go|trip|vacation|journey|adventure|getaway)\b',
                    r'\b(capital|province|state|island|continent|territory)\b'
                ],
                'weight': 2
            },
            'activities': {
                'patterns': [
                    r'\b(museum|gallery|park|beach|mountain|hiking|climbing|swimming)\b',
                    r'\b(restaurant|cafe|bar|nightlife|shopping|market|festival|concert)\b',
                    r'\b(tour|excursion|sightseeing|photography|art|culture|history)\b',
                    r'\b(adventure|experience|activity|attraction|landmark|monument)\b',
                    r'\b(sports|outdoor|nature|wildlife|scenic|entertainment)\b'
                ],
                'weight': 2
            },
            'planning': {
                'patterns': [
                    r'\b(budget|cost|price|money|expensive|cheap|affordable)\b',
                    r'\b(time|season|weather|climate|temperature|when|schedule)\b',
                    r'\b(visa|passport|document|requirement|booking|reservation)\b',
                    r'\b(flight|train|bus|transport|accommodation|hotel|hostel)\b',
                    r'\b(pack|luggage|clothes|essentials|prepare|plan)\b'
                ],
                'weight': 1.5
            },
            'preferences': {
                'patterns': [
                    r'\b(prefer|like|love|enjoy|hate|dislike|avoid|interested)\b',
                    r'\b(want|need|looking|hoping|expecting|wish|dream)\b',
                    r'\b(favorite|best|worst|amazing|terrible|beautiful|ugly)\b',
                    r'\b(recommend|suggest|advice|opinion|thoughts|ideas)\b'
                ],
                'weight': 1
            },
            'cultural': {
                'patterns': [
                    r'\b(culture|tradition|custom|etiquette|language|people|local)\b',
                    r'\b(religion|spiritual|temple|church|ceremony|festival)\b',
                    r'\b(food|cuisine|dish|meal|cooking|recipe|authentic)\b',
                    r'\b(art|music|dance|literature|heritage|historical)\b'
                ],
                'weight': 1.5
            },
            'practical': {
                'patterns': [
                    r'\b(safety|safe|dangerous|security|crime|precaution)\b',
                    r'\b(health|medical|insurance|emergency|hospital|pharmacy)\b',
                    r'\b(communication|internet|phone|language|translate)\b',
                    r'\b(currency|exchange|payment|tip|bargain|negotiate)\b'
                ],
                'weight': 1
            }
        }
        
        # Score each topic based on pattern matches
        topic_scores = {}
        for topic, config in topic_keywords.items():
            score = 0
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches * config['weight']
            
            if score > 0:
                topic_scores[topic] = score
        
        # Add topics that meet threshold
        threshold = 1.0
        for topic, score in topic_scores.items():
            if score >= threshold:
                travel_topics.append(topic)
        
        # Add contextual topics based on combinations
        if 'destinations' in travel_topics and 'preferences' in travel_topics:
            travel_topics.append('destination_preferences')
        
        if 'activities' in travel_topics and 'cultural' in travel_topics:
            travel_topics.append('cultural_activities')
        
        if 'planning' in travel_topics and 'practical' in travel_topics:
            travel_topics.append('trip_logistics')
        
        return travel_topics[:10]  # Limit to prevent overloading
        
        return travel_topics


class ContextExtractor:
    """Enhanced context extraction with smart pattern recognition"""
    
    DESTINATION_PATTERNS = [
        # Direct travel action patterns - more precise with negative lookbehind/lookahead
        r'\b(?:visit|visiting|go to|going to|traveling to|trip to|fly to|heading to|bound for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?!\s+(?:before|after|while|when|if|unless|during|since|until))\b',
        r'\b(?:from|leaving)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:to|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        
        # Preposition-based patterns - more conservative with context checks
        r'\b(?:in|at|around|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?!\s+(?:before|after|while|when|if|unless|during|since|until|order|case|time|general))\b',
        r'\babout\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?!\s+(?:before|after|while|when|if|unless|during|since|until|time|it|that|this))\b',
        
        # Question patterns with context awareness
        r'\bhow\'s\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?!\s+(?:before|after|while|when|going|doing|working))\b',
        
        # Possessive patterns - more specific with better context
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\'s\s+(?:weather|climate|food|culture|people|attractions|museums|restaurants|nightlife|beaches|mountains|history|customs|traditions)\b',
        
        # Comparison patterns
        r'\b(?:than|versus|vs\.?|compared to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:vs\.?|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        
        # Context-specific patterns with better boundaries
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:trip|vacation|journey|adventure|getaway)\b',
        r'\b(?:through|across)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    ]
    
    BUDGET_PATTERNS = [
        r'\$(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'(\d+(?:,\d+)*)\s*dollars?',
        r'budget.*?(\d+(?:,\d+)*)',
        r'(cheap|budget|luxury|expensive|mid-range|affordable)'
    ]
    
    DATE_PATTERNS = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}(?:st|nd|rd|th)?)',
        r'\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})',
        r'\b(next|this)\s+(week|month|year|spring|summer|fall|autumn|winter)',
        r'\b(in|during)\s+(January|February|March|April|May|June|July|August|September|October|November|December)'
    ]
    
    def extract_destinations(self, message: str) -> List[str]:
        """Extract destination names from message with improved accuracy and no hardcoded lists"""
        destinations = []
        message_original = message
        message_lower = message.lower()
        
        # Apply all destination patterns
        for pattern in self.DESTINATION_PATTERNS:
            matches = re.findall(pattern, message_original, re.IGNORECASE)
            for match in matches:
                # Handle both single strings and tuples from regex groups
                if isinstance(match, tuple):
                    # For patterns with groups, take all non-empty groups
                    for m in match:
                        if m and m.strip():
                            destinations.append(m.strip())
                else:
                    if match and match.strip():
                        destinations.append(match.strip())
        
        # Enhanced false positive filtering
        false_positives = {
            # Common words that get capitalized
            'trip', 'travel', 'vacation', 'visit', 'going', 'planning', 'to', 'in', 'at', 'from',
            'the', 'and', 'or', 'but', 'with', 'like', 'about', 'what', 'how', 'when', 'where',
            'pack', 'weather', 'best', 'season', 'time', 'good', 'great', 'nice', 'beautiful',
            'there', 'here', 'place', 'places', 'destination', 'country', 'city', 'area',
            # Common names and titles that aren't places
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december', 'today', 'tomorrow', 'yesterday',
            # Common adjectives and verbs
            'amazing', 'incredible', 'wonderful', 'terrible', 'expensive', 'cheap', 'hot', 'cold',
            'people', 'person', 'family', 'friend', 'hotel', 'restaurant', 'museum', 'beach',
            # Internet/social media terms
            'instagram', 'facebook', 'google', 'youtube', 'reddit', 'twitter',
            # General terms
            'money', 'cost', 'price', 'budget', 'food', 'culture', 'language', 'airport',
            'flight', 'train', 'bus', 'car', 'taxi', 'uber', 'hotel', 'hostel', 'airbnb',
            # Travel-related non-destinations that get mistakenly captured
            'first solo', 'solo trip', 'solo travel', 'first time', 'next adventure',
            'adventure trip', 'photography trip', 'landscape photography', 'northern lights',
            'my trip', 'my vacation', 'my travel', 'group trip', 'family trip'
        }
        
        # Smart destination validation
        def is_likely_destination(dest: str) -> bool:
            dest_lower = dest.lower().strip()
            
            # Skip if it's a known false positive
            if dest_lower in false_positives:
                return False
            
            # Special check for common travel phrases that aren't destinations
            travel_phrases = [
                'first solo', 'solo trip', 'solo travel', 'first time', 'next adventure',
                'adventure trip', 'photography trip', 'landscape photography', 'northern lights',
                'my trip', 'my vacation', 'my travel', 'group trip', 'family trip',
                'bucket list', 'dream destination', 'perfect place'
            ]
            
            if any(phrase in dest_lower for phrase in travel_phrases):
                return False
            
            # Must be at least 3 characters
            if len(dest_lower) < 3:
                return False
            
            # Skip if it starts with common non-destination words
            if dest_lower.startswith(('the ', 'a ', 'an ', 'my ', 'your ', 'our ', 'their ', 'first ', 'next ', 'last ')):
                return False
            
            # Skip if it's all lowercase or all uppercase (likely not a proper place name)
            if dest == dest.lower() or dest == dest.upper():
                return False
            
            # Must start with capital letter (proper noun)
            if not dest[0].isupper():
                return False
            
            # Skip if it contains numbers (unlikely to be a place name in casual conversation)
            if any(char.isdigit() for char in dest):
                return False
            
            # Skip if it's a single character
            if len(dest) == 1:
                return False
            
            # Additional context check - if the word appears in typical travel contexts
            travel_context_indicators = [
                'visit', 'go to', 'travel', 'trip', 'vacation', 'weather', 'climate',
                'culture', 'food', 'people', 'attractions', 'museums', 'restaurants',
                'hotels', 'flights', 'currency', 'language', 'visa', 'passport'
            ]
            
            # Check if this destination appears near travel-related words
            dest_pos = message_lower.find(dest_lower)
            if dest_pos != -1:
                # Look at 50 characters before and after the destination
                context_start = max(0, dest_pos - 50)
                context_end = min(len(message_lower), dest_pos + len(dest_lower) + 50)
                context = message_lower[context_start:context_end]
                
                # If it appears near travel words, it's more likely to be a destination
                if any(indicator in context for indicator in travel_context_indicators):
                    return True
            
            # Default: if it passes basic checks, consider it a potential destination
            return True
        
        # Filter and clean destinations with context validation
        clean_destinations = []
        for dest in destinations:
            dest_clean = dest.strip().title()
            
            # Additional context validation - check what comes after the potential destination
            if self._validate_destination_context(dest_clean, message_original):
                if is_likely_destination(dest_clean) and dest_clean not in clean_destinations:
                    clean_destinations.append(dest_clean)
        
        return clean_destinations[:5]  # Limit to prevent false positive overload
    
    def _validate_destination_context(self, destination: str, original_message: str) -> bool:
        """Validate that the destination makes sense in its surrounding context"""
        dest_lower = destination.lower()
        message_lower = original_message.lower()
        
        # Find the position of the destination in the message
        dest_pos = message_lower.find(dest_lower)
        if dest_pos == -1:
            return False
        
        # Check what comes immediately after the destination
        after_dest_pos = dest_pos + len(dest_lower)
        if after_dest_pos < len(message_lower):
            # Get the next 20 characters to check context
            after_context = message_lower[after_dest_pos:after_dest_pos + 20].strip()
            
            # If it's followed by temporal/conditional words, it's likely not a standalone destination
            exclusion_words = [
                'before', 'after', 'while', 'when', 'if', 'unless', 'during', 
                'since', 'until', 'and then', 'or', 'but', 'however',
                'in order', 'in case', 'in time', 'in general'
            ]
            
            for word in exclusion_words:
                if after_context.startswith(word):
                    return False
        
        # Check what comes before the destination  
        before_context = message_lower[max(0, dest_pos - 20):dest_pos].strip()
        
        # If it's preceded by certain words, it might not be a destination
        if before_context.endswith(('about', 'like', 'such as', 'including')):
            # These could be valid, but need stronger evidence
            # Only accept if there are strong travel indicators nearby
            travel_indicators = ['visit', 'travel', 'trip', 'vacation', 'customs', 'culture']
            if not any(indicator in message_lower for indicator in travel_indicators):
                return False
        
        return True
    
    def extract_budget_info(self, message: str) -> Optional[str]:
        """Extract budget information from message"""
        for pattern in self.BUDGET_PATTERNS:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group()
        return None
    
    def extract_dates(self, message: str) -> Optional[Dict[str, str]]:
        """Extract travel dates from message"""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return {"raw_date": match.group(), "pattern_matched": pattern}
        return None
    
    def update_context(self, context: ConversationContext, message: str) -> ConversationContext:
        """Update context based on new message content with improved destination tracking"""
        # Extract destinations
        destinations = self.extract_destinations(message)
        if destinations:
            new_destination = destinations[0]  # Take first/primary destination
            
            # Only update if it's actually a new destination
            if context.current_destination and context.current_destination != new_destination:
                # Add current destination to previous destinations before switching
                if context.current_destination not in context.previous_destinations:
                    context.previous_destinations.append(context.current_destination)
            
            context.current_destination = new_destination
        
        # Extract budget
        budget = self.extract_budget_info(message)
        if budget:
            context.budget_range = budget
        
        # Extract dates
        dates = self.extract_dates(message)
        if dates:
            context.travel_dates = dates
        
        # Extract interests from the message
        interests = self._extract_interests(message)
        for interest in interests:
            if interest not in context.interests:
                context.interests.append(interest)
        
        # Increment conversation depth
        context.conversation_depth += 1
        
        return context
    
    def _extract_interests(self, message: str) -> List[str]:
        """Extract travel interests from message"""
        interests = []
        interest_keywords = {
            'museums': ['museum', 'art', 'gallery', 'culture', 'history'],
            'food': ['restaurant', 'food', 'cuisine', 'dining', 'eat'],
            'nightlife': ['nightlife', 'bar', 'club', 'party', 'drink'],
            'nature': ['nature', 'park', 'hiking', 'outdoor', 'beach', 'mountain'],
            'shopping': ['shopping', 'market', 'shop', 'buy', 'souvenir'],
            'architecture': ['architecture', 'building', 'church', 'temple', 'monument'],
            'couples': ['couple', 'romantic', 'honeymoon', 'date'],
            'family': ['family', 'kids', 'children', 'child'],
            'adventure': ['adventure', 'extreme', 'thrill', 'adrenaline']
        }
        
        message_lower = message.lower()
        for interest, keywords in interest_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                interests.append(interest)
        
        return interests


class StateManager:
    """Manages conversation state transitions based on context and intents"""
    
    STATE_TRANSITIONS = {
        ConversationState.GREETING: {
            UserIntent.DESTINATION_INQUIRY: ConversationState.DESTINATION_PLANNING,
            UserIntent.ACTIVITY_REQUEST: ConversationState.ACTIVITY_DISCOVERY,
            UserIntent.PRACTICAL_ADVICE: ConversationState.PRACTICAL_PLANNING
        },
        ConversationState.DESTINATION_PLANNING: {
            UserIntent.ACTIVITY_REQUEST: ConversationState.ACTIVITY_DISCOVERY,
            UserIntent.WEATHER_CHECK: ConversationState.CONTEXT_ENRICHMENT,
            UserIntent.CULTURAL_INFO: ConversationState.CONTEXT_ENRICHMENT,
            UserIntent.PRACTICAL_ADVICE: ConversationState.PRACTICAL_PLANNING
        },
        ConversationState.ACTIVITY_DISCOVERY: {
            UserIntent.DESTINATION_INQUIRY: ConversationState.RECOMMENDATION_REFINEMENT,
            UserIntent.PRACTICAL_ADVICE: ConversationState.PRACTICAL_PLANNING,
            UserIntent.WEATHER_CHECK: ConversationState.CONTEXT_ENRICHMENT
        },
        ConversationState.PRACTICAL_PLANNING: {
            UserIntent.ACTIVITY_REQUEST: ConversationState.ACTIVITY_DISCOVERY,
            UserIntent.DESTINATION_INQUIRY: ConversationState.RECOMMENDATION_REFINEMENT
        },
        ConversationState.CONTEXT_ENRICHMENT: {
            UserIntent.ACTIVITY_REQUEST: ConversationState.ACTIVITY_DISCOVERY,
            UserIntent.DESTINATION_INQUIRY: ConversationState.RECOMMENDATION_REFINEMENT
        },
        ConversationState.RECOMMENDATION_REFINEMENT: {
            # Can transition to any state based on user needs
        }
    }
    
    def determine_next_state(self, current_state: ConversationState, 
                           intents: List[UserIntent], 
                           context: ConversationContext) -> ConversationState:
        """Determine next conversation state based on current state and intents"""
        
        if not intents:
            return current_state
        
        primary_intent = intents[0]
        
        # Check for valid transitions
        if current_state in self.STATE_TRANSITIONS:
            transitions = self.STATE_TRANSITIONS[current_state]
            if primary_intent in transitions:
                return transitions[primary_intent]
        
        # Context-based overrides
        if context.conversation_depth > 8:
            return ConversationState.RECOMMENDATION_REFINEMENT
        
        # Default fallback logic
        if primary_intent == UserIntent.DESTINATION_INQUIRY:
            return ConversationState.DESTINATION_PLANNING
        elif primary_intent == UserIntent.ACTIVITY_REQUEST:
            return ConversationState.ACTIVITY_DISCOVERY
        elif primary_intent in [UserIntent.PRACTICAL_ADVICE, UserIntent.BUDGET_PLANNING]:
            return ConversationState.PRACTICAL_PLANNING
        else:
            return ConversationState.CONTEXT_ENRICHMENT


class ConversationEngine:
    """Main conversation orchestrator with advanced context and state management"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.intent_detector = IntentDetector()
        self.context_extractor = ContextExtractor()
        self.state_manager = StateManager()
        self.prompt_orchestrator = PromptChainOrchestrator()
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Get existing session or create new one"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        new_session_id = session_id or str(uuid.uuid4())
        context = ConversationContext(session_id=new_session_id)
        session = ConversationSession(
            session_id=new_session_id,
            context=context
        )
        self.sessions[new_session_id] = session
        return session
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> Tuple[ConversationSession, List[UserIntent]]:
        """Process user message and update conversation state"""
        
        # Get or create session
        session = self.get_or_create_session(session_id)
        
        # Add user message to history
        user_message = Message(role=MessageRole.USER, content=message)
        session.messages.append(user_message)
        
        # Validate and track user preferences to prevent contradictions
        self._track_user_preferences(session, message)
        
        # Detect intents
        intents = self.intent_detector.detect_intents(message, session.context)
        session.detected_intents = intents
        
        # Update context
        session.context = self.context_extractor.update_context(session.context, message)
        
        # Update conversation state
        new_state = self.state_manager.determine_next_state(
            session.state, intents, session.context
        )
        session.state = new_state
        session.updated_at = datetime.now()
        
        return session, intents
    
    def build_prompt_chain(self, session: ConversationSession, user_message: str, psychological_profile: Dict = None) -> str:
        """Build prompt chain for LLM with psychological insights"""
        
        # Detect if this is a simple informational query
        is_simple_query = self._is_simple_informational_query(user_message, session.detected_intents)
        
        if is_simple_query:
            # Use prompt for straightforward questions
            return self._build_simple_prompt(session, user_message)
        
        # Use full prompt for complex queries
        # Get system prompt
        system_prompt = self.prompt_orchestrator.get_system_prompt(session.context)
        
        # Build reasoning chain with psychological profile
        reasoning_chain = self.prompt_orchestrator.build_reasoning_chain(
            user_message, session.context, session.detected_intents, psychological_profile
        )
        
        # Combine with conversation history (limited)
        history_context = self._build_history_context(session)
        
        full_prompt = f"""
{system_prompt}

CONVERSATION HISTORY:
{history_context}

REASONING FRAMEWORK:
{reasoning_chain}

USER MESSAGE: {user_message}

Please provide your response following the reasoning framework above:
"""
        
        return full_prompt
    
    def _is_simple_informational_query(self, user_message: str, intents: List[UserIntent]) -> bool:
        """Detect simple informational queries that don't need complex reasoning"""
        # Weather queries
        if UserIntent.WEATHER_CHECK in intents:
            return True
        
        # Simple factual questions
        simple_patterns = [
            r'\bwhat.*weather\b',
            r'\bhow.*weather\b',
            r'\btemperature\b',
            r'\bwhat time\b',
            r'\bcurrency\b',
            r'\blanguage\b',
            r'\btimezone\b'
        ]
        
        message_lower = user_message.lower()
        return any(re.search(pattern, message_lower) for pattern in simple_patterns)
    
    def _build_simple_prompt(self, session: ConversationSession, user_message: str) -> str:
        """Build simple, concise prompt for factual queries with destination context awareness"""
        history_context = self._build_history_context(session)
        current_dest = session.context.current_destination
        
        # Build destination-aware instructions
        destination_context = ""
        if current_dest:
            destination_context = f"""
⚠️  CRITICAL DESTINATION CONTEXT: The user is asking about {current_dest}
- Your answer MUST be specific to {current_dest}
- NEVER give generic responses when {current_dest} is the established destination
- Always mention "{current_dest}" by name in your response
"""
        
        return f"""You are Sofia, a helpful travel consultant. Answer the user's question directly and concisely.

{destination_context}
CONVERSATION CONTEXT:
{history_context}

INSTRUCTIONS:
- Answer the question directly in 1-2 sentences
- Be specific to the established destination context
- Add ONE brief, relevant follow-up or tip
- Keep total response under 100 words
- Be helpful but concise

USER QUESTION: {user_message}

Provide a clear, concise answer:"""
    
    def _build_history_context(self, session: ConversationSession) -> str:
        """Build rich conversation history for context with destination awareness"""
        if not session.messages:
            return "This is the start of a new conversation."
        
        # Get last few messages for context
        recent_messages = session.messages[-8:]  # Increased to 8 messages for better context
        history_lines = []
        
        for msg in recent_messages:
            role_prefix = "User" if msg.role == MessageRole.USER else "Assistant"
            # Include full message content for better context
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            history_lines.append(f"{role_prefix}: {content}")
        
        # Enhanced pronoun and reference resolution
        last_user_message = None
        last_assistant_message = None
        for msg in reversed(session.messages):
            if msg.role == MessageRole.USER and not last_user_message:
                last_user_message = msg.content
            elif msg.role == MessageRole.ASSISTANT and not last_assistant_message:
                last_assistant_message = msg.content
            if last_user_message and last_assistant_message:
                break
        
        # Extract implicit context clues
        implicit_context = self._extract_implicit_context(session.messages)
        
        # Add context summary
        context_summary = f"""

CONVERSATION CONTEXT SUMMARY:
- PRIMARY DESTINATION FOCUS: {session.context.current_destination or 'Not specified'} 
- RECENTLY DISCUSSED LOCATIONS: {', '.join(session.context.previous_destinations[-3:]) if session.context.previous_destinations else 'None'}
- User interests: {', '.join(session.context.interests) if session.context.interests else 'Not yet identified'}
- Budget mentioned: {session.context.budget_range or 'Not specified'}
- Travel dates: {session.context.travel_dates.get('raw_date', 'Not specified') if session.context.travel_dates else 'Not specified'}
- Conversation depth: {session.context.conversation_depth} exchanges
- Current conversation state: {session.state.value}

CRITICAL PRONOUN & REFERENCE RESOLUTION:
- When user says "there": ALWAYS refers to "{session.context.current_destination or 'the primary destination being discussed'}"
- When user says "it": Refers to the last mentioned topic/activity: "{implicit_context.get('last_topic', 'the current topic of discussion')}"
- Weather questions without location: ALWAYS assume "{session.context.current_destination or 'the destination being discussed'}"
- "The weather" or "How's the weather": MUST refer to "{session.context.current_destination}" weather

⚠️  CRITICAL: NEVER GIVE GENERIC RESPONSES WHEN A DESTINATION IS ESTABLISHED
⚠️  ALWAYS mention the specific destination name in your response
⚠️  If current_destination is set, your response MUST be specific to that location

IMPLICIT CONTEXT AWARENESS:
- Last user message: "{last_user_message[:100] + '...' if last_user_message and len(last_user_message) > 100 else last_user_message or 'None'}"
- Last assistant topic: "{implicit_context.get('last_assistant_topic', 'None')}"
- Conversation momentum: "{implicit_context.get('momentum', 'building')}"

RECENT CONVERSATION HISTORY:
{chr(10).join(history_lines)}

USER PREFERENCE MEMORY (NEVER CONTRADICT THESE):
{self._extract_confirmed_preferences(session)}

CONVERSATION PROGRESSION RULES:
- Exchange {session.context.conversation_depth}: {"Building context and preferences" if session.context.conversation_depth <= 2 else "Providing specific recommendations" if session.context.conversation_depth <= 5 else "Deepening planning details"}
- NEVER repeat exact information already discussed
- Each response should advance the conversation forward
- Build on previously established context rather than starting fresh

MANDATORY RESPONSE RULES:
1. NEVER give generic responses when context shows a specific destination
2. ALWAYS use the destination name explicitly in your response, don't just say "there"
3. If user asks about weather without specifying location, use the current destination focus
4. Reference previous conversation naturally and build on established context
5. NEVER contradict previously confirmed user preferences (especially interest types)
6. Don't ask about preferences you've already confirmed (e.g., don't ask about city vs landscape if user already said landscapes)
"""
        
        return context_summary
    
    def _extract_confirmed_preferences(self, session: ConversationSession) -> str:
        """Extract and remember confirmed user preferences to prevent contradictions"""
        preferences = []
        
        # Extract from conversation history
        for msg in session.messages:
            if msg.role == MessageRole.USER:
                content = msg.content.lower()
                
                # Interest types
                if any(word in content for word in ['landscape', 'photography', 'photo']):
                    preferences.append("CONFIRMED: User wants landscape photography")
                if any(word in content for word in ['city', 'urban', 'vibrant city']):
                    preferences.append("CONFIRMED: User wants city experiences")
                if any(word in content for word in ['culture', 'cultural', 'local']):
                    preferences.append("CONFIRMED: User wants cultural experiences")
                if any(word in content for word in ['adventure', 'hiking', 'outdoor']):
                    preferences.append("CONFIRMED: User wants adventure activities")
                
                # Trip type
                if any(word in content for word in ['solo', 'first', 'alone']):
                    preferences.append("CONFIRMED: First solo trip")
                if any(word in content for word in ['overwhelmed', 'nervous', 'worried']):
                    preferences.append("CONFIRMED: User feeling nervous/overwhelmed")
                
                # Specific interests mentioned
                if 'incredible landscapes' in content:
                    preferences.append("CONFIRMED: Wants incredible landscapes (NOT cities)")
                if 'love photography' in content:
                    preferences.append("CONFIRMED: Photography enthusiast")
        
        # From stored context
        if session.context.interests:
            for interest in session.context.interests:
                preferences.append(f"STORED: {interest}")
        
        if session.context.current_destination:
            preferences.append(f"DESTINATION FOCUS: {session.context.current_destination}")
        
        return "\n".join(preferences) if preferences else "No confirmed preferences yet"
    
    def _track_user_preferences(self, session: ConversationSession, message: str):
        """Track and validate user preferences to prevent contradictions"""
        message_lower = message.lower()
        
        # Track travel style preferences
        if any(word in message_lower for word in ['landscape', 'photography', 'incredible landscapes']):
            if 'landscape_photography' not in session.context.interests:
                session.context.interests = session.context.interests or []
                session.context.interests.append('landscape_photography')
        
        if any(word in message_lower for word in ['solo', 'first', 'alone', 'overwhelmed']):
            if 'solo_travel' not in session.context.interests:
                session.context.interests = session.context.interests or []
                session.context.interests.append('solo_travel')
        
        # Track explicit destination mentions
        destinations = ['iceland', 'banff', 'new zealand', 'norway', 'patagonia', 'dolomites', 
                       'paris', 'tokyo', 'new york', 'london', 'barcelona']
        
        for dest in destinations:
            if dest in message_lower and not session.context.current_destination:
                # Validate destination matches user's interests
                if self._validate_destination_relevance(dest, session.context.interests):
                    session.context.current_destination = dest.title()
                    break
    
    def _validate_destination_relevance(self, destination: str, interests: List[str]) -> bool:
        """Validate that suggested destination matches user's stated interests"""
        if not interests:
            return True  # No conflicts yet
        
        landscape_destinations = ['iceland', 'banff', 'new zealand', 'norway', 'patagonia', 'dolomites']
        city_destinations = ['paris', 'tokyo', 'new york', 'london', 'barcelona']
        
        # If user wants landscape photography, don't accept city destinations
        if 'landscape_photography' in interests and destination.lower() in city_destinations:
            return False
        
        # If user wants city experiences, landscape destinations might still be ok
        # (some cities have both urban and natural elements)
        
        return True
    
    def _extract_implicit_context(self, messages: List) -> Dict[str, str]:
        """Extract implicit context clues from conversation history"""
        implicit_context = {
            'last_topic': 'the current topic of discussion',
            'last_assistant_topic': 'None',
            'momentum': 'building'
        }
        
        if not messages:
            return implicit_context
        
        # Look at recent messages for topic extraction
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        
        # Find last assistant response for topic extraction
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT:
                # Extract main topics from assistant response
                content = msg.content.lower()
                topics = []
                
                # Look for specific activities/attractions/topics
                activity_patterns = [
                    r'(museum[s]?)', r'(restaurant[s]?)', r'(beach[es]?)', 
                    r'(mountain[s]?)', r'(park[s]?)', r'(market[s]?)',
                    r'(nightlife)', r'(culture)', r'(food)', r'(weather)',
                    r'(northern lights?)', r'(aurora[s]?)', r'(kirkjufell)',
                    r'(jökulsárlón)', r'(glacier[s]?)', r'(iceland)', 
                    r'(reykjavik)', r'(photography)', r'(september)'
                ]
                
                for pattern in activity_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        topics.extend([match for match in matches if match])
                
                if topics:
                    implicit_context['last_assistant_topic'] = topics[-1].title()
                    implicit_context['last_topic'] = topics[-1].title()
                break
        
        # Analyze conversation momentum
        user_messages = [msg for msg in recent_messages if msg.role == MessageRole.USER]
        if user_messages:
            last_user_msg = user_messages[-1].content.lower()
            if any(word in last_user_msg for word in ['thanks', 'perfect', 'great', 'exactly']):
                implicit_context['momentum'] = 'positive'
            elif any(word in last_user_msg for word in ['confused', 'not what', 'different']):
                implicit_context['momentum'] = 'needs_clarification'
            elif len(last_user_msg) < 20:  # Short questions often indicate focused inquiry
                implicit_context['momentum'] = 'focused'
            else:
                implicit_context['momentum'] = 'exploring'
        
        return implicit_context
    
    def add_assistant_response(self, session: ConversationSession, response: str):
        """Add assistant response to conversation history"""
        assistant_message = Message(role=MessageRole.ASSISTANT, content=response)
        session.messages.append(assistant_message)
        session.updated_at = datetime.now()
