"""
Advanced Error Recovery System for Production-Grade Reliability
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from app.models.conversation import ConversationSession, ConversationContext, UserIntent


class ErrorType(str, Enum):
    """Classification of error types for targeted recovery"""
    API_TIMEOUT = "api_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    INVALID_RESPONSE = "invalid_response" 
    CONTEXT_CONFUSION = "context_confusion"
    INTENT_AMBIGUITY = "intent_ambiguity"
    DATA_UNAVAILABLE = "data_unavailable"
    USER_FRUSTRATION = "user_frustration"
    SYSTEM_OVERLOAD = "system_overload"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different error scenarios"""
    GRACEFUL_FALLBACK = "graceful_fallback"
    CONTEXT_RESET = "context_reset"
    CLARIFICATION_REQUEST = "clarification_request"
    ALTERNATIVE_PATH = "alternative_path"
    ESCALATION = "escalation"
    LEARNING_OPPORTUNITY = "learning_opportunity"


class ErrorPattern:
    """Pattern matching for error detection and classification"""
    
    USER_FRUSTRATION_INDICATORS = [
        "this doesn't help", "not what I asked", "you don't understand",
        "this is wrong", "that's not helpful", "I already told you",
        "you're not listening", "this is confusing", "start over",
        "why this long", "too verbose", "too much text", "keep it short"
    ]
    
    CONTEXT_CONFUSION_INDICATORS = [
        "what are we talking about", "I'm lost", "can we start over",
        "I don't follow", "that doesn't make sense", "completely off topic"
    ]
    
    INTENT_AMBIGUITY_INDICATORS = [
        "I meant", "actually", "no, I was asking about", "that's not what I",
        "let me clarify", "I'm looking for", "I need help with"
    ]


class ConversationRecoveryEngine:
    """Conversation recovery system with learning capabilities"""
    
    def __init__(self):
        self.error_history = {}
        self.recovery_patterns = {}
        self.success_metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def detect_conversation_issues(self, user_message: str, session: ConversationSession, 
                                 ai_response: Optional[str] = None) -> List[Tuple[ErrorType, float]]:
        """Detect potential issues in the conversation with confidence scores"""
        
        detected_issues = []
        user_msg_lower = user_message.lower()
        
        # User frustration detection
        frustration_score = self._calculate_frustration_score(user_msg_lower, session)
        if frustration_score > 0.6:
            detected_issues.append((ErrorType.USER_FRUSTRATION, frustration_score))
        
        # Context confusion detection
        confusion_score = self._calculate_confusion_score(user_msg_lower, session)
        if confusion_score > 0.5:
            detected_issues.append((ErrorType.CONTEXT_CONFUSION, confusion_score))
        
        # Intent ambiguity detection
        ambiguity_score = self._calculate_ambiguity_score(user_msg_lower, session)
        if ambiguity_score > 0.4:
            detected_issues.append((ErrorType.INTENT_AMBIGUITY, ambiguity_score))
        
        # Response quality issues
        if ai_response:
            response_issues = self._detect_response_issues(ai_response, user_message)
            detected_issues.extend(response_issues)
        
        return detected_issues
    
    def generate_recovery_response(self, error_type: ErrorType, confidence: float,
                                 user_message: str, session: ConversationSession) -> Dict[str, Any]:
        """Generate intelligent recovery response based on error type and context"""
        
        recovery_data = {
            'strategy': self._select_recovery_strategy(error_type, confidence, session),
            'response': '',
            'context_actions': [],
            'follow_up_suggestions': [],
            'learning_notes': []
        }
        
        if error_type == ErrorType.USER_FRUSTRATION:
            recovery_data.update(self._handle_user_frustration(user_message, session))
        
        elif error_type == ErrorType.CONTEXT_CONFUSION:
            recovery_data.update(self._handle_context_confusion(user_message, session))
        
        elif error_type == ErrorType.INTENT_AMBIGUITY:
            recovery_data.update(self._handle_intent_ambiguity(user_message, session))
        
        elif error_type == ErrorType.INVALID_RESPONSE:
            recovery_data.update(self._handle_invalid_response(user_message, session))
        
        elif error_type == ErrorType.API_TIMEOUT:
            recovery_data.update(self._handle_api_timeout(user_message, session))
        
        else:
            recovery_data.update(self._handle_generic_error(user_message, session))
        
        # Log recovery attempt for learning
        self._log_recovery_attempt(error_type, recovery_data, session.session_id)
        
        return recovery_data
    
    def _calculate_frustration_score(self, user_msg_lower: str, session: ConversationSession) -> float:
        """Calculate user frustration level based on language patterns"""
        
        score = 0.0
        
        # Direct frustration indicators
        for indicator in ErrorPattern.USER_FRUSTRATION_INDICATORS:
            if indicator in user_msg_lower:
                score += 0.3
        
        # Repeated requests (session context)
        if len(session.messages) > 4:
            recent_user_messages = [m.content.lower() for m in session.messages[-4:] 
                                  if m.role.value == "user"]
            
            # Check for repetitive patterns
            for i, msg in enumerate(recent_user_messages[:-1]):
                similarity = self._calculate_message_similarity(msg, user_msg_lower)
                if similarity > 0.7:
                    score += 0.2
        
        # Escalating language intensity
        caps_ratio = sum(1 for c in user_msg_lower if c.isupper()) / max(len(user_msg_lower), 1)
        if caps_ratio > 0.3:
            score += 0.2
        
        # Multiple punctuation (!!!, ???)
        if '!!!' in user_msg_lower or '???' in user_msg_lower:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confusion_score(self, user_msg_lower: str, session: ConversationSession) -> float:
        """Calculate context confusion level"""
        
        score = 0.0
        
        # Direct confusion indicators
        for indicator in ErrorPattern.CONTEXT_CONFUSION_INDICATORS:
            if indicator in user_msg_lower:
                score += 0.4
        
        # Topic jumping detection
        if len(session.messages) > 2:
            last_ai_msg = next((m.content for m in reversed(session.messages) 
                              if m.role.value == "assistant"), "")
            
            # Check if user message seems unrelated to last AI response
            relevance = self._calculate_message_relevance(user_msg_lower, last_ai_msg.lower())
            if relevance < 0.2:
                score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_ambiguity_score(self, user_msg_lower: str, session: ConversationSession) -> float:
        """Calculate intent ambiguity level"""
        
        score = 0.0
        
        # Direct ambiguity indicators
        for indicator in ErrorPattern.INTENT_AMBIGUITY_INDICATORS:
            if indicator in user_msg_lower:
                score += 0.3
        
        # Vague language patterns
        vague_terms = ['something', 'anything', 'whatever', 'some', 'kind of', 'sort of']
        vague_count = sum(1 for term in vague_terms if term in user_msg_lower)
        score += min(vague_count * 0.1, 0.3)
        
        # Multiple questions without clear priority
        question_count = user_msg_lower.count('?')
        if question_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_response_issues(self, ai_response: str, user_message: str) -> List[Tuple[ErrorType, float]]:
        """Detect issues in AI response quality"""
        
        issues = []
        
        # Response too short for complex query
        if len(user_message) > 100 and len(ai_response) < 50:
            issues.append((ErrorType.INVALID_RESPONSE, 0.7))
        
        # Generic/template responses
        generic_phrases = [
            "I'd be happy to help", "Great question", "Let me help you with that",
            "There are many options", "It depends on", "That's a good point"
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in ai_response.lower())
        if generic_count > 2:
            issues.append((ErrorType.INVALID_RESPONSE, 0.5))
        
        return issues
    
    def _select_recovery_strategy(self, error_type: ErrorType, confidence: float, 
                                session: ConversationSession) -> RecoveryStrategy:
        """Select optimal recovery strategy based on error type and context"""
        
        # High confidence errors need immediate action
        if confidence > 0.8:
            if error_type == ErrorType.USER_FRUSTRATION:
                return RecoveryStrategy.ESCALATION
            elif error_type == ErrorType.CONTEXT_CONFUSION:
                return RecoveryStrategy.CONTEXT_RESET
        
        # Medium confidence - try clarification first
        if confidence > 0.5:
            return RecoveryStrategy.CLARIFICATION_REQUEST
        
        # Low confidence - graceful fallback
        return RecoveryStrategy.GRACEFUL_FALLBACK
    
    def _handle_user_frustration(self, user_message: str, session: ConversationSession) -> Dict[str, Any]:
        """Handle user frustration with empathetic recovery"""
        
        return {
            'response': "I can sense this isn't quite hitting the mark for you, and I apologize. Let me take a step back and make sure I understand exactly what you're looking for. Sometimes the best travel advice comes from really understanding your unique situation. Could you help me by sharing what specific aspect of your travel planning feels most important to get right?",
            
            'context_actions': ['reset_conversation_tone', 'increase_attention_to_detail'],
            
            'follow_up_suggestions': [
                "Start fresh with your main travel question",
                "Tell me what went wrong with my previous responses",
                "Let me know your top priority for this trip"
            ],
            
            'learning_notes': [f"User frustration detected in session {session.session_id}",
                             "Previous responses may have been too generic or off-target"]
        }
    
    def _handle_context_confusion(self, user_message: str, session: ConversationSession) -> Dict[str, Any]:
        """Handle context confusion with clarity restoration"""
        
        # Extract what we actually know about their situation
        context_summary = self._generate_context_summary(session)
        
        return {
            'response': f"I think we might have gotten a bit tangled up in our conversation. Let me clarify where we are: {context_summary}. Does this match what you're looking for, or should we start fresh with your main travel question?",
            
            'context_actions': ['summarize_conversation', 'clarify_objectives'],
            
            'follow_up_suggestions': [
                "Yes, that's right - let's continue",
                "No, let me explain what I actually need",
                "Let's start over with my main question"
            ],
            
            'learning_notes': ["Context confusion - conversation may have drifted off topic"]
        }
    
    def _handle_intent_ambiguity(self, user_message: str, session: ConversationSession) -> Dict[str, Any]:
        """Handle unclear user intent with targeted clarification"""
        
        return {
            'response': "I want to make sure I understand exactly what you're looking for help with. Based on what you've shared, I can help you with several different aspects of travel planning. Which of these is your main priority right now?",
            
            'context_actions': ['clarify_intent', 'offer_structured_options'],
            
            'follow_up_suggestions': [
                "Finding the right destination",
                "Planning activities and experiences", 
                "Practical travel logistics"
            ],
            
            'learning_notes': ["Intent ambiguity - user needs clearer options"]
        }
    
    def _handle_invalid_response(self, user_message: str, session: ConversationSession) -> Dict[str, Any]:
        """Handle cases where AI response was inadequate"""
        
        return {
            'response': "Let me give you a more helpful response to your question. I want to make sure I provide the specific, actionable travel advice you're looking for rather than general information.",
            
            'context_actions': ['regenerate_response', 'increase_specificity'],
            
            'follow_up_suggestions': [
                "Give me more specific details to work with",
                "Ask about a particular aspect",
                "Tell me what kind of answer would be most helpful"
            ],
            
            'learning_notes': ["Response quality issue - need more specific/detailed answers"]
        }
    
    def _handle_api_timeout(self, user_message: str, session: ConversationSession) -> Dict[str, Any]:
        """Handle API timeout gracefully"""
        
        return {
            'response': "I'm experiencing a brief delay in processing your request. While I work on getting you a complete answer, let me share some immediate travel insights that might be helpful, and then I'll follow up with more detailed information.",
            
            'context_actions': ['provide_cached_response', 'queue_for_retry'],
            
            'follow_up_suggestions': [
                "Ask a more specific question",
                "Focus on one aspect at a time",
                "Try rephrasing your request"
            ],
            
            'learning_notes': ["API timeout - may need to break down complex requests"]
        }
    
    def _handle_generic_error(self, user_message: str, session: ConversationSession) -> Dict[str, Any]:
        """Handle generic errors with graceful degradation"""
        
        return {
            'response': "I want to make sure I give you the most helpful travel advice possible. Let me approach your question from a different angle to better serve your travel planning needs.",
            
            'context_actions': ['alternative_approach'],
            
            'follow_up_suggestions': [
                "Help me understand your main travel goal",
                "Share more context about your situation",
                "Ask about a specific part of travel planning"
            ],
            
            'learning_notes': ["Generic error recovery attempted"]
        }
    
    def _generate_context_summary(self, session: ConversationSession) -> str:
        """Generate a clear summary of current conversation context"""
        
        summary_parts = []
        
        if session.context.current_destination:
            summary_parts.append(f"we're discussing travel to {session.context.current_destination}")
        
        if session.detected_intents:
            intent_descriptions = {
                UserIntent.DESTINATION_INQUIRY: "finding destinations",
                UserIntent.ACTIVITY_REQUEST: "planning activities",
                UserIntent.WEATHER_CHECK: "checking weather conditions",
                UserIntent.PRACTICAL_ADVICE: "practical travel logistics"
            }
            
            intents = [intent_descriptions.get(intent, intent.value) for intent in session.detected_intents]
            summary_parts.append(f"focusing on {', '.join(intents)}")
        
        if session.context.budget_range:
            summary_parts.append(f"with a {session.context.budget_range} budget")
        
        return " and ".join(summary_parts) if summary_parts else "your travel planning needs"
    
    def _calculate_message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate similarity between two messages (basic implementation)"""
        words1 = set(msg1.split())
        words2 = set(msg2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_message_relevance(self, user_msg: str, ai_msg: str) -> float:
        """Calculate how relevant user message is to AI response"""
        # Simplified relevance calculation
        user_words = set(user_msg.split())
        ai_words = set(ai_msg.split())
        
        if not user_words or not ai_words:
            return 0.0
        
        overlap = len(user_words.intersection(ai_words))
        return overlap / len(user_words) if user_words else 0.0
    
    def _log_recovery_attempt(self, error_type: ErrorType, recovery_data: Dict[str, Any], 
                            session_id: str):
        """Log recovery attempts for system learning and improvement"""
        
        if session_id not in self.error_history:
            self.error_history[session_id] = []
        
        self.error_history[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'recovery_strategy': recovery_data.get('strategy'),
            'learning_notes': recovery_data.get('learning_notes', [])
        })
        
        self.logger.info(f"Recovery attempt logged: {error_type} in session {session_id}")
    
    def get_recovery_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics about recovery patterns and success rates"""
        
        if session_id:
            return self.error_history.get(session_id, [])
        
        # Aggregate analytics across all sessions
        all_errors = []
        for session_errors in self.error_history.values():
            all_errors.extend(session_errors)
        
        error_types = {}
        for error in all_errors:
            error_type = error['error_type']
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        return {
            'total_recovery_attempts': len(all_errors),
            'error_type_distribution': error_types,
            'most_common_error': max(error_types.keys(), key=error_types.get) if error_types else None
        }