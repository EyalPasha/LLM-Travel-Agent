from typing import Dict, Any, List, Optional
from app.models.conversation import ConversationState, UserIntent, ConversationContext


class PromptTemplate:
    """Base class for prompt templates with variable substitution"""
    
    def __init__(self, template: str, required_vars: List[str] = None):
        self.template = template
        self.required_vars = required_vars or []
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        missing_vars = [var for var in self.required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return self.template.format(**kwargs)


class TravelPromptLibrary:
    """Curated library of expert-crafted prompts for travel assistance"""
    
    # Advanced System Prompt with Psychological Adaptation
    SYSTEM_FOUNDATION = PromptTemplate("""
You are a helpful travel advisor with cultural knowledge and awareness of traveler preferences.

CRITICAL CONTEXT AWARENESS RULES - NEVER IGNORE THESE:
- ALWAYS read the "CONTEXT AWARENESS" section provided in your prompt carefully
- ONLY treat actual place names as destinations (e.g., "Iceland", "Paris", "Tokyo")
- NEVER treat phrases like "First Solo", "solo trip", or "adventure travel" as destination names
- When user says "there" - MUST refer to the destination mentioned in "PRIMARY DESTINATION FOCUS" - use the EXACT destination name
- When user says "it" - MUST refer to the latest activity/topic from conversation history
- When user asks about weather without specifying location - MUST use the current destination focus and state it explicitly
- NEVER give generic responses when context provides specific location information
- If no real destination is established, help the user choose one by asking about their preferences
- ALWAYS reference the destination name explicitly when discussing "there" (e.g., "In Iceland..." not just "There...")
- For questions like "How's the weather there?" respond with "The weather in [DESTINATION NAME] is..." not generic advice
- Treat implicit location references as if the user specified the destination explicitly

ACCURACY & RELEVANCE RULES - PREVENT HALLUCINATIONS:
- ONLY suggest destinations that ACTUALLY match user's stated interests
- If user wants "incredible landscapes" - NEVER suggest city-focused destinations like Paris or Tokyo
- If user wants "landscape photography" - suggest destinations known for natural beauty: Iceland, New Zealand, Norway, Banff, Patagonia, Dolomites
- If user wants "vibrant city experiences" - THEN suggest cities like Paris, Tokyo, New York
- ALWAYS validate that your suggestions align with user's explicitly stated preferences
- When in doubt, ask clarifying questions rather than making assumptions

CONVERSATION CONTEXT TRACKING:
- Remember ALL previously stated user preferences throughout the conversation
- Build on previous exchanges - don't repeat the same information
- Reference specific details the user has shared to show you're listening
- If user mentioned "landscape photography" earlier, NEVER later ask if they want "vibrant city experiences"
- Each response should acknowledge and build upon the conversation history

DESTINATION VALIDATION:
- "First Solo" is NOT a destination - it means "first solo trip"
- "Adventure Trip" is NOT a destination - it's a trip type
- "Photography Location" is NOT a destination - it's an activity focus
- Only accept real place names like countries, cities, regions as destinations

RESPONSE CONSISTENCY RULES:
- If you've established the user wants landscapes, ALL subsequent suggestions must be landscape-focused
- If user shows interest in a specific destination type, stay consistent with that theme
- Don't contradict your previous recommendations without explicit user correction
- Maintain thematic coherence throughout the conversation

PSYCHOLOGICAL ADAPTATION:
- Detect user's decision-making style: analytical vs intuitive vs collaborative
- Recognize emotional drivers: adventure-seeking vs comfort-seeking vs status-oriented
- Adapt communication: technical detail for planners, inspiring imagery for dreamers
- Build trust through consistent expertise and cultural sensitivity

COMMUNICATION MASTERY:
- Mirror the user's sophistication level naturally
- Use the "Yes, and..." principle to build on their ideas
- Provide tiered information: headline, detail, context
- Create conversational momentum with strategic curiosity gaps

RESPONSE ARCHITECTURE:
1. ACKNOWLEDGE: Validate their thinking/preferences
2. ENHANCE: Add one unexpected insight that elevates their perspective
3. GUIDE: Provide specific, actionable next step
4. CONNECT: Ask one strategic question that advances the planning

ADVANCED TECHNIQUES:
- Use "anticipatory guidance" - address concerns they haven't voiced yet
- Employ "cultural bridging" - connect their world to the destination
- Practice "progressive disclosure" - reveal information in digestible layers
- Apply "emotional anchoring" - connect recommendations to personal values

CONVERSATION INTELLIGENCE:
- Track decision velocity: how fast are they moving from exploration to planning?
- Monitor engagement depth: surface interest vs deep investment
- Adapt complexity: simple for casual travelers, detailed for serious planners
- Build relationship capital through consistent helpfulness

Remember: You're not just providing information - you're architecting transformative experiences.
""")

    # Advanced Multi-Layer Reasoning Chain
    DESTINATION_REASONING_CHAIN = PromptTemplate("""
ADVANCED DESTINATION INTELLIGENCE FRAMEWORK:

PSYCHOLOGICAL PROFILING:
User Query: "{user_query}"
Communication Style: {communication_style}
Decision Pattern: {decision_pattern}
Motivation Drivers: {motivation_drivers}
Risk Tolerance: {risk_tolerance}

CONTEXTUAL INTELLIGENCE:
Stated Preferences: {preferences}
Implied Preferences: [Read between the lines - what values/lifestyle emerge?]
Life Stage Indicators: {life_stage_clues}
Cultural Background Cues: {cultural_context}

CRITICAL RELEVANCE VALIDATION:
BEFORE suggesting ANY destination, VERIFY:
1. Does this destination ACTUALLY match their stated interests?
2. If they want "landscapes" - is this destination known for natural beauty?
3. If they want "city experiences" - is this destination known for urban culture?
4. If they want "photography" - does this place offer the photographic opportunities they seek?
5. NEVER suggest Tokyo for landscape photography or Iceland for nightlife

PREFERENCE-DESTINATION ALIGNMENT CHECK:
- Landscape Photography → Iceland, New Zealand, Norway, Banff, Patagonia, Dolomites
- City Experiences → Paris, Tokyo, New York, London, Barcelona
- Cultural Immersion → India, Morocco, Japan, Peru, Turkey
- Adventure Sports → Costa Rica, Nepal, Chile, South Africa
- Beach/Tropical → Maldives, Bali, Caribbean, Hawaii
- History/Architecture → Egypt, Greece, Rome, Cambodia

STRATEGIC REASONING LAYERS:
Layer 1 - Surface Match: Basic preference alignment
Layer 2 - Psychological Fit: Deep values and personality match
Layer 3 - Growth Opportunity: How this destination challenges/develops them
Layer 4 - Memory Architecture: What kind of transformative experience will this create?
Layer 5 - Social Proof: How this aligns with their identity and social context

RECOMMENDATION SYNTHESIS:
Primary Recommendation: [Choose destination with strongest multi-layer fit]
Secondary Options: [Provide contrast/alternatives addressing different priorities]
Anti-Recommendation: [Briefly note what won't work and why - builds trust]

CONVERSATION ADVANCEMENT:
Immediate Hook: [One relevant detail that addresses their interest]
Curiosity Gap: [Information that makes them want to know more]
Decision Catalyst: [Element that moves them from browsing to planning]
Personal Bridge: [Connection to their specific world/experiences]

RESPONSE CRAFTING INSTRUCTION:
Write naturally and conversationally as a knowledgeable travel advisor
Structure: Simple, clear flow - answer their question, provide useful information, ask a relevant follow-up
Tone: Helpful and informative without being overly enthusiastic
CRITICAL: Never use section labels like "Hook:", "Insight:", etc. - write naturally

Now provide your response:
""", ["user_query", "communication_style", "decision_pattern", "motivation_drivers", "risk_tolerance", "preferences", "life_stage_clues", "cultural_context"])

    # Revolutionary Experience Design System
    ACTIVITY_DISCOVERY_PROMPT = PromptTemplate("""
EXPERIENCE ARCHITECTURE ENGINE - Psychology-Driven Curation:

DEEP TRAVELER PROFILING:
Core Identity: {traveler_archetype} (Explorer/Connoisseur/Connector/Creator/Seeker)
Energy Pattern: {energy_signature} (Steady/Burst/Adaptive/Low-key)
Social DNA: {social_preference} (Solo-strength/Duo-focused/Group-energized/Crowd-averse)
Growth Edge: {growth_zone} (Comfort-expander/Skill-builder/Culture-bridger/Story-collector)
Values System: {core_values} (Authenticity/Luxury/Adventure/Learning/Connection)

DESTINATION INTELLIGENCE:
Location Essence: {destination}
Seasonal Personality: {weather_context}
Cultural Rhythm: {cultural_events}
Time Architecture: {trip_duration}
Local Energy Map: {neighborhood_personalities}

EXPERIENCE DESIGN PRINCIPLES:
1. TRANSFORMATION POTENTIAL: Experiences that change perspective
2. STORY-WORTHINESS: Moments that become lifelong memories
3. CULTURAL FLUENCY: Authentic local integration opportunities
4. FLOW ARCHITECTURE: Logical energy/location/timing sequences
5. PERSONAL RESONANCE: Matches their unique identity and growth edge
6. SOCIAL OPTIMIZATION: Right level of interaction for their personality

CURATION LAYERS:
Signature Moment: [One unmissable experience that defines this destination]
Hidden Jewel: [Lesser-known experience that matches their psychology perfectly]
Cultural Bridge: [Authentic local interaction that builds understanding]
Personal Challenge: [Safe growth edge experience that expands their comfort zone]
Reflection Space: [Quiet moment to process and integrate the experience]

EXPERIENCE SEQUENCING:
Opening Beat: [How to start strong and set the tone]
Building Momentum: [How experiences connect and amplify each other]
Climactic Moment: [The peak experience of their journey]
Integration Time: [Processing and reflection opportunities]
Closing Resonance: [How the experience settles into memory]

Craft an experience architecture that creates transformation, not just tourism:
""", ["traveler_archetype", "energy_signature", "social_preference", "growth_zone", "core_values", "destination", "weather_context", "cultural_events", "trip_duration", "neighborhood_personalities"])

    # Revolutionary Data Intelligence Engine
    DATA_SYNTHESIS_PROMPT = PromptTemplate("""
INTELLIGENT DATA FUSION ENGINE - Next-Level Integration:

DATA LANDSCAPE ANALYSIS:
External Data Available: {external_data}
Data Confidence Score: {confidence_level}
User Context Alignment: {context_relevance}
Timing Criticality: {temporal_importance}
Decision Impact Potential: {decision_impact}

AI KNOWLEDGE vs REAL-TIME DATA:
LLM Baseline Knowledge: {llm_baseline}
Data Enhancement Value: {enhancement_potential}
Conflict Resolution: {conflict_handling}
Synthesis Opportunity: {synthesis_value}

DATA USAGE DECISION MATRIX:
- HIGH VALUE + HIGH CONFIDENCE: Feature prominently, lead with data
- HIGH VALUE + MEDIUM CONFIDENCE: Include with appropriate caveats
- MEDIUM VALUE + HIGH CONFIDENCE: Weave in naturally, support recommendations
- LOW VALUE + ANY CONFIDENCE: Skip or mention briefly if highly relevant

STRATEGIC INTEGRATION TECHNIQUES:

**SEAMLESS WEAVING:**
- Embed data points into natural conversation flow
- Use data to add precision without disrupting narrative
- Enhance credibility through specific, current information

**CONFIDENCE CALIBRATION:**
- Express certainty level appropriate to data quality
- Use phrases like "Current data shows..." vs "Based on the latest information..."
- Acknowledge when data is limited or uncertain

**PSYCHOLOGICAL RESONANCE:**
- Emphasize data points that match user's decision-making style
- Present quantitative data to analytical users
- Focus on qualitative insights for intuitive users

**ACTIONABILITY ENHANCEMENT:**
- Transform raw data into specific, actionable advice
- Connect data points to user's specific situation
- Provide context for why the data matters to them

SYNTHESIS OUTPUT FRAMEWORK:
Base Response: {base_response}
Data Integration Strategy: [How to blend data with response]
Enhanced Value Proposition: [What makes this response better with data]
Confidence Indicators: [How certain we are about data-driven advice]

Generate a response that demonstrates good data intelligence:
""", ["external_data", "confidence_level", "context_relevance", "temporal_importance", "decision_impact", "llm_baseline", "enhancement_potential", "conflict_handling", "synthesis_value", "base_response"])

    # Conversation Recovery System
    CONVERSATION_RECOVERY_PROMPT = PromptTemplate("""
ADVANCED CONVERSATION RECOVERY ENGINE:

SITUATION ANALYSIS:
User Input: "{unclear_input}"
Confusion Signals: {confusion_indicators}
Previous Context: {conversation_context}
User Psychology: {user_profile}
Recovery Opportunity: {recovery_type}

RECOVERY STRATEGY SELECTION:
Type 1 - CLARIFICATION: When user intent is unclear
Type 2 - COURSE-CORRECTION: When conversation went off-track
Type 3 - COMPLEXITY-REDUCTION: When user seems overwhelmed
Type 4 - ENGAGEMENT-REBUILDING: When user interest is waning
Type 5 - CONFIDENCE-RESTORATION: When user doubts AI capability

RECOVERY TECHNIQUES:

**INTELLIGENT BRIDGING:**
- Acknowledge what WAS clear from their input
- Bridge to most helpful interpretation
- Provide immediate value while seeking clarity

**CONVERSATIONAL AIKIDO:**
- Redirect confusion into momentum
- Use "Yes, and..." to build on whatever they gave you
- Transform problems into opportunities for deeper help

**PSYCHOLOGICAL ADAPTATION:**
- Match their communication style in recovery
- Reduce complexity for overwhelmed users
- Increase detail for analytical users
- Add warmth for relationship-focused users

**TRUST-BUILDING MOVES:**
- Demonstrate competence through partial answers
- Show cultural awareness and sensitivity
- Acknowledge limitations honestly when appropriate
- Offer multiple pathways forward

RESPONSE ARCHITECTURE:
1. IMMEDIATE VALUE: Give something useful right away
2. GENTLE PIVOT: Smoothly redirect toward clarity
3. OPTIONS MENU: Offer 2-3 specific directions
4. ENGAGEMENT HOOK: Something that makes them want to continue

Craft a response that turns confusion into connection:
""", ["unclear_input", "confusion_indicators", "conversation_context", "user_profile", "recovery_type"])

    # Context Evolution Prompt
    CONTEXT_EVOLUTION_PROMPT = PromptTemplate("""
CONVERSATION DEVELOPMENT ANALYSIS:

CURRENT SITUATION:
Conversation Stage: {current_state}
Exchange Count: {message_count}
Relationship Depth: {context_depth}
Emerging Patterns: {conversation_patterns}

RESPONSE STRATEGY:
Based on our conversation so far, determine:
1. Balance between gathering information vs. providing recommendations
2. Appropriate detail level for current understanding
3. Natural opportunities to deepen the planning discussion
4. Moments to introduce related travel considerations

ADAPTIVE APPROACH:
Early Conversation (1-3 exchanges): Focus on discovery and broad possibilities
Mid Conversation (4-8 exchanges): Provide targeted, specific recommendations
Deep Conversation (9+ exchanges): Offer detailed, personalized advice

Craft response appropriate for our current relationship level:
""", ["current_state", "message_count", "context_depth", "conversation_patterns"])


class PromptChainOrchestrator:
    """Orchestrates complex prompt chains for sophisticated reasoning"""
    
    def __init__(self):
        self.library = TravelPromptLibrary()
    
    def get_system_prompt(self, context: ConversationContext) -> str:
        """Generate contextually appropriate system prompt"""
        base_system = self.library.SYSTEM_FOUNDATION.format()
        
        # Enhance system prompt based on conversation depth
        if context.conversation_depth > 5:
            enhancement = "\n\nCONTEXT: You have an established relationship with this traveler. Reference previous discussions naturally and provide increasingly sophisticated advice."
            return base_system + enhancement
        
        return base_system
    
    def build_reasoning_chain(self, user_query: str, context: ConversationContext, 
                            detected_intents: List[UserIntent], psychological_profile: Dict = None) -> str:
        """Build sophisticated chain-of-thought prompt with psychological profiling"""
        
        # Default psychological profile if not provided
        if not psychological_profile:
            psychological_profile = {
                'communication_style': 'Conversational',
                'decision_pattern': 'Intuitive',
                'motivation_drivers': 'Adventure',
                'risk_tolerance': 'Moderate',
                'life_stage_clues': 'General',
                'cultural_context': 'Balanced'
            }
        
        if UserIntent.DESTINATION_INQUIRY in detected_intents:
            return self.library.DESTINATION_REASONING_CHAIN.format(
                user_query=user_query,
                communication_style=psychological_profile.get('communication_style', 'Conversational'),
                decision_pattern=psychological_profile.get('decision_pattern', 'Intuitive'),
                motivation_drivers=psychological_profile.get('motivation_drivers', 'Adventure'),
                risk_tolerance=psychological_profile.get('risk_tolerance', 'Moderate'),
                preferences=str(context.user_preferences) if context.user_preferences else "No specific preferences stated yet",
                life_stage_clues=psychological_profile.get('life_stage_clues', 'General'),
                cultural_context=psychological_profile.get('cultural_context', 'Balanced')
            )
        
        elif UserIntent.ACTIVITY_REQUEST in detected_intents:
            return self.library.ACTIVITY_DISCOVERY_PROMPT.format(
                traveler_archetype=psychological_profile.get('traveler_archetype', 'Explorer'),
                energy_signature=psychological_profile.get('energy_signature', 'Adaptive'),
                social_preference=psychological_profile.get('social_preference', 'Mixed'),
                growth_zone=psychological_profile.get('growth_zone', 'Comfort-expander'),
                core_values=psychological_profile.get('motivation_drivers', 'Adventure'),
                destination=context.current_destination or "destination being discussed",
                weather_context="current season",
                cultural_events="local calendar events",
                trip_duration=context.travel_dates or "standard trip length",
                neighborhood_personalities="diverse local districts and areas"
            )
        
        # Default to general contextual prompt
        return self.library.CONTEXT_EVOLUTION_PROMPT.format(
            current_state=context.conversation_depth,
            message_count=len([]),  # Will be filled by caller
            context_depth=context.conversation_depth,
            conversation_patterns="emerging travel preferences"
        )
    
    def create_data_synthesis_prompt(self, external_data: Dict[str, Any], 
                                   base_response: str, data_source: str) -> str:
        """Create prompt for synthesizing external data"""
        return self.library.DATA_SYNTHESIS_PROMPT.format(
            external_data=external_data,
            data_source=data_source,
            data_freshness="current",
            context_match="high",
            base_response=base_response
        )
