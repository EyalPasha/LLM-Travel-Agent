#!/usr/bin/env python3
"""
Quick performance test for the travel agent suggestion system
"""

import asyncio
import time
from app.services.llm import OpenRouterService

async def test_suggestion_speed():
    """Test how fast the suggestion generation is now"""
    
    service = OpenRouterService()
    
    test_response = """Hello! That's completely normal to feel overwhelmed for your first solo trip - it's a big step into the unknown! The good news is there are some amazing destinations that are particularly well-suited for first-time solo travelers.

To help me recommend the perfect place for you, what kind of experience are you most drawn to? Are you looking for:
- Cultural immersion with cities and history
- Natural adventures and stunning landscapes
- Beach relaxation and tropical vibes
- Food and culinary experiences
- Something more active like hiking or outdoor adventures

For example, if you're looking for a bit of everything with great infrastructure for solo travelers, Portugal is fantastic - it's affordable, has beautiful cities like Lisbon and Porto, delicious food, and is very safe for solo exploration. Or if you dream of mountains and nature, places like New Zealand or Norway offer incredible landscapes with well-marked trails.

What sounds most appealing to you?"""
    
    history = [
        {'role': 'user', 'content': "Hi! I'm planning my first solo trip and feeling a bit overwhelmed."}
    ]
    
    print("🚀 Testing suggestion generation speed...")
    start_time = time.time()
    
    suggestions = await service.generate_contextual_suggestions(test_response, history)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️  Generation took: {duration:.2f} seconds")
    print(f"📝 Generated suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    if duration < 1.0:
        print("✅ Performance is good! (under 1 second)")
    elif duration < 3.0:
        print("⚠️  Performance is acceptable (1-3 seconds)")
    else:
        print("❌ Performance is slow (over 3 seconds)")

if __name__ == "__main__":
    asyncio.run(test_suggestion_speed())
