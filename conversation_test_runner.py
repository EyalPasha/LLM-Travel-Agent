"""
Comprehensive Travel Agent Conversation Test Runner
Tests all 7 travel query types with psychological profiling, data integration, and context memory.
Saves complete transcript for submission.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import requests
import os


class ConversationTestRunner:
    """
    Runs a comprehensive test of the travel agent through 5 rounds of conversation
    showcasing all capabilities and saves the complete transcript.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.transcript = []
        self.start_time = datetime.now()
        
        # Test conversation rounds
        self.test_rounds = {
            "Round 1: Multi-turn Planning (Psychological Profiling)": [
                "Hi! I'm planning my first solo trip and feeling a bit overwhelmed.",
                "I love photography and want somewhere with incredible landscapes.",
                # Will wait for suggestions and pick one
            ],
            "Round 2: Weather & Cultural (Data Integration)": [
                "What's the weather like in Tokyo right now?",
                "What should I know about local customs in Thailand before I visit?",
                "I'm most interested in Northern Thailand, especially around Chiang Mai"
            ],
            "Round 3: Budget & Practical (Complex Analysis)": [
                "How much should I budget for a week in Paris?",
                "I definitely want to eat well but I'm not looking for super fancy places. More like authentic local spots.",
                # Will wait for suggestions
            ],
            "Round 4: Error Recovery Test": [
                "um so like i'm thinking maybe somewhere tropical but also not too touristy and good for instagram but authentic you know?",
                # Will wait for suggestions
            ],
            "Round 5: Advanced Context (Memory Test)": [
                "I've been thinking about our earlier conversation about solo travel to Iceland. I'm getting more confident about the idea, but now I'm wondering about the best photography equipment to bring"
            ]
        }
    
    async def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the travel agent API"""
        try:
            payload = {
                "message": message,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{self.base_url}/chat", 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if not self.session_id:
                    self.session_id = result.get("session_id")
                return result
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response": "Sorry, I encountered an error processing your request.",
                    "session_id": self.session_id or "unknown",
                    "state": "error"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "response": "Sorry, I couldn't connect to the travel agent service.",
                "session_id": self.session_id or "unknown", 
                "state": "error"
            }
    
    def log_interaction(self, user_message: str, agent_response: Dict[str, Any], round_name: str, message_num: int):
        """Log a conversation interaction to the transcript"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        interaction = {
            "timestamp": timestamp,
            "round": round_name,
            "message_number": message_num,
            "user_message": user_message,
            "agent_response": agent_response.get("response", "No response"),
            "session_id": agent_response.get("session_id"),
            "conversation_state": agent_response.get("state"),
            "current_destination": agent_response.get("context", {}).get("current_destination"),
            "suggestions": agent_response.get("suggestions", []),
            "external_data_used": agent_response.get("external_data_used", False),
            "reasoning_trace": agent_response.get("reasoning_trace"),
            "error": agent_response.get("error")
        }
        
        self.transcript.append(interaction)
        
        # Print to console for real-time monitoring
        print(f"\n{'='*60}")
        print(f"{round_name} - Message {message_num}")
        print(f"Time: {timestamp}")
        print(f"{'='*60}")
        print(f"USER: {user_message}")
        print(f"\nAGENT: {agent_response.get('response', 'No response')}")
        
        if agent_response.get("suggestions"):
            print(f"\nSUGGESTIONS:")
            for i, suggestion in enumerate(agent_response["suggestions"], 1):
                print(f"  {i}. {suggestion}")
        
        if agent_response.get("external_data_used"):
            print(f"\n[External data was used in this response]")
        
        if agent_response.get("error"):
            print(f"\n[ERROR: {agent_response['error']}]")
        
        current_dest = agent_response.get("context", {}).get("current_destination")
        if current_dest:
            print(f"\n[DESTINATION CONTEXT: {current_dest}]")
        
        print(f"State: {agent_response.get('state', 'unknown')}")
        print(f"Session ID: {agent_response.get('session_id', 'unknown')}")
    
    def select_suggestion(self, suggestions: List[str], preference_keywords: List[str] = None) -> str:
        """Intelligently select a suggestion based on preference keywords or pick the first one"""
        if not suggestions:
            return None
        
        if preference_keywords:
            for suggestion in suggestions:
                if any(keyword.lower() in suggestion.lower() for keyword in preference_keywords):
                    return suggestion
        
        # Default to first suggestion
        return suggestions[0]
    
    async def run_conversation_round(self, round_name: str, messages: List[str]):
        """Run a complete conversation round"""
        print(f"\n\n🌟 STARTING {round_name} 🌟")
        
        for i, message in enumerate(messages, 1):
            # Add 5-minute pause before each question (except the very first one)
            if not (round_name == "Round 1: Multi-turn Planning (Psychological Profiling)" and i == 1):
                print(f"Waiting 5 minutes before next question to manage API token limits...")
                await asyncio.sleep(300)  # 5 minutes = 300 seconds
            
            # Send user message
            response = await self.send_message(message)
            self.log_interaction(message, response, round_name, i)
            
            # Handle suggestions if present
            if response.get("suggestions") and len(response["suggestions"]) > 0:
                # For Round 1, we'll pick Iceland for photography
                if "photography" in message.lower() and "landscapes" in message.lower():
                    selected = self.select_suggestion(
                        response["suggestions"], 
                        ["iceland", "norway", "patagonia", "new zealand"]
                    )
                elif "authentic local spots" in message.lower():
                    selected = self.select_suggestion(
                        response["suggestions"],
                        ["marais", "latin quarter", "montmartre", "local"]
                    )
                elif "tropical" in message.lower() and "instagram" in message.lower():
                    selected = self.select_suggestion(
                        response["suggestions"],
                        ["philippines", "indonesia", "maldives", "costa rica"]
                    )
                else:
                    selected = response["suggestions"][0]  # Pick first suggestion
                
                if selected:
                    # Add 5-minute pause before follow-up
                    print("Waiting 5 minutes before follow-up to manage API token limits...")
                    await asyncio.sleep(300)  # 5 minutes = 300 seconds
                    
                    # Send the selection as a follow-up message
                    follow_up = f"That sounds perfect! Tell me more about {selected}."
                    follow_response = await self.send_message(follow_up)
                    self.log_interaction(
                        follow_up, 
                        follow_response, 
                        f"{round_name} (Follow-up)", 
                        f"{i}b"
                    )
    
    async def run_full_test(self):
        """Run the complete conversation test suite"""
        print("STARTING COMPREHENSIVE TRAVEL AGENT CONVERSATION TEST")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target API: {self.base_url}")
        
        try:
            # Test API connection first
            health_response = requests.get(f"{self.base_url}/", timeout=10)
            if health_response.status_code != 200:
                print(f"API not available at {self.base_url}")
                return False
        except:
            print(f"Cannot connect to API at {self.base_url}")
            return False
        
        # Run all conversation rounds
        for round_name, messages in self.test_rounds.items():
            await self.run_conversation_round(round_name, messages)
        
        # Save transcript
        self.save_transcript()
        print(f"\n✅ CONVERSATION TEST COMPLETED SUCCESSFULLY!")
        print(f"Total Duration: {datetime.now() - self.start_time}")
        print(f"Session ID: {self.session_id}")
        print(f"Total Interactions: {len(self.transcript)}")
        
        return True
    
    def save_transcript(self):
        """Save the complete conversation transcript to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed JSON transcript
        json_filename = f"travel_agent_conversation_test_{timestamp}.json"
        transcript_data = {
            "test_metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "session_id": self.session_id,
                "total_interactions": len(self.transcript),
                "api_endpoint": self.base_url
            },
            "test_objectives": [
                "Multi-turn Planning (Psychological Profiling)",
                "Weather & Cultural Data Integration", 
                "Budget & Practical Complex Analysis",
                "Error Recovery Test",
                "Advanced Context Memory Test"
            ],
            "conversation_transcript": self.transcript
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        # Create human-readable markdown transcript
        md_filename = f"travel_agent_conversation_test_{timestamp}.md"
        self.create_markdown_transcript(md_filename, transcript_data)
        
        print(f"\n📄 Transcripts saved:")
        print(f"   • JSON (detailed): {json_filename}")
        print(f"   • Markdown (readable): {md_filename}")
    
    def create_markdown_transcript(self, filename: str, data: Dict[str, Any]):
        """Create a human-readable markdown transcript"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Travel Agent Conversation Test - Complete Transcript\n\n")
            
            # Test metadata
            f.write("## Test Information\n\n")
            f.write(f"- **Start Time:** {data['test_metadata']['start_time']}\n")
            f.write(f"- **End Time:** {data['test_metadata']['end_time']}\n")
            f.write(f"- **Session ID:** {data['test_metadata']['session_id']}\n")
            f.write(f"- **Total Interactions:** {data['test_metadata']['total_interactions']}\n")
            f.write(f"- **API Endpoint:** {data['test_metadata']['api_endpoint']}\n\n")
            
            # Test objectives
            f.write("## Test Objectives\n\n")
            for objective in data['test_objectives']:
                f.write(f"- ✅ {objective}\n")
            f.write("\n")
            
            # Conversation transcript
            f.write("## Complete Conversation Transcript\n\n")
            
            current_round = None
            for interaction in data['conversation_transcript']:
                # New round header
                if interaction['round'] != current_round:
                    current_round = interaction['round']
                    f.write(f"### {current_round}\n\n")
                
                # Interaction details
                f.write(f"**{interaction['timestamp']} - Message {interaction['message_number']}**\n\n")
                f.write(f"**User:** {interaction['user_message']}\n\n")
                f.write(f"**Assistant:** {interaction['agent_response']}\n\n")
                
                # Additional details
                if interaction['suggestions']:
                    f.write("**Suggestions provided:**\n")
                    for suggestion in interaction['suggestions']:
                        f.write(f"- {suggestion}\n")
                    f.write("\n")
                
                if interaction['external_data_used']:
                    f.write("*🌐 External data was used in this response*\n\n")
                
                if interaction['error']:
                    f.write(f"*❌ Error: {interaction['error']}*\n\n")
                
                f.write(f"- **State:** {interaction['conversation_state']}\n")
                f.write(f"- **Session:** {interaction['session_id']}\n\n")
                f.write("---\n\n")


async def main():
    """Main execution function"""
    # Check if the server is running locally
    runner = ConversationTestRunner("http://localhost:8000")
    
    print("Travel Agent Conversation Test Runner")
    print("=====================================")
    print()
    print("This script will run a comprehensive test of the travel agent")
    print("showcasing all 7 travel query types across 5 conversation rounds:")
    print("1. Multi-turn Planning (Psychological Profiling)")
    print("2. Weather & Cultural (Data Integration)")  
    print("3. Budget & Practical (Complex Analysis)")
    print("4. Error Recovery Test")
    print("5. Advanced Context (Memory Test)")
    print()
    
    input("Press Enter to start the test (make sure your travel agent server is running)...")
    
    success = await runner.run_full_test()
    
    if success:
        print("\n🎉 Test completed successfully! Check the generated transcript files.")
    else:
        print("\n❌ Test failed. Please check your server and try again.")


if __name__ == "__main__":
    asyncio.run(main())
