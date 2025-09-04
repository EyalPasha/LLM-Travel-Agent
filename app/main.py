"""
Main FastAPI Application - Travel Intelligence Assistant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from contextlib import asynccontextmanager
import logging
import time
import re
from typing import List, Optional, Dict
from datetime import datetime

from app.core.config import settings
from app.core.conversation import ConversationEngine
from app.services.llm import HuggingFaceService
from app.services.external_apis import DataAugmentationService
from app.models.conversation import ChatRequest, ChatResponse, ConversationState, MessageRole, UserIntent


# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Travel Intelligence Assistant")
    yield
    logger.info("Shutting down Travel Intelligence Assistant")


# Initialize FastAPI app
app = FastAPI(
    title="Travel Intelligence Assistant",
    description="A travel consultant with travel knowledge",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
conversation_engine = ConversationEngine()
llm_service = HuggingFaceService()
data_service = DataAugmentationService()

# Initialize systems  
from app.core.error_recovery import ConversationRecoveryEngine

recovery_engine = ConversationRecoveryEngine()


@app.get("/favicon.ico")
async def favicon():
    """Return custom favicon"""
    try:
        with open("R.png", "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    except FileNotFoundError:
        # Fallback to simple icon if R.png not found
        return Response(content="", media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve modern web interface for Sofia Travel Assistant"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sofia • Travel Intelligence Assistant</title>
        <link href="https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                --secondary-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
                --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                --background: #f8fafc;
                --surface: #ffffff;
                --surface-alt: #f1f5f9;
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --text-light: #94a3b8;
                --border: #e2e8f0;
                --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --border-radius: 16px;
                --border-radius-sm: 12px;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Rubik', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--background);
                color: var(--text-primary);
                line-height: 1.6;
                overflow-x: hidden;
            }

            .main-container {
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 24px 24px 24px;
                gap: 24px;
                padding-top: 80px;
            }

            .header {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px 32px;
                background: var(--surface);
                box-shadow: var(--shadow-lg);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid var(--border);
            }
            }

            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--primary-gradient);
            }

            .header-content {
                flex: 1;
            }

            .header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                background: var(--primary-gradient);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
                letter-spacing: -0.025em;
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .header .subtitle {
                font-size: 1.125rem;
                color: var(--text-secondary);
                font-weight: 400;
                margin-bottom: 16px;
            }

            .header .description {
                max-width: 600px;
                color: var(--text-light);
                font-size: 0.95rem;
                line-height: 1.6;
            }

            .header-actions {
                display: flex;
                flex-direction: column;
                gap: 12px;
                align-items: flex-end;
            }

            .new-chat-btn {
                padding: 12px 24px;
                background: var(--secondary-gradient);
                color: white;
                border: none;
                border-radius: 50px;
                font-family: inherit;
                font-weight: 500;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                box-shadow: var(--shadow);
            }

            .new-chat-btn:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }

            .chat-status {
                font-size: 0.8rem;
                color: var(--text-light);
                text-align: right;
            }

            .chat-wrapper {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: var(--surface);
                border-radius: var(--border-radius);
                box-shadow: var(--shadow-lg);
                overflow: hidden;
                min-height: 600px;
            }

            .chat-header {
                padding: 24px 32px;
                background: var(--surface-alt);
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .chat-header .avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: var(--accent-gradient);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.25rem;
                font-weight: 600;
            }

            .chat-header .info h3 {
                font-size: 1.125rem;
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 4px;
            }

            .chat-header .info p {
                color: var(--text-secondary);
                font-size: 0.875rem;
            }

            .chat-messages {
                flex: 1;
                padding: 32px;
                overflow-y: auto;
                scroll-behavior: smooth;
                background: linear-gradient(to bottom, var(--surface) 0%, #fafbfc 100%);
            }

            .message {
                display: flex;
                margin-bottom: 8px;
                align-items: flex-start;
                gap: 16px;
                animation: slideIn 0.3s ease-out;
            }

            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .message.user {
                flex-direction: row-reverse;
            }

            .message-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 0.875rem;
                flex-shrink: 0;
            }

            .message.user .message-avatar {
                background: var(--secondary-gradient);
                color: white;
            }

            .message.assistant .message-avatar {
                background: var(--accent-gradient);
                color: white;
            }

            .message-content {
                max-width: 75%;
                padding: 16px 20px;
                border-radius: var(--border-radius-sm);
                position: relative;
                word-wrap: break-word;
                word-break: break-word;
                overflow-wrap: break-word;
                line-height: 1.4;
                white-space: normal;
            }

            .message.user .message-content {
                background: var(--primary-gradient);
                color: white;
                border-bottom-right-radius: 6px;
            }

            .message.assistant .message-content {
                background: var(--surface);
                border: 1px solid var(--border);
                color: var(--text-primary);
                border-bottom-left-radius: 6px;
                box-shadow: var(--shadow);
            }

            .message-content p {
                margin-bottom: 2px;
            }

            .message-content p:last-child {
                margin-bottom: 0;
            }

            .message-content ul, .message-content ol {
                margin: 4px 0 4px 20px;
            }

            .message-content li {
                margin-bottom: 2px;
            }

            .message-content h1, .message-content h2, .message-content h3, .message-content h4 {
                font-weight: 600;
                margin: 12px 0 8px 0;
                color: var(--text-primary);
            }

            .message-content h1 { font-size: 1.5rem; }
            .message-content h2 { font-size: 1.25rem; }
            .message-content h3 { font-size: 1.125rem; }
            .message-content h4 { font-size: 1rem; }

            .message-content h1:first-child, 
            .message-content h2:first-child, 
            .message-content h3:first-child, 
            .message-content h4:first-child {
                margin-top: 0;
            }

            .suggestions {
                margin-top: 16px;
                padding: 20px;
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-radius: var(--border-radius-sm);
                border-left: 4px solid #0ea5e9;
            }

            .suggestions h4 {
                color: #0c4a6e;
                margin-bottom: 12px;
                font-weight: 600;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            }

            .suggestions ul {
                list-style: none;
                margin: 0;
            }

            .suggestions li {
                margin-bottom: 10px;
                color: #0369a1;
                font-weight: 500;
                position: relative;
                padding: 12px 16px 12px 40px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s ease;
                background: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(14, 165, 233, 0.2);
            }

            .suggestions li:hover {
                background: rgba(14, 165, 233, 0.1);
                border-color: #0ea5e9;
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(14, 165, 233, 0.2);
            }

            .suggestions li::before {
                content: '→';
                position: absolute;
                left: 14px;
                top: 12px;
                color: #0ea5e9;
                font-weight: bold;
                font-size: 1.1em;
                transition: transform 0.2s ease;
            }

            .suggestions li:hover::before {
                transform: translateX(2px);
            }

            .chat-input {
                padding: 24px 32px;
                background: var(--surface);
                border-top: 1px solid var(--border);
                display: flex;
                gap: 16px;
                align-items: center;
            }

            .input-wrapper {
                flex: 1;
                position: relative;
            }

            .chat-input input {
                width: 100%;
                padding: 16px 24px;
                border: 2px solid var(--border);
                border-radius: 50px;
                font-family: inherit;
                font-size: 1rem;
                background: var(--surface-alt);
                color: var(--text-primary);
                transition: all 0.2s ease;
                outline: none;
            }

            .chat-input input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
                background: var(--surface);
            }

            .send-button {
                width: 56px;
                height: 56px;
                border: none;
                border-radius: 50%;
                background: var(--primary-gradient);
                color: white;
                font-size: 1.25rem;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: var(--shadow);
            }

            .send-button:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }

            .send-button:active {
                transform: translateY(0);
            }

            .typing-indicator {
                display: none;
                padding: 20px 24px;
                background: var(--surface-alt);
                border-radius: var(--border-radius-sm);
                margin-bottom: 24px;
                align-items: center;
                gap: 12px;
                max-width: 140px;
            }

            .typing-indicator .message-avatar {
                position: relative;
                overflow: visible;
            }

            .typing-indicator .message-avatar::after {
                content: '';
                position: absolute;
                top: -4px;
                left: -4px;
                right: -4px;
                bottom: -4px;
                border: 2px solid #0ea5e9;
                border-radius: 50%;
                animation: pulse-ring 2s infinite ease-in-out;
            }

            .typing-dots {
                display: flex;
                gap: 4px;
            }

            .typing-dots span {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--text-light);
                animation: typing 1.4s infinite ease-in-out;
            }

            .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
            .typing-dots span:nth-child(2) { animation-delay: -0.16s; }

            @keyframes typing {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }

            @keyframes pulse-ring {
                0% {
                    transform: scale(1);
                    opacity: 1;
                }
                50% {
                    transform: scale(1.1);
                    opacity: 0.7;
                }
                100% {
                    transform: scale(1);
                    opacity: 1;
                }
            }

            @media (max-width: 768px) {
                .main-container {
                    padding: 16px;
                    gap: 16px;
                }

                .header {
                    flex-direction: column;
                    text-align: center;
                    gap: 20px;
                    padding: 24px;
                }

                .header-actions {
                    align-items: center;
                }

                .header h1 {
                    font-size: 2rem;
                }

                .chat-messages {
                    padding: 16px;
                }

                .message-content {
                    max-width: 90%;
                    padding: 16px 20px;
                }

                .chat-input {
                    padding: 16px 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="chat-wrapper">
                <div class="chat-header">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div class="avatar">S</div>
                        <div class="info">
                            <h3>Sofia</h3>
                            <p>Ready to help plan your next adventure</p>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <button class="new-chat-btn" onclick="startNewChat()">
                            <i class="fas fa-plus"></i> New Chat
                        </button>
                        <div class="chat-status" id="chat-status">
                            Ready to help
                        </div>
                    </div>
                </div>

                <div class="chat-messages" id="chat-messages">
                    <div class="message assistant">
                        <div class="message-avatar">S</div>
                        <div class="message-content">
                            <p>Hello! I'm Sofia, your personal travel consultant. I'm excited to help you discover amazing destinations and plan unforgettable experiences.</p>
                            <p>I can assist you with:</p>
                            <ul>
                                <li>Destination recommendations tailored to your interests</li>
                                <li>Activity planning and must-see attractions</li>
                                <li>Real-time weather and seasonal advice</li>
                                <li>Cultural insights and local customs</li>
                                <li>Practical travel logistics and tips</li>
                            </ul>
                            <p>What kind of adventure are you dreaming about?</p>
                        </div>
                    </div>
                </div>

                <div class="typing-indicator" id="typing-indicator">
                    <div class="message-avatar">S</div>
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>

                <div class="chat-input">
                    <div class="input-wrapper">
                        <input 
                            type="text" 
                            id="message-input" 
                            placeholder="Ask me about destinations, activities, or travel tips..." 
                            onkeypress="handleKeyPress(event)"
                            autocomplete="off"
                        >
                    </div>
                    <button class="send-button" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>

        <script>
            let sessionId = null;
            
            // Initialize chat from localStorage on page load
            window.addEventListener('load', function() {
                loadChatFromStorage();
                updateChatStatus();
            });
            
            function loadChatFromStorage() {
                const savedChat = localStorage.getItem('sofia_chat');
                const savedSessionId = localStorage.getItem('sofia_session_id');
                
                if (savedChat && savedSessionId) {
                    sessionId = savedSessionId;
                    const messages = JSON.parse(savedChat);
                    const messagesContainer = document.getElementById('chat-messages');
                    
                    // Clear initial message
                    messagesContainer.innerHTML = '';
                    
                    // Restore messages
                    messages.forEach(msg => {
                        if (msg.type === 'message') {
                            addMessageToDOM(msg.content, msg.role, false);
                        } else if (msg.type === 'suggestions') {
                            addSuggestionsToLastMessage(msg.suggestions);
                        }
                    });
                    
                    updateChatStatus(`Chat restored • ${messages.filter(m => m.type === 'message').length} messages`);
                }
            }
            
            function saveChatToStorage() {
                const messagesContainer = document.getElementById('chat-messages');
                const messages = [];
                
                messagesContainer.querySelectorAll('.message').forEach(messageDiv => {
                    const isUser = messageDiv.classList.contains('user');
                    const contentDiv = messageDiv.querySelector('.message-content');
                    
                    // Get main message content (excluding suggestions)
                    const messageText = contentDiv.querySelector('p').textContent;
                    
                    messages.push({
                        type: 'message',
                        role: isUser ? 'user' : 'assistant',
                        content: messageText
                    });
                    
                    // Check for suggestions
                    const suggestionsDiv = contentDiv.querySelector('.suggestions');
                    if (suggestionsDiv) {
                        const suggestions = Array.from(suggestionsDiv.querySelectorAll('li')).map(li => li.textContent);
                        messages.push({
                            type: 'suggestions',
                            suggestions: suggestions
                        });
                    }
                });
                
                localStorage.setItem('sofia_chat', JSON.stringify(messages));
                if (sessionId) {
                    localStorage.setItem('sofia_session_id', sessionId);
                }
            }
            
            function startNewChat() {
                // Clear storage
                localStorage.removeItem('sofia_chat');
                localStorage.removeItem('sofia_session_id');
                sessionId = null;
                
                // Clear chat interface
                const messagesContainer = document.getElementById('chat-messages');
                messagesContainer.innerHTML = `
                    <div class="message assistant">
                        <div class="message-avatar">S</div>
                        <div class="message-content">
                            <p>Hello! I'm Sofia, your personal travel consultant. I'm excited to help you discover amazing destinations and plan unforgettable experiences.</p>
                            <p>I can assist you with:</p>
                            <ul>
                                <li>Destination recommendations tailored to your interests</li>
                                <li>Activity planning and must-see attractions</li>
                                <li>Real-time weather and seasonal advice</li>
                                <li>Cultural insights and local customs</li>
                                <li>Practical travel logistics and tips</li>
                            </ul>
                            <p>What kind of adventure are you dreaming about?</p>
                        </div>
                    </div>
                `;
                
                updateChatStatus('Ready to help');
            }
            
            function updateChatStatus(status) {
                const statusDiv = document.getElementById('chat-status');
                if (statusDiv) {
                    statusDiv.textContent = status;
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            function formatResponse(text) {
                // Handle headers first (before paragraph processing)
                text = text.replace(/^#### (.*$)/gm, '<h4>$1</h4>');
                text = text.replace(/^### (.*$)/gm, '<h3>$1</h3>');
                text = text.replace(/^## (.*$)/gm, '<h2>$1</h2>');
                text = text.replace(/^# (.*$)/gm, '<h1>$1</h1>');
                
                // Add proper formatting for better readability
                text = text.split('\\n\\n').join('</p><p>');
                text = text.split('\\n').join('<br>');
                
                // Handle bold text **text** (using simple string replacement)
                text = text.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
                // Handle italic text *text*
                text = text.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
                // Handle bullet points
                text = text.replace(/^- /gm, '• ');
                
                // Clean up any headers that got wrapped in <p> tags
                text = text.replace(/<p>(<h[1-4]>.*?<\\/h[1-4]>)<\\/p>/g, '$1');
                
                return text;
            }
            
            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';
                updateChatStatus('Sofia is thinking...');
                
                // Show typing indicator
                showTypingIndicator();
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            message: message, 
                            session_id: sessionId 
                        })
                    });
                    
                    const data = await response.json();
                    sessionId = data.session_id;
                    
                    // Hide typing indicator
                    hideTypingIndicator();
                    
                    // Add assistant response with formatting
                    addMessage(data.response, 'assistant');
                    
                    // Add suggestions if available
                    if (data.suggestions && data.suggestions.length > 0) {
                        addSuggestions(data.suggestions);
                    }
                    
                    // Save chat to storage
                    saveChatToStorage();
                    updateChatStatus('Ready for next question');
                    
                } catch (error) {
                    hideTypingIndicator();
                    addMessage('I apologize, but I encountered a technical issue. Please try again in a moment.', 'assistant');
                    updateChatStatus('Error occurred');
                }
            }
            
            function addMessage(content, role) {
                addMessageToDOM(content, role, true);
            }
            
            function addMessageToDOM(content, role, shouldSave = true) {
                const messagesContainer = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}`;
                
                const avatarDiv = document.createElement('div');
                avatarDiv.className = 'message-avatar';
                avatarDiv.textContent = role === 'user' ? 'Y' : 'S';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                if (role === 'assistant') {
                    contentDiv.innerHTML = '<p>' + formatResponse(content) + '</p>';
                } else {
                    contentDiv.innerHTML = '<p>' + content + '</p>';
                }
                
                messageDiv.appendChild(avatarDiv);
                messageDiv.appendChild(contentDiv);
                messagesContainer.appendChild(messageDiv);
                
                // Scroll to bottom
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
                if (shouldSave) {
                    saveChatToStorage();
                }
            }
            
            function addSuggestions(suggestions) {
                addSuggestionsToLastMessage(suggestions);
                saveChatToStorage();
            }
            
            function addSuggestionsToLastMessage(suggestions) {
                const messagesContainer = document.getElementById('chat-messages');
                const lastMessage = messagesContainer.lastElementChild;
                const contentDiv = lastMessage.querySelector('.message-content');
                
                const suggestionsDiv = document.createElement('div');
                suggestionsDiv.className = 'suggestions';
                const suggestionId = 'suggestions-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                suggestionsDiv.id = suggestionId;
                
                const suggestionsHTML = `
                    <h4>Suggestions for you</h4>
                    <ul>
                        ${suggestions.map((s, index) => `<li onclick="sendSuggestion('${suggestionId}', ${index})" title="Click to send this message">${s}</li>`).join('')}
                    </ul>
                `;
                
                suggestionsDiv.innerHTML = suggestionsHTML;
                suggestionsDiv.setAttribute('data-suggestions', JSON.stringify(suggestions));
                contentDiv.appendChild(suggestionsDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function sendSuggestion(suggestionId, index) {
                // Find the specific suggestions div by ID
                const suggestionsDiv = document.getElementById(suggestionId);
                if (suggestionsDiv) {
                    const suggestions = JSON.parse(suggestionsDiv.getAttribute('data-suggestions'));
                    const suggestionText = suggestions[index];
                    
                    if (suggestionText) {
                        // Set the input value and send the message
                        const input = document.getElementById('message-input');
                        input.value = suggestionText;
                        sendMessage();
                    }
                }
            }
            
            function showTypingIndicator() {
                const indicator = document.getElementById('typing-indicator');
                const messagesContainer = document.getElementById('chat-messages');
                indicator.style.display = 'flex';
                messagesContainer.appendChild(indicator);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function hideTypingIndicator() {
                const indicator = document.getElementById('typing-indicator');
                indicator.style.display = 'none';
                if (indicator.parentNode) {
                    indicator.parentNode.removeChild(indicator);
                }
                // Re-add to chat-wrapper for next use
                document.querySelector('.chat-wrapper').appendChild(indicator);
            }
        </script>
    </body>
    </html>
    """


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with conversation management"""
    
    start_time = time.time()
    external_data_used = False
    
    try:
        # Check for conversation issues first
        session_exists = request.session_id and request.session_id in conversation_engine.sessions
        if session_exists:
            existing_session = conversation_engine.sessions[request.session_id]
            conversation_issues = recovery_engine.detect_conversation_issues(
                request.message, existing_session
            )
            
            # Handle critical issues with recovery
            critical_issues = [issue for issue in conversation_issues if issue[1] > 0.7]
            if critical_issues:
                error_type, confidence = critical_issues[0]
                recovery_data = recovery_engine.generate_recovery_response(
                    error_type, confidence, request.message, existing_session
                )
                
                return ChatResponse(
                    response=recovery_data['response'],
                    session_id=existing_session.session_id,
                    state=existing_session.state,
                    suggestions=recovery_data.get('follow_up_suggestions', [])
                )
        
        # Process the message through conversation engine
        session, detected_intents = conversation_engine.process_message(
            request.message, request.session_id
        )
        
        # Clear invalid destinations that are actually travel descriptions
        def is_invalid_destination(dest: str) -> bool:
            """Check if a destination is actually a travel description or phrase"""
            if not dest:
                return True
            
            dest_lower = dest.lower()
            
            # Check for common travel phrases that aren't destinations
            invalid_patterns = [
                r'.*\b(before|after|while|when|if|unless|during|since|until)\b',
                r'^(first|next|last|my|your|our|their)\s+.*',
                r'.*\b(trip|travel|vacation|journey|adventure|solo|group)\b.*',
                r'.*\b(photography|landscape|northern lights|aurora)\b.*',
                r'.*\b(time|moment|opportunity|experience|chance)\b.*'
            ]
            
            return any(re.search(pattern, dest_lower) for pattern in invalid_patterns)
        
        if session.context.current_destination and is_invalid_destination(session.context.current_destination):
            logger.warning(f"Clearing invalid destination: {session.context.current_destination}")
            session.context.current_destination = None
        
        # Enhanced context management with psychological profiling
        from app.core.conversation import SmartContextManager
        context_manager = SmartContextManager()
        evolved_context = context_manager.extract_evolving_context(session.messages, session.session_id)
        
        # Check if weather was recently mentioned for this destination
        weather_recently_mentioned = (
            session.context.weather_mentioned_for == session.context.current_destination and
            session.context.weather_mentioned_at and
            (datetime.now() - session.context.weather_mentioned_at).total_seconds() < 3600  # Within last hour
        )
        
        # Check if user is explicitly asking about weather
        user_asking_weather = any(keyword in request.message.lower() for keyword in [
            'weather', 'temperature', 'climate', 'rain', 'sunny', 'cold', 'hot', 'degrees'
        ])
        
        # Generate psychological profile
        psychological_profile = context_manager.psychological_profiler.analyze_user_psychology(
            session.messages, session.session_id
        )
        
        # Build prompt chain with psychological insights
        prompt_chain = conversation_engine.build_prompt_chain(
            session, request.message, psychological_profile
        )
        
        # Add conversation quality instructions
        quality_instructions = f"""
        
CONVERSATION QUALITY GUIDELINES:
- Keep responses focused and actionable
- Reference previous conversation context naturally
- Ask ONE specific follow-up question to maintain flow
- Avoid repetition of previously covered information
- Adapt tone to user's engagement level
- Be helpful but not overwhelming

WEATHER INFORMATION POLICY:
- Weather recently mentioned for {session.context.current_destination}: {'Yes' if weather_recently_mentioned else 'No'}
- User explicitly asking about weather: {'Yes' if user_asking_weather else 'No'}
- ONLY mention weather if user asks about it OR if no weather was mentioned for this destination yet
- If weather was recently provided, focus on the user's actual question instead
- When providing weather, use real-time data and be concise (1-2 sentences max)

MANDATORY CONTEXT CHECK:
Before responding, you MUST check the "CONTEXT AWARENESS" section below for:
1. Current destination focus (what "there" means)
2. Recent conversation topics (what "it" refers to)
3. Any location-specific queries without explicit location names
If context shows a specific destination, USE IT. Never give generic responses when context provides location information.

WEATHER RESPONSES:
- For weather questions: Give temperature, condition, and ONE brief travel tip (max 2-3 sentences)
- Don't write paragraphs about activities unless specifically asked
- Example: "It's 24°C and sunny in Berlin today! Perfect weather for exploring the city. Any specific area you'd like to visit?"
        """
        
        # Add current date context for accurate temporal awareness
        from app.core.date_context import date_manager
        date_context = date_manager.get_comprehensive_date_context()
        
        prompt_chain += quality_instructions + date_context
        
        # Use data orchestration instead of basic data gathering
        data_orchestration_result = await data_service.intelligent_data_orchestration(
            request.message, session.context.current_destination, psychological_profile, session
        )
        external_data = data_orchestration_result.get('data', {})
        data_metadata = data_orchestration_result.get('metadata', {})
        
        # Generate response with error handling
        messages = [
            {"role": "system", "content": prompt_chain},
        ]
        
        # Add comprehensive conversation history for context (last 6 messages for better context)
        recent_history = session.messages[-6:] if len(session.messages) > 6 else session.messages
        for msg in recent_history:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Add conversation context summary for AI awareness - CRITICAL FOR CONTEXT
        context_summary = conversation_engine._build_history_context(session)
        
        # FORCE context awareness with explicit override message
        if session.context.current_destination:
            context_override = f"""
CRITICAL CONTEXT ENFORCEMENT - MANDATORY COMPLIANCE:

CURRENT CONVERSATION STATE:
{context_summary}

ABSOLUTE REQUIREMENTS:
1. Current destination focus: {session.context.current_destination}
2. If user says "there" and destination is set, you MUST respond about {session.context.current_destination}
3. If user asks about weather without location and destination is set, respond about {session.context.current_destination}
4. NEVER give generic travel advice when a specific destination is established
5. Always mention the destination name explicitly (e.g., "In {session.context.current_destination}..." not "There...")

RESPONSE REQUIREMENTS:
- For weather queries: Give specific temperature/condition + brief travel tip (2-3 sentences MAX)
- For photography questions: Reference the destination's specific landscape features
- Always maintain conversational context - this is an ongoing conversation, not a first interaction

IMMEDIATE COMPLIANCE REQUIRED - NO EXCEPTIONS.
"""
        else:
            context_override = f"""
NO DESTINATION SET - DESTINATION DISCOVERY MODE:

CURRENT CONVERSATION STATE:
{context_summary}

ABSOLUTE REQUIREMENTS:
1. No destination is currently established
2. Focus on helping the user choose a destination based on their interests
3. Ask specific questions about their travel preferences (adventure, culture, relaxation, etc.)
4. Suggest 2-3 specific destinations that match their stated interests
5. NEVER pretend a destination is set when it isn't

RESPONSE REQUIREMENTS:
- Help narrow down destination choices based on their preferences
- Ask about travel style, interests, budget, or timing to recommend destinations
- Be specific and helpful in suggesting actual places to visit

IMMEDIATE COMPLIANCE REQUIRED - NO EXCEPTIONS.
"""
        
        messages.append({
            "role": "system", 
            "content": context_override
        })
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        base_response = await llm_service.generate_response(messages)
        
        # Enhanced response validation and recovery
        if not base_response:
            final_response = "I want to give you the best travel advice possible. Could you help me understand exactly what you're looking for? I can assist with destinations, activities, planning, or any specific travel questions you have."
        else:
            # CRITICAL: Check for generic responses and fix them
            if session.context.current_destination and _is_generic_response(base_response):
                logger.warning(f"Detected generic response despite established destination: {session.context.current_destination}")
                
                # Force a contextual response based on the specific query
                if "weather" in request.message.lower() or "temperature" in request.message.lower():
                    # Handle weather queries for established destinations with REAL weather data
                    try:
                        weather_data = await data_service.weather_service.get_current_weather(session.context.current_destination)
                        if weather_data:
                            temp_str = f"{weather_data.temperature:.0f}°C"
                            condition_str = weather_data.description
                            final_response = f"It's {temp_str} and {condition_str} in {session.context.current_destination} today! Perfect weather for exploring the city. What specific type of photography are you most interested in?"
                            
                            # Track that weather was mentioned for this destination
                            session.context.weather_mentioned_for = session.context.current_destination
                            session.context.weather_mentioned_at = datetime.now()
                        else:
                            final_response = f"I'd love to help you plan photography in {session.context.current_destination}! What specific type of photography are you most interested in?"
                    except Exception as e:
                        logger.error(f"Weather API error: {e}")
                        final_response = f"I'd love to help you plan photography in {session.context.current_destination}! What specific type of photography are you most interested in?"
                elif "best time" in request.message.lower() and "visit" in request.message.lower():
                    # Handle timing questions
                    final_response = f"For photography in {session.context.current_destination}, the best times are typically during golden hours for stunning light, and different seasons offer unique opportunities. What style of photography captures your interest most?"
                elif "there" in request.message.lower() or "it" in request.message.lower():
                    # Handle implicit location references
                    final_response = f"For {session.context.current_destination}, {base_response.replace('I can assist with destination recommendations, activity suggestions, cultural insights, and practical travel advice.', '').replace('What aspect of your trip would you like to explore first?', '').strip()} What specific aspect would you like to explore further?"
                else:
                    # General context fix
                    final_response = f"Regarding {session.context.current_destination}: {base_response}"
            else:
                final_response = base_response
            external_data_used = False
            
            # Enhance response with external data if available
            if external_data:
                formatted_data = {}
                for data_type, data in external_data.items():
                    if data_type == "weather":
                        formatted_data["weather"] = data_service.format_weather_for_llm(data)
                        # Track weather mention when external weather data is used
                        if session.context.current_destination:
                            session.context.weather_mentioned_for = session.context.current_destination
                            session.context.weather_mentioned_at = datetime.now()
                    elif data_type == "forecast":
                        # Extract location from the forecast data or session context
                        location = session.context.current_destination or "the location"
                        # Pass the user message for advanced date parsing
                        formatted_data["forecast"] = data_service.format_forecast_for_llm(data, location, request.message)
                    elif data_type == "country":
                        formatted_data["country"] = data_service.format_country_info_for_llm(data)
                
                if formatted_data:
                    enhanced_response = await llm_service.enhance_response_with_data(
                        base_response, formatted_data
                    )
                    if enhanced_response:
                        final_response = enhanced_response
                        external_data_used = True
        
        # Add response to conversation history
        conversation_engine.add_assistant_response(session, final_response)
        
        # Track conversation quality metrics
        quality_metrics = context_manager.track_conversation_quality(
            session.session_id, request.message, final_response
        )
        
        # Generate contextually intelligent suggestions
        conversation_history = [
            {"role": msg.role.value, "content": msg.content} 
            for msg in session.messages[-3:]  # Last 3 messages for focused context
        ]
        
        # Enhanced suggestion generation with conversation momentum
        suggestions = await llm_service.generate_contextual_suggestions(
            final_response, conversation_history
        )
        
        # Quality assurance for suggestions
        if not suggestions or len(suggestions) < 2:
            suggestions = _generate_smart_fallback_suggestions(
                evolved_context, detected_intents, quality_metrics
            )
        
        return ChatResponse(
            response=final_response,
            session_id=session.session_id,
            state=session.state,
            suggestions=suggestions,
            external_data_used=external_data_used
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get conversation session details"""
    if session_id not in conversation_engine.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = conversation_engine.sessions[session_id]
    return {
        "session_id": session.session_id,
        "state": session.state,
        "message_count": len(session.messages),
        "current_destination": session.context.current_destination,
        "detected_intents": session.detected_intents,
        "conversation_depth": session.context.conversation_depth
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": len(conversation_engine.sessions)
    }




def _generate_smart_fallback_suggestions(context: Dict, intents: List, quality_metrics: Dict) -> List[str]:
    """Generate intelligent fallback suggestions based on conversation context"""
    
    suggestions = []
    
    # Context-aware suggestions
    if context.get("destinations_mentioned"):
        dest = context["destinations_mentioned"][-1]
        suggestions.extend([
            f"What's the best time to visit {dest}?",
            f"Tell me about local culture in {dest}",
            f"What should I budget for {dest}?"
        ])
    
    # Intent-based suggestions
    elif UserIntent.DESTINATION_INQUIRY in intents:
        suggestions.extend([
            "I'm interested in cultural experiences",
            "Show me adventure destinations",
            "What's good for relaxation?"
        ])
    
    elif UserIntent.ACTIVITY_REQUEST in intents:
        suggestions.extend([
            "Tell me about local food specialties",
            "What are the must-see attractions?",
            "Any hidden gems to discover?"
        ])
    
    # Engagement-based suggestions
    if quality_metrics.get("user_engagement", 0) > 0.7:
        suggestions.extend([
            "Help me plan the perfect itinerary",
            "What else should I know?",
            "Any insider tips?"
        ])
    
    return suggestions[:3]  # Return top 3


def _is_generic_response(response: str) -> bool:
    """Detect generic travel responses that ignore established context"""
    generic_indicators = [
        "I'm excited to help plan your next adventure",
        "I can assist with destination recommendations",
        "What aspect of your trip would you like to explore first",
        "destination recommendations, activity suggestions, cultural insights",
        "travel advice. What aspect of your trip",
        "What kind of adventure are you dreaming about",
        "I'd love to help plan your next adventure",
        "What aspect of your trip would you like to explore first?",
        "travel advice",
        "What are you most excited to discover"
    ]
    
    response_lower = response.lower()
    # Check if the response contains generic indicators and lacks specific destination context
    has_generic_content = any(indicator.lower() in response_lower for indicator in generic_indicators)
    
    # Also check for overly vague responses that don't address the specific question
    is_too_vague = (
        len(response.split()) < 30 and 
        response.count("?") > 0 and
        not any(word in response_lower for word in ['weather', 'temperature', 'climate', 'photography', 'iceland', 'landscape'])
    )
    
    return has_generic_content or is_too_vague


def _generate_suggestions(state: ConversationState, intents: List, context) -> List[str]:
    """Generate contextual follow-up suggestions"""
    
    suggestions = []
    
    if state == ConversationState.DESTINATION_PLANNING:
        suggestions.extend([
            "What's the best time of year to visit?",
            "What are the must-see attractions?",
            "Tell me about local customs and etiquette"
        ])
    
    elif state == ConversationState.ACTIVITY_DISCOVERY:
        suggestions.extend([
            "What's the local food scene like?",
            "Are there any hidden gems off the beaten path?",
            "What should I pack for this destination?"
        ])
    
    elif state == ConversationState.PRACTICAL_PLANNING:
        suggestions.extend([
            "How should I budget for this trip?",
            "What's the best way to get around locally?",
            "Do I need any special travel documents?"
        ])
    
    elif state == ConversationState.CONTEXT_ENRICHMENT:
        suggestions.extend([
            "Can you recommend specific neighborhoods to explore?",
            "What are some unique cultural experiences?",
            "How can I connect with locals?"
        ])
    
    # Add destination-specific suggestions if we know the destination
    if context.current_destination:
        suggestions.append(f"What else should I know about {context.current_destination}?")
    
    return suggestions[:3]  # Limit to 3 suggestions


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
