"""
External API Services for Data Augmentation
"""

import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.core.config import settings
from app.models.conversation import WeatherData, CountryInfo
from app.core.date_context import date_manager


class WeatherService:
    """OpenWeatherMap API integration for weather data"""
    
    def __init__(self):
        self.api_key = settings.WEATHER_API_KEY
        self.base_url = settings.WEATHER_BASE_URL
    
    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather for a location with enhanced error handling"""
        if not self.api_key:
            # Return mock data for testing when no API key
            return WeatherData(
                location=location,
                temperature=22.0,
                condition="partly cloudy",
                description="partly cloudy",
                humidity=65,
                wind_speed=3.5
            )
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/weather",
                    params={
                        "q": location,
                        "appid": self.api_key,
                        "units": "metric"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return WeatherData(
                        location=data["name"],
                        temperature=data["main"]["temp"],
                        condition=data["weather"][0]["main"],
                        humidity=data["main"]["humidity"],
                        wind_speed=data["wind"]["speed"],
                        description=data["weather"][0]["description"]
                    )
                else:
                    # Return fallback data for invalid locations
                    return WeatherData(
                        location=location,
                        temperature=20.0,
                        condition="unknown",
                        description="weather data unavailable",
                        humidity=50,
                        wind_speed=0.0
                    )
                    
        except Exception as e:
            print(f"Weather API error: {e}")
            # Return fallback data
            return WeatherData(
                location=location,
                temperature=18.0,
                condition="unknown",
                description="weather service temporarily unavailable",
                humidity=55,
                wind_speed=2.0
            )
    
    async def get_weather_forecast(self, location: str, days: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Get weather forecast for location"""
        if not self.api_key:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/forecast",
                    params={
                        "q": location,
                        "appid": self.api_key,
                        "units": "metric",
                        "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    forecasts = []
                    
                    for item in data["list"]:
                        forecasts.append({
                            "datetime": item["dt_txt"],
                            "temperature": item["main"]["temp"],
                            "condition": item["weather"][0]["main"],
                            "description": item["weather"][0]["description"],
                            "humidity": item["main"]["humidity"],
                            "wind_speed": item["wind"]["speed"]
                        })
                    
                    return forecasts
        except Exception as e:
            print(f"Weather forecast API error: {e}")
            return None


class CountryService:
    """REST Countries API integration for country information"""
    
    def __init__(self):
        self.base_url = settings.COUNTRIES_BASE_URL
    
    async def get_country_info(self, country_name: str) -> Optional[CountryInfo]:
        """Get comprehensive country information"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/name/{country_name}",
                    params={"fullText": "false"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        country = data[0]  # Take first match
                        
                        return CountryInfo(
                            name=country["name"]["common"],
                            capital=country.get("capital", ["Unknown"])[0],
                            population=country.get("population", 0),
                            currencies=list(country.get("currencies", {}).keys()),
                            languages=list(country.get("languages", {}).values()),
                            timezone=country.get("timezones", ["Unknown"])[0],
                            continent=country.get("continents", ["Unknown"])[0]
                        )
        except Exception as e:
            print(f"Country API error: {e}")
            return None
    
    async def get_neighboring_countries(self, country_name: str) -> Optional[List[str]]:
        """Get list of neighboring countries"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/name/{country_name}",
                    params={"fullText": "false"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        country = data[0]
                        borders = country.get("borders", [])
                        
                        if borders:
                            # Get country names from country codes
                            neighbor_names = []
                            for border_code in borders:
                                neighbor_response = await client.get(
                                    f"{self.base_url}/alpha/{border_code}"
                                )
                                if neighbor_response.status_code == 200:
                                    neighbor_data = neighbor_response.json()
                                    neighbor_names.append(neighbor_data[0]["name"]["common"])
                            
                            return neighbor_names
        except Exception as e:
            print(f"Neighboring countries API error: {e}")
            return None


class DataAugmentationService:
    """Revolutionary intelligent service for strategic external data usage"""
    
    def __init__(self):
        self.weather_service = WeatherService()
        self.country_service = CountryService()
        self.data_confidence_cache = {}  # Cache confidence scores
        self.user_data_preferences = {}  # Track user preferences for data types
        self.data_usage_analytics = {}   # Track data usage patterns
        self.smart_caching = {}          # Intelligent caching system
    
    async def should_fetch_weather(self, user_message: str, destination: Optional[str]) -> bool:
        """Determine if weather data would be valuable for the response"""
        weather_keywords = [
            "weather", "temperature", "climate", "rain", "sunny", "cold", "hot",
            "pack", "clothing", "what to wear", "best time", "season", "forecast",
            "today", "tomorrow", "degrees", "celsius", "fahrenheit"
        ]
        
        message_lower = user_message.lower()
        return any(keyword in message_lower for keyword in weather_keywords) and destination is not None
    
    async def should_fetch_country_info(self, user_message: str, destination: Optional[str]) -> bool:
        """Determine if country information would be valuable"""
        country_keywords = [
            "culture", "customs", "language", "currency", "population",
            "capital", "neighbors", "timezone", "continent", "borders"
        ]
        
        message_lower = user_message.lower()
        return any(keyword in message_lower for keyword in country_keywords) and destination is not None
    
    async def gather_relevant_data(self, user_message: str, destination: Optional[str]) -> Dict[str, Any]:
        """Intelligently gather relevant external data"""
        data = {}
        
        if not destination:
            return data
        
        tasks = []
        
        # Check if weather data is needed
        if await self.should_fetch_weather(user_message, destination):
            # Check if it's a forecast request
            if any(word in user_message.lower() for word in ['forecast', 'tomorrow', 'tmrw', 'next', 'week', 'days', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                # Request longer forecast for better date coverage
                tasks.append(("forecast", self.weather_service.get_weather_forecast(destination, days=5)))
            else:
                tasks.append(("weather", self.weather_service.get_current_weather(destination)))
        
        # Check if country info is needed
        if await self.should_fetch_country_info(user_message, destination):
            tasks.append(("country", self.country_service.get_country_info(destination)))
        
        # Execute tasks concurrently
        if tasks:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            for i, (data_type, result) in enumerate(zip([task[0] for task in tasks], results)):
                if not isinstance(result, Exception) and result is not None:
                    data[data_type] = result
        
        return data
    
    def calculate_data_value_score(self, user_message: str, data_type: str, 
                                   psychological_profile: Dict = None) -> float:
        """Calculate the value score for using specific external data"""
        
        # Base relevance scoring
        relevance_keywords = {
            'weather': ['weather', 'temperature', 'climate', 'rain', 'pack', 'wear', 'season', 'time', 'forecast', 'today', 'tomorrow', 'degrees', 'celsius', 'fahrenheit', 'sunny', 'cold', 'hot'],
            'country': ['culture', 'customs', 'language', 'currency', 'visa', 'safety', 'local']
        }
        
        message_lower = user_message.lower()
        base_score = 0.0
        
        if data_type in relevance_keywords:
            keyword_matches = sum(1 for keyword in relevance_keywords[data_type] 
                                if keyword in message_lower)
            base_score = min(keyword_matches * 0.2, 1.0)
        
        # Psychological profiling boost
        if psychological_profile:
            decision_style = psychological_profile.get('decision_pattern', 'Intuitive')
            
            # Analytical users value data more
            if decision_style == 'Analytical':
                base_score *= 1.3
            elif decision_style == 'Intuitive':
                base_score *= 0.8
        
        # Context-based adjustments
        planning_indicators = ['plan', 'prepare', 'book', 'decide', 'when']
        if any(indicator in message_lower for indicator in planning_indicators):
            base_score *= 1.2
        
        return min(base_score, 1.0)
    
    async def intelligent_data_decision(self, user_message: str, destination: Optional[str],
                                      psychological_profile: Dict = None) -> Dict[str, Any]:
        """Make intelligent decisions about which data to fetch and how to use it"""
        
        data_strategy = {
            'weather': {'fetch': False, 'priority': 0.0},
            'forecast': {'fetch': False, 'priority': 0.0},
            'country': {'fetch': False, 'priority': 0.0}
        }
        
        if not destination:
            return data_strategy
        
        msg_lower = user_message.lower()
        
        # Enhanced weather detection - MORE RESTRICTIVE
        weather_keywords = [
            'weather', 'temperature', 'rain', 'sunny', 'cloudy', 'snow', 'hot', 'cold', 
            'climate', 'forecast', 'degrees', 'celsius', 'fahrenheit', 'humid', 'windy',
            'what\'s it like', 'how\'s the weather', 'what\'s the temperature'
        ]
        
        # Enhanced country/culture detection  
        country_keywords = [
            'culture', 'custom', 'tradition', 'language', 'currency', 'capital', 
            'population', 'food', 'religion', 'history', 'people', 'local'
        ]
        
        # Check for explicit weather requests - MUCH MORE RESTRICTIVE
        weather_score = 0.0
        explicit_weather_request = False
        for keyword in weather_keywords:
            if keyword in msg_lower:
                weather_score = max(weather_score, 0.9)
                explicit_weather_request = True
                break
        
        # Additional check: Only include weather if it's REALLY about weather
        if not explicit_weather_request:
            # Check for contextual weather needs (like packing, timing, etc.)
            contextual_weather = [
                'what to pack', 'what should i bring', 'what to wear', 
                'best time to visit', 'when to go', 'good time for'
            ]
            for phrase in contextual_weather:
                if phrase in msg_lower:
                    weather_score = max(weather_score, 0.6)
                    break
        
        # Don't include weather for general travel questions
        if any(phrase in msg_lower for phrase in [
            'hidden gems', 'photo spots', 'photography', 'activities', 'things to do',
            'restaurants', 'hotels', 'attractions', 'sightseeing', 'itinerary',
            'travel tips', 'recommendations', 'suggestions', 'advice'
        ]) and weather_score < 0.9:  # Only override if not explicitly asking for weather
            weather_score = 0.0
        
        # Check for forecast-specific requests
        forecast_keywords = ['forecast', 'tomorrow', 'next week', 'next few days', 'this week']
        forecast_score = 0.0
        for keyword in forecast_keywords:
            if keyword in msg_lower:
                forecast_score = max(forecast_score, 0.9)
        
        # Check for country/cultural information requests
        country_score = 0.0
        for keyword in country_keywords:
            if keyword in msg_lower:
                country_score = max(country_score, 0.8)
        
        # Fallback scoring using original method for complex queries
        if weather_score == 0.0 and forecast_score == 0.0 and country_score == 0.0:
            weather_score = max(weather_score, self.calculate_data_value_score(
                user_message, 'weather', psychological_profile
            ))
            country_score = max(country_score, self.calculate_data_value_score(
                user_message, 'country', psychological_profile
            ))
        
        # Decision thresholds - lowered for better responsiveness
        FETCH_THRESHOLD = 0.05
        
        # Make data fetching decisions
        if forecast_score > FETCH_THRESHOLD:
            data_strategy['forecast']['fetch'] = True
            data_strategy['forecast']['priority'] = forecast_score
        elif weather_score > FETCH_THRESHOLD:
            data_strategy['weather']['fetch'] = True
            data_strategy['weather']['priority'] = weather_score
        
        if country_score > FETCH_THRESHOLD:
            data_strategy['country']['fetch'] = True
            data_strategy['country']['priority'] = country_score
        
        return data_strategy
    
    def format_weather_for_llm(self, weather: WeatherData) -> str:
        """Format weather data for LLM consumption - concise format"""
        return f"""WEATHER in {weather.location}: {weather.temperature}°C, {weather.description}. Humidity {weather.humidity}%, wind {weather.wind_speed}m/s.

Keep response concise - just answer the weather question with a brief, helpful follow-up if relevant."""
    
    def format_forecast_for_llm(self, forecast_data: List[Dict[str, Any]], location: str, user_message: str = "") -> str:
        """Format weather forecast data for LLM consumption with advanced date parsing"""
        if not forecast_data:
            return ""
        
        # Get time context from user message
        time_context = date_manager.get_weather_time_context(user_message)
        target_date = time_context['formatted_target']
        time_description = time_context['time_description']
        
        if target_date:
            # Get forecast for specific target date
            target_forecast = [entry for entry in forecast_data if entry['datetime'].startswith(target_date)]
            
            forecast_text = f"\nWEATHER FORECAST DATA for {location}:\n"
            forecast_text += f"{time_description.title()} ({target_date}):\n"
            
            if target_forecast:
                for entry in target_forecast:
                    time_str = entry['datetime'].split(' ')[1][:5]  # Extract time HH:MM
                    forecast_text += f"- {time_str}: {entry['temperature']}°C, {entry['description']}\n"
                
                # Calculate daily summary
                temps = [entry['temperature'] for entry in target_forecast]
                high_temp = max(temps)
                low_temp = min(temps)
                
                forecast_text += f"\n{time_description.title()}'s Summary: High {high_temp}°C, Low {low_temp}°C\n"
                forecast_text += f"\nThis forecast data can help you plan activities and clothing for {time_description}.\n"
            else:
                forecast_text += f"No detailed forecast available for {time_description}.\n"
        else:
            # Default to today's forecast
            current_date = datetime.now()
            today_date = current_date.strftime('%Y-%m-%d')
            today_forecast = [entry for entry in forecast_data if entry['datetime'].startswith(today_date)]
            
            forecast_text = f"\nWEATHER FORECAST DATA for {location}:\n"
            forecast_text += f"Today ({today_date}):\n"
            
            if today_forecast:
                for entry in today_forecast:
                    time_str = entry['datetime'].split(' ')[1][:5]  # Extract time HH:MM
                    forecast_text += f"- {time_str}: {entry['temperature']}°C, {entry['description']}\n"
                
                # Calculate daily summary
                temps = [entry['temperature'] for entry in today_forecast]
                high_temp = max(temps)
                low_temp = min(temps)
                
                forecast_text += f"\nToday's Summary: High {high_temp}°C, Low {low_temp}°C\n"
                forecast_text += "\nThis real-time forecast data can inform activity planning and clothing recommendations for today.\n"
            else:
                # Fallback to first 8 entries if no today data
                today_forecast = forecast_data[:8]
                for entry in today_forecast:
                    time_str = entry['datetime'].split(' ')[1][:5]  # Extract time HH:MM
                    forecast_text += f"- {time_str}: {entry['temperature']}°C, {entry['description']}\n"
                
                temps = [entry['temperature'] for entry in today_forecast]
                high_temp = max(temps)
                low_temp = min(temps)
                
                forecast_text += f"\nSummary: High {high_temp}°C, Low {low_temp}°C\n"
                forecast_text += "\nThis forecast data can inform activity planning and clothing recommendations.\n"
        
        return forecast_text
    
    def format_country_info_for_llm(self, country: CountryInfo) -> str:
        """Format country information for LLM consumption"""
        return f"""
COUNTRY INFORMATION for {country.name}:
- Capital: {country.capital}
- Population: {country.population:,}
- Currencies: {', '.join(country.currencies)}
- Languages: {', '.join(country.languages)}
- Timezone: {country.timezone}
- Continent: {country.continent}

This data can inform cultural context, practical planning, and local insights.
"""
    
    def evaluate_data_freshness_importance(self, user_message: str, destination: str) -> Dict[str, float]:
        """Evaluate how important fresh data is for different data types"""
        
        freshness_importance = {
            'weather': 0.9,   # Weather changes frequently
            'forecast': 0.9,  # Forecast is also time-sensitive
            'country': 0.3,   # Country info is relatively static
            'events': 0.8,    # Events are time-sensitive
            'prices': 0.7     # Prices change but not daily
        }
        
        # Adjust based on user intent
        msg_lower = user_message.lower()
        
        # Time-sensitive queries increase weather importance
        if any(word in msg_lower for word in ['today', 'now', 'current', 'this week']):
            freshness_importance['weather'] = 1.0
        
        # Planning queries may care less about exact current conditions
        if any(word in msg_lower for word in ['planning', 'next year', 'future', 'considering']):
            freshness_importance['weather'] = 0.6
        
        return freshness_importance
    
    def smart_data_cache_decision(self, data_type: str, location: str, age_hours: float) -> bool:
        """Make intelligent decisions about using cached vs fresh data"""
        
        cache_thresholds = {
            'weather': 1.0,     # Cache weather for 1 hour
            'forecast': 1.0,    # Cache forecast for 1 hour
            'country': 24 * 7,  # Cache country info for 1 week
            'events': 6.0,      # Cache events for 6 hours
            'attractions': 24.0  # Cache attractions for 24 hours
        }
        
        threshold = cache_thresholds.get(data_type, 12.0)
        
        # Use cached data if it's fresh enough
        return age_hours <= threshold
    
    async def intelligent_data_orchestration(self, user_message: str, destination: Optional[str],
                                           psychological_profile: Dict = None, session = None) -> Dict[str, Any]:
        """Orchestrate intelligent data gathering with caching and prioritization"""
        
        if not destination:
            return {'data': {}, 'metadata': {}, 'strategy_used': {}}
        
        # Check if weather was recently mentioned (and user isn't explicitly asking for weather)
        user_asking_weather = any(keyword in user_message.lower() for keyword in [
            'weather', 'temperature', 'climate', 'rain', 'sunny', 'cold', 'hot', 'degrees'
        ])
        
        weather_recently_mentioned = False
        if session and hasattr(session, 'context'):
            from datetime import datetime
            weather_recently_mentioned = (
                session.context.weather_mentioned_for == destination and
                session.context.weather_mentioned_at and
                (datetime.now() - session.context.weather_mentioned_at).total_seconds() < 3600  # Within last hour
            )
        
        # Analyze data needs
        data_strategy = await self.intelligent_data_decision(
            user_message, destination, psychological_profile
        )
        
        # Override weather strategy if weather was recently mentioned and user isn't explicitly asking
        if weather_recently_mentioned and not user_asking_weather:
            data_strategy['weather']['fetch'] = False
            data_strategy['forecast']['fetch'] = False
        
        freshness_requirements = self.evaluate_data_freshness_importance(user_message, destination)
        
        gathered_data = {}
        data_metadata = {}
        
        for data_type, strategy in data_strategy.items():
            if not strategy['fetch']:
                continue
            
            # Check cache first
            cache_key = f"{data_type}:{destination.lower()}"
            cached_result = self._check_smart_cache(cache_key, freshness_requirements.get(data_type, 1.0))
            
            if cached_result:
                gathered_data[data_type] = cached_result['data']
                data_metadata[data_type] = {
                    'source': 'cache',
                    'age_hours': cached_result['age_hours'],
                    'confidence': cached_result.get('confidence', 0.8)
                }
            else:
                # Fetch fresh data with proper error handling
                fresh_data = None
                try:
                    if data_type == 'weather':
                        fresh_data = await self.weather_service.get_current_weather(destination)
                    elif data_type == 'forecast':
                        fresh_data = await self.weather_service.get_weather_forecast(destination, days=5)
                    elif data_type == 'country':
                        fresh_data = await self.country_service.get_country_info(destination)
                    
                    if fresh_data:
                        gathered_data[data_type] = fresh_data
                        data_metadata[data_type] = {
                            'source': 'api',
                            'age_hours': 0.0,
                            'confidence': 0.9
                        }
                        # Cache the fresh data
                        self._update_smart_cache(cache_key, fresh_data, 0.9)
                        
                except Exception as e:
                    # Log error but don't fail completely
                    import logging
                    logging.warning(f"Failed to fetch {data_type} data for {destination}: {e}")
                    
                    # Try to use slightly stale cached data as fallback
                    fallback_cache = self._check_smart_cache(cache_key, freshness_requirements.get(data_type, 1.0) * 2)
                    if fallback_cache:
                        gathered_data[data_type] = fallback_cache['data']
                        data_metadata[data_type] = {
                            'source': 'stale_cache',
                            'age_hours': fallback_cache['age_hours'],
                            'confidence': 0.6
                        }
        
        # Track usage analytics
        self._track_data_usage(user_message, destination, data_strategy, gathered_data)
        
        return {
            'data': gathered_data,
            'metadata': data_metadata,
            'strategy_used': data_strategy
        }
    
    def _check_smart_cache(self, cache_key: str, max_age_hours: float) -> Optional[Dict[str, Any]]:
        """Check if cached data is still valid and useful"""
        
        if cache_key not in self.smart_caching:
            return None
        
        cached_entry = self.smart_caching[cache_key]
        age_hours = (datetime.now() - cached_entry['timestamp']).total_seconds() / 3600
        
        if age_hours <= max_age_hours:
            return {
                'data': cached_entry['data'],
                'age_hours': age_hours,
                'confidence': cached_entry['confidence']
            }
        
        return None
    
    def _update_smart_cache(self, cache_key: str, data: Any, confidence: float):
        """Update cache with fresh data"""
        
        from datetime import datetime
        
        self.smart_caching[cache_key] = {
            'data': data,
            'timestamp': datetime.now(),
            'confidence': confidence,
            'access_count': self.smart_caching.get(cache_key, {}).get('access_count', 0) + 1
        }
        
        # Prevent cache from growing too large
        if len(self.smart_caching) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.smart_caching.keys(),
                key=lambda k: self.smart_caching[k]['timestamp']
            )[:20]
            
            for key in oldest_keys:
                del self.smart_caching[key]
    
    def _track_data_usage(self, user_message: str, destination: str, 
                         strategy: Dict[str, Any], gathered_data: Dict[str, Any]):
        """Track data usage patterns for analytics and optimization"""
        
        usage_key = f"{destination.lower()}:{datetime.now().strftime('%Y-%m-%d')}"
        
        if usage_key not in self.data_usage_analytics:
            self.data_usage_analytics[usage_key] = {
                'destination': destination,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'data_types_requested': {},
                'data_types_successful': {},
                'user_intent_patterns': [],
                'response_quality_feedback': []
            }
        
        analytics = self.data_usage_analytics[usage_key]
        
        # Track what was requested vs what was successfully gathered
        for data_type, strategy_info in strategy.items():
            if strategy_info['fetch']:
                analytics['data_types_requested'][data_type] = analytics['data_types_requested'].get(data_type, 0) + 1
                
                if data_type in gathered_data:
                    analytics['data_types_successful'][data_type] = analytics['data_types_successful'].get(data_type, 0) + 1
        
        # Extract user intent patterns for learning
        intent_keywords = self._extract_intent_keywords(user_message)
        analytics['user_intent_patterns'].append({
            'keywords': intent_keywords,
            'timestamp': datetime.now().isoformat()
        })
    
    def _extract_intent_keywords(self, user_message: str) -> List[str]:
        """Extract key intent-revealing words from user message"""
        
        intent_keywords = []
        msg_lower = user_message.lower()
        
        keyword_categories = {
            'planning': ['planning', 'prepare', 'organize', 'schedule'],
            'immediate': ['now', 'today', 'current', 'right now'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'climate'],
            'cultural': ['culture', 'customs', 'local', 'traditional'],
            'practical': ['visa', 'money', 'transportation', 'safety'],
            'experiential': ['experience', 'feel', 'authentic', 'unique']
        }
        
        for category, keywords in keyword_categories.items():
            if any(keyword in msg_lower for keyword in keywords):
                intent_keywords.append(category)
        
        return intent_keywords
    
    def get_data_usage_insights(self) -> Dict[str, Any]:
        """Get insights about data usage patterns for optimization"""
        
        total_requests = 0
        success_rates = {}
        popular_destinations = {}
        
        for analytics in self.data_usage_analytics.values():
            total_requests += sum(analytics['data_types_requested'].values())
            
            destination = analytics['destination']
            popular_destinations[destination] = popular_destinations.get(destination, 0) + 1
            
            for data_type, requested in analytics['data_types_requested'].items():
                successful = analytics['data_types_successful'].get(data_type, 0)
                
                if data_type not in success_rates:
                    success_rates[data_type] = {'requested': 0, 'successful': 0}
                
                success_rates[data_type]['requested'] += requested
                success_rates[data_type]['successful'] += successful
        
        # Calculate success rates
        for data_type in success_rates:
            rate_data = success_rates[data_type]
            rate_data['success_rate'] = (rate_data['successful'] / rate_data['requested'] 
                                       if rate_data['requested'] > 0 else 0.0)
        
        return {
            'total_data_requests': total_requests,
            'success_rates_by_type': success_rates,
            'popular_destinations': dict(sorted(popular_destinations.items(), 
                                              key=lambda x: x[1], reverse=True)[:10]),
            'cache_statistics': {
                'entries': len(self.smart_caching),
                'hit_rate': self._calculate_cache_hit_rate()
            }
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for performance optimization"""
        
        total_accesses = sum(entry.get('access_count', 0) for entry in self.smart_caching.values())
        if total_accesses == 0:
            return 0.0
        
        # Simplified calculation - in production would track hits vs misses more precisely
        return min(0.8, total_accesses / (total_accesses + 10))  # Rough estimate
