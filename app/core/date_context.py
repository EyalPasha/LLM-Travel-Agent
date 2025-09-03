"""
Advanced Date Parsing and Context Management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import re


class DateContextManager:
    """Advanced date parsing and context management for natural language date references"""
    
    def __init__(self):
        self.current_date = datetime.now()
        
        # Day name mappings
        self.weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6,
            'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6
        }
        
        # Month name mappings
        self.months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
    
    def parse_natural_date(self, text: str) -> Optional[datetime]:
        """Parse natural language date references"""
        text_lower = text.lower().strip()
        
        # Handle specific day references
        if 'today' in text_lower:
            return self.current_date
        
        if 'tomorrow' in text_lower or 'tmrw' in text_lower:
            return self.current_date + timedelta(days=1)
        
        if 'yesterday' in text_lower:
            return self.current_date - timedelta(days=1)
        
        # Handle "next" references
        if 'next week' in text_lower:
            return self.current_date + timedelta(weeks=1)
        
        if 'next month' in text_lower:
            # Add approximately 30 days for next month
            return self.current_date + timedelta(days=30)
        
        if 'next year' in text_lower:
            return self.current_date.replace(year=self.current_date.year + 1)
        
        # Handle "this" references
        if 'this week' in text_lower:
            return self.current_date
        
        if 'this month' in text_lower:
            return self.current_date
        
        # Handle specific weekday references
        for day_name, day_num in self.weekdays.items():
            if day_name in text_lower:
                return self._get_next_weekday(day_num, text_lower)
        
        # Handle weekend references
        if 'weekend' in text_lower:
            # Find the next Saturday
            return self._get_next_weekday(5, 'next')  # Saturday is day 5
        
        # Handle month references
        for month_name, month_num in self.months.items():
            if month_name in text_lower:
                return self._get_date_for_month(month_num, text_lower)
        
        # Handle relative day references (in X days)
        day_match = re.search(r'in (\d+) days?', text_lower)
        if day_match:
            days = int(day_match.group(1))
            return self.current_date + timedelta(days=days)
        
        # Handle relative week references (in X weeks)
        week_match = re.search(r'in (\d+) weeks?', text_lower)
        if week_match:
            weeks = int(week_match.group(1))
            return self.current_date + timedelta(weeks=weeks)
        
        return None
    
    def _get_next_weekday(self, target_day: int, text: str) -> datetime:
        """Get the next occurrence of a specific weekday"""
        current_day = self.current_date.weekday()
        
        if 'next' in text:
            # Next week's occurrence of this day
            days_ahead = target_day - current_day + 7
        else:
            # This week's occurrence (or next week if already passed)
            days_ahead = target_day - current_day
            if days_ahead <= 0:  # If day has passed this week, get next week
                days_ahead += 7
        
        return self.current_date + timedelta(days=days_ahead)
    
    def _get_date_for_month(self, target_month: int, text: str) -> datetime:
        """Get date for a specific month reference"""
        current_month = self.current_date.month
        current_year = self.current_date.year
        
        if 'next' in text:
            # Next year's occurrence of this month
            year = current_year + 1 if target_month <= current_month else current_year
            if target_month <= current_month:
                year = current_year + 1
            else:
                year = current_year
        else:
            # This year's occurrence (or next year if already passed)
            if target_month < current_month:
                year = current_year + 1
            else:
                year = current_year
        
        return datetime(year, target_month, 1)
    
    def get_weather_time_context(self, user_message: str) -> Dict[str, Any]:
        """Get comprehensive time context for weather requests"""
        parsed_date = self.parse_natural_date(user_message)
        
        context = {
            'current_date': self.current_date,
            'target_date': parsed_date,
            'is_today': False,
            'is_tomorrow': False,
            'is_future': False,
            'days_from_now': 0,
            'formatted_target': None,
            'time_description': 'today'
        }
        
        if parsed_date:
            days_diff = (parsed_date.date() - self.current_date.date()).days
            context['days_from_now'] = days_diff
            context['formatted_target'] = parsed_date.strftime('%Y-%m-%d')
            
            if days_diff == 0:
                context['is_today'] = True
                context['time_description'] = 'today'
            elif days_diff == 1:
                context['is_tomorrow'] = True
                context['time_description'] = 'tomorrow'
            elif days_diff > 1:
                context['is_future'] = True
                if days_diff <= 7:
                    context['time_description'] = f'in {days_diff} days ({parsed_date.strftime("%A")})'
                else:
                    context['time_description'] = f'on {parsed_date.strftime("%B %d, %Y")}'
        
        return context
    
    def get_comprehensive_date_context(self) -> str:
        """Get comprehensive date context for LLM"""
        current_date = datetime.now()
        
        # Calculate various relative dates
        tomorrow = current_date + timedelta(days=1)
        next_week = current_date + timedelta(weeks=1)
        next_month_approx = current_date + timedelta(days=30)
        
        # Get upcoming weekdays
        upcoming_days = []
        for i in range(1, 8):  # Next 7 days
            future_date = current_date + timedelta(days=i)
            upcoming_days.append(f"{future_date.strftime('%A')}: {future_date.strftime('%Y-%m-%d')}")
        
        context = f"""

COMPREHENSIVE DATE & TIME CONTEXT:
- Current Date: {current_date.strftime('%A, %B %d, %Y')}
- Current Time: {current_date.strftime('%H:%M')} (24-hour format)
- Current Year: {current_date.year}
- Current Month: {current_date.strftime('%B')} ({current_date.month})
- Current Day of Week: {current_date.strftime('%A')}

RELATIVE DATE REFERENCES:
- Today: {current_date.strftime('%Y-%m-%d')}
- Tomorrow: {tomorrow.strftime('%Y-%m-%d (%A)')}
- Next Week (approximately): {next_week.strftime('%Y-%m-%d')}
- Next Month (approximately): {next_month_approx.strftime('%Y-%m-%d')}

UPCOMING WEEKDAYS:
{chr(10).join(f"- {day}" for day in upcoming_days)}

When users mention days like "Monday", "Tuesday", dates like "next week", "next month", 
or relative terms like "tomorrow", "in 3 days", use this context to understand exactly 
which date they mean. Always provide weather and information for the correct target date.
        """
        
        return context


# Global instance
date_manager = DateContextManager()
