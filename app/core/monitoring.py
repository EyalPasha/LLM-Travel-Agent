"""
Production-Ready Monitoring and Analytics System
"""

import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import defaultdict, deque

from app.models.conversation import ConversationSession, UserIntent


class MetricType(str, Enum):
    """Types of metrics we track"""
    RESPONSE_TIME = "response_time"
    CONVERSATION_QUALITY = "conversation_quality" 
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    DATA_USAGE = "data_usage"
    INTENT_ACCURACY = "intent_accuracy"
    PSYCHOLOGICAL_PROFILING = "psychological_profiling"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None


@dataclass
class SystemAlert:
    """System alert for monitoring issues"""
    level: AlertLevel
    message: str
    metric_name: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Advanced metrics collection and analysis system"""
    
    def __init__(self, max_metrics_per_type: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self.alerts: List[SystemAlert] = []
        self.thresholds: Dict[str, Dict[str, float]] = self._setup_default_thresholds()
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []
        self.logger = logging.getLogger(__name__)
        
        # Real-time tracking
        self.current_sessions = {}
        self.session_metrics = defaultdict(dict)
        
        # Performance baselines
        self.performance_baselines = {}
        self._initialize_baselines()
    
    def _setup_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup default alerting thresholds"""
        return {
            MetricType.RESPONSE_TIME: {
                "warning": 3.0,   # 3 seconds
                "error": 5.0,     # 5 seconds
                "critical": 10.0  # 10 seconds
            },
            MetricType.ERROR_RATE: {
                "warning": 0.05,  # 5%
                "error": 0.10,    # 10%
                "critical": 0.20  # 20%
            },
            MetricType.USER_SATISFACTION: {
                "warning": 0.6,   # Below 60%
                "error": 0.4,     # Below 40%
                "critical": 0.2   # Below 20%
            },
            MetricType.CONVERSATION_QUALITY: {
                "warning": 0.7,   # Below 70%
                "error": 0.5,     # Below 50%
                "critical": 0.3   # Below 30%
            }
        }
    
    def _initialize_baselines(self):
        """Initialize performance baselines for comparison"""
        self.performance_baselines = {
            MetricType.RESPONSE_TIME: 2.0,        # Target 2s response time
            MetricType.CONVERSATION_QUALITY: 0.8, # Target 80% quality
            MetricType.USER_SATISFACTION: 0.85,   # Target 85% satisfaction
            MetricType.INTENT_ACCURACY: 0.9       # Target 90% intent accuracy
        }
    
    def record_metric(self, metric_type: MetricType, value: float, 
                     session_id: Optional[str] = None, **metadata):
        """Record a performance metric"""
        
        metric = PerformanceMetric(
            name=metric_type,
            value=value,
            session_id=session_id,
            metadata=metadata
        )
        
        self.metrics[metric_type].append(metric)
        
        # Session-level tracking
        if session_id:
            if session_id not in self.session_metrics:
                self.session_metrics[session_id] = {}
            self.session_metrics[session_id][metric_type] = metric
        
        # Check for alerting conditions
        self._check_alert_conditions(metric_type, value, metadata)
        
        self.logger.debug(f"Recorded metric {metric_type}: {value}")
    
    def _check_alert_conditions(self, metric_type: str, value: float, metadata: Dict[str, Any]):
        """Check if metric violates any alerting thresholds"""
        
        if metric_type not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_type]
        
        # Check critical first, then error, then warning
        for level in [AlertLevel.CRITICAL, AlertLevel.ERROR, AlertLevel.WARNING]:
            if level in thresholds:
                threshold = thresholds[level]
                
                # For most metrics, lower is better (response time, error rate)
                # But for quality metrics, higher is better
                is_quality_metric = metric_type in [
                    MetricType.USER_SATISFACTION, 
                    MetricType.CONVERSATION_QUALITY,
                    MetricType.INTENT_ACCURACY
                ]
                
                threshold_exceeded = (
                    (is_quality_metric and value < threshold) or
                    (not is_quality_metric and value > threshold)
                )
                
                if threshold_exceeded:
                    alert = SystemAlert(
                        level=level,
                        message=f"{metric_type} threshold exceeded: {value} {'<' if is_quality_metric else '>'} {threshold}",
                        metric_name=metric_type,
                        value=value,
                        threshold=threshold,
                        context=metadata
                    )
                    
                    self.alerts.append(alert)
                    self._trigger_alert_callbacks(alert)
                    
                    self.logger.warning(f"Alert triggered: {alert.message}")
                    break  # Only trigger highest severity alert
    
    def _trigger_alert_callbacks(self, alert: SystemAlert):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def get_metric_summary(self, metric_type: MetricType, 
                          hours_back: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric type"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_metrics = [
            m for m in self.metrics[metric_type]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent metrics found"}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else None,
            "trend": self._calculate_trend(values),
            "baseline_comparison": self._compare_to_baseline(metric_type, values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values"""
        if len(values) < 5:
            return "insufficient_data"
        
        # Compare first and last quarters
        quarter = len(values) // 4
        first_quarter_avg = sum(values[:quarter]) / quarter
        last_quarter_avg = sum(values[-quarter:]) / quarter
        
        diff_percent = ((last_quarter_avg - first_quarter_avg) / first_quarter_avg) * 100
        
        if diff_percent > 5:
            return "improving"
        elif diff_percent < -5:
            return "degrading"
        else:
            return "stable"
    
    def _compare_to_baseline(self, metric_type: str, values: List[float]) -> Dict[str, Any]:
        """Compare current performance to baseline"""
        
        if metric_type not in self.performance_baselines:
            return {"status": "no_baseline"}
        
        baseline = self.performance_baselines[metric_type]
        current_avg = sum(values) / len(values) if values else 0
        
        diff_percent = ((current_avg - baseline) / baseline) * 100
        
        # For quality metrics, positive diff is good. For latency/error metrics, negative diff is good
        is_quality_metric = metric_type in [
            MetricType.USER_SATISFACTION, 
            MetricType.CONVERSATION_QUALITY,
            MetricType.INTENT_ACCURACY
        ]
        
        if is_quality_metric:
            status = "above_baseline" if diff_percent > 0 else "below_baseline"
        else:
            status = "below_baseline" if diff_percent < 0 else "above_baseline"
        
        return {
            "baseline": baseline,
            "current": current_avg,
            "diff_percent": abs(diff_percent),
            "status": status
        }
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific session"""
        
        if session_id not in self.session_metrics:
            return {"error": "Session not found"}
        
        session_data = self.session_metrics[session_id]
        
        analytics = {
            "session_id": session_id,
            "metrics_recorded": len(session_data),
            "performance_summary": {},
            "quality_indicators": {},
            "recommendations": []
        }
        
        # Performance summary
        for metric_type, metric in session_data.items():
            analytics["performance_summary"][metric_type] = {
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "metadata": metric.metadata
            }
        
        # Quality indicators
        if MetricType.CONVERSATION_QUALITY in session_data:
            quality = session_data[MetricType.CONVERSATION_QUALITY].value
            analytics["quality_indicators"]["conversation_quality"] = quality
            
            if quality < 0.6:
                analytics["recommendations"].append("Consider conversation recovery strategies")
        
        if MetricType.RESPONSE_TIME in session_data:
            response_time = session_data[MetricType.RESPONSE_TIME].value
            analytics["quality_indicators"]["response_speed"] = "fast" if response_time < 2 else "slow"
            
            if response_time > 3:
                analytics["recommendations"].append("Response time optimization needed")
        
        return analytics
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",  # Will be updated based on metrics
            "metric_summaries": {},
            "active_alerts": len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)]),
            "session_statistics": self._get_session_statistics(),
            "performance_trends": {},
            "recommendations": []
        }
        
        # Get summaries for all metric types
        health_score = 0
        total_metrics = 0
        
        for metric_type in MetricType:
            if self.metrics[metric_type]:
                summary = self.get_metric_summary(metric_type, hours_back=1)
                report["metric_summaries"][metric_type] = summary
                
                # Calculate health contribution
                baseline_comp = summary.get("baseline_comparison", {})
                if baseline_comp.get("status") == "above_baseline":
                    health_score += 1
                total_metrics += 1
        
        # Determine overall health
        if total_metrics > 0:
            health_ratio = health_score / total_metrics
            if health_ratio > 0.8:
                report["overall_health"] = "excellent"
            elif health_ratio > 0.6:
                report["overall_health"] = "good"  
            elif health_ratio > 0.4:
                report["overall_health"] = "fair"
            else:
                report["overall_health"] = "needs_attention"
        
        # Generate recommendations
        if report["active_alerts"] > 5:
            report["recommendations"].append("High alert volume - investigate system issues")
        
        if report["overall_health"] in ["fair", "needs_attention"]:
            report["recommendations"].append("Performance optimization recommended")
        
        return report
    
    def _get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        
        now = datetime.now()
        active_sessions = 0
        recent_sessions = 0
        
        for session_id, metrics in self.session_metrics.items():
            if not metrics:
                continue
            
            latest_timestamp = max(m.timestamp for m in metrics.values())
            time_diff = (now - latest_timestamp).total_seconds() / 60  # minutes
            
            if time_diff < 30:  # Active within 30 minutes
                active_sessions += 1
            if time_diff < 24 * 60:  # Recent within 24 hours
                recent_sessions += 1
        
        return {
            "total_tracked_sessions": len(self.session_metrics),
            "active_sessions": active_sessions,
            "recent_sessions": recent_sessions
        }


class PerformanceMonitor:
    """Context manager and decorator for performance monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
    
    def time_function(self, metric_name: str = MetricType.RESPONSE_TIME, **metadata):
        """Decorator to time function execution"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    self.collector.record_metric(
                        metric_name, 
                        duration,
                        success=success,
                        error=error,
                        function_name=func.__name__,
                        **metadata
                    )
                return result
            return wrapper
        return decorator
    
    def monitor_conversation_quality(self, session: ConversationSession, 
                                   user_message: str, ai_response: str,
                                   quality_metrics: Dict[str, Any]):
        """Monitor conversation quality metrics"""
        
        # Overall conversation quality score
        engagement = quality_metrics.get('user_engagement', 0.5)
        relevance = quality_metrics.get('response_relevance', 0.5) 
        progress = quality_metrics.get('conversation_progress', 0.5)
        
        overall_quality = (engagement + relevance + progress) / 3
        
        self.collector.record_metric(
            MetricType.CONVERSATION_QUALITY,
            overall_quality,
            session_id=session.session_id,
            conversation_depth=session.context.conversation_depth,
            user_intents=len(session.detected_intents),
            message_length=len(user_message),
            response_length=len(ai_response)
        )
        
        # Individual quality components
        self.collector.record_metric(
            MetricType.USER_SATISFACTION,
            engagement,
            session_id=session.session_id,
            satisfaction_indicators=quality_metrics.get('user_satisfaction_indicators', [])
        )
    
    def monitor_intent_accuracy(self, detected_intents: List[UserIntent], 
                              expected_intents: List[UserIntent],
                              session_id: str):
        """Monitor intent detection accuracy"""
        
        if not expected_intents:
            return  # Can't measure accuracy without ground truth
        
        # Simple accuracy calculation
        correct_detections = len(set(detected_intents).intersection(set(expected_intents)))
        total_expected = len(expected_intents)
        
        accuracy = correct_detections / total_expected if total_expected > 0 else 0
        
        self.collector.record_metric(
            MetricType.INTENT_ACCURACY,
            accuracy,
            session_id=session_id,
            detected_count=len(detected_intents),
            expected_count=len(expected_intents),
            correct_detections=correct_detections
        )


def setup_monitoring() -> tuple[MetricsCollector, PerformanceMonitor]:
    """Setup monitoring system with default configuration"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('travel_assistant.log')
        ]
    )
    
    collector = MetricsCollector()
    monitor = PerformanceMonitor(collector)
    
    # Setup alert callbacks
    def log_alert(alert: SystemAlert):
        logger = logging.getLogger("alerts")
        logger.warning(f"ALERT [{alert.level}]: {alert.message}")
    
    collector.alert_callbacks.append(log_alert)
    
    return collector, monitor


# Global monitoring instances
metrics_collector, performance_monitor = setup_monitoring()