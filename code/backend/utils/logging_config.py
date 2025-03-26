import logging
import logging.config
from typing import Dict, Any
import json
from datetime import datetime
import os
from pathlib import Path

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Add file handler to configuration
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": f"logs/app_{datetime.now().strftime('%Y%m%d')}.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
        
        # Add file handler to root logger
        config["loggers"][""]["handlers"].append("file")
        
        # Apply configuration
        logging.config.dictConfig(config)
        
    except Exception as e:
        print(f"Error setting up logging configuration: {str(e)}")
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

def log_event(
    logger: logging.Logger,
    event_type: str,
    event_data: Dict[str, Any],
    level: int = logging.INFO
) -> None:
    """Log an event with structured data"""
    try:
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": event_data
        }
        
        logger.log(level, json.dumps(event))
        
    except Exception as e:
        logger.error(f"Error logging event: {str(e)}")

class EventLogger:
    """Event logger for tracking system events"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        
    def log_recommendation(
        self,
        user_id: str,
        recommendations: list,
        context: Dict[str, Any]
    ) -> None:
        """Log recommendation event"""
        event_data = {
            "user_id": user_id,
            "recommendations": recommendations,
            "context": context
        }
        log_event(self.logger, "recommendation", event_data)
        
    def log_user_interaction(
        self,
        user_id: str,
        interaction_type: str,
        item_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Log user interaction event"""
        event_data = {
            "user_id": user_id,
            "interaction_type": interaction_type,
            "item_id": item_id,
            "metadata": metadata
        }
        log_event(self.logger, "user_interaction", event_data)
        
    def log_model_update(
        self,
        model_id: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> None:
        """Log model update event"""
        event_data = {
            "model_id": model_id,
            "metrics": metrics,
            "metadata": metadata
        }
        log_event(self.logger, "model_update", event_data)
        
    def log_error(
        self,
        error_type: str,
        error_message: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Log error event"""
        event_data = {
            "error_type": error_type,
            "error_message": error_message,
            "metadata": metadata
        }
        log_event(self.logger, "error", event_data, level=logging.ERROR)
        
    def log_performance_metrics(
        self,
        metrics: Dict[str, Any],
        period: Dict[str, str]
    ) -> None:
        """Log performance metrics"""
        event_data = {
            "metrics": metrics,
            "period": period
        }
        log_event(self.logger, "performance_metrics", event_data)
        
    def log_system_health(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Log system health metrics"""
        event_data = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        log_event(self.logger, "system_health", event_data)

class MetricsLogger:
    """Logger for tracking model and system metrics"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self.metrics_history = {}
        
    def log_metric(
        self,
        metric_name: str,
        value: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log a single metric"""
        event_data = {
            "metric_name": metric_name,
            "value": value,
            "metadata": metadata or {}
        }
        
        # Store in history
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        self.metrics_history[metric_name].append(
            (datetime.now(), value, metadata)
        )
        
        log_event(self.logger, "metric", event_data)
        
    def get_metric_history(
        self,
        metric_name: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> list:
        """Get historical values for a metric"""
        history = self.metrics_history.get(metric_name, [])
        
        if start_time:
            history = [h for h in history if h[0] >= start_time]
        if end_time:
            history = [h for h in history if h[0] <= end_time]
            
        return history
        
    def calculate_metric_statistics(
        self,
        metric_name: str,
        window: int = None
    ) -> Dict[str, float]:
        """Calculate statistics for a metric"""
        history = self.metrics_history.get(metric_name, [])
        
        if window:
            cutoff = datetime.now() - window
            history = [h for h in history if h[0] >= cutoff]
            
        if not history:
            return {}
            
        values = [h[1] for h in history]
        
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        } 