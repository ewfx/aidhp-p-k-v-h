from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Storage configurations
STORAGE_CONFIG = {
    "DATA_DIR": os.path.join(BASE_DIR, "data"),
    "BACKUP_DIR": os.path.join(BASE_DIR, "data", "backups")
}

# API configurations
API_CONFIG = {
    "VERSION": "1.0.0",
    "TITLE": "Hyper-Personalization API",
    "PREFIX": "/api/v1",
    "CORS_ORIGINS": ["*"],  # Modify in production
}

# Model configurations
MODEL_CONFIG = {
    "embedding_dim": 128,
    "n_factors": 100,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
}

# Feature engineering configurations
FEATURE_CONFIG = {
    "text_features": [
        "product_description",
        "review_text",
        "social_media_posts"
    ],
    "categorical_features": [
        "category",
        "brand",
        "location",
        "device_type"
    ],
    "numerical_features": [
        "price",
        "rating",
        "age",
        "purchase_frequency"
    ]
}

# Social media API configurations
SOCIAL_MEDIA_CONFIG = {
    "twitter": {
        "api_key": os.getenv("TWITTER_API_KEY"),
        "api_secret": os.getenv("TWITTER_API_SECRET"),
        "access_token": os.getenv("TWITTER_ACCESS_TOKEN"),
        "access_token_secret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    },
    "facebook": {
        "app_id": os.getenv("FACEBOOK_APP_ID"),
        "app_secret": os.getenv("FACEBOOK_APP_SECRET"),
        "access_token": os.getenv("FACEBOOK_ACCESS_TOKEN")
    }
}

# Recommendation system configurations
RECOMMENDATION_CONFIG = {
    "min_interactions": 5,
    "n_recommendations": 10,
    "similarity_metric": "cosine",
    "cold_start_strategy": "content_based"
}

# Monitoring configurations
MONITORING_CONFIG = {
    "mlflow": {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "experiment_name": "hyper_personalization"
    },
    "wandb": {
        "project": "hyper_personalization",
        "entity": os.getenv("WANDB_ENTITY")
    }
}

# Cache configurations
CACHE_CONFIG = {
    "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "TTL": 3600  # Time to live in seconds
}

# Security configurations
SECURITY_CONFIG = {
    "JWT_SECRET": os.getenv("JWT_SECRET", "your-secret-key"),
    "JWT_ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 30
}

# Logging configurations
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True
        },
    }
} 