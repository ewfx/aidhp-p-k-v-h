from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "organization": os.getenv("OPENAI_ORGANIZATION", None),
    
    # Model configurations
    "models": {
        "text": {
            "chat": "gpt-4-turbo-preview",  # For chat and general text generation
            "embedding": "text-embedding-3-small",  # For embeddings
        },
        "image": {
            "generation": "dall-e-3",  # For image generation
            "editing": "dall-e-2",     # For image editing
        },
        "audio": {
            "transcription": "whisper-1",  # For audio transcription
            "tts": "tts-1",               # For text-to-speech
        }
    },
    
    # Default parameters
    "defaults": {
        "text": {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "image": {
            "quality": "standard",
            "size": "1024x1024",
            "style": "natural",
        },
        "audio": {
            "response_format": "mp3",
            "voice": "alloy",
            "speed": 1.0,
        }
    },
    
    # Rate limiting settings
    "rate_limits": {
        "text": {
            "rpm": 200,     # Requests per minute
            "tpm": 160000,  # Tokens per minute
        },
        "image": {
            "rpm": 50,
        },
        "audio": {
            "rpm": 50,
        }
    }
}

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration"""
    if not OPENAI_CONFIG["api_key"]:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
    return OPENAI_CONFIG 