import openai
from openai import OpenAI
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqGeneration,
    pipeline,
    GPT2LMHeadModel,
    T5ForConditionalGeneration
)
from diffusers import (
    StableDiffusionPipeline,
    AudioLDMPipeline,
    TextToVideoZeroPipeline
)
from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
from PIL import Image
import io
import numpy as np
import base64
from src.config.openai_config import get_openai_config
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class GenerativeAIHub:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Generative AI Hub with OpenAI integration
        """
        self.config = config
        self.openai_config = get_openai_config()
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.openai_config["api_key"],
            organization=self.openai_config["organization"]
        )
        
        # Initialize rate limiters
        self._init_rate_limiters()
        
    def _init_rate_limiters(self):
        """Initialize rate limiters for different API endpoints"""
        self.rate_limits = {
            "text": {
                "last_request": 0,
                "requests_this_minute": 0,
                "tokens_this_minute": 0,
                "minute_start": time.time()
            },
            "image": {
                "last_request": 0,
                "requests_this_minute": 0,
                "minute_start": time.time()
            },
            "audio": {
                "last_request": 0,
                "requests_this_minute": 0,
                "minute_start": time.time()
            }
        }

    async def _check_rate_limit(self, api_type: str, tokens: int = 0) -> None:
        """Check and handle rate limits"""
        current_time = time.time()
        rate_limit = self.rate_limits[api_type]
        
        # Reset counters if a minute has passed
        if current_time - rate_limit["minute_start"] >= 60:
            rate_limit["requests_this_minute"] = 0
            rate_limit["tokens_this_minute"] = 0 if "tokens_this_minute" in rate_limit else 0
            rate_limit["minute_start"] = current_time
        
        # Check rate limits
        if rate_limit["requests_this_minute"] >= self.openai_config["rate_limits"][api_type]["rpm"]:
            wait_time = 60 - (current_time - rate_limit["minute_start"])
            await asyncio.sleep(wait_time)
            return await self._check_rate_limit(api_type, tokens)
        
        if "tpm" in self.openai_config["rate_limits"][api_type] and \
           rate_limit["tokens_this_minute"] + tokens > self.openai_config["rate_limits"][api_type]["tpm"]:
            wait_time = 60 - (current_time - rate_limit["minute_start"])
            await asyncio.sleep(wait_time)
            return await self._check_rate_limit(api_type, tokens)
        
        # Update counters
        rate_limit["requests_this_minute"] += 1
        if "tokens_this_minute" in rate_limit:
            rate_limit["tokens_this_minute"] += tokens
        rate_limit["last_request"] = current_time

    async def generate_personalized_content(
        self,
        user_preferences: Dict[str, Any],
        content_type: str = "text",
        max_length: int = 500
    ) -> Dict[str, Any]:
        """
        Generate personalized content using OpenAI models
        """
        try:
            if content_type == "text":
                return await self._generate_text_content(user_preferences, max_length)
            elif content_type == "image":
                return await self._generate_image_content(user_preferences)
            elif content_type == "audio":
                return await self._generate_audio_content(user_preferences)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        except Exception as e:
            logger.error(f"Error generating {content_type} content: {str(e)}")
            raise

    async def _generate_text_content(
        self,
        user_preferences: Dict[str, Any],
        max_length: int
    ) -> Dict[str, str]:
        """Generate personalized text content using GPT-4"""
        prompt = self._create_content_prompt(user_preferences)
        
        # Estimate tokens and check rate limit
        estimated_tokens = len(prompt.split()) * 1.5
        await self._check_rate_limit("text", int(estimated_tokens))
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.openai_config["models"]["text"]["chat"],
            messages=[
                {"role": "system", "content": "You are a personalized content generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length,
            **self.openai_config["defaults"]["text"]
        )
        
        return {"text": response.choices[0].message.content}

    async def _generate_image_content(
        self,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Image.Image]:
        """Generate personalized image content using DALL-E"""
        prompt = self._create_image_prompt(user_preferences)
        
        # Check rate limit
        await self._check_rate_limit("image")
        
        response = await asyncio.to_thread(
            self.client.images.generate,
            prompt=prompt,
            model=self.openai_config["models"]["image"]["generation"],
            **self.openai_config["defaults"]["image"]
        )
        
        return {"image_url": response.data[0].url}

    async def _generate_audio_content(
        self,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, bytes]:
        """Generate personalized audio content using TTS"""
        text = await self._generate_text_content(user_preferences, 100)
        
        # Check rate limit
        await self._check_rate_limit("audio")
        
        response = await asyncio.to_thread(
            self.client.audio.speech.create,
            input=text["text"],
            model=self.openai_config["models"]["audio"]["tts"],
            **self.openai_config["defaults"]["audio"]
        )
        
        return {"audio": response.content}

    async def generate_explanation(
        self,
        recommendation: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Generate personalized explanation using GPT-4"""
        prompt = self._create_explanation_prompt(recommendation, user_context)
        
        # Estimate tokens and check rate limit
        estimated_tokens = len(prompt.split()) * 1.5
        await self._check_rate_limit("text", int(estimated_tokens))
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.openai_config["models"]["text"]["chat"],
            messages=[
                {"role": "system", "content": "You are an expert at explaining recommendations."},
                {"role": "user", "content": prompt}
            ],
            **self.openai_config["defaults"]["text"]
        )
        
        return response.choices[0].message.content

    async def generate_response(
        self,
        user_query: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate personalized response using GPT-4"""
        prompt = self._create_response_prompt(user_query, context)
        
        # Estimate tokens and check rate limit
        estimated_tokens = len(prompt.split()) * 1.5
        await self._check_rate_limit("text", int(estimated_tokens))
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.openai_config["models"]["text"]["chat"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing personalized responses."},
                {"role": "user", "content": prompt}
            ],
            **self.openai_config["defaults"]["text"]
        )
        
        return response.choices[0].message.content

    def _create_content_prompt(self, user_preferences: Dict[str, Any]) -> str:
        """Create detailed prompt for content generation"""
        interests = user_preferences.get("interests", [])
        style = user_preferences.get("style", "neutral")
        tone = user_preferences.get("tone", "professional")
        
        return (
            f"Generate engaging content about {', '.join(interests)} "
            f"in a {style} style with a {tone} tone. "
            f"Consider the user's preferences and make it personalized."
        )

    def _create_image_prompt(self, user_preferences: Dict[str, Any]) -> str:
        """Create detailed prompt for image generation"""
        style = user_preferences.get("visual_style", "realistic")
        subject = user_preferences.get("subject", "landscape")
        mood = user_preferences.get("mood", "natural")
        
        return (
            f"A {style} {subject} with {mood} mood, "
            f"high quality, detailed, professional photography, "
            f"4K, high resolution"
        )

    def _create_explanation_prompt(
        self,
        recommendation: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Create detailed prompt for explanation generation"""
        return (
            f"Given a user who {user_context.get('user_description', '')} "
            f"and has interests in {', '.join(user_context.get('interests', []))}, "
            f"explain why {recommendation['item']} was recommended. "
            f"Consider their preferences for {user_context.get('style', 'neutral')} content "
            f"and recent interactions with {', '.join(user_context.get('recent_interactions', []))}."
        )

    def _create_response_prompt(self, user_query: str, context: Dict[str, Any]) -> str:
        """Create detailed prompt for response generation"""
        return (
            f"Given a user who {context.get('user_description', '')} "
            f"with recent interactions: {', '.join(context.get('recent_interactions', []))}, "
            f"provide a personalized response to their query: {user_query}"
        ) 