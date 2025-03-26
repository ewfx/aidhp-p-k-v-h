from typing import Dict, List, Any, Optional, Union
import logging
from src.models.generative.hybrid_hub import HybridGenerativeHub, ModelProvider
from PIL import Image
import numpy as np
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class GenerativeService:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Generative Service"""
        self.config = config
        self.gen_hub = HybridGenerativeHub(config)
        self.content_cache = {}
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour default

    async def generate_personalized_content(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        content_types: List[str] = ["text"],
        model_provider: str = ModelProvider.HYBRID
    ) -> Dict[str, Any]:
        """
        Generate personalized content across multiple modalities
        """
        try:
            results = {}
            tasks = []

            for content_type in content_types:
                cache_key = f"{user_id}_{content_type}_{json.dumps(preferences, sort_keys=True)}_{model_provider}"
                cached_content = self._get_from_cache(cache_key)
                
                if cached_content:
                    results[content_type] = cached_content
                else:
                    # Create prompt based on content type and preferences
                    prompt = self._create_prompt(content_type, preferences)
                    
                    task = self.gen_hub.generate_content(
                        prompt=prompt,
                        content_type=content_type,
                        user_preferences=preferences
                    )
                    tasks.append((content_type, task))

            # Execute remaining tasks concurrently
            if tasks:
                completed_tasks = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )

                for i, (content_type, _) in enumerate(tasks):
                    if isinstance(completed_tasks[i], Exception):
                        logger.error(f"Error generating {content_type}: {str(completed_tasks[i])}")
                        continue
                        
                    results[content_type] = completed_tasks[i]
                    self._add_to_cache(
                        f"{user_id}_{content_type}_{json.dumps(preferences, sort_keys=True)}_{model_provider}",
                        completed_tasks[i]
                    )

            return results
        except Exception as e:
            logger.error(f"Error in generate_personalized_content: {str(e)}")
            raise

    async def generate_recommendation_explanation(
        self,
        user_id: str,
        recommendation: Dict[str, Any],
        user_context: Dict[str, Any],
        model_provider: str = ModelProvider.HYBRID
    ) -> str:
        """
        Generate personalized explanation for a recommendation
        """
        try:
            cache_key = (
                f"explanation_{user_id}_"
                f"{recommendation['item']}_{json.dumps(user_context, sort_keys=True)}_{model_provider}"
            )
            
            cached_explanation = self._get_from_cache(cache_key)
            if cached_explanation:
                return cached_explanation

            prompt = self._create_explanation_prompt(recommendation, user_context)
            
            result = await self.gen_hub.generate_content(
                prompt=prompt,
                content_type="text",
                user_preferences=user_context
            )
            
            explanation = result["text"]
            self._add_to_cache(cache_key, explanation)
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise

    async def generate_interactive_response(
        self,
        user_id: str,
        query: str,
        context: Dict[str, Any],
        model_provider: str = ModelProvider.HYBRID
    ) -> str:
        """
        Generate personalized response to user query
        """
        try:
            cache_key = f"response_{user_id}_{query}_{json.dumps(context, sort_keys=True)}_{model_provider}"
            
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response

            prompt = self._create_response_prompt(query, context)
            
            result = await self.gen_hub.generate_content(
                prompt=prompt,
                content_type="text",
                user_preferences=context
            )
            
            response = result["text"]
            self._add_to_cache(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def generate_dynamic_ui_elements(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        element_type: str,
        model_provider: str = ModelProvider.HYBRID
    ) -> Dict[str, Any]:
        """
        Generate dynamic UI elements based on user preferences
        """
        try:
            if element_type == "visual":
                return await self._generate_visual_elements(user_id, preferences, model_provider)
            elif element_type == "textual":
                return await self._generate_textual_elements(user_id, preferences, model_provider)
            else:
                raise ValueError(f"Unsupported UI element type: {element_type}")
        except Exception as e:
            logger.error(f"Error generating UI elements: {str(e)}")
            raise

    async def _generate_visual_elements(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        model_provider: str
    ) -> Dict[str, Any]:
        """Generate visual UI elements"""
        style_preferences = {
            "visual_style": preferences.get("ui_style", "modern"),
            "subject": "interface element"
        }
        
        prompt = self._create_prompt("image", style_preferences)
        
        result = await self.gen_hub.generate_content(
            prompt=prompt,
            content_type="image",
            user_preferences=style_preferences
        )
        
        return {"visual_elements": result}

    async def _generate_textual_elements(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        model_provider: str
    ) -> Dict[str, Any]:
        """Generate textual UI elements"""
        style_preferences = {
            "style": preferences.get("text_style", "professional"),
            "interests": preferences.get("interests", [])
        }
        
        prompt = self._create_prompt("text", style_preferences)
        
        result = await self.gen_hub.generate_content(
            prompt=prompt,
            content_type="text",
            user_preferences=style_preferences
        )
        
        return {"textual_elements": result}

    async def provide_feedback(
        self,
        feedback: Dict[str, Any]
    ) -> None:
        """
        Provide feedback to update model weights
        """
        await self.gen_hub.update_model_weights(feedback)

    def _create_prompt(self, content_type: str, preferences: Dict[str, Any]) -> str:
        """Create appropriate prompt based on content type and preferences"""
        if content_type == "text":
            interests = preferences.get("interests", [])
            style = preferences.get("style", "neutral")
            tone = preferences.get("tone", "professional")
            return (
                f"Generate engaging content about {', '.join(interests)} "
                f"in a {style} style with a {tone} tone. "
                f"Consider the user's preferences and make it personalized."
            )
        
        elif content_type == "image":
            style = preferences.get("visual_style", "realistic")
            subject = preferences.get("subject", "landscape")
            mood = preferences.get("mood", "natural")
            return (
                f"A {style} {subject} with {mood} mood, "
                f"high quality, detailed, professional photography, "
                f"4K, high resolution"
            )
        
        elif content_type == "audio":
            text = preferences.get("text", "")
            voice = preferences.get("voice", "natural")
            emotion = preferences.get("emotion", "neutral")
            return f"Generate {voice} voice audio with {emotion} emotion: {text}"
        
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def _create_explanation_prompt(
        self,
        recommendation: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Create prompt for explanation generation"""
        return (
            f"Given a user who {user_context.get('user_description', '')} "
            f"and has interests in {', '.join(user_context.get('interests', []))}, "
            f"explain why {recommendation['item']} was recommended. "
            f"Consider their preferences for {user_context.get('style', 'neutral')} content "
            f"and recent interactions with {', '.join(user_context.get('recent_interactions', []))}."
        )

    def _create_response_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Create prompt for response generation"""
        return (
            f"Given a user who {context.get('user_description', '')} "
            f"with recent interactions: {', '.join(context.get('recent_interactions', []))}, "
            f"provide a personalized response to their query: {query}"
        )

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        if key in self.content_cache:
            content, timestamp = self.content_cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return content
            else:
                del self.content_cache[key]
        return None

    def _add_to_cache(self, key: str, content: Any) -> None:
        """Add item to cache with current timestamp"""
        self.content_cache[key] = (content, datetime.now()) 