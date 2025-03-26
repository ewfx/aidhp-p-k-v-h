from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqGeneration,
    pipeline,
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection
)
from diffusers import (
    StableDiffusionXLPipeline,
    AudioLDMPipeline,
    ConsistencyDecoderVAE
)
import logging
import asyncio
from PIL import Image
import io
import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime
from src.config.openai_config import get_openai_config
from openai import OpenAI
import time
from cryptography.fernet import Fernet
import pyotp
import qrcode
from plaid import Client as PlaidClient
import stripe

logger = logging.getLogger(__name__)

class ModelProvider:
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    HYBRID = "hybrid"

class BankingSecurity:
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.totp_secret = pyotp.random_base32()

    def encrypt_sensitive_data(self, data: str) -> str:
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

    def generate_2fa_qr(self, user_id: str) -> str:
        totp = pyotp.TOTP(self.totp_secret)
        provisioning_uri = totp.provisioning_uri(user_id)
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        return qr.make_image(fill_color="black", back_color="white")

    def verify_2fa(self, token: str) -> bool:
        totp = pyotp.TOTP(self.totp_secret)
        return totp.verify(token)

class HybridGenerativeHub:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Hybrid Generative Hub with both OpenAI and Hugging Face models
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_provider = config.get("model_provider", ModelProvider.HYBRID)
        
        # Initialize security
        self.security = BankingSecurity()
        
        # Initialize OpenAI
        self._init_openai()
        
        # Initialize Hugging Face models
        self._init_hf_models()
        
        # Initialize rate limiters
        self._init_rate_limiters()
        
        # Initialize model weights for hybrid approach
        self._init_hybrid_weights()
        
        # Initialize banking clients
        self._init_banking_clients()

    def _init_banking_clients(self):
        """Initialize banking-related clients"""
        try:
            # Initialize Plaid client
            self.plaid_client = PlaidClient(
                client_id=os.getenv("PLAID_CLIENT_ID"),
                secret=os.getenv("PLAID_SECRET"),
                environment=os.getenv("PLAID_ENV", "sandbox")
            )
            
            # Initialize Stripe client
            stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
            
        except Exception as e:
            logger.error(f"Error initializing banking clients: {str(e)}")
            raise

    async def generate_banking_content(
        self,
        prompt: str,
        content_type: str,
        user_preferences: Dict[str, Any],
        sensitive_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate banking-specific content with security measures
        """
        # Encrypt sensitive data if present
        if sensitive_data:
            encrypted_data = {
                k: self.security.encrypt_sensitive_data(v)
                for k, v in sensitive_data.items()
            }
        else:
            encrypted_data = None

        # Generate content using appropriate model
        content = await self.generate_content(
            prompt=prompt,
            content_type=content_type,
            user_preferences=user_preferences
        )

        # Add security metadata
        content["security_metadata"] = {
            "encrypted": bool(encrypted_data),
            "timestamp": datetime.utcnow().isoformat(),
            "content_hash": self.security.encrypt_sensitive_data(str(content))
        }

        return content

    async def generate_transaction_summary(
        self,
        transactions: List[Dict[str, Any]],
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized transaction summaries
        """
        prompt = self._create_transaction_prompt(transactions, user_preferences)
        return await self.generate_banking_content(
            prompt=prompt,
            content_type="text",
            user_preferences=user_preferences,
            sensitive_data={"transactions": json.dumps(transactions)}
        )

    async def generate_financial_insights(
        self,
        account_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized financial insights
        """
        prompt = self._create_insights_prompt(account_data, user_preferences)
        return await self.generate_banking_content(
            prompt=prompt,
            content_type="text",
            user_preferences=user_preferences,
            sensitive_data={"account_data": json.dumps(account_data)}
        )

    def _create_transaction_prompt(
        self,
        transactions: List[Dict[str, Any]],
        user_preferences: Dict[str, Any]
    ) -> str:
        """Create prompt for transaction summary generation"""
        return (
            f"Generate a personalized summary of {len(transactions)} transactions "
            f"for a user with preferences: {json.dumps(user_preferences)}. "
            f"Focus on spending patterns, categories, and notable changes."
        )

    def _create_insights_prompt(
        self,
        account_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> str:
        """Create prompt for financial insights generation"""
        return (
            f"Generate personalized financial insights based on account data "
            f"for a user with preferences: {json.dumps(user_preferences)}. "
            f"Focus on spending trends, savings opportunities, and financial goals."
        )

    def _init_openai(self):
        """Initialize OpenAI client and config"""
        try:
            self.openai_config = get_openai_config()
            self.openai_client = OpenAI(
                api_key=self.openai_config["api_key"],
                organization=self.openai_config["organization"]
            )
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {str(e)}")
            self.openai_client = None

    def _init_hf_models(self):
        """Initialize Hugging Face models"""
        try:
            # Quantization config for efficient memory usage
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            # Text generation model (Mixtral-8x7B)
            self.text_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                device_map="auto",
                quantization_config=quantization_config
            )
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mixtral-8x7B-Instruct-v0.1"
            )

            # Text-to-text model (FLAN-T5)
            self.t2t_model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-xl",
                device_map="auto",
                quantization_config=quantization_config
            )
            self.t2t_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

            # Image generation (Stable Diffusion XL)
            self.image_generator = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16",
                use_safetensors=True
            ).to(self.device)

            # Audio generation (AudioLDM)
            self.audio_generator = AudioLDMPipeline.from_pretrained(
                "cvssp/audioldm-s-full-v2",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            # CLIP for understanding images and text
            self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

        except Exception as e:
            logger.error(f"Error initializing Hugging Face models: {str(e)}")
            raise

    def _init_rate_limiters(self):
        """Initialize rate limiters for API calls"""
        self.rate_limits = {
            "openai": {
                "text": {"last_call": 0, "calls_per_minute": 0, "start_time": time.time()},
                "image": {"last_call": 0, "calls_per_minute": 0, "start_time": time.time()},
                "audio": {"last_call": 0, "calls_per_minute": 0, "start_time": time.time()}
            },
            "huggingface": {
                "text": {"last_call": 0, "calls_per_minute": 0, "start_time": time.time()},
                "image": {"last_call": 0, "calls_per_minute": 0, "start_time": time.time()},
                "audio": {"last_call": 0, "calls_per_minute": 0, "start_time": time.time()}
            }
        }

    def _init_hybrid_weights(self):
        """Initialize weights for hybrid model selection"""
        self.model_weights = {
            "text": {
                "openai": 0.7,  # Higher weight for OpenAI due to better quality
                "huggingface": 0.3
            },
            "image": {
                "openai": 0.6,
                "huggingface": 0.4
            },
            "audio": {
                "openai": 0.5,
                "huggingface": 0.5
            }
        }

    async def generate_content(
        self,
        prompt: str,
        content_type: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate content using the appropriate model based on configuration
        """
        if self.model_provider == ModelProvider.OPENAI and self.openai_client:
            return await self._generate_openai_content(prompt, content_type, user_preferences)
        elif self.model_provider == ModelProvider.HUGGINGFACE:
            return await self._generate_hf_content(prompt, content_type, user_preferences)
        else:
            return await self._generate_hybrid_content(prompt, content_type, user_preferences)

    async def _generate_openai_content(
        self,
        prompt: str,
        content_type: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content using OpenAI models"""
        try:
            if content_type == "text":
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=self.openai_config["models"]["text"]["chat"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    **self.openai_config["defaults"]["text"]
                )
                return {"text": response.choices[0].message.content}
            
            elif content_type == "image":
                response = await asyncio.to_thread(
                    self.openai_client.images.generate,
                    prompt=prompt,
                    **self.openai_config["defaults"]["image"]
                )
                return {"image_url": response.data[0].url}
            
            elif content_type == "audio":
                response = await asyncio.to_thread(
                    self.openai_client.audio.speech.create,
                    input=prompt,
                    **self.openai_config["defaults"]["audio"]
                )
                return {"audio": response.content}
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            raise

    async def _generate_hf_content(
        self,
        prompt: str,
        content_type: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content using Hugging Face models"""
        try:
            if content_type == "text":
                inputs = self.text_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.text_model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )
                
                return {
                    "text": self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
                }
            
            elif content_type == "image":
                image = self.image_generator(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]
                
                return {"image": image}
            
            elif content_type == "audio":
                audio = self.audio_generator(
                    prompt,
                    num_inference_steps=50,
                    audio_length_in_s=5.0
                ).audios[0]
                
                return {"audio": audio}
            
        except Exception as e:
            logger.error(f"Hugging Face generation error: {str(e)}")
            raise

    async def _generate_hybrid_content(
        self,
        prompt: str,
        content_type: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content using both OpenAI and Hugging Face models"""
        results = {}
        weights = self.model_weights[content_type]
        
        # Generate content from both sources
        if self.openai_client:
            try:
                openai_result = await self._generate_openai_content(
                    prompt, content_type, user_preferences
                )
                results["openai"] = openai_result
            except Exception as e:
                logger.error(f"OpenAI generation failed: {str(e)}")
                weights["huggingface"] = 1.0
                weights["openai"] = 0.0

        try:
            hf_result = await self._generate_hf_content(
                prompt, content_type, user_preferences
            )
            results["huggingface"] = hf_result
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {str(e)}")
            if "openai" in results:
                weights["openai"] = 1.0
                weights["huggingface"] = 0.0
            else:
                raise

        # Combine results based on weights and content type
        if content_type == "text":
            return await self._combine_text_results(results, weights)
        elif content_type == "image":
            return await self._combine_image_results(results, weights)
        elif content_type == "audio":
            return await self._combine_audio_results(results, weights)

    async def _combine_text_results(
        self,
        results: Dict[str, Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, str]:
        """Combine text results from different models"""
        combined_text = ""
        
        if "openai" in results and weights["openai"] > 0:
            combined_text += results["openai"]["text"]
        
        if "huggingface" in results and weights["huggingface"] > 0:
            if combined_text:
                combined_text += "\n\n"
            combined_text += results["huggingface"]["text"]
        
        return {"text": combined_text}

    async def _combine_image_results(
        self,
        results: Dict[str, Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Union[str, Image.Image]]:
        """Return the best image based on weights"""
        if weights["openai"] > weights["huggingface"] and "openai" in results:
            return {"image_url": results["openai"]["image_url"]}
        elif "huggingface" in results:
            return {"image": results["huggingface"]["image"]}
        return results["openai" if "openai" in results else "huggingface"]

    async def _combine_audio_results(
        self,
        results: Dict[str, Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, bytes]:
        """Return the best audio based on weights"""
        if weights["openai"] > weights["huggingface"] and "openai" in results:
            return {"audio": results["openai"]["audio"]}
        elif "huggingface" in results:
            return {"audio": results["huggingface"]["audio"]}
        return results["openai" if "openai" in results else "huggingface"]

    async def update_model_weights(
        self,
        feedback: Dict[str, Any]
    ) -> None:
        """Update model weights based on feedback"""
        content_type = feedback.get("content_type")
        provider = feedback.get("provider")
        score = feedback.get("score", 0.0)
        
        if content_type in self.model_weights and provider in self.model_weights[content_type]:
            current_weight = self.model_weights[content_type][provider]
            other_provider = "huggingface" if provider == "openai" else "openai"
            
            # Update weights using exponential moving average
            alpha = 0.1  # Learning rate
            new_weight = current_weight * (1 - alpha) + score * alpha
            
            # Normalize weights
            self.model_weights[content_type][provider] = new_weight
            self.model_weights[content_type][other_provider] = 1.0 - new_weight 