from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, List, Any, Optional
from src.services.generative_service import GenerativeService
from src.utils.auth import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/content/generate")
async def generate_content(
    user_id: str = Depends(get_current_user),
    preferences: Dict[str, Any] = Body(...),
    content_types: List[str] = Body(default=["text"])
):
    """Generate personalized content across multiple modalities"""
    try:
        service = GenerativeService({})  # Initialize with config
        result = await service.generate_personalized_content(
            user_id,
            preferences,
            content_types
        )
        return {"status": "success", "content": result}
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/explain")
async def explain_recommendation(
    user_id: str = Depends(get_current_user),
    recommendation: Dict[str, Any] = Body(...),
    user_context: Dict[str, Any] = Body(...)
):
    """Generate personalized explanation for a recommendation"""
    try:
        service = GenerativeService({})
        explanation = await service.generate_recommendation_explanation(
            user_id,
            recommendation,
            user_context
        )
        return {"status": "success", "explanation": explanation}
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/response")
async def generate_response(
    user_id: str = Depends(get_current_user),
    query: str = Body(...),
    context: Dict[str, Any] = Body(...)
):
    """Generate personalized response to user query"""
    try:
        service = GenerativeService({})
        response = await service.generate_interactive_response(
            user_id,
            query,
            context
        )
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ui/elements")
async def generate_ui_elements(
    user_id: str = Depends(get_current_user),
    preferences: Dict[str, Any] = Body(...),
    element_type: str = Body(...)
):
    """Generate dynamic UI elements based on user preferences"""
    try:
        service = GenerativeService({})
        elements = await service.generate_dynamic_ui_elements(
            user_id,
            preferences,
            element_type
        )
        return {"status": "success", "elements": elements}
    except Exception as e:
        logger.error(f"Error generating UI elements: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 