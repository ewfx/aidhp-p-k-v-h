from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from services.genai_service import genai_service

router = APIRouter()

class UserPreferences(BaseModel):
    user_id: str
    age: int
    income: float
    risk_tolerance: str
    investment_goals: List[str]

class PersonalizedContent(BaseModel):
    content: str
    recommendations: List[str]
    insights: Dict[str, str]

class UIElements(BaseModel):
    layout: Dict[str, str]
    components: List[Dict[str, str]]
    theme: Dict[str, str]

@router.post("/content", response_model=PersonalizedContent)
async def get_personalized_content(request: UserPreferences):
    try:
        # Generate personalized content using Gen AI
        content = await genai_service.generate_personalized_content(request.dict())
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ui-elements", response_model=UIElements)
async def get_dynamic_ui_elements(request: UserPreferences):
    try:
        # Generate dynamic UI elements using Gen AI
        ui_config = await genai_service.generate_dynamic_ui(request.dict())
        return ui_config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 