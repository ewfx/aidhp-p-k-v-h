from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

router = APIRouter()

class UserProfile(BaseModel):
    age: int
    income: float
    riskTolerance: str
    investmentGoals: List[str]
    timeHorizon: str
    interests: List[str]

class ContentRequest(BaseModel):
    accountId: str
    userProfile: UserProfile

class ContentItem(BaseModel):
    id: str
    type: str
    title: str
    description: str
    category: str
    tags: List[str]
    priority: int
    relevanceScore: float
    content: Dict[str, Any]

class PersonalizedContentResponse(BaseModel):
    recommendations: List[ContentItem]
    categories: List[str]
    summary: str

class UserPreferences(BaseModel):
    theme: str
    layout: str
    accessibility: str
    features: List[str]

class UIRequest(BaseModel):
    accountId: str
    userPreferences: UserPreferences

class UIElement(BaseModel):
    id: str
    type: str
    position: str
    visibility: bool
    properties: Dict[str, Any]
    customization: Dict[str, Any]

class DynamicUIResponse(BaseModel):
    elements: List[UIElement]
    theme: Dict[str, Any]
    layout: Dict[str, Any]
    accessibility: Dict[str, Any]

@router.post("/content", response_model=PersonalizedContentResponse)
async def get_personalized_content(request: ContentRequest):
    # Generate personalized content based on user profile
    recommendations = []
    
    # Investment education content for retirement goal
    if "retirement" in request.userProfile.investmentGoals:
        recommendations.append(
            ContentItem(
                id="CONTENT-1",
                type="article",
                title="Retirement Planning Essentials",
                description="Learn the key strategies for successful retirement planning",
                category="retirement",
                tags=["retirement", "investment", "planning"],
                priority=1,
                relevanceScore=0.95,
                content={
                    "body": "Comprehensive guide to retirement planning...",
                    "readTime": "10 minutes",
                    "difficulty": "intermediate",
                    "actionItems": [
                        "Calculate retirement needs",
                        "Review investment portfolio",
                        "Consider tax implications"
                    ]
                }
            )
        )

    # Real estate investment content
    if "real_estate" in request.userProfile.interests:
        recommendations.append(
            ContentItem(
                id="CONTENT-2",
                type="video",
                title="Real Estate Investment Strategies",
                description="Expert insights on real estate investment opportunities",
                category="real_estate",
                tags=["real_estate", "investment", "market_analysis"],
                priority=2,
                relevanceScore=0.88,
                content={
                    "videoUrl": "https://example.com/videos/real-estate-investing",
                    "duration": "15:00",
                    "highlights": [
                        "Market trends analysis",
                        "Investment property selection",
                        "Risk management strategies"
                    ]
                }
            )
        )

    # Technology sector insights
    if "technology" in request.userProfile.interests:
        recommendations.append(
            ContentItem(
                id="CONTENT-3",
                type="market_analysis",
                title="Tech Sector Investment Outlook",
                description="Analysis of technology sector investment opportunities",
                category="technology",
                tags=["technology", "market_analysis", "trends"],
                priority=3,
                relevanceScore=0.85,
                content={
                    "sectors": ["AI", "Cloud Computing", "Cybersecurity"],
                    "marketTrends": ["Growing AI adoption", "Digital transformation"],
                    "riskFactors": ["Market volatility", "Regulatory changes"]
                }
            )
        )

    return PersonalizedContentResponse(
        recommendations=recommendations,
        categories=list(set(item.category for item in recommendations)),
        summary=f"Personalized content based on your interests in {', '.join(request.userProfile.interests)}"
    )

@router.post("/ui-elements", response_model=DynamicUIResponse)
async def get_dynamic_ui_elements(request: UIRequest):
    # Generate dynamic UI elements based on user preferences
    elements = []
    
    # Dashboard components
    if "spending_insights" in request.userPreferences.features:
        elements.append(
            UIElement(
                id="UI-1",
                type="widget",
                position="main",
                visibility=True,
                properties={
                    "name": "spending_insights",
                    "refreshInterval": 300,
                    "dataSource": "spending_analytics"
                },
                customization={
                    "expanded": True,
                    "showChart": True,
                    "chartType": "line"
                }
            )
        )

    # Investment recommendations widget
    if "investment_recommendations" in request.userPreferences.features:
        elements.append(
            UIElement(
                id="UI-2",
                type="widget",
                position="sidebar",
                visibility=True,
                properties={
                    "name": "investment_recommendations",
                    "refreshInterval": 3600,
                    "dataSource": "investment_analytics"
                },
                customization={
                    "expanded": False,
                    "showNotifications": True
                }
            )
        )

    # Theme configuration
    theme_config = {
        "dark": {
            "background": "#121212",
            "primary": "#BB86FC",
            "secondary": "#03DAC6",
            "text": "#FFFFFF"
        },
        "light": {
            "background": "#FFFFFF",
            "primary": "#6200EE",
            "secondary": "#03DAC6",
            "text": "#000000"
        }
    }

    # Layout configuration
    layout_config = {
        "compact": {
            "spacing": "8px",
            "maxWidth": "1200px",
            "columnCount": 2
        },
        "comfortable": {
            "spacing": "16px",
            "maxWidth": "1400px",
            "columnCount": 3
        }
    }

    # Accessibility configuration
    accessibility_config = {
        "high_contrast": {
            "contrastRatio": 7,
            "fontSize": "16px",
            "fontWeight": "bold"
        },
        "normal": {
            "contrastRatio": 4.5,
            "fontSize": "14px",
            "fontWeight": "normal"
        }
    }

    return DynamicUIResponse(
        elements=elements,
        theme=theme_config[request.userPreferences.theme],
        layout=layout_config[request.userPreferences.layout],
        accessibility=accessibility_config[request.userPreferences.accessibility]
    ) 