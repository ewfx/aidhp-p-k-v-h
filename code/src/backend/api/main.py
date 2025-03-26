from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import logging
import sys
import os
from src.api.routes import generative

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.config import API_CONFIG, LOGGING_CONFIG
from src.models.recommendation import RecommendationEngine
from src.models.user_embedding import UserEmbeddingModel
from src.services.personalization import PersonalizationService
from src.utils.auth import get_current_user
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Driven Hyper-Personalization API",
    description="API for generating personalized content and recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
recommendation_engine = RecommendationEngine()
user_embedding_model = UserEmbeddingModel()
personalization_service = PersonalizationService(
    recommendation_engine=recommendation_engine,
    user_embedding_model=user_embedding_model
)

# Include routers
app.include_router(
    generative.router,
    prefix="/api/v1/generative",
    tags=["generative"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "AI-Driven Hyper-Personalization API",
        "version": "1.0.0"
    }

@app.get("/api/v1/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    n: int = 10,
    current_user: Dict = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get personalized recommendations for a user"""
    try:
        recommendations = await personalization_service.get_recommendations(
            user_id=user_id,
            n_recommendations=n
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/user-profile/{user_id}")
async def get_user_profile(
    user_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get enriched user profile with preferences and segments"""
    try:
        profile = await personalization_service.get_user_profile(user_id)
        return profile
    except Exception as e:
        logger.error(f"Error getting profile for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/feedback")
async def record_feedback(
    feedback: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, str]:
    """Record user feedback for recommendations"""
    try:
        await personalization_service.record_feedback(feedback)
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/segments")
async def get_user_segments(
    current_user: Dict = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get all available user segments"""
    try:
        segments = await personalization_service.get_segments()
        return segments
    except Exception as e:
        logger.error(f"Error getting user segments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/real-time-personalization")
async def get_real_time_personalization(
    context: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get real-time personalization based on current context"""
    try:
        personalization = await personalization_service.get_real_time_personalization(context)
        return personalization
    except Exception as e:
        logger.error(f"Error getting real-time personalization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/performance")
async def get_recommendation_performance(
    start_date: str,
    end_date: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get performance metrics for the recommendation system"""
    try:
        metrics = await personalization_service.get_performance_metrics(
            start_date=start_date,
            end_date=end_date
        )
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 