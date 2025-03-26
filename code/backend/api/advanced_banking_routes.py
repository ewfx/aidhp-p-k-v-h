from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.services.advanced_banking_service import AdvancedBankingService
from src.utils.auth import get_current_user, check_permissions
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/banking/advanced", tags=["advanced_banking"])

# Initialize service
advanced_banking_service = AdvancedBankingService({})  # Config will be loaded from environment

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class FraudDetectionRequest(BaseModel):
    account_id: str
    transaction_data: Dict[str, Any]

class LoanRecommendationsRequest(BaseModel):
    account_id: str
    loan_purpose: str
    user_preferences: Dict[str, Any]

class SpendingPatternsRequest(BaseModel):
    account_id: str
    time_period: str = "90d"

@router.post("/fraud/detect")
async def detect_fraud(
    request: FraudDetectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Detect potential fraudulent transactions
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["read:fraud"])
        
        # Detect fraud
        fraud_analysis = await advanced_banking_service.detect_fraud(
            user_id=current_user["id"],
            account_id=request.account_id,
            transaction_data=request.transaction_data
        )
        
        return fraud_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/loans/recommendations")
async def get_loan_recommendations(
    request: LoanRecommendationsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Get personalized loan recommendations
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["read:loans"])
        
        # Generate loan recommendations
        recommendations = await advanced_banking_service.generate_loan_recommendations(
            user_id=current_user["id"],
            account_id=request.account_id,
            loan_purpose=request.loan_purpose,
            user_preferences=request.user_preferences
        )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/patterns/analyze")
async def analyze_spending_patterns(
    request: SpendingPatternsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Analyze spending patterns and detect anomalies
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["read:patterns"])
        
        # Analyze spending patterns
        analysis = await advanced_banking_service.analyze_spending_patterns(
            user_id=current_user["id"],
            account_id=request.account_id,
            time_period=request.time_period
        )
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 