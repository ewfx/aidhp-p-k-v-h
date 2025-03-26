from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from src.services.banking_service import BankingService
from src.utils.auth import get_current_user, check_permissions
from src.models.generative.hybrid_hub import BankingSecurity
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/banking", tags=["banking"])

# Initialize services
banking_service = BankingService({})  # Config will be loaded from environment
banking_security = BankingSecurity()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AccountSummaryRequest(BaseModel):
    account_id: str
    user_preferences: Dict[str, Any]

class TransactionAnalysisRequest(BaseModel):
    account_id: str
    start_date: datetime
    end_date: datetime
    user_preferences: Dict[str, Any]

class BudgetRecommendationsRequest(BaseModel):
    account_id: str
    user_preferences: Dict[str, Any]

class InvestmentInsightsRequest(BaseModel):
    portfolio_data: Dict[str, Any]
    user_preferences: Dict[str, Any]

@router.post("/account/summary")
async def get_account_summary(
    request: AccountSummaryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Get personalized account summary with AI insights
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["read:account"])
        
        # Generate account summary
        summary = await banking_service.get_account_summary(
            user_id=current_user["id"],
            account_id=request.account_id,
            user_preferences=request.user_preferences
        )
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transactions/analysis")
async def get_transaction_analysis(
    request: TransactionAnalysisRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Get AI-powered transaction analysis
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["read:transactions"])
        
        # Generate transaction analysis
        analysis = await banking_service.get_transaction_analysis(
            user_id=current_user["id"],
            account_id=request.account_id,
            start_date=request.start_date,
            end_date=request.end_date,
            user_preferences=request.user_preferences
        )
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/budget/recommendations")
async def get_budget_recommendations(
    request: BudgetRecommendationsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Get personalized budget recommendations
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["read:budget"])
        
        # Generate budget recommendations
        recommendations = await banking_service.generate_budget_recommendations(
            user_id=current_user["id"],
            account_id=request.account_id,
            user_preferences=request.user_preferences
        )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/investments/insights")
async def get_investment_insights(
    request: InvestmentInsightsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Get personalized investment insights
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["read:investments"])
        
        # Generate investment insights
        insights = await banking_service.generate_investment_insights(
            user_id=current_user["id"],
            portfolio_data=request.portfolio_data,
            user_preferences=request.user_preferences
        )
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security/2fa/setup")
async def setup_2fa(
    current_user: Dict[str, Any] = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Set up 2FA for the user
    """
    try:
        # Verify user permissions
        await check_permissions(current_user, ["manage:security"])
        
        # Generate 2FA QR code
        qr_code = banking_security.generate_2fa_qr(current_user["id"])
        
        return {
            "qr_code": qr_code,
            "message": "2FA setup successful"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security/2fa/verify")
async def verify_2fa(
    token: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Verify 2FA token
    """
    try:
        # Verify 2FA token
        is_valid = banking_security.verify_2fa(token)
        
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid 2FA token")
        
        return {
            "message": "2FA verification successful"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 