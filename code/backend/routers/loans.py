from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from services.genai_service import genai_service

router = APIRouter()

class LoanRequest(BaseModel):
    user_id: str
    loan_type: str
    amount: float
    term: int

class LoanOffer(BaseModel):
    id: str
    type: str
    amount: float
    interest_rate: float
    term: int
    monthly_payment: float
    requirements: List[str]
    personalized_insights: str

@router.post("/recommendations", response_model=List[LoanOffer])
async def get_loan_recommendations(request: LoanRequest):
    try:
        # Generate personalized recommendations using Gen AI
        recommendations = await genai_service.generate_loan_recommendations(
            user_id=request.user_id,
            loan_type=request.loan_type,
            amount=request.amount,
            term=request.term
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply")
async def apply_for_loan(request: LoanRequest):
    try:
        # Process loan application
        application_result = await genai_service.process_loan_application(request.dict())
        return application_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 