from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

class LoanPreferences(BaseModel):
    loanType: str
    amount: float
    term: int
    purpose: str

class UserProfile(BaseModel):
    creditScore: int
    annualIncome: float
    employmentStatus: str
    existingDebts: float

class LoanRequest(BaseModel):
    accountId: str
    userProfile: UserProfile
    preferences: LoanPreferences

class LoanOffer(BaseModel):
    id: str
    type: str
    amount: float
    interestRate: float
    term: int
    monthlyPayment: float
    requirements: List[str]
    personalizedInsights: List[str]
    probability: float

class LoanRecommendationResponse(BaseModel):
    recommendations: List[LoanOffer]
    summary: str
    nextSteps: List[str]

@router.post("/recommendations", response_model=LoanRecommendationResponse)
async def get_loan_recommendations(request: LoanRequest):
    # Calculate debt-to-income ratio
    dti_ratio = request.userProfile.existingDebts / request.userProfile.annualIncome

    # Initialize response
    recommendations = []
    
    # Basic loan eligibility check
    if request.userProfile.creditScore < 600 or dti_ratio > 0.43:
        raise HTTPException(
            status_code=400,
            detail="Not eligible for loans at this time due to credit score or debt-to-income ratio"
        )

    # Generate personalized loan offers based on user profile and preferences
    base_rate = 0.05  # 5% base interest rate
    
    # Adjust rate based on credit score
    credit_adjustment = (750 - request.userProfile.creditScore) * 0.0001
    
    # Adjust rate based on loan term
    term_adjustment = (request.preferences.term - 24) * 0.0002
    
    # Calculate final rate
    interest_rate = base_rate + credit_adjustment + term_adjustment

    # Generate primary loan offer
    monthly_payment = (request.preferences.amount * (1 + interest_rate)) / request.preferences.term
    
    primary_offer = LoanOffer(
        id="LOAN-1",
        type=request.preferences.loanType,
        amount=request.preferences.amount,
        interestRate=interest_rate * 100,  # Convert to percentage
        term=request.preferences.term,
        monthlyPayment=round(monthly_payment, 2),
        requirements=[
            "Proof of income",
            "Bank statements for last 3 months",
            "Valid ID"
        ],
        personalizedInsights=[
            f"Based on your credit score of {request.userProfile.creditScore}, you qualify for our competitive rates",
            f"Your debt-to-income ratio of {dti_ratio:.2%} is within acceptable range"
        ],
        probability=0.85
    )
    recommendations.append(primary_offer)

    # Generate alternative offer with different terms
    alt_term = request.preferences.term + 12
    alt_monthly_payment = (request.preferences.amount * (1 + interest_rate)) / alt_term
    
    alternative_offer = LoanOffer(
        id="LOAN-2",
        type=request.preferences.loanType,
        amount=request.preferences.amount,
        interestRate=(interest_rate - 0.005) * 100,  # Slightly lower rate for longer term
        term=alt_term,
        monthlyPayment=round(alt_monthly_payment, 2),
        requirements=[
            "Proof of income",
            "Bank statements for last 3 months",
            "Valid ID"
        ],
        personalizedInsights=[
            "This alternative offer provides lower monthly payments with a longer term",
            "Slightly lower interest rate available with extended term"
        ],
        probability=0.75
    )
    recommendations.append(alternative_offer)

    return LoanRecommendationResponse(
        recommendations=recommendations,
        summary=f"Found {len(recommendations)} loan offers matching your criteria",
        nextSteps=[
            "Review and compare the loan offers",
            "Prepare required documentation",
            "Schedule a consultation with our loan advisor",
            "Begin the application process online"
        ]
    ) 