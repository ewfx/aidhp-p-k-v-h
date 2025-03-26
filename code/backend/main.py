from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from services.data_service import data_service
from services.genai_service import genai_service
from routers import loans, personalization
import pandas as pd
from datetime import datetime, timedelta
import logging
import uvicorn
from analysis.customer_analysis import customer_analyzer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from services.chatbot_service import chatbot_service
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Banking Dashboard API",
    description="API for the Banking Dashboard with AI-powered features",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SpendingAnalysisRequest(BaseModel):
    user_id: str
    time_period: str = "30d"  # e.g., "7d", "30d", "90d"

class FraudDetectionRequest(BaseModel):
    transaction_data: List[Dict]

class LoanRequest(BaseModel):
    user_id: str
    loan_type: str
    amount: float
    term: int

class UserPreferences(BaseModel):
    user_id: str
    age: int
    income: float
    risk_tolerance: str
    investment_goals: List[str]

# Add new ChatbotRequest model
class ChatbotRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[Dict] = None

# Add new ChatbotResponse model
class ChatbotResponse(BaseModel):
    response: str
    context: Dict
    suggested_actions: List[str]

# Add new ChatMessage model
class ChatMessage(BaseModel):
    message: str
    customer_id: Optional[str] = None

# Include routers
app.include_router(
    loans.router,
    prefix="/api/v1/banking/loans",
    tags=["loans"]
)

app.include_router(
    personalization.router,
    prefix="/api/v1/banking/personalization",
    tags=["personalization"]
)

# Mount static files
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(BASE_DIR, "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

# API Endpoints
@app.post("/api/v1/banking/advanced/patterns/analyze")
async def analyze_spending_patterns(request: SpendingAnalysisRequest):
    try:
        # Get transaction data
        df = data_service.get_user_transactions(request.user_id)
        
        # Filter by time period
        end_date = datetime.now()
        if request.time_period == "7d":
            start_date = end_date - timedelta(days=7)
        elif request.time_period == "30d":
            start_date = end_date - timedelta(days=30)
        else:  # 90d
            start_date = end_date - timedelta(days=90)
            
        df = df[(df['date'] >= start_date.strftime("%Y-%m-%d")) & 
                (df['date'] <= end_date.strftime("%Y-%m-%d"))]
        
        # Analyze patterns
        patterns = {
            "total_spending": float(df['amount'].sum()),
            "avg_transaction": float(df['amount'].mean()),
            "category_breakdown": df.groupby('category')['amount'].sum().to_dict(),
            "daily_trend": df.groupby('date')['amount'].sum().to_dict()
        }
        
        # Generate insights using Gen AI
        insights = await genai_service.analyze_spending_patterns(
            user_id=request.user_id,
            time_period=request.time_period
        )
        
        return {
            "patterns": patterns,
            "insights": insights["insights"],
            "time_period": request.time_period
        }
    except Exception as e:
        logger.error(f"Error in analyze_spending_patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/banking/advanced/fraud/detect")
async def detect_fraud(request: FraudDetectionRequest):
    try:
        # Analyze transactions for fraud
        result = await genai_service.detect_fraud(request.transaction_data)
        return result
    except Exception as e:
        logger.error(f"Error in detect_fraud: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/banking/advanced/loans/recommendations")
async def get_loan_recommendations(request: LoanRequest):
    try:
        # Generate loan recommendations
        result = await genai_service.generate_loan_recommendations(
            user_id=request.user_id,
            loan_type=request.loan_type,
            amount=request.amount,
            term=request.term
        )
        return result
    except Exception as e:
        logger.error(f"Error in get_loan_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generative/content")
async def get_personalized_content(request: UserPreferences):
    try:
        # Generate personalized content
        result = await genai_service.generate_personalized_content(request.dict())
        return result
    except Exception as e:
        logger.error(f"Error in get_personalized_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generative/ui-elements")
async def get_dynamic_ui_elements(request: UserPreferences):
    try:
        # Generate dynamic UI elements
        result = await genai_service.generate_dynamic_ui(request.dict())
        return result
    except Exception as e:
        logger.error(f"Error in get_dynamic_ui_elements: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chatbot/interact")
async def chat_with_bot(request: ChatbotRequest):
    try:
        # Get user data if available
        user_data = data_service.get_user_data(request.user_id)

        # Generate response using Gen AI
        response = await genai_service.generate_chatbot_response(
            message=request.message,
            context=request.context or {},
            user_data=user_data
        )

        return ChatbotResponse(
            response=response["message"],
            context=response["context"],
            suggested_actions=response["suggested_actions"]
        )
    except Exception as e:
        logger.error(f"Error in chat_with_bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/v1/chat")
async def chat(chat_message: ChatMessage):
    """Handle chat messages"""
    try:
        response = await chatbot_service.process_message(chat_message.message, chat_message.customer_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/customer/{customer_id}/analysis")
async def analyze_customer(customer_id: str):
    """Get comprehensive analysis for a customer"""
    try:
        analysis = await genai_service.get_customer_insights(customer_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Customer not found")
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/customer/{customer_id}/recommendations")
async def get_recommendations(customer_id: str):
    """Get personalized recommendations for a customer"""
    try:
        recommendations = data_service.get_customer_recommendations(customer_id)
        if not recommendations:
            raise HTTPException(status_code=404, detail="Customer not found")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/load")
async def load_data(file: UploadFile = File(...)):
    """Load customer data from Excel file"""
    try:
        # Save the uploaded file temporarily
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        # Load the data using data service
        success = data_service.load_data(file_location)
        
        # Clean up the temporary file
        os.remove(file_location)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to load data")

        return {"status": "success", "message": "Data loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/customer/{customer_id}/insights")
async def get_customer_insights(customer_id: str):
    """Get AI-powered insights for a specific customer"""
    try:
        insights = await genai_service.get_customer_insights(customer_id)
        if "error" in insights:
            raise HTTPException(status_code=404, detail=insights["error"])
        return insights
    except Exception as e:
        logger.error(f"Error getting customer insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/customer/{customer_id}/profile")
async def get_customer_profile(customer_id: str):
    """Get complete customer profile"""
    try:
        profile = data_service.get_customer_profile(customer_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        return profile
    except Exception as e:
        logger.error(f"Error getting customer profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def create_required_directories():
    """Create required directories if they don't exist"""
    directories = [
        "data/datasets",
        "static",
        "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main function to run the application"""
    try:
        # Create required directories
        create_required_directories()
        
        # Initialize data service
        print("Initializing data service...")
        data_service.initialize()
        
        # Start the server
        print("\nStarting Customer Analytics Chatbot...")
        print("Access the application at: http://localhost:8001")
        print("Press Ctrl+C to stop the server\n")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8001,
            reload=True
        )
        
    except Exception as e:
        print(f"\nError starting the application: {str(e)}")
        print("Please check the error message and try again.")
        print("If the problem persists, check the README.md file for troubleshooting steps.")

if __name__ == "__main__":
    main() 