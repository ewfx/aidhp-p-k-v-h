from typing import Dict, List, Optional, Any
import openai
from transformers import pipeline
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import random
from .data_service import data_service
import numpy as np

load_dotenv()

class GenAIService:
    def __init__(self):
        # Initialize OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not found. Using basic analysis.")
        else:
            self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        try:
            # Initialize Hugging Face models for sentiment analysis and classification
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            print(f"Warning: Could not initialize Hugging Face models: {str(e)}")
            self.sentiment_analyzer = None
            self.zero_shot_classifier = None

    async def get_customer_insights(self, customer_id: str) -> Dict[str, Any]:
        """Get AI-powered insights for a customer"""
        try:
            # Get customer profile
            profile = data_service.get_customer_profile(customer_id)
            if not profile:
                return {
                    "error": "Customer not found",
                    "customer_id": customer_id
                }

            # Get transaction data
            transactions = data_service.get_user_transactions(customer_id)
            if transactions.empty:
                return {
                    "error": "No transaction data available",
                    "customer_id": customer_id
                }

            # Calculate transaction metrics
            total_spent = transactions['amount'].sum()
            avg_transaction = transactions['amount'].mean()
            category_breakdown = transactions.groupby('category')['amount'].sum().to_dict()
            recent_transactions = transactions.nlargest(5, 'date')[
                ['category', 'amount', 'date']
            ].to_dict('records')

            # Get sentiment data
            sentiment_data = data_service.get_customer_recommendations(customer_id)
            sentiment_metrics = sentiment_data.get('sentiment_metrics', {}) if sentiment_data else {}

            # Generate insights based on the data
            insights = {
                "profile_insights": {
                    "age_group": self._get_age_group(profile.get('age')),
                    "income_level": self._get_income_level(profile.get('income')),
                    "occupation_insights": self._get_occupation_insights(profile.get('occupation'))
                },
                "spending_insights": {
                    "total_spent": total_spent,
                    "avg_transaction": avg_transaction,
                    "top_categories": sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)[:3],
                    "spending_trend": self._analyze_spending_trend(transactions)
                },
                "sentiment_insights": {
                    "overall_sentiment": sentiment_metrics.get('avg_sentiment_score', 0),
                    "platform_breakdown": sentiment_metrics.get('sentiment_by_platform', {})
                }
            }

            return insights

        except Exception as e:
            print(f"Error generating customer insights: {str(e)}")
            return {
                "error": str(e),
                "customer_id": customer_id
            }

    def _get_age_group(self, age: Optional[int]) -> str:
        """Categorize age into groups"""
        if not age:
            return "Unknown"
        if age < 25:
            return "Young Professional"
        elif age < 35:
            return "Early Career"
        elif age < 45:
            return "Mid Career"
        elif age < 55:
            return "Established Professional"
        else:
            return "Senior Professional"

    def _get_income_level(self, income: Optional[float]) -> str:
        """Categorize income into levels"""
        if not income:
            return "Unknown"
        if income < 50000:
            return "Entry Level"
        elif income < 100000:
            return "Mid Level"
        elif income < 150000:
            return "Upper Mid Level"
        else:
            return "High Level"

    def _get_occupation_insights(self, occupation: Optional[str]) -> Dict[str, Any]:
        """Generate insights based on occupation"""
        if not occupation:
            return {"category": "Unknown", "risk_level": "Unknown"}
        
        occupation = occupation.lower()
        if any(tech in occupation for tech in ['engineer', 'developer', 'programmer', 'software']):
            return {"category": "Technology", "risk_level": "Moderate"}
        elif any(finance in occupation for finance in ['banker', 'finance', 'accountant', 'analyst']):
            return {"category": "Finance", "risk_level": "Conservative"}
        elif any(health in occupation for health in ['doctor', 'nurse', 'healthcare', 'medical']):
            return {"category": "Healthcare", "risk_level": "Conservative"}
        elif any(edu in occupation for edu in ['teacher', 'professor', 'educator', 'academic']):
            return {"category": "Education", "risk_level": "Conservative"}
        else:
            return {"category": "Other", "risk_level": "Moderate"}

    def _analyze_spending_trend(self, transactions: pd.DataFrame) -> str:
        """Analyze spending trend over time"""
        if transactions.empty:
            return "No transaction data available"
            
        # Group by date and calculate daily spending
        daily_spending = transactions.groupby('date')['amount'].sum()
        
        # Calculate trend
        if len(daily_spending) < 2:
            return "Insufficient data for trend analysis"
            
        # Calculate simple linear regression
        x = np.arange(len(daily_spending))
        y = daily_spending.values
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0:
            return "Increasing spending trend"
        elif slope < 0:
            return "Decreasing spending trend"
        else:
            return "Stable spending pattern"

    async def _generate_ai_insights(self, customer_data: Dict, recommendations: Dict) -> Dict:
        """Generate detailed insights using OpenAI"""
        try:
            # Prepare the prompt with customer data
            prompt = f"""
            Analyze the following customer data and provide detailed insights:

            Customer Profile:
            {customer_data['profile']}

            Social Media Sentiment:
            - Average sentiment score: {customer_data['sentiment_analysis']['avg_sentiment_score']}
            - Recent sentiments: {customer_data['sentiment_analysis']['recent_sentiments']}

            Transaction Analysis:
            - Total spent: ${customer_data['transaction_analysis']['total_spent']}
            - Category breakdown: {customer_data['transaction_analysis']['category_breakdown']}

            Recommended Categories:
            {recommendations['top_categories']}

            Customer Segments:
            {recommendations['customer_segments']}

            Please provide:
            1. Key behavioral patterns
            2. Risk factors
            3. Opportunities for engagement
            4. Personalized recommendations
            """

            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert customer analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            # Parse the response into structured insights
            ai_response = response.choices[0].message.content
            return {
                'ai_generated_insights': ai_response,
                'confidence_score': 0.9,
                'generated_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error generating AI insights: {str(e)}")
            return self._generate_basic_insights(customer_data, recommendations)

    def _generate_basic_insights(self, customer_data: Dict, recommendations: Dict) -> Dict:
        """Generate basic insights without AI models"""
        profile = customer_data['profile']
        sentiment = customer_data['sentiment_analysis']
        transactions = customer_data['transaction_analysis']

        insights = {
            'behavioral_patterns': [
                f"Customer primarily shops in: {', '.join(list(transactions['category_breakdown'].keys())[:3])}",
                f"Preferred payment method: {max(transactions['payment_preferences'].items(), key=lambda x: x[1])[0]}",
                f"Average transaction amount: ${transactions['avg_transaction']:.2f}"
            ],
            'risk_factors': [
                "Low engagement" if sentiment['avg_sentiment_score'] < 0.5 else "Good engagement",
                "Declining activity" if len(transactions['recent_transactions']) < 2 else "Active customer"
            ],
            'opportunities': [
                f"Potential interest in {cat}" for cat, _ in recommendations['top_categories']
            ],
            'generated_timestamp': datetime.now().isoformat()
        }

        return insights

    async def generate_chatbot_response(
        self,
        message: str,
        context: Dict = None,
        user_data: Dict = None
    ) -> Dict:
        """Generate a response for the chatbot"""
        try:
            if not self.openai_api_key:
                # Return mock response if OpenAI is not configured
                return self._get_mock_response(message, context)

            # Prepare the system message with enhanced banking knowledge
            system_message = """You are an expert banking assistant with deep knowledge of:
            1. Account Management:
               - Checking and savings accounts
               - Interest rates and APY
               - Account fees and minimum balances
               - Overdraft protection
               - Direct deposit setup
            
            2. Transaction Services:
               - Online and mobile banking
               - Bill payments
               - Wire transfers
               - ACH transactions
               - International transfers
            
            3. Loans and Credit:
               - Personal loans
               - Home loans
               - Auto loans
               - Credit cards
               - Lines of credit
               - Interest rates and terms
            
            4. Investment Services:
               - Savings accounts
               - CDs
               - Investment accounts
               - Portfolio management
               - Market analysis
            
            5. Security and Fraud:
               - Account security
               - Fraud prevention
               - Identity theft protection
               - Two-factor authentication
               - Suspicious activity alerts
            
            6. Financial Planning:
               - Budgeting
               - Savings goals
               - Retirement planning
               - Tax planning
               - Estate planning
            
            Always be professional, clear, and concise. If you need more information, ask for it politely.
            Provide specific, actionable advice when possible."""
            
            # Prepare user context with enhanced profile
            user_context = ""
            if user_data:
                user_context = f"""
                User Profile:
                - Age: {user_data.get('age')}
                - Income: {user_data.get('income')}
                - Risk Tolerance: {user_data.get('risk_tolerance')}
                - Investment Goals: {user_data.get('investment_goals')}
                
                Banking History:
                - Account Type: {user_data.get('account_type', 'Standard')}
                - Years with Bank: {user_data.get('years_with_bank', 'N/A')}
                - Credit Score Range: {user_data.get('credit_score_range', 'N/A')}
                - Preferred Services: {user_data.get('preferred_services', [])}
                """
            
            # Prepare conversation history
            conversation_history = context.get("conversation_history", [])
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in conversation_history[-5:]  # Keep last 5 messages for context
            ])
            
            # Prepare the prompt with enhanced context
            prompt = f"""
            {user_context}
            
            Previous conversation:
            {history_text}
            
            User: {message}
            
            Assistant:"""
            
            # Generate response using OpenAI with enhanced parameters
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # Increased for more detailed responses
                temperature=0.7,  # Balanced between creativity and consistency
                presence_penalty=0.6,  # Encourage diverse responses
                frequency_penalty=0.6  # Reduce repetition
            )
            
            # Generate suggested actions with enhanced context
            actions_prompt = f"""
            Based on the user's message: "{message}"
            And their profile: {user_context}
            Generate 2-3 relevant suggested actions or follow-up questions that would be helpful.
            Focus on:
            1. Immediate next steps
            2. Related banking services
            3. Financial planning opportunities
            """
            
            actions_response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate 2-3 relevant suggested actions or follow-up questions."},
                    {"role": "user", "content": actions_prompt}
                ],
                max_tokens=150
            )
            
            # Parse suggested actions
            suggested_actions = [
                action.strip() for action in actions_response.choices[0].message.content.split("\n")
                if action.strip()
            ]
            
            # Update context with enhanced metadata
            new_context = context or {}
            new_context["conversation_history"] = conversation_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response.choices[0].message.content}
            ]
            new_context["last_topic"] = self._extract_topic(message)
            new_context["interaction_count"] = new_context.get("interaction_count", 0) + 1
            
            return {
                "message": response.choices[0].message.content,
                "context": new_context,
                "suggested_actions": suggested_actions
            }
        except Exception as e:
            print(f"Error in generate_chatbot_response: {str(e)}")
            return self._get_mock_response(message, context)

    def _get_mock_response(self, message: str, context: Dict = None) -> Dict:
        """Generate a mock response when OpenAI is not available"""
        message_lower = message.lower()
        
        # Define some basic mock responses
        responses = {
            "balance": {
                "message": "I can help you check your account balance. However, this is a demo version, so I can't show real balance information. In a real system, I would securely fetch your current balance.",
                "actions": ["View Transactions", "Set Balance Alert", "Contact Support"]
            },
            "loan": {
                "message": "I can help you with loan information. We offer various types of loans including personal loans, home loans, and auto loans. Would you like to learn more about any specific type?",
                "actions": ["Personal Loan Rates", "Home Loan Options", "Auto Loan Calculator"]
            },
            "fraud": {
                "message": "I understand you're concerned about fraud. Your security is our top priority. Would you like to learn about our fraud protection measures or report suspicious activity?",
                "actions": ["Report Suspicious Activity", "Security Tips", "Contact Support"]
            },
            "transfer": {
                "message": "I can help you with transfers. In this demo version, I can explain how transfers work but cannot process actual transactions.",
                "actions": ["Transfer Guide", "View Limits", "Contact Support"]
            }
        }

        # Find the most relevant mock response
        response_key = next(
            (key for key in responses.keys() if key in message_lower),
            "default"
        )

        if response_key == "default":
            return {
                "message": "I'm here to help with your banking needs. What would you like to know about?",
                "context": context or {},
                "suggested_actions": [
                    "Check Balance",
                    "Transfer Money",
                    "Loan Information",
                    "Report Fraud"
                ]
            }

        return {
            "message": responses[response_key]["message"],
            "context": context or {},
            "suggested_actions": responses[response_key]["actions"]
        }

    async def generate_loan_recommendations(
        self,
        user_id: str,
        loan_type: str,
        amount: float,
        term: int
    ) -> Dict:
        """Generate personalized loan recommendations"""
        # Mock data for demonstration
        offers = [
            {
                "id": "loan1",
                "type": "Personal Loan",
                "amount": amount,
                "interestRate": 7.5,
                "term": term,
                "monthlyPayment": amount * (1 + 0.075 * term/12) / term,
                "requirements": [
                    "Minimum credit score: 650",
                    "Income verification",
                    "Employment history"
                ],
                "personalizedInsights": [
                    "Based on your profile, you qualify for our best rates",
                    "Consider a longer term for lower monthly payments"
                ]
            },
            {
                "id": "loan2",
                "type": "Home Equity Loan",
                "amount": amount * 1.2,
                "interestRate": 6.8,
                "term": term,
                "monthlyPayment": amount * 1.2 * (1 + 0.068 * term/12) / term,
                "requirements": [
                    "Home equity of at least 20%",
                    "Property appraisal",
                    "Income verification"
                ],
                "personalizedInsights": [
                    "This option provides better rates for home improvements",
                    "Tax-deductible interest for qualified expenses"
                ]
            }
        ]

        return {
            "offers": offers,
            "activeApplication": None
        }

    async def detect_fraud(
        self,
        transaction_data: List[Dict]
    ) -> Dict:
        """Detect potential fraudulent transactions"""
        # Mock data for demonstration
        alerts = [
            {
                "id": "alert1",
                "severity": "high",
                "title": "Suspicious Large Transfer",
                "description": "Large transfer to unknown account detected",
                "timestamp": datetime.now().isoformat(),
                "status": "new",
                "recommendedActions": [
                    "Verify transaction details",
                    "Contact customer support",
                    "Review recent activity"
                ],
                "riskScore": 85
            },
            {
                "id": "alert2",
                "severity": "medium",
                "title": "Unusual Location",
                "description": "Transaction from new location detected",
                "timestamp": datetime.now().isoformat(),
                "status": "new",
                "recommendedActions": [
                    "Confirm location",
                    "Review recent transactions"
                ],
                "riskScore": 65
            }
        ]

        return {
            "alerts": alerts,
            "risk_level": "medium",
            "recommendations": [
                "Enable two-factor authentication",
                "Review account activity regularly",
                "Set up transaction alerts"
            ]
        }

    async def analyze_spending_patterns(
        self,
        user_id: str,
        time_period: str = "30d"
    ) -> Dict:
        """Analyze spending patterns"""
        # Mock data for demonstration
        days = int(time_period.replace("d", ""))
        dates = [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(days)]
        
        daily_trend = {
            date: random.uniform(100, 1000) for date in dates
        }
        
        category_breakdown = {
            "Housing": random.uniform(2000, 4000),
            "Food": random.uniform(500, 1500),
            "Transportation": random.uniform(300, 1000),
            "Entertainment": random.uniform(200, 800),
            "Utilities": random.uniform(200, 600)
        }
        
        total_spending = sum(daily_trend.values())
        
        return {
            "patterns": {
                "daily_trend": daily_trend,
                "category_breakdown": category_breakdown,
                "total_spending": total_spending,
                "average_daily_spending": total_spending / days
            },
            "insights": {
                "Housing": ["Your housing expenses are within normal range"],
                "Food": ["Consider meal planning to reduce food expenses"],
                "Transportation": ["Your transportation costs are below average"],
                "Entertainment": ["Entertainment spending has increased this month"],
                "Utilities": ["Utility costs are consistent with previous months"]
            }
        }

    async def generate_personalized_content(
        self,
        user_preferences: Dict
    ) -> Dict:
        """Generate personalized content based on user preferences"""
        # Mock data for demonstration
        return {
            "content": "Based on your profile, here are some personalized recommendations...",
            "recommendations": [
                "Consider a high-yield savings account",
                "Look into investment opportunities",
                "Review your insurance coverage"
            ],
            "insights": {
                "savings": ["You could save more by reducing entertainment expenses"],
                "investments": ["Your risk tolerance suggests balanced portfolio"],
                "insurance": ["Consider increasing your coverage"]
            }
        }

    async def generate_dynamic_ui(
        self,
        user_preferences: Dict
    ) -> Dict:
        """Generate dynamic UI elements based on user preferences"""
        # Mock data for demonstration
        return {
            "layout": {
                "theme": "light",
                "primary_color": "#1976d2",
                "secondary_color": "#dc004e"
            },
            "components": [
                {
                    "type": "card",
                    "title": "Savings Goals",
                    "priority": "high"
                },
                {
                    "type": "chart",
                    "title": "Spending Analysis",
                    "priority": "medium"
                }
            ],
            "theme": {
                "mode": "light",
                "primary": "#1976d2",
                "secondary": "#dc004e",
                "background": "#ffffff"
            }
        }

    def _extract_topic(self, message: str) -> str:
        """Extract the main topic from the user's message"""
        topics = {
            "account": ["account", "balance", "checking", "savings"],
            "transaction": ["transaction", "transfer", "payment", "bill"],
            "loan": ["loan", "credit", "mortgage", "financing"],
            "investment": ["investment", "portfolio", "stock", "market"],
            "security": ["security", "fraud", "password", "authentication"],
            "planning": ["budget", "planning", "retirement", "savings"]
        }
        
        message_lower = message.lower()
        for topic, keywords in topics.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic
        return "general"

# Create singleton instance
genai_service = GenAIService() 