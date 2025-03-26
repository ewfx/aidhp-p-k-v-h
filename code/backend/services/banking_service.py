from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from src.models.generative.hybrid_hub import HybridGenerativeHub
from plaid import Client as PlaidClient
import stripe
import json

logger = logging.getLogger(__name__)

class BankingService:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Banking Service with AI capabilities
        """
        self.config = config
        self.generative_hub = HybridGenerativeHub(config)
        self._init_banking_clients()

    def _init_banking_clients(self):
        """Initialize banking-related clients"""
        try:
            # Initialize Plaid client
            self.plaid_client = PlaidClient(
                client_id=self.config.get("PLAID_CLIENT_ID"),
                secret=self.config.get("PLAID_SECRET"),
                environment=self.config.get("PLAID_ENV", "sandbox")
            )
            
            # Initialize Stripe client
            stripe.api_key = self.config.get("STRIPE_SECRET_KEY")
            
        except Exception as e:
            logger.error(f"Error initializing banking clients: {str(e)}")
            raise

    async def get_account_summary(
        self,
        user_id: str,
        account_id: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized account summary with AI insights
        """
        try:
            # Get account data from Plaid
            account_data = await self._get_plaid_account_data(account_id)
            
            # Generate AI-powered insights
            insights = await self.generative_hub.generate_financial_insights(
                account_data=account_data,
                user_preferences=user_preferences
            )
            
            return {
                "account_data": account_data,
                "ai_insights": insights,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating account summary: {str(e)}")
            raise

    async def get_transaction_analysis(
        self,
        user_id: str,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI-powered transaction analysis
        """
        try:
            # Get transactions from Plaid
            transactions = await self._get_plaid_transactions(
                account_id,
                start_date,
                end_date
            )
            
            # Generate AI-powered summary
            summary = await self.generative_hub.generate_transaction_summary(
                transactions=transactions,
                user_preferences=user_preferences
            )
            
            return {
                "transactions": transactions,
                "ai_summary": summary,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating transaction analysis: {str(e)}")
            raise

    async def generate_budget_recommendations(
        self,
        user_id: str,
        account_id: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized budget recommendations
        """
        try:
            # Get spending data
            spending_data = await self._get_spending_data(account_id)
            
            # Generate AI recommendations
            prompt = self._create_budget_prompt(spending_data, user_preferences)
            recommendations = await self.generative_hub.generate_banking_content(
                prompt=prompt,
                content_type="text",
                user_preferences=user_preferences,
                sensitive_data={"spending_data": json.dumps(spending_data)}
            )
            
            return {
                "recommendations": recommendations,
                "spending_analysis": spending_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating budget recommendations: {str(e)}")
            raise

    async def generate_investment_insights(
        self,
        user_id: str,
        portfolio_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized investment insights
        """
        try:
            # Generate AI insights
            prompt = self._create_investment_prompt(portfolio_data, user_preferences)
            insights = await self.generative_hub.generate_banking_content(
                prompt=prompt,
                content_type="text",
                user_preferences=user_preferences,
                sensitive_data={"portfolio_data": json.dumps(portfolio_data)}
            )
            
            return {
                "insights": insights,
                "portfolio_analysis": portfolio_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating investment insights: {str(e)}")
            raise

    async def _get_plaid_account_data(self, account_id: str) -> Dict[str, Any]:
        """Get account data from Plaid"""
        try:
            response = await self.plaid_client.accounts_get(account_id)
            return response
        except Exception as e:
            logger.error(f"Error getting Plaid account data: {str(e)}")
            raise

    async def _get_plaid_transactions(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get transactions from Plaid"""
        try:
            response = await self.plaid_client.transactions_get(
                account_id,
                start_date=start_date.date(),
                end_date=end_date.date()
            )
            return response["transactions"]
        except Exception as e:
            logger.error(f"Error getting Plaid transactions: {str(e)}")
            raise

    async def _get_spending_data(self, account_id: str) -> Dict[str, Any]:
        """Get spending data for budget analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            transactions = await self._get_plaid_transactions(
                account_id,
                start_date,
                end_date
            )
            
            # Analyze spending patterns
            spending_by_category = {}
            for transaction in transactions:
                category = transaction.get("category", ["Uncategorized"])[0]
                amount = transaction.get("amount", 0)
                spending_by_category[category] = spending_by_category.get(category, 0) + amount
            
            return {
                "total_spending": sum(spending_by_category.values()),
                "spending_by_category": spending_by_category,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting spending data: {str(e)}")
            raise

    def _create_budget_prompt(
        self,
        spending_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> str:
        """Create prompt for budget recommendations"""
        return (
            f"Generate personalized budget recommendations based on spending data: "
            f"{json.dumps(spending_data)} for a user with preferences: "
            f"{json.dumps(user_preferences)}. Focus on spending optimization, "
            f"savings opportunities, and financial goals."
        )

    def _create_investment_prompt(
        self,
        portfolio_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> str:
        """Create prompt for investment insights"""
        return (
            f"Generate personalized investment insights based on portfolio data: "
            f"{json.dumps(portfolio_data)} for a user with preferences: "
            f"{json.dumps(user_preferences)}. Focus on portfolio optimization, "
            f"risk management, and investment opportunities."
        ) 