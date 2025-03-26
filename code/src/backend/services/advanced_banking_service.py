from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
from src.models.generative.hybrid_hub import HybridGenerativeHub
from src.config.banking_config import get_banking_config
import json

logger = logging.getLogger(__name__)

class AdvancedBankingService:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Advanced Banking Service with AI capabilities
        """
        self.config = config
        self.generative_hub = HybridGenerativeHub(config)
        self.banking_config = get_banking_config()

    async def detect_fraud(
        self,
        user_id: str,
        account_id: str,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect potential fraudulent transactions using AI
        """
        try:
            # Analyze transaction patterns
            risk_score = await self._calculate_risk_score(transaction_data)
            
            # Generate AI-powered fraud analysis
            prompt = self._create_fraud_prompt(transaction_data, risk_score)
            analysis = await self.generative_hub.generate_banking_content(
                prompt=prompt,
                content_type="text",
                user_preferences={"risk_tolerance": "low"},
                sensitive_data={"transaction_data": json.dumps(transaction_data)}
            )
            
            # Determine if immediate action is needed
            requires_action = risk_score > self.banking_config["FRAUD_DETECTION"]["HIGH_RISK_THRESHOLD"]
            
            return {
                "risk_score": risk_score,
                "analysis": analysis,
                "requires_action": requires_action,
                "recommended_actions": self._get_recommended_actions(risk_score),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {str(e)}")
            raise

    async def generate_loan_recommendations(
        self,
        user_id: str,
        account_id: str,
        loan_purpose: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized loan recommendations
        """
        try:
            # Get user's financial profile
            financial_profile = await self._get_financial_profile(account_id)
            
            # Calculate creditworthiness
            credit_score = await self._calculate_credit_score(financial_profile)
            
            # Generate AI-powered loan recommendations
            prompt = self._create_loan_prompt(
                financial_profile,
                credit_score,
                loan_purpose,
                user_preferences
            )
            recommendations = await self.generative_hub.generate_banking_content(
                prompt=prompt,
                content_type="text",
                user_preferences=user_preferences,
                sensitive_data={"financial_profile": json.dumps(financial_profile)}
            )
            
            return {
                "credit_score": credit_score,
                "recommendations": recommendations,
                "loan_options": self._get_loan_options(credit_score, loan_purpose),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating loan recommendations: {str(e)}")
            raise

    async def analyze_spending_patterns(
        self,
        user_id: str,
        account_id: str,
        time_period: str = "90d"
    ) -> Dict[str, Any]:
        """
        Analyze spending patterns and detect anomalies
        """
        try:
            # Get transaction history
            transactions = await self._get_transaction_history(account_id, time_period)
            
            # Calculate spending patterns
            patterns = self._calculate_spending_patterns(transactions)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(patterns)
            
            # Generate insights
            prompt = self._create_pattern_prompt(patterns, anomalies)
            insights = await self.generative_hub.generate_banking_content(
                prompt=prompt,
                content_type="text",
                user_preferences={"analysis_depth": "detailed"},
                sensitive_data={"patterns": json.dumps(patterns)}
            )
            
            return {
                "patterns": patterns,
                "anomalies": anomalies,
                "insights": insights,
                "recommendations": self._get_pattern_recommendations(patterns),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spending patterns: {str(e)}")
            raise

    async def _calculate_risk_score(self, transaction_data: Dict[str, Any]) -> float:
        """Calculate risk score for fraud detection"""
        try:
            # Implement risk scoring logic
            risk_factors = {
                "amount": self._calculate_amount_risk(transaction_data),
                "location": self._calculate_location_risk(transaction_data),
                "time": self._calculate_time_risk(transaction_data),
                "pattern": self._calculate_pattern_risk(transaction_data)
            }
            
            # Weighted average of risk factors
            weights = self.banking_config["FRAUD_DETECTION"]["RISK_FACTOR_WEIGHTS"]
            risk_score = sum(score * weights[factor] for factor, score in risk_factors.items())
            
            return min(max(risk_score, 0), 1)  # Normalize between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            raise

    async def _calculate_credit_score(self, financial_profile: Dict[str, Any]) -> float:
        """Calculate creditworthiness score"""
        try:
            # Implement credit scoring logic
            credit_factors = {
                "payment_history": self._calculate_payment_history_score(financial_profile),
                "credit_utilization": self._calculate_credit_utilization_score(financial_profile),
                "account_age": self._calculate_account_age_score(financial_profile),
                "diversity": self._calculate_credit_diversity_score(financial_profile)
            }
            
            # Weighted average of credit factors
            weights = self.banking_config["LOAN_ANALYSIS"]["CREDIT_FACTOR_WEIGHTS"]
            credit_score = sum(score * weights[factor] for factor, score in credit_factors.items())
            
            return min(max(credit_score, 0), 1)  # Normalize between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating credit score: {str(e)}")
            raise

    def _calculate_spending_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate spending patterns from transactions"""
        try:
            patterns = {
                "daily": self._calculate_daily_patterns(transactions),
                "weekly": self._calculate_weekly_patterns(transactions),
                "monthly": self._calculate_monthly_patterns(transactions),
                "category": self._calculate_category_patterns(transactions)
            }
            return patterns
        except Exception as e:
            logger.error(f"Error calculating spending patterns: {str(e)}")
            raise

    def _detect_anomalies(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in spending patterns"""
        try:
            anomalies = []
            
            # Check for unusual transaction amounts
            amount_anomalies = self._detect_amount_anomalies(patterns)
            anomalies.extend(amount_anomalies)
            
            # Check for unusual timing
            timing_anomalies = self._detect_timing_anomalies(patterns)
            anomalies.extend(timing_anomalies)
            
            # Check for unusual categories
            category_anomalies = self._detect_category_anomalies(patterns)
            anomalies.extend(category_anomalies)
            
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise

    def _create_fraud_prompt(
        self,
        transaction_data: Dict[str, Any],
        risk_score: float
    ) -> str:
        """Create prompt for fraud analysis"""
        return (
            f"Analyze potential fraudulent activity based on transaction data: "
            f"{json.dumps(transaction_data)} with risk score: {risk_score}. "
            f"Focus on unusual patterns, suspicious amounts, and location-based risks."
        )

    def _create_loan_prompt(
        self,
        financial_profile: Dict[str, Any],
        credit_score: float,
        loan_purpose: str,
        user_preferences: Dict[str, Any]
    ) -> str:
        """Create prompt for loan recommendations"""
        return (
            f"Generate personalized loan recommendations based on financial profile: "
            f"{json.dumps(financial_profile)}, credit score: {credit_score}, "
            f"loan purpose: {loan_purpose}, and user preferences: "
            f"{json.dumps(user_preferences)}. Focus on suitable loan types, "
            f"interest rates, and repayment terms."
        )

    def _create_pattern_prompt(
        self,
        patterns: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for spending pattern analysis"""
        return (
            f"Analyze spending patterns: {json.dumps(patterns)} and detected "
            f"anomalies: {json.dumps(anomalies)}. Provide insights on spending "
            f"behavior, trends, and potential areas for optimization."
        )

    def _get_recommended_actions(self, risk_score: float) -> List[str]:
        """Get recommended actions based on risk score"""
        if risk_score > 0.8:
            return ["Block transaction", "Contact customer", "Review account activity"]
        elif risk_score > 0.6:
            return ["Flag transaction", "Monitor account", "Request additional verification"]
        elif risk_score > 0.4:
            return ["Review transaction", "Monitor patterns", "Update risk assessment"]
        else:
            return ["Continue monitoring", "Update risk profile"]

    def _get_loan_options(
        self,
        credit_score: float,
        loan_purpose: str
    ) -> List[Dict[str, Any]]:
        """Get available loan options based on credit score and purpose"""
        options = []
        if credit_score > 0.7:
            options.extend([
                {"type": "Personal Loan", "rate": "5.99%", "term": "36 months"},
                {"type": "Home Equity", "rate": "4.99%", "term": "60 months"},
                {"type": "Business Loan", "rate": "6.99%", "term": "48 months"}
            ])
        elif credit_score > 0.5:
            options.extend([
                {"type": "Personal Loan", "rate": "8.99%", "term": "36 months"},
                {"type": "Secured Loan", "rate": "7.99%", "term": "48 months"}
            ])
        else:
            options.extend([
                {"type": "Secured Loan", "rate": "10.99%", "term": "36 months"},
                {"type": "Credit Builder", "rate": "12.99%", "term": "24 months"}
            ])
        return options

    def _get_pattern_recommendations(
        self,
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on spending patterns"""
        recommendations = []
        
        # Analyze daily patterns
        if patterns["daily"].get("variance", 0) > 0.5:
            recommendations.append({
                "type": "daily_spending",
                "message": "Consider setting daily spending limits",
                "priority": "high"
            })
        
        # Analyze category patterns
        for category, amount in patterns["category"].items():
            if amount > patterns["monthly"]["average"] * 2:
                recommendations.append({
                    "type": "category_spending",
                    "message": f"High spending in {category} category",
                    "priority": "medium"
                })
        
        return recommendations 