from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

BANKING_CONFIG: Dict[str, Any] = {
    # Plaid Configuration
    "PLAID_CLIENT_ID": os.getenv("PLAID_CLIENT_ID"),
    "PLAID_SECRET": os.getenv("PLAID_SECRET"),
    "PLAID_ENV": os.getenv("PLAID_ENV", "sandbox"),
    
    # Stripe Configuration
    "STRIPE_SECRET_KEY": os.getenv("STRIPE_SECRET_KEY"),
    "STRIPE_WEBHOOK_SECRET": os.getenv("STRIPE_WEBHOOK_SECRET"),
    
    # Security Settings
    "SECURITY": {
        "JWT_SECRET": os.getenv("JWT_SECRET"),
        "JWT_ALGORITHM": "HS256",
        "ACCESS_TOKEN_EXPIRE_MINUTES": 30,
        "REFRESH_TOKEN_EXPIRE_DAYS": 7,
        "PASSWORD_HASH_ALGORITHM": "bcrypt",
        "2FA_EXPIRY_MINUTES": 5,
        "MAX_LOGIN_ATTEMPTS": 5,
        "LOCKOUT_MINUTES": 30
    },
    
    # Rate Limiting
    "RATE_LIMITS": {
        "API_REQUESTS_PER_MINUTE": 60,
        "LOGIN_ATTEMPTS_PER_MINUTE": 5,
        "2FA_ATTEMPTS_PER_MINUTE": 3
    },
    
    # Transaction Analysis
    "TRANSACTION_ANALYSIS": {
        "DEFAULT_LOOKBACK_DAYS": 90,
        "MAX_LOOKBACK_DAYS": 365,
        "CATEGORY_MAPPING": {
            "FOOD": ["restaurants", "groceries", "food"],
            "TRANSPORTATION": ["gas", "transit", "parking"],
            "SHOPPING": ["retail", "online", "shopping"],
            "BILLS": ["utilities", "rent", "mortgage"],
            "ENTERTAINMENT": ["entertainment", "recreation", "leisure"],
            "HEALTHCARE": ["medical", "health", "pharmacy"],
            "OTHER": ["other", "uncategorized"]
        }
    },
    
    # Budget Analysis
    "BUDGET_ANALYSIS": {
        "SPENDING_THRESHOLDS": {
            "HIGH": 0.8,  # 80% of budget
            "MEDIUM": 0.6,  # 60% of budget
            "LOW": 0.4  # 40% of budget
        },
        "ALERT_THRESHOLDS": {
            "OVERSPENT": 1.1,  # 110% of budget
            "APPROACHING_LIMIT": 0.9  # 90% of budget
        }
    },
    
    # Investment Analysis
    "INVESTMENT_ANALYSIS": {
        "RISK_LEVELS": {
            "LOW": 0.2,
            "MEDIUM": 0.5,
            "HIGH": 0.8
        },
        "PORTFOLIO_REBALANCE_THRESHOLD": 0.1,  # 10% deviation
        "MINIMUM_INVESTMENT_AMOUNT": 100.0
    },
    
    # Fraud Detection
    "FRAUD_DETECTION": {
        "HIGH_RISK_THRESHOLD": 0.8,
        "MEDIUM_RISK_THRESHOLD": 0.6,
        "LOW_RISK_THRESHOLD": 0.4,
        "RISK_FACTOR_WEIGHTS": {
            "amount": 0.3,
            "location": 0.2,
            "time": 0.2,
            "pattern": 0.3
        },
        "AMOUNT_THRESHOLDS": {
            "UNUSUAL": 2.0,  # 2x average transaction
            "SUSPICIOUS": 5.0  # 5x average transaction
        },
        "LOCATION_RULES": {
            "MAX_DISTANCE_KM": 1000,
            "SUSPICIOUS_COUNTRIES": ["North Korea", "Iran", "Syria"],
            "HIGH_RISK_REGIONS": ["conflict_zones", "sanctioned_areas"]
        },
        "TIME_RULES": {
            "UNUSUAL_HOURS": [0, 1, 2, 3, 4, 5],  # Midnight to 5 AM
            "MAX_TRANSACTIONS_PER_HOUR": 10,
            "MAX_AMOUNT_PER_HOUR": 10000.0
        }
    },
    
    # Loan Analysis
    "LOAN_ANALYSIS": {
        "CREDIT_FACTOR_WEIGHTS": {
            "payment_history": 0.35,
            "credit_utilization": 0.30,
            "account_age": 0.15,
            "diversity": 0.20
        },
        "MINIMUM_CREDIT_SCORE": 0.5,
        "LOAN_TYPES": {
            "PERSONAL": {
                "MIN_AMOUNT": 1000.0,
                "MAX_AMOUNT": 50000.0,
                "MIN_TERM": 12,
                "MAX_TERM": 60
            },
            "HOME_EQUITY": {
                "MIN_AMOUNT": 10000.0,
                "MAX_AMOUNT": 500000.0,
                "MIN_TERM": 36,
                "MAX_TERM": 180
            },
            "BUSINESS": {
                "MIN_AMOUNT": 5000.0,
                "MAX_AMOUNT": 1000000.0,
                "MIN_TERM": 24,
                "MAX_TERM": 120
            }
        },
        "INTEREST_RATES": {
            "EXCELLENT": 0.0599,
            "GOOD": 0.0699,
            "FAIR": 0.0899,
            "POOR": 0.1299
        }
    },
    
    # AI Model Settings
    "AI_MODELS": {
        "TRANSACTION_ANALYSIS": {
            "MODEL": "gpt-4-turbo-preview",
            "MAX_TOKENS": 1000,
            "TEMPERATURE": 0.7
        },
        "BUDGET_RECOMMENDATIONS": {
            "MODEL": "gpt-4-turbo-preview",
            "MAX_TOKENS": 800,
            "TEMPERATURE": 0.7
        },
        "INVESTMENT_INSIGHTS": {
            "MODEL": "gpt-4-turbo-preview",
            "MAX_TOKENS": 1000,
            "TEMPERATURE": 0.7
        },
        "FRAUD_DETECTION": {
            "MODEL": "gpt-4-turbo-preview",
            "MAX_TOKENS": 500,
            "TEMPERATURE": 0.3
        },
        "LOAN_RECOMMENDATIONS": {
            "MODEL": "gpt-4-turbo-preview",
            "MAX_TOKENS": 800,
            "TEMPERATURE": 0.7
        }
    },
    
    # Logging Configuration
    "LOGGING": {
        "LEVEL": "INFO",
        "FILE_PATH": "logs/banking.log",
        "MAX_BYTES": 10485760,  # 10MB
        "BACKUP_COUNT": 5
    },
    
    # Cache Settings
    "CACHE": {
        "ENABLED": True,
        "TTL": 300,  # 5 minutes
        "MAX_SIZE": 1000  # Maximum number of items
    }
}

def get_banking_config() -> Dict[str, Any]:
    """
    Get the banking configuration
    """
    return BANKING_CONFIG 