import pandas as pd
import numpy as np
from datetime import datetime
import os
from transformers import pipeline
import openai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class CustomerAnalyzer:
    def __init__(self, data_file=None):
        """Initialize the analyzer with data file"""
        if data_file is None:
            # Get the absolute path to the backend directory
            backend_dir = Path(__file__).parent.parent
            datasets_dir = backend_dir / 'data' / 'datasets'
            
            # Find all Excel files in the datasets directory
            excel_files = list(datasets_dir.glob('*.xlsx'))
            if not excel_files:
                raise FileNotFoundError(f"No Excel files found in {datasets_dir}")
            
            # Try each Excel file until we find one with valid data
            for excel_file in excel_files:
                try:
                    if self._validate_excel_file(excel_file):
                        data_file = excel_file
                        break
                except Exception as e:
                    print(f"Warning: Could not validate {excel_file}: {str(e)}")
                    continue
            
            if data_file is None:
                raise ValueError("No valid Excel files found with required data structure")
        
        # Load data
        self._load_data(data_file)
        
        # Initialize AI models
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            print("Sentiment analysis model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load sentiment analysis model: {str(e)}")
            self.sentiment_analyzer = None

        # Initialize OpenAI if key is available
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            print("OpenAI API configured successfully")
        else:
            print("Warning: OpenAI API key not found")

    def _validate_excel_file(self, file_path: Path) -> bool:
        """Validate if an Excel file has the required structure"""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Define required columns for each sheet
            required_columns = {
                'Sheet1': ['Customer id', 'age', 'income', 'education', 'occupation', 'interests', 'preferences'],
                'Sheet2': ['Customer id', 'platform', 'content', 'timestamp', 'sentiment_score', 'intent'],
                'Sheet3': ['Customer id', 'purchase_date', 'amount', 'category', 'payment_mode']
            }
            
            # Check each required sheet
            for sheet_name, columns in required_columns.items():
                if sheet_name not in sheet_names:
                    print(f"Warning: Sheet '{sheet_name}' not found in {file_path}")
                    return False
                
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                missing_columns = [col for col in columns if col not in df.columns]
                if missing_columns:
                    print(f"Warning: Sheet '{sheet_name}' is missing required columns: {missing_columns}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validating Excel file: {str(e)}")
            return False

    def _load_data(self, data_file: Path):
        """Load data from Excel file"""
        excel_file = pd.ExcelFile(data_file)
        sheet_names = excel_file.sheet_names
        
        # Map sheet names to data attributes
        sheet_mapping = {
            'Sheet1': 'customer_profiles',
            'Sheet2': 'social_media',
            'Sheet3': 'transactions'
        }
        
        # Load each sheet if it exists
        for sheet_name, attr_name in sheet_mapping.items():
            if sheet_name in sheet_names:
                df = pd.read_excel(data_file, sheet_name=sheet_name)
                setattr(self, attr_name, df)
            else:
                print(f"Warning: Sheet '{sheet_name}' not found in {data_file}")

    async def analyze_customer(self, customer_id: str) -> dict:
        """Perform comprehensive analysis for a single customer"""
        # Get customer data
        customer = self.customer_profiles[
            self.customer_profiles['Customer id'] == customer_id
        ].iloc[0]
        
        # Get customer's social media activity
        social_posts = self.social_media[
            self.social_media['Customer id'] == customer_id
        ]
        
        # Get customer's transactions
        customer_transactions = self.transactions[
            self.transactions['Customer id'] == customer_id
        ]

        # Perform analysis
        profile_analysis = self._analyze_profile(customer)
        sentiment_analysis = self._analyze_social_sentiment(social_posts)
        transaction_analysis = self._analyze_transactions(customer_transactions)
        
        # Generate AI insights if OpenAI is available
        if self.openai_api_key:
            ai_insights = await self._generate_ai_insights(
                customer, social_posts, customer_transactions,
                profile_analysis, sentiment_analysis, transaction_analysis
            )
        else:
            ai_insights = {"error": "OpenAI API key not configured"}

        return {
            "customer_id": customer_id,
            "profile_analysis": profile_analysis,
            "sentiment_analysis": sentiment_analysis,
            "transaction_analysis": transaction_analysis,
            "ai_insights": ai_insights,
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_profile(self, customer: pd.Series) -> dict:
        """Analyze customer profile data"""
        # Age group categorization
        age = customer['age']
        if age < 25:
            age_group = "Young Adult"
        elif age < 40:
            age_group = "Adult"
        elif age < 60:
            age_group = "Middle Aged"
        else:
            age_group = "Senior"

        # Income level categorization
        income = customer['income']
        if income < 40000:
            income_level = "Low"
        elif income < 80000:
            income_level = "Medium"
        elif income < 150000:
            income_level = "High"
        else:
            income_level = "Very High"

        return {
            "age_group": age_group,
            "income_level": income_level,
            "education": customer['education'],
            "occupation": customer['occupation'],
            "interests": customer['interests'].split(', '),
            "banking_preferences": customer['preferences']
        }

    def _analyze_social_sentiment(self, posts: pd.DataFrame) -> dict:
        """Analyze social media sentiment"""
        if posts.empty:
            return {"error": "No social media data available"}

        # Calculate sentiment metrics
        sentiment_scores = posts['sentiment_score']
        sentiment_by_platform = posts.groupby('platform')['sentiment_score'].mean()
        
        # Analyze content using transformer if available
        if self.sentiment_analyzer and not posts.empty:
            try:
                latest_posts = posts.nlargest(5, 'timestamp')
                transformer_sentiments = self.sentiment_analyzer(
                    latest_posts['content'].tolist()
                )
            except Exception as e:
                print(f"Error in transformer analysis: {str(e)}")
                transformer_sentiments = None
        else:
            transformer_sentiments = None

        return {
            "overall_sentiment": {
                "mean": sentiment_scores.mean(),
                "std": sentiment_scores.std(),
                "min": sentiment_scores.min(),
                "max": sentiment_scores.max()
            },
            "sentiment_by_platform": sentiment_by_platform.to_dict(),
            "post_frequency": len(posts),
            "transformer_analysis": transformer_sentiments,
            "intent_distribution": posts['intent'].value_counts().to_dict()
        }

    def _analyze_transactions(self, transactions: pd.DataFrame) -> dict:
        """Analyze transaction patterns"""
        if transactions.empty:
            return {"error": "No transaction data available"}

        # Temporal analysis
        transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
        daily_amounts = transactions.groupby('purchase_date')['amount'].sum()
        
        # Category analysis
        category_stats = transactions.groupby('category').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        
        # Payment mode analysis
        payment_preferences = transactions['payment_mode'].value_counts()

        # Calculate spending trends
        spending_trend = self._calculate_spending_trend(daily_amounts)

        return {
            "total_spent": transactions['amount'].sum(),
            "average_transaction": transactions['amount'].mean(),
            "transaction_count": len(transactions),
            "category_analysis": {
                cat: {
                    "count": stats[('amount', 'count')],
                    "total": stats[('amount', 'sum')],
                    "average": stats[('amount', 'mean')]
                }
                for cat, stats in category_stats.iterrows()
            },
            "payment_preferences": payment_preferences.to_dict(),
            "spending_trend": spending_trend
        }

    def _calculate_spending_trend(self, daily_amounts: pd.Series) -> str:
        """Calculate the spending trend from daily transaction amounts"""
        if len(daily_amounts) < 2:
            return "Insufficient data"

        # Calculate the trend using simple linear regression
        x = np.arange(len(daily_amounts))
        y = daily_amounts.values
        z = np.polyfit(x, y, 1)
        
        # Determine trend direction and magnitude
        slope = z[0]
        if abs(slope) < 0.1:
            return "Stable"
        elif slope > 0:
            return "Increasing" if slope > 1 else "Slightly Increasing"
        else:
            return "Decreasing" if slope < -1 else "Slightly Decreasing"

    async def _generate_ai_insights(
        self, customer: pd.Series, 
        social_posts: pd.DataFrame, 
        transactions: pd.DataFrame,
        profile_analysis: dict,
        sentiment_analysis: dict,
        transaction_analysis: dict
    ) -> dict:
        """Generate AI-powered insights using OpenAI or fallback to rule-based analysis"""
        try:
            # Create OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)

            # Prepare the prompt with comprehensive customer data
            prompt = f"""
            Analyze the following customer data and provide detailed insights:

            Customer Profile:
            - Age: {customer['age']} ({profile_analysis['age_group']})
            - Income: ${customer['income']} ({profile_analysis['income_level']})
            - Education: {profile_analysis['education']}
            - Occupation: {profile_analysis['occupation']}
            - Interests: {', '.join(profile_analysis['interests'])}

            Social Media Analysis:
            - Overall Sentiment: {sentiment_analysis['overall_sentiment']['mean']:.2f}
            - Post Frequency: {sentiment_analysis['post_frequency']}
            - Intent Distribution: {sentiment_analysis['intent_distribution']}

            Transaction Analysis:
            - Total Spent: ${transaction_analysis['total_spent']:,.2f}
            - Average Transaction: ${transaction_analysis['average_transaction']:,.2f}
            - Transaction Count: {transaction_analysis['transaction_count']}
            - Spending Trend: {transaction_analysis['spending_trend']}

            Please provide:
            1. Key customer insights
            2. Risk assessment
            3. Personalized recommendations
            4. Engagement opportunities
            """

            # Make the API call using the new format
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return {
                "insights": response.choices[0].message.content,
                "confidence_score": 0.9,
                "generated_at": datetime.now().isoformat(),
                "source": "openai"
            }

        except Exception as e:
            print(f"Error generating AI insights: {str(e)}")
            # Fallback to rule-based analysis
            return self._generate_rule_based_insights(
                customer, profile_analysis, sentiment_analysis, transaction_analysis
            )

    def _generate_rule_based_insights(
        self, 
        customer: pd.Series,
        profile_analysis: dict,
        sentiment_analysis: dict,
        transaction_analysis: dict
    ) -> dict:
        """Generate insights using rule-based analysis when AI is unavailable"""
        insights = []
        risks = []
        recommendations = []
        opportunities = []

        # Analyze age and income
        age = customer['age']
        income = customer['income']
        
        # Key insights based on profile
        if age < 30 and income > 80000:
            insights.append("Young high-earner with strong financial potential")
        elif age > 50 and income > 100000:
            insights.append("Established professional with significant financial capacity")
        
        if profile_analysis['education'] in ['Master', 'PhD']:
            insights.append("Highly educated customer with potential interest in sophisticated financial products")

        # Risk assessment
        if transaction_analysis['spending_trend'] == "Increasing":
            risks.append("Increasing spending pattern - monitor for sustainability")
        
        avg_sentiment = sentiment_analysis['overall_sentiment']['mean']
        if avg_sentiment < 0.5:
            risks.append("Lower customer satisfaction based on social media sentiment")
        
        if transaction_analysis['average_transaction'] > income * 0.1:
            risks.append("High average transaction amount relative to income")

        # Recommendations based on transaction patterns
        top_categories = sorted(
            transaction_analysis['category_analysis'].items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:3]
        
        for category, stats in top_categories:
            recommendations.append(f"Consider rewards program for {category} purchases")

        if 'Credit Card' not in transaction_analysis['payment_preferences']:
            recommendations.append("Potential for credit card product offering")

        # Engagement opportunities
        interests = profile_analysis['interests']
        if 'Technology' in interests:
            opportunities.append("Digital banking features and mobile app engagement")
        if 'Investment' in interests or income > 100000:
            opportunities.append("Investment advisory services and wealth management")
        if 'Travel' in interests:
            opportunities.append("Travel rewards programs and premium travel cards")

        analysis_text = f"""
Key Customer Insights:
{chr(10).join(f'- {insight}' for insight in insights)}

Risk Assessment:
{chr(10).join(f'- {risk}' for risk in risks)}

Personalized Recommendations:
{chr(10).join(f'- {rec}' for rec in recommendations)}

Engagement Opportunities:
{chr(10).join(f'- {opp}' for opp in opportunities)}
"""

        return {
            "insights": analysis_text,
            "confidence_score": 0.7,
            "generated_at": datetime.now().isoformat(),
            "source": "rule-based"
        }

# Create singleton instance
customer_analyzer = CustomerAnalyzer() 