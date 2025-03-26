import openai
from typing import Dict, Any, Optional, List
from datetime import datetime
from .data_service import data_service
from .genai_service import genai_service

class ChatbotService:
    def __init__(self):
        self.openai_api_key = genai_service.openai_api_key
        self.use_openai = bool(self.openai_api_key)
        if self.use_openai:
            self.client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            print("Warning: OpenAI API key not found. Using basic analysis mode.")

    async def process_message(self, message: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user message and return a response"""
        try:
            # First, try to understand the user's intent and extract any customer ID
            intent_analysis = await self._analyze_intent(message)
            
            # If no customer ID was provided in the message but we have one from context
            if not intent_analysis.get('customer_id') and customer_id:
                intent_analysis['customer_id'] = customer_id
            
            # If the message is just a customer ID, treat it as a request for customer analysis
            if self._is_customer_id_only(message):
                intent_analysis['intent'] = 'get_customer_analysis'
                intent_analysis['customer_id'] = message.upper()
            
            # Validate customer ID format if present
            if intent_analysis.get('customer_id'):
                if not self._validate_customer_id(intent_analysis['customer_id']):
                    return {
                        "error": "Invalid customer ID format",
                        "message": "Please provide a valid customer ID in the format CUST#### (e.g., CUST0001)",
                        "timestamp": datetime.now().isoformat()
                    }

            # Process the message based on intent
            if intent_analysis['intent'] == 'get_customer_analysis':
                return await self._handle_customer_analysis(intent_analysis)
            elif intent_analysis['intent'] == 'get_recommendations':
                return await self._handle_recommendations(intent_analysis)
            elif intent_analysis['intent'] == 'get_transaction_summary':
                return await self._handle_transaction_summary(intent_analysis)
            elif intent_analysis['intent'] == 'get_sentiment_analysis':
                return await self._handle_sentiment_analysis(intent_analysis)
            else:
                return await self._handle_general_query(message)

        except Exception as e:
            error_message = str(e)
            if "insufficient_quota" in error_message or "429" in error_message:
                return {
                    "error": "OpenAI API quota exceeded",
                    "message": "I apologize, but I'm currently experiencing high demand. Let me provide you with basic information instead:\n\n" + 
                             self._get_fallback_response(message, customer_id),
                    "timestamp": datetime.now().isoformat()
                }
            return {
                "error": str(e),
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

    def _is_customer_id_only(self, message: str) -> bool:
        """Check if the message is just a customer ID"""
        message = message.strip().upper()
        return bool(message.startswith('CUST') and len(message) == 8 and message[4:].isdigit())

    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze the user's message to determine intent and extract relevant information"""
        try:
            if not self.use_openai:
                return self._basic_intent_analysis(message)

            # Create OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)

            # Prepare the prompt for intent analysis
            prompt = f"""
            Analyze the following user message and extract:
            1. The user's intent (get_customer_analysis, get_recommendations, get_transaction_summary, get_sentiment_analysis, or general_query)
            2. Any customer ID mentioned (format: CUST####)
            3. Any specific time period mentioned
            4. Any specific categories or topics mentioned

            Message: {message}

            Respond in JSON format with the following structure:
            {{
                "intent": "intent_type",
                "customer_id": "customer_id or null",
                "time_period": "time_period or null",
                "categories": ["category1", "category2", ...]
            }}
            """

            # Make the API call
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an intent analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )

            # Parse the response
            intent_analysis = eval(response.choices[0].message.content)
            return intent_analysis

        except Exception as e:
            print(f"Error analyzing intent: {str(e)}")
            return self._basic_intent_analysis(message)

    def _basic_intent_analysis(self, message: str) -> Dict[str, Any]:
        """Basic intent analysis without OpenAI"""
        message = message.lower()
        
        # Extract customer ID if present
        customer_id = None
        if "cust" in message:
            try:
                # Find the position of 'cust'
                cust_index = message.find("cust")
                # Extract the next 4 digits after 'cust'
                potential_id = message[cust_index:cust_index+8].upper()
                # Validate the format (CUST followed by 4 digits)
                if len(potential_id) == 8 and potential_id.startswith("CUST") and potential_id[4:].isdigit():
                    customer_id = potential_id
            except:
                pass

        # Determine intent based on keywords
        if any(word in message for word in ["analysis", "analyze", "insights", "profile", "info", "details"]):
            intent = "get_customer_analysis"
        elif any(word in message for word in ["recommend", "suggest", "offer", "recommendation"]):
            intent = "get_recommendations"
        elif any(word in message for word in ["transaction", "spending", "purchase", "history"]):
            intent = "get_transaction_summary"
        elif any(word in message for word in ["sentiment", "feel", "opinion", "feedback"]):
            intent = "get_sentiment_analysis"
        else:
            intent = "general_query"

        return {
            "intent": intent,
            "customer_id": customer_id,
            "time_period": None,
            "categories": []
        }

    def _get_fallback_response(self, message: str, customer_id: Optional[str] = None) -> str:
        """Provide a fallback response when OpenAI API is unavailable"""
        try:
            # Try to extract customer ID from message if not provided
            if not customer_id:
                # First try to find a customer ID in the message
                message_upper = message.upper()
                if "CUST" in message_upper:
                    try:
                        # Find the position of CUST
                        cust_index = message_upper.find("CUST")
                        # Extract the next 4 digits
                        potential_id = message_upper[cust_index:cust_index+8]
                        # Validate the format
                        if len(potential_id) == 8 and potential_id.startswith("CUST") and potential_id[4:].isdigit():
                            customer_id = potential_id
                    except:
                        customer_id = None

            if customer_id:
                # Ensure customer ID is in correct format
                customer_id = customer_id.upper()
                if not customer_id.startswith("CUST"):
                    customer_id = f"CUST{customer_id}"
                
                # Try to get basic customer data
                customer_data = data_service.get_customer_profile(customer_id)
                if customer_data:
                    # Format income value safely
                    income = customer_data.get('income')
                    income_str = f"${float(income):,.2f}" if income and str(income).replace('.', '').isdigit() else 'N/A'

                    return f"""
Basic information for customer {customer_id}:

Profile:
- Age: {customer_data.get('age', 'N/A')}
- Income: {income_str}
- Education: {customer_data.get('education', 'N/A')}
- Occupation: {customer_data.get('occupation', 'N/A')}

Recent Transactions:
{self._get_recent_transactions(customer_id)}

Would you like to know more about:
1. Their spending patterns
2. Personalized recommendations
3. Social media sentiment

Just ask for any of these details!
"""
                else:
                    return f"I couldn't find any data for customer {customer_id}. Please verify the customer ID and try again."
            else:
                return "I can help you with:\n1. Customer analysis (provide a customer ID)\n2. Transaction summaries\n3. Basic customer information\n4. Personalized recommendations\n5. Sentiment analysis\n\nPlease include a customer ID (e.g., CUST0001) in your query for specific information."
        except Exception as e:
            print(f"Error in fallback response: {str(e)}")
            return "I apologize, but I'm having trouble accessing the customer data. Please try again later."

    def _get_recent_transactions(self, customer_id: str) -> str:
        """Get recent transactions for a customer"""
        try:
            transactions = data_service.get_user_transactions(customer_id)
            if not transactions.empty:
                recent_transactions = transactions.tail(5)
                return "\n".join([
                    f"- {row['date']}: ${row['amount']:,.2f} ({row['category']})"
                    for _, row in recent_transactions.iterrows()
                ])
            return "No recent transactions found."
        except Exception:
            return "Transaction data unavailable."

    def _validate_customer_id(self, customer_id: str) -> bool:
        """Validate customer ID format"""
        customer_id = customer_id.upper()
        return bool(
            customer_id.startswith('CUST') and 
            len(customer_id) == 8 and 
            customer_id[4:].isdigit()
        )

    def _format_customer_id(self, customer_id: str) -> str:
        """Format customer ID to standard format"""
        customer_id = customer_id.upper()
        if not customer_id.startswith('CUST'):
            customer_id = f"CUST{customer_id}"
        if len(customer_id) < 8:
            customer_id = f"{customer_id:0>8}"
        return customer_id

    async def _handle_customer_analysis(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for customer analysis"""
        customer_id = intent_analysis.get('customer_id')
        if not customer_id:
            return {
                "error": "No customer ID provided",
                "message": "Please provide a customer ID to get their analysis. For example:\n- Show me CUST0001\n- Tell me about customer CUST0001\n- What's the profile of CUST0001",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Format customer ID
            customer_id = self._format_customer_id(customer_id)
            
            # Get customer profile data
            customer_data = data_service.get_customer_profile(customer_id)
            if not customer_data:
                return {
                    "error": "Customer not found",
                    "message": f"I couldn't find any data for customer {customer_id}. Please verify the customer ID and try again.\n\nAvailable customer IDs are: CUST0001 through CUST0010",
                    "timestamp": datetime.now().isoformat()
                }

            # Get customer insights
            insights = await genai_service.get_customer_insights(customer_id)
            
            # Format income value safely
            income = customer_data.get('income')
            income_str = f"${float(income):,.2f}" if income and str(income).replace('.', '').isdigit() else 'N/A'

            # Format the response in a conversational way
            response = f"""
Here's what I found for customer {customer_id}:

Profile Overview:
- Age: {customer_data.get('age', 'N/A')}
- Income: {income_str}
- Education: {customer_data.get('education', 'N/A')}
- Occupation: {customer_data.get('occupation', 'N/A')}

Key Insights:
{insights.get('insights', 'No specific insights available at this time.')}

Recent Transactions:
{self._get_recent_transactions(customer_id)}

Would you like to know more about:
1. Their spending patterns
2. Personalized recommendations
3. Social media sentiment

Just ask for any of these details!
"""

            return {
                "message": response,
                "timestamp": datetime.now().isoformat(),
                "customer_id": customer_id
            }

        except Exception as e:
            print(f"Error in customer analysis: {str(e)}")
            return {
                "error": str(e),
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_recommendations(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for customer recommendations"""
        customer_id = intent_analysis.get('customer_id')
        if not customer_id:
            return {
                "error": "No customer ID provided",
                "message": "Please provide a customer ID to get their recommendations.",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Get recommendations from data service
            recommendations_data = data_service.get_customer_recommendations(customer_id)
            if not recommendations_data:
                return {
                    "error": "Customer not found",
                    "message": f"I couldn't find any data for customer {customer_id}.",
                    "timestamp": datetime.now().isoformat()
                }

            # Format the response in a conversational way
            response = f"""
Here are personalized recommendations for customer {customer_id}:

Profile Overview:
- Age: {recommendations_data['profile'].get('age', 'N/A')}
- Income: ${recommendations_data['profile'].get('income', 'N/A'):,.2f}
- Education: {recommendations_data['profile'].get('education', 'N/A')}
- Occupation: {recommendations_data['profile'].get('occupation', 'N/A')}

Spending Summary:
- Total Spent: ${recommendations_data['transaction_metrics']['total_spent']:,.2f}
- Average Transaction: ${recommendations_data['transaction_metrics']['avg_transaction']:,.2f}

Top Spending Categories:
{self._format_category_breakdown(recommendations_data['transaction_metrics']['category_breakdown'])}

Personalized Recommendations:
{self._format_recommendations(recommendations_data['recommendations'])}

Would you like to know more about:
1. Their spending patterns
2. Social media sentiment
3. Detailed transaction history

Just ask for any of these details!
"""

            return {
                "message": response,
                "timestamp": datetime.now().isoformat(),
                "customer_id": customer_id
            }

        except Exception as e:
            print(f"Error handling recommendations: {str(e)}")
            return {
                "error": str(e),
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

    def _format_category_breakdown(self, category_breakdown: Dict[str, float]) -> str:
        """Format category breakdown for display"""
        sorted_categories = sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([
            f"- {category}: ${amount:,.2f}"
            for category, amount in sorted_categories[:5]
        ])

    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """Format recommendations for display"""
        if not recommendations:
            return "No specific recommendations available at this time."
            
        formatted_recs = []
        for i, rec in enumerate(recommendations, 1):
            formatted_recs.append(f"{i}. {rec['reason']} (Confidence: {rec['confidence']:.1%})")
        return "\n".join(formatted_recs)

    async def _handle_transaction_summary(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for transaction summaries"""
        customer_id = intent_analysis.get('customer_id')
        if not customer_id:
            return {
                "error": "No customer ID provided",
                "message": "Please provide a customer ID to get their transaction summary.",
                "timestamp": datetime.now().isoformat()
            }

        # Get customer transactions
        transactions = data_service.get_user_transactions(customer_id)
        if transactions.empty:
            return {
                "error": "No transactions found",
                "message": f"I couldn't find any transactions for customer {customer_id}.",
                "timestamp": datetime.now().isoformat()
            }

        # Calculate summary statistics
        total_spent = transactions['amount'].sum()
        avg_transaction = transactions['amount'].mean()
        category_breakdown = transactions.groupby('category')['amount'].sum()

        # Format the response
        response = f"""
Here's the transaction summary for customer {customer_id}:

Total Spent: ${total_spent:,.2f}
Average Transaction: ${avg_transaction:,.2f}
Number of Transactions: {len(transactions)}

Spending by Category:
{chr(10).join(f"- {cat}: ${amount:,.2f}" for cat, amount in category_breakdown.items())}
"""

        return {
            "message": response,
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id
        }

    async def _handle_sentiment_analysis(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for sentiment analysis"""
        customer_id = intent_analysis.get('customer_id')
        if not customer_id:
            return {
                "error": "No customer ID provided",
                "message": "Please provide a customer ID to get their sentiment analysis. For example:\n- Show me sentiment for CUST0001\n- What's the sentiment analysis for CUST0001",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Format customer ID
            customer_id = self._format_customer_id(customer_id)
            
            # Get customer profile first to validate customer exists
            customer_data = data_service.get_customer_profile(customer_id)
            if not customer_data:
                return {
                    "error": "Customer not found",
                    "message": f"I couldn't find any data for customer {customer_id}. Please verify the customer ID and try again.\n\nAvailable customer IDs are: CUST0001 through CUST0010",
                    "timestamp": datetime.now().isoformat()
                }

            # Get sentiment data
            sentiment_data = data_service.get_customer_recommendations(customer_id)
            if not sentiment_data or not sentiment_data.get('sentiment_metrics'):
                return {
                    "error": "No sentiment data available",
                    "message": f"I couldn't find any sentiment data for customer {customer_id}. This might be because:\n1. The customer hasn't posted on social media yet\n2. The sentiment data is still being collected\n3. There was an error retrieving the sentiment data",
                    "timestamp": datetime.now().isoformat()
                }

            sentiment_metrics = sentiment_data['sentiment_metrics']
            
            # Format the response
            response = f"""
Sentiment Analysis for {customer_id}:

Overall Sentiment Score: {sentiment_metrics['avg_sentiment_score']:.2f}

Platform Breakdown:
{self._format_sentiment_by_platform(sentiment_metrics['sentiment_by_platform'])}

Recent Sentiment Activity:
{self._format_recent_sentiments(sentiment_metrics['recent_sentiments'])}

Would you like to know more about:
1. Their transaction patterns
2. Personalized recommendations
3. Complete customer profile

Just ask for any of these details!
"""

            return {
                "message": response,
                "timestamp": datetime.now().isoformat(),
                "customer_id": customer_id
            }

        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                "error": str(e),
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

    def _format_sentiment_by_platform(self, sentiment_by_platform: Dict[str, float]) -> str:
        """Format sentiment scores by platform"""
        if not sentiment_by_platform:
            return "No platform-specific sentiment data available."
            
        formatted = []
        for platform, score in sentiment_by_platform.items():
            sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
            formatted.append(f"- {platform}: {sentiment} ({score:.2f})")
        return "\n".join(formatted)

    def _format_recent_sentiments(self, recent_sentiments: List[Dict]) -> str:
        """Format recent sentiment activities"""
        if not recent_sentiments:
            return "No recent sentiment activities found."
            
        formatted = []
        for sent in recent_sentiments:
            sentiment = "Positive" if sent['sentiment_score'] > 0 else "Negative" if sent['sentiment_score'] < 0 else "Neutral"
            formatted.append(f"- {sent['platform']}: {sentiment} ({sent['sentiment_score']:.2f})")
        return "\n".join(formatted)

    async def _handle_general_query(self, message: str) -> Dict[str, Any]:
        """Handle general queries using OpenAI"""
        try:
            # Create OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)

            # Prepare the prompt
            prompt = f"""
            You are a helpful banking assistant. The user has asked: {message}
            
            Please provide a helpful response that:
            1. Addresses their question directly
            2. Suggests related queries they might be interested in
            3. Maintains a friendly, conversational tone
            """

            # Make the API call
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful banking assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )

            return {
                "message": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "error": str(e),
                "message": "I apologize, but I'm having trouble processing your request. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

# Create singleton instance
chatbot_service = ChatbotService() 