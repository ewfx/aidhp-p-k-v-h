import asyncio
from services.genai_service import genai_service
from services.data_service import data_service
import pandas as pd

async def test_customer_insights():
    print("Testing customer insights with Excel data...")
    
    # First, verify Excel data is loaded
    if not data_service.data_loaded:
        print("Loading Excel data...")
        data_service.load_data("sample_customer_data.xlsx")
    
    # Get a sample customer ID from the Excel data
    if data_service.customer_profiles is not None:
        sample_customer_id = data_service.customer_profiles['Customer id'].iloc[0]
        print(f"\nTesting with customer ID: {sample_customer_id}")
        
        # Get customer insights
        insights = await genai_service.get_customer_insights(sample_customer_id)
        
        # Print results
        print("\nCustomer Insights:")
        print("-" * 50)
        print(f"Customer ID: {insights['customer_id']}")
        print("\nData Summary:")
        print(f"Profile: {insights['data_summary']['profile']}")
        print(f"Sentiment Analysis: {insights['data_summary']['sentiment_analysis']}")
        print(f"Transaction Analysis: {insights['data_summary']['transaction_analysis']}")
        print("\nAI Generated Insights:")
        print(insights['insights'])
        print("\nRecommendations:")
        print(insights['recommendations'])
    else:
        print("Error: Excel data not loaded properly")

async def test_chatbot():
    print("\nTesting chatbot with Excel data...")
    
    # Get a sample customer ID
    if data_service.customer_profiles is not None:
        sample_customer_id = data_service.customer_profiles['Customer id'].iloc[0]
        print(f"\nTesting with customer ID: {sample_customer_id}")
        
        # Test message
        test_message = "What are my recent spending patterns?"
        
        # Get chatbot response
        response = await genai_service.generate_chatbot_response(
            message=test_message,
            context={"customer_id": sample_customer_id},
            user_data=data_service.get_user_data(sample_customer_id)
        )
        
        print("\nChatbot Response:")
        print("-" * 50)
        print(f"Message: {response['message']}")
        print("\nSuggested Actions:")
        for action in response['suggested_actions']:
            print(f"- {action}")
    else:
        print("Error: Excel data not loaded properly")

async def main():
    print("Starting Gen AI service tests...")
    await test_customer_insights()
    await test_chatbot()
    print("\nTests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 