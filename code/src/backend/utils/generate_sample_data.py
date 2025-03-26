import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_customer_profiles(num_customers=100):
    """Generate customer profile data"""
    # Lists for generating realistic data
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']
    interests = ['Technology', 'Travel', 'Fashion', 'Sports', 'Food', 'Music', 'Art', 'Reading', 'Fitness', 'Photography']
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD', 'Associate']
    occupations = ['Software Engineer', 'Teacher', 'Doctor', 'Sales Manager', 'Financial Analyst', 
                  'Marketing Manager', 'Business Owner', 'Lawyer', 'Accountant', 'Consultant']

    data = []
    for i in range(num_customers):
        # Generate random customer data
        customer_id = f'CUST{str(i+1).zfill(4)}'
        age = random.randint(18, 75)
        gender = random.choice(['Male', 'Female', 'Other'])
        location = random.choice(locations)
        
        # Generate 2-4 random interests
        num_interests = random.randint(2, 4)
        customer_interests = ', '.join(random.sample(interests, num_interests))
        
        # Generate preferences based on interests
        preferences = f"Prefers {random.choice(['online', 'in-person'])} banking, "
        preferences += f"Interested in {random.choice(['savings', 'investments', 'loans', 'credit cards'])}"
        
        income = random.randint(30000, 200000)
        education = random.choice(education_levels)
        occupation = random.choice(occupations)

        data.append({
            'Customer id': customer_id,
            'age': age,
            'gender': gender,
            'location': location,
            'interests': customer_interests,
            'preferences': preferences,
            'income': income,
            'education': education,
            'occupation': occupation
        })

    return pd.DataFrame(data)

def generate_social_media_sentiment(customer_profiles, num_posts_per_customer=10):
    """Generate social media sentiment data"""
    platforms = ['Twitter', 'Facebook', 'Instagram', 'LinkedIn']
    
    # Sample content templates for different intents
    content_templates = {
        'product_inquiry': [
            "Interested in {product} options from {bank}",
            "Looking for information about {bank}'s {product}",
            "Can anyone recommend {bank}'s {product}?"
        ],
        'service_feedback': [
            "Great experience with {bank}'s customer service!",
            "Had issues with {bank}'s online banking today",
            "Really impressed with {bank}'s mobile app"
        ],
        'general_mention': [
            "Just opened a new account with {bank}",
            "Banking with {bank} has been great so far",
            "Considering switching to {bank}"
        ]
    }

    products = ['savings account', 'credit card', 'mortgage', 'personal loan', 'investment account']
    bank_name = "Our Bank"

    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    for _, customer in customer_profiles.iterrows():
        for _ in range(num_posts_per_customer):
            # Generate random post data
            intent = random.choice(list(content_templates.keys()))
            content_template = random.choice(content_templates[intent])
            content = content_template.format(
                bank=bank_name,
                product=random.choice(products)
            )

            # Generate sentiment score (biased towards positive)
            base_sentiment = random.uniform(0, 1)
            if 'Great' in content or 'impressed' in content:
                sentiment_score = random.uniform(0.7, 1.0)
            elif 'issues' in content or 'problem' in content:
                sentiment_score = random.uniform(0, 0.3)
            else:
                sentiment_score = base_sentiment

            data.append({
                'Customer id': customer['Customer id'],
                'post_id': f"POST{len(data)+1}",
                'platform': random.choice(platforms),
                'content': content,
                'timestamp': start_date + timedelta(
                    seconds=random.randint(0, int((end_date - start_date).total_seconds()))
                ),
                'sentiment_score': sentiment_score,
                'intent': intent
            })

    return pd.DataFrame(data)

def generate_transaction_history(customer_profiles, num_transactions_per_customer=50):
    """Generate transaction history data"""
    categories = {
        'Groceries': ('Walmart', 'Whole Foods', 'Kroger', 'Trader Joes'),
        'Electronics': ('Best Buy', 'Apple Store', 'Amazon', 'NewEgg'),
        'Entertainment': ('Netflix', 'AMC Movies', 'Spotify', 'Steam'),
        'Travel': ('United Airlines', 'Marriott', 'Expedia', 'Airbnb'),
        'Dining': ('McDonalds', 'Starbucks', 'Chipotle', 'Local Restaurant'),
        'Shopping': ('Target', 'Amazon', 'Macys', 'Nike'),
        'Utilities': ('Electric Company', 'Water Service', 'Gas Company', 'Internet Provider')
    }
    
    payment_modes = ['Credit Card', 'Debit Card', 'Net Banking', 'UPI', 'Mobile Wallet']

    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    for _, customer in customer_profiles.iterrows():
        # Adjust number of transactions based on income
        num_transactions = int(num_transactions_per_customer * (1 + customer['income']/100000))
        
        for _ in range(num_transactions):
            # Select random category and merchant
            category = random.choice(list(categories.keys()))
            merchant = random.choice(categories[category])
            
            # Generate amount based on category and income
            if category in ['Electronics', 'Travel']:
                amount = random.uniform(100, 2000)
            elif category in ['Groceries', 'Shopping']:
                amount = random.uniform(20, 500)
            else:
                amount = random.uniform(10, 200)
            
            # Adjust amount based on income
            amount *= (1 + customer['income']/100000)

            data.append({
                'Customer id': customer['Customer id'],
                'product_id': f"PROD{len(data)+1}",
                'Transaction type': 'Purchase',
                'category': category,
                'amount': round(amount, 2),
                'purchase_date': start_date + timedelta(
                    seconds=random.randint(0, int((end_date - start_date).total_seconds()))
                ),
                'payment_mode': random.choice(payment_modes)
            })

    return pd.DataFrame(data)

def generate_sample_dataset(output_file='sample_customer_data.xlsx'):
    """Generate complete sample dataset and save to Excel"""
    print("Generating customer profiles...")
    customer_profiles = generate_customer_profiles()
    
    print("Generating social media sentiment data...")
    social_media_sentiment = generate_social_media_sentiment(customer_profiles)
    
    print("Generating transaction history...")
    transaction_history = generate_transaction_history(customer_profiles)
    
    print("Saving data to Excel file...")
    with pd.ExcelWriter(output_file) as writer:
        customer_profiles.to_excel(writer, sheet_name='Sheet1', index=False)
        social_media_sentiment.to_excel(writer, sheet_name='Sheet2', index=False)
        transaction_history.to_excel(writer, sheet_name='Sheet3', index=False)
    
    print(f"Sample dataset has been generated and saved to {output_file}")
    print(f"Total customers: {len(customer_profiles)}")
    print(f"Total social media posts: {len(social_media_sentiment)}")
    print(f"Total transactions: {len(transaction_history)}")

if __name__ == "__main__":
    generate_sample_dataset() 