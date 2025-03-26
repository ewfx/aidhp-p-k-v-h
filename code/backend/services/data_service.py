import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import os

class DataService:
    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.customer_profiles = None
        self.social_media_sentiment = None
        self.transaction_history = None
        self.data_loaded = False
        
        # Load Excel data by default
        backend_dir = Path(__file__).parent.parent
        datasets_dir = backend_dir / 'data' / 'datasets'
        
        # Create datasets directory if it doesn't exist
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the first Excel file in the datasets directory
        excel_files = list(datasets_dir.glob('*.xlsx'))
        if excel_files:
            print(f"Found Excel file: {excel_files[0]}")
            self.load_data()
        else:
            print(f"Warning: No Excel files found in {datasets_dir}. Using mock data.")
            self._initialize_mock_data()

    def _initialize_mock_data(self):
        """Initialize mock data with proper structure"""
        try:
            print("Initializing mock data...")
            # Create 10 mock customer profiles
            self.customer_profiles = pd.DataFrame({
                'customer_id': [f'CUST{str(i+1).zfill(4)}' for i in range(10)],
                'age': np.random.randint(25, 65, 10),
                'income': np.random.randint(50000, 150000, 10),
                'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 10),
                'occupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Business', 'Artist'], 10)
            })

            # Create social media sentiment data
            sentiment_data = []
            for customer_id in self.customer_profiles['customer_id']:
                num_posts = np.random.randint(3, 8)
                for _ in range(num_posts):
                    sentiment_data.append({
                        'customer_id': customer_id,
                        'platform': np.random.choice(['Twitter', 'Facebook', 'Instagram', 'LinkedIn']),
                        'post_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30)),
                        'content': f"Sample post for {customer_id}",
                        'sentiment_score': np.random.uniform(-1, 1)
                    })
            self.social_media_sentiment = pd.DataFrame(sentiment_data)

            # Create transaction history
            transaction_data = []
            for customer_id in self.customer_profiles['customer_id']:
                num_transactions = np.random.randint(5, 15)
                for _ in range(num_transactions):
                    transaction_data.append({
                        'customer_id': customer_id,
                        'date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 90)),
                        'amount': np.random.uniform(10, 500),
                        'category': np.random.choice(['Food', 'Transport', 'Shopping', 'Entertainment', 'Bills'])
                    })
            self.transaction_history = pd.DataFrame(transaction_data)

            print("Mock data initialized successfully")
            print(f"Number of customers: {len(self.customer_profiles)}")
            print(f"Number of transactions: {len(self.transaction_history)}")
            print(f"Number of sentiment posts: {len(self.social_media_sentiment)}")
            self.data_loaded = True

        except Exception as e:
            print(f"Error initializing mock data: {str(e)}")
            raise

    def load_dataset(self, name: str, file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """Load a dataset from file and store it in memory."""
        full_path = self.data_dir / file_path
        
        if file_type == "csv":
            df = pd.read_csv(full_path)
        elif file_type == "json":
            df = pd.read_json(full_path)
        elif file_type == "excel":
            df = pd.read_excel(full_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        self.datasets[name] = df
        return df

    def save_dataset(self, name: str, file_path: str, file_type: str = "csv"):
        """Save a dataset to file."""
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not found")
        
        full_path = self.data_dir / file_path
        
        if file_type == "csv":
            self.datasets[name].to_csv(full_path, index=False)
        elif file_type == "json":
            self.datasets[name].to_json(full_path, orient='records')
        elif file_type == "excel":
            self.datasets[name].to_excel(full_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get a dataset by name"""
        if dataset_name == "transactions":
            return self.transactions_df
        elif dataset_name == "users":
            return self.users_df
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not found")
        
        df = self.datasets[name]
        return {
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": df.describe().to_dict()
        }

    def preprocess_dataset(self, name: str, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply preprocessing operations to a dataset."""
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not found")
        
        df = self.datasets[name].copy()
        
        for op in operations:
            if op["type"] == "fill_missing":
                df[op["column"]] = df[op["column"]].fillna(op["value"])
            elif op["type"] == "drop_rows":
                df = df.dropna(subset=op["columns"])
            elif op["type"] == "convert_type":
                df[op["column"]] = df[op["column"]].astype(op["dtype"])
            elif op["type"] == "normalize":
                df[op["column"]] = (df[op["column"]] - df[op["column"]].mean()) / df[op["column"]].std()
            elif op["type"] == "encode_categorical":
                df[op["column"]] = pd.Categorical(df[op["column"]]).codes
            else:
                raise ValueError(f"Unsupported operation type: {op['type']}")
        
        return df

    def save_metadata(self, name: str, metadata: Dict[str, Any]):
        """Save metadata for a dataset."""
        self.metadata[name] = metadata
        metadata_path = self.data_dir / f"{name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, name: str) -> Dict[str, Any]:
        """Load metadata for a dataset."""
        if name in self.metadata:
            return self.metadata[name]
        
        metadata_path = self.data_dir / f"{name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata[name] = json.load(f)
            return self.metadata[name]
        
        raise KeyError(f"Metadata for dataset {name} not found")

    def get_user_data(self, user_id: str) -> Optional[Dict]:
        """Get user data by ID"""
        if not self.data_loaded or self.customer_profiles is None:
            return None
            
        user_data = self.customer_profiles[
            self.customer_profiles['Customer id'] == user_id
        ]
        
        if not user_data.empty:
            return user_data.iloc[0].to_dict()
        return None

    def get_user_transactions(self, user_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get transactions for a specific user"""
        if not self.data_loaded or self.transaction_history is None:
            print("Transaction data not loaded")
            return pd.DataFrame()
            
        try:
            print(f"Looking up transactions for customer: {user_id}")
            print(f"Total transactions in history: {len(self.transaction_history)}")
            print(f"Sample transaction dates: {self.transaction_history['date'].head().tolist()}")
            
            transactions = self.transaction_history[
                self.transaction_history['customer_id'] == user_id
            ]
            
            if transactions.empty:
                print(f"No transactions found for customer {user_id}")
                return pd.DataFrame()
                
            if start_date:
                transactions = transactions[transactions['date'] >= start_date]
            if end_date:
                transactions = transactions[transactions['date'] <= end_date]
                
            print(f"Found {len(transactions)} transactions for customer {user_id}")
            print(f"Transaction dates: {transactions['date'].tolist()}")
            return transactions
            
        except Exception as e:
            print(f"Error getting user transactions: {str(e)}")
            return pd.DataFrame()

    def load_data(self):
        """Load data from Excel file"""
        try:
            print("Starting data loading process...")
            excel_path = os.path.join(self.data_dir, "sample_customer_data.xlsx")
            print(f"Looking for Excel file at: {excel_path}")
            
            if not os.path.exists(excel_path):
                print("Excel file not found, initializing mock data...")
                self._initialize_mock_data()
                return

            print("Excel file found, loading data...")
            # Load all sheets
            self.customer_profiles = pd.read_excel(excel_path, sheet_name="Sheet1")
            self.social_media_sentiment = pd.read_excel(excel_path, sheet_name="Sheet2")
            self.transaction_history = pd.read_excel(excel_path, sheet_name="Sheet3")

            # Validate required columns
            required_columns = {
                "customer_profiles": ["customer_id", "age", "income", "education", "occupation"],
                "social_media_sentiment": ["customer_id", "platform", "post_date", "content", "sentiment_score"],
                "transaction_history": ["customer_id", "date", "amount", "category"]
            }

            for df_name, columns in required_columns.items():
                df = getattr(self, df_name)
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    print(f"Warning: Missing required columns in {df_name}: {missing_cols}")
                    self._initialize_mock_data()
                    return

            # Convert date columns to datetime
            try:
                self.transaction_history['date'] = pd.to_datetime(self.transaction_history['date'])
                self.social_media_sentiment['post_date'] = pd.to_datetime(self.social_media_sentiment['post_date'])
            except Exception as e:
                print(f"Error converting dates: {str(e)}")
                self._initialize_mock_data()
                return

            # Validate data types
            try:
                self.customer_profiles['customer_id'] = self.customer_profiles['customer_id'].astype(str)
                self.customer_profiles['age'] = self.customer_profiles['age'].astype(int)
                self.customer_profiles['income'] = self.customer_profiles['income'].astype(float)
                self.transaction_history['amount'] = self.transaction_history['amount'].astype(float)
                self.social_media_sentiment['sentiment_score'] = self.social_media_sentiment['sentiment_score'].astype(float)
            except Exception as e:
                print(f"Error validating data types: {str(e)}")
                self._initialize_mock_data()
                return

            # Ensure customer IDs are in correct format
            self.customer_profiles['customer_id'] = self.customer_profiles['customer_id'].apply(
                lambda x: f"CUST{int(x):04d}" if str(x).isdigit() else x
            )
            self.transaction_history['customer_id'] = self.transaction_history['customer_id'].apply(
                lambda x: f"CUST{int(x):04d}" if str(x).isdigit() else x
            )
            self.social_media_sentiment['customer_id'] = self.social_media_sentiment['customer_id'].apply(
                lambda x: f"CUST{int(x):04d}" if str(x).isdigit() else x
            )

            # Validate customer IDs exist across all datasets
            valid_customers = set(self.customer_profiles['customer_id'])
            invalid_transactions = set(self.transaction_history['customer_id']) - valid_customers
            invalid_sentiment = set(self.social_media_sentiment['customer_id']) - valid_customers

            if invalid_transactions or invalid_sentiment:
                print(f"Warning: Found invalid customer IDs in transactions: {invalid_transactions}")
                print(f"Warning: Found invalid customer IDs in sentiment: {invalid_sentiment}")
                self._initialize_mock_data()
                return

            print(f"Successfully loaded data for {len(valid_customers)} customers")
            print(f"Sample customer IDs: {list(valid_customers)[:5]}")
            self.data_loaded = True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self._initialize_mock_data()

    def get_customer_profile(self, customer_id: str) -> Optional[Dict]:
        """Get complete customer profile including aggregated sentiment and transaction data"""
        if not self.data_loaded:
            print("Data not loaded")
            return None

        try:
            print(f"Looking up profile for customer: {customer_id}")
            print(f"Available customer IDs: {self.customer_profiles['customer_id'].tolist()}")
            
            # Get basic profile
            profile = self.customer_profiles[
                self.customer_profiles['customer_id'] == customer_id
            ]
            
            if profile.empty:
                print(f"No profile found for customer {customer_id}")
                return None
                
            profile = profile.iloc[0].to_dict()
            print(f"Found profile for customer {customer_id}")
            print(f"Profile data: {profile}")
            return profile
            
        except Exception as e:
            print(f"Error getting customer profile: {str(e)}")
            return None

    def get_customer_recommendations(self, customer_id: str) -> Optional[Dict]:
        """Generate recommendations based on customer data"""
        try:
            # Get basic profile data
            profile = self.get_customer_profile(customer_id)
            if not profile:
                print(f"No profile found for customer {customer_id}")
                return None

            # Get transaction data
            transactions = self.get_user_transactions(customer_id)
            if transactions.empty:
                print(f"No transactions found for customer {customer_id}")
                return None

            # Calculate transaction metrics
            transaction_metrics = {
                'total_spent': transactions['amount'].sum(),
                'avg_transaction': transactions['amount'].mean(),
                'category_breakdown': transactions.groupby('category')['amount'].sum().to_dict(),
                'recent_transactions': transactions.nlargest(5, 'date')[
                    ['category', 'amount', 'date']
                ].to_dict('records')
            }

            # Get sentiment data if available
            sentiment_metrics = None
            if self.social_media_sentiment is not None:
                customer_sentiment = self.social_media_sentiment[
                    self.social_media_sentiment['customer_id'] == customer_id
                ]
                if not customer_sentiment.empty:
                    sentiment_metrics = {
                        'avg_sentiment_score': customer_sentiment['sentiment_score'].mean(),
                        'recent_sentiments': customer_sentiment.nlargest(5, 'post_date')[
                            ['platform', 'content', 'sentiment_score']
                        ].to_dict('records'),
                        'sentiment_by_platform': customer_sentiment.groupby('platform')['sentiment_score'].mean().to_dict()
                    }

            # Generate recommendations
            recommendations = self._generate_personalized_recommendations(
                profile, 
                sentiment_metrics, 
                transaction_metrics
            )

            return {
                'customer_id': customer_id,
                'profile': profile,
                'transaction_metrics': transaction_metrics,
                'sentiment_metrics': sentiment_metrics,
                'recommendations': recommendations
            }

        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return None

    def _generate_personalized_recommendations(
        self, 
        profile: Dict, 
        sentiment: Optional[Dict], 
        transactions: Dict
    ) -> List[Dict]:
        """Generate personalized recommendations based on all available data"""
        recommendations = []

        # Product recommendations based on transaction history
        for category, amount in transactions['category_breakdown'].items():
            if amount > transactions['avg_transaction']:
                recommendations.append({
                    'type': 'product',
                    'category': category,
                    'reason': f'Based on your spending history in {category}',
                    'confidence': 0.8
                })

        # Recommendations based on social media sentiment if available
        if sentiment:
            positive_sentiments = [
                s for s in sentiment['recent_sentiments'] 
                if s['sentiment_score'] > 0.7
            ]
            for sent in positive_sentiments:
                recommendations.append({
                    'type': 'social',
                    'category': sent['platform'],
                    'reason': f'Based on your positive engagement on {sent["platform"]}',
                    'confidence': sent['sentiment_score']
                })

        # Lifestyle recommendations based on profile
        if profile.get('occupation'):
            recommendations.append({
                'type': 'lifestyle',
                'category': profile['occupation'],
                'reason': f'Based on your profession as {profile["occupation"]}',
                'confidence': 0.9
            })

        return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:5]

# Create a singleton instance
data_service = DataService() 