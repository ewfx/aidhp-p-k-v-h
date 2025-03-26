from services.data_service import data_service
import pandas as pd
import numpy as np

def example_usage():
    # 1. Load your dataset
    # Example: Loading a CSV file
    try:
        df = data_service.load_dataset(
            name="transactions",
            file_path="transactions.csv",
            file_type="csv"
        )
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Creating sample dataset...")
        # Create a sample dataset if none exists
        dates = pd.date_range(start='2024-01-01', end='2024-03-24', freq='D')
        transactions = pd.DataFrame({
            'date': dates,
            'amount': np.random.normal(100, 30, len(dates)),
            'category': np.random.choice(['Food', 'Transport', 'Shopping', 'Bills'], len(dates)),
            'merchant': np.random.choice(['Walmart', 'Amazon', 'Netflix', 'Uber', 'Restaurant'], len(dates)),
            'location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], len(dates))
        })
        data_service.save_dataset(
            name="transactions",
            file_path="transactions.csv",
            file_type="csv"
        )
        df = transactions

    # 2. Get dataset information
    info = data_service.get_dataset_info("transactions")
    print("\nDataset Information:")
    print(f"Number of rows: {info['rows']}")
    print(f"Columns: {info['columns']}")
    print(f"Data types: {info['dtypes']}")

    # 3. Preprocess the dataset
    # Example: Fill missing values and convert date column
    preprocessed_df = data_service.preprocess_dataset(
        name="transactions",
        operations=[
            {
                "type": "fill_missing",
                "column": "amount",
                "value": df["amount"].mean()
            },
            {
                "type": "convert_type",
                "column": "date",
                "dtype": "datetime64[ns]"
            }
        ]
    )

    # 4. Save metadata about the dataset
    metadata = {
        "description": "Daily transaction data",
        "source": "Sample data",
        "last_updated": "2024-03-24",
        "columns": {
            "date": "Transaction date",
            "amount": "Transaction amount in USD",
            "category": "Transaction category",
            "merchant": "Merchant name",
            "location": "Transaction location"
        }
    }
    data_service.save_metadata("transactions", metadata)

    # 5. Example analysis using the preprocessed data
    print("\nExample Analysis:")
    print("\nAverage transaction amount by category:")
    print(preprocessed_df.groupby('category')['amount'].mean())
    
    print("\nTransaction count by merchant:")
    print(preprocessed_df['merchant'].value_counts().head())
    
    print("\nDaily transaction volume:")
    daily_volume = preprocessed_df.groupby(preprocessed_df['date'].dt.date)['amount'].sum()
    print(daily_volume.head())

if __name__ == "__main__":
    example_usage() 