import re
import numpy as np
from typing import Dict, Any, List, Union
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text: str) -> str:
    """Preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment of text"""
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

def extract_features_from_text(
    text: str,
    include_sentiment: bool = True
) -> Dict[str, Union[str, float]]:
    """Extract features from text data"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    features = {
        "processed_text": processed_text,
        "word_count": len(processed_text.split()),
        "char_count": len(processed_text)
    }
    
    # Add sentiment if requested
    if include_sentiment:
        sentiment = analyze_sentiment(text)
        features.update(sentiment)
    
    return features

def normalize_features(
    features: np.ndarray,
    scaler: StandardScaler = None,
    fit: bool = False
) -> np.ndarray:
    """Normalize numerical features"""
    if scaler is None or fit:
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    return scaler.transform(features)

def encode_categorical(
    categories: List[str],
    encoder: LabelEncoder = None,
    fit: bool = False
) -> np.ndarray:
    """Encode categorical variables"""
    if encoder is None or fit:
        encoder = LabelEncoder()
        return encoder.fit_transform(categories)
    return encoder.transform(categories)

def preprocess_user_data(
    user_data: Dict[str, Any],
    feature_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Preprocess user data for model input"""
    processed_data = {}
    
    # Process text features
    text_features = []
    for feature in feature_config["text_features"]:
        if feature in user_data:
            text_features.append(
                extract_features_from_text(user_data[feature])
            )
    processed_data["text_features"] = text_features
    
    # Process numerical features
    numerical_features = []
    for feature in feature_config["numerical_features"]:
        if feature in user_data:
            numerical_features.append(user_data[feature])
    processed_data["numerical_features"] = normalize_features(
        np.array(numerical_features).reshape(1, -1)
    )
    
    # Process categorical features
    categorical_features = []
    for feature in feature_config["categorical_features"]:
        if feature in user_data:
            categorical_features.append(user_data[feature])
    processed_data["categorical_features"] = encode_categorical(
        categorical_features
    )
    
    return processed_data

def preprocess_item_data(
    item_data: Dict[str, Any],
    feature_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Preprocess item data for model input"""
    processed_data = {}
    
    # Process text features
    text_features = []
    for feature in feature_config["text_features"]:
        if feature in item_data:
            text_features.append(
                extract_features_from_text(item_data[feature])
            )
    processed_data["text_features"] = text_features
    
    # Process numerical features
    numerical_features = []
    for feature in feature_config["numerical_features"]:
        if feature in item_data:
            numerical_features.append(item_data[feature])
    processed_data["numerical_features"] = normalize_features(
        np.array(numerical_features).reshape(1, -1)
    )
    
    # Process categorical features
    categorical_features = []
    for feature in feature_config["categorical_features"]:
        if feature in item_data:
            categorical_features.append(item_data[feature])
    processed_data["categorical_features"] = encode_categorical(
        categorical_features
    )
    
    return processed_data

def extract_temporal_features(
    timestamp: Union[str, datetime]
) -> Dict[str, int]:
    """Extract temporal features from timestamp"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
        
    return {
        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,
        "year": timestamp.year,
        "weekday": timestamp.weekday(),
        "is_weekend": int(timestamp.weekday() >= 5)
    }

def create_interaction_matrix(
    interactions: List[Dict[str, Any]],
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating"
) -> pd.DataFrame:
    """Create user-item interaction matrix"""
    df = pd.DataFrame(interactions)
    return df.pivot(
        index=user_col,
        columns=item_col,
        values=rating_col
    ).fillna(0)

def calculate_similarity(
    vector1: np.ndarray,
    vector2: np.ndarray,
    metric: str = "cosine"
) -> float:
    """Calculate similarity between two vectors"""
    if metric == "cosine":
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
    elif metric == "euclidean":
        return 1 / (1 + np.linalg.norm(vector1 - vector2))
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

def aggregate_features(
    features: List[Dict[str, Any]],
    weights: List[float] = None
) -> Dict[str, Any]:
    """Aggregate multiple feature dictionaries"""
    if weights is None:
        weights = [1.0] * len(features)
        
    aggregated = {}
    for feature_dict, weight in zip(features, weights):
        for key, value in feature_dict.items():
            if key not in aggregated:
                aggregated[key] = 0
            if isinstance(value, (int, float)):
                aggregated[key] += value * weight
                
    # Normalize weights
    total_weight = sum(weights)
    for key in aggregated:
        aggregated[key] /= total_weight
        
    return aggregated

def extract_user_interests(
    text_data: List[str],
    n_topics: int = 5
) -> List[str]:
    """Extract user interests from text data"""
    # Combine all text
    combined_text = " ".join([preprocess_text(text) for text in text_data])
    
    # Get word frequencies
    words = combined_text.split()
    word_freq = pd.Series(words).value_counts()
    
    # Return top topics
    return word_freq.head(n_topics).index.tolist()

def create_feature_vector(
    data: Dict[str, Any],
    feature_config: Dict[str, List[str]]
) -> np.ndarray:
    """Create a feature vector from raw data"""
    features = []
    
    # Process each feature type
    for feature_type, feature_list in feature_config.items():
        for feature in feature_list:
            if feature in data:
                if isinstance(data[feature], (int, float)):
                    features.append(data[feature])
                elif isinstance(data[feature], str):
                    # For categorical features, you might want to use
                    # one-hot encoding or other appropriate encoding
                    pass
                    
    return np.array(features) 