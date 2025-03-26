import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
import json

from src.config.config import MODEL_CONFIG
from src.utils.data_processing import preprocess_text

logger = logging.getLogger(__name__)

class MultiModalEncoder(nn.Module):
    """Encoder for different types of user data"""
    
    def __init__(
        self,
        text_embedding_dim: int,
        numerical_dim: int,
        categorical_dim: int,
        output_dim: int = MODEL_CONFIG["embedding_dim"]
    ):
        super().__init__()
        
        # Text encoder (using pre-trained transformer)
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_projection = nn.Linear(768, text_embedding_dim)
        
        # Numerical features encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_dim, text_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Categorical features encoder
        self.categorical_encoder = nn.Sequential(
            nn.Linear(categorical_dim, text_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(text_embedding_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def forward(
        self,
        text_data: Dict[str, torch.Tensor],
        numerical_features: torch.Tensor,
        categorical_features: torch.Tensor
    ) -> torch.Tensor:
        # Encode text
        text_output = self.text_encoder(**text_data)
        text_embedding = self.text_projection(text_output.pooler_output)
        
        # Encode numerical features
        numerical_embedding = self.numerical_encoder(numerical_features)
        
        # Encode categorical features
        categorical_embedding = self.categorical_encoder(categorical_features)
        
        # Combine all embeddings
        combined = torch.cat(
            [text_embedding, numerical_embedding, categorical_embedding],
            dim=1
        )
        
        # Final embedding
        return self.fusion(combined)

class UserEmbeddingModel:
    """Model for learning user embeddings from multiple data sources"""
    
    def __init__(self):
        self.model = None
        self.numerical_scaler = StandardScaler()
        self.embeddings_cache = {}
        self.feature_processors = {}
        
    async def train(
        self,
        training_data: Dict[str, Any],
        feature_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Train the user embedding model"""
        try:
            # Initialize model if not exists
            if self.model is None:
                self.model = MultiModalEncoder(
                    text_embedding_dim=MODEL_CONFIG["embedding_dim"],
                    numerical_dim=len(training_data["numerical_features"][0]),
                    categorical_dim=len(training_data["categorical_features"][0])
                )
            
            # Prepare training data
            text_data = self._prepare_text_data(training_data["text_data"])
            numerical_features = torch.tensor(
                self.numerical_scaler.fit_transform(
                    training_data["numerical_features"]
                )
            )
            categorical_features = torch.tensor(
                training_data["categorical_features"]
            )
            
            # Training loop
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=MODEL_CONFIG["learning_rate"]
            )
            
            for epoch in range(MODEL_CONFIG["epochs"]):
                self.model.train()
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = self.model(
                    text_data,
                    numerical_features,
                    categorical_features
                )
                
                # Calculate loss (e.g., contrastive loss, reconstruction loss)
                loss = self._calculate_loss(embeddings, training_data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                logger.info(f"Epoch {epoch + 1}/{MODEL_CONFIG['epochs']}, Loss: {loss.item():.4f}")
                
            # Update embeddings cache
            self._update_embeddings_cache(training_data["user_ids"], embeddings)
            
        except Exception as e:
            logger.error(f"Error training user embedding model: {str(e)}")
            raise
            
    async def get_user_embedding(
        self,
        user_data: Dict[str, Any],
        use_cache: bool = True
    ) -> np.ndarray:
        """Get embedding for a user"""
        try:
            user_id = user_data.get("user_id")
            
            # Return cached embedding if available and requested
            if use_cache and user_id in self.embeddings_cache:
                return self.embeddings_cache[user_id]
            
            # Prepare features
            text_data = self._prepare_text_data([user_data["text_data"]])
            numerical_features = torch.tensor(
                self.numerical_scaler.transform([user_data["numerical_features"]])
            )
            categorical_features = torch.tensor(
                [user_data["categorical_features"]]
            )
            
            # Get embedding
            self.model.eval()
            with torch.no_grad():
                embedding = self.model(
                    text_data,
                    numerical_features,
                    categorical_features
                )
                
            # Cache embedding if user_id is provided
            if user_id:
                self.embeddings_cache[user_id] = embedding.numpy()
                
            return embedding.numpy()
            
        except Exception as e:
            logger.error(f"Error getting user embedding: {str(e)}")
            raise
            
    def _prepare_text_data(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare text data for the model"""
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Tokenize
        tokenized = self.model.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return tokenized
        
    def _calculate_loss(
        self,
        embeddings: torch.Tensor,
        training_data: Dict[str, Any]
    ) -> torch.Tensor:
        """Calculate loss for the embedding model"""
        # Implement appropriate loss function
        # This could be contrastive loss, reconstruction loss, etc.
        # depending on your specific requirements
        pass
        
    def _update_embeddings_cache(
        self,
        user_ids: List[str],
        embeddings: torch.Tensor
    ) -> None:
        """Update the embeddings cache"""
        embeddings_np = embeddings.detach().numpy()
        for i, user_id in enumerate(user_ids):
            self.embeddings_cache[user_id] = embeddings_np[i]
            
    async def analyze_user_preferences(
        self,
        user_embedding: np.ndarray,
        item_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze user preferences based on their embedding"""
        try:
            # Extract key preferences and interests
            preferences = {
                "categories": self._extract_category_preferences(
                    user_embedding,
                    item_features
                ),
                "price_sensitivity": self._calculate_price_sensitivity(
                    user_embedding
                ),
                "brand_affinity": self._calculate_brand_affinity(
                    user_embedding,
                    item_features
                ),
                "trending_score": self._calculate_trending_score(
                    user_embedding
                )
            }
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error analyzing user preferences: {str(e)}")
            raise
            
    def _extract_category_preferences(
        self,
        user_embedding: np.ndarray,
        item_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract category preferences from user embedding"""
        # Implementation depends on your category structure
        pass
        
    def _calculate_price_sensitivity(
        self,
        user_embedding: np.ndarray
    ) -> float:
        """Calculate user's price sensitivity"""
        # Implementation depends on your pricing model
        pass
        
    def _calculate_brand_affinity(
        self,
        user_embedding: np.ndarray,
        item_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate user's brand preferences"""
        # Implementation depends on your brand structure
        pass
        
    def _calculate_trending_score(
        self,
        user_embedding: np.ndarray
    ) -> float:
        """Calculate user's affinity for trending items"""
        # Implementation depends on your trending metrics
        pass 