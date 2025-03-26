import numpy as np
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

from src.config.config import MODEL_CONFIG, RECOMMENDATION_CONFIG
from src.utils.data_processing import normalize_features

logger = logging.getLogger(__name__)

class HybridRecommender(nn.Module):
    """Neural network-based hybrid recommender system"""
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = MODEL_CONFIG["embedding_dim"],
        n_factors: int = MODEL_CONFIG["n_factors"]
    ):
        super().__init__()
        
        # User and item embeddings for collaborative filtering
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Content-based feature transformation
        self.content_transform = nn.Sequential(
            nn.Linear(n_factors, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        
        # Transform content features
        content_emb = self.content_transform(item_features)
        
        # Concatenate all features
        combined = torch.cat([user_emb, item_emb, content_emb], dim=1)
        
        # Get predictions
        return self.predictor(combined)

class RecommendationEngine:
    """Main recommendation engine that combines different recommendation approaches"""
    
    def __init__(self):
        self.model = None
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.item_features = {}
        self.user_interaction_history = {}
        self.cold_start_threshold = RECOMMENDATION_CONFIG["min_interactions"]
        
    async def train(
        self,
        training_data: Dict[str, Any],
        user_features: Dict[str, Any],
        item_features: Dict[str, Any]
    ) -> None:
        """Train the recommendation model"""
        try:
            # Prepare training data
            n_users = len(user_features)
            n_items = len(item_features)
            
            # Initialize model if not exists
            if self.model is None:
                self.model = HybridRecommender(n_users, n_items)
            
            # Convert data to tensors
            user_ids = torch.tensor(training_data["user_ids"])
            item_ids = torch.tensor(training_data["item_ids"])
            labels = torch.tensor(training_data["labels"])
            item_features_tensor = torch.tensor(training_data["item_features"])
            
            # Training loop
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=MODEL_CONFIG["learning_rate"]
            )
            criterion = nn.BCELoss()
            
            for epoch in range(MODEL_CONFIG["epochs"]):
                # Training step
                self.model.train()
                optimizer.zero_grad()
                
                predictions = self.model(user_ids, item_ids, item_features_tensor)
                loss = criterion(predictions, labels)
                
                loss.backward()
                optimizer.step()
                
                logger.info(f"Epoch {epoch + 1}/{MODEL_CONFIG['epochs']}, Loss: {loss.item():.4f}")
                
            # Update embeddings cache
            self._update_embeddings()
            
        except Exception as e:
            logger.error(f"Error training recommendation model: {str(e)}")
            raise
            
    async def get_recommendations(
        self,
        user_id: str,
        n_recommendations: int = RECOMMENDATION_CONFIG["n_recommendations"],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a user"""
        try:
            # Check if user has enough interactions
            if self._is_cold_start_user(user_id):
                return await self._get_cold_start_recommendations(
                    user_id,
                    n_recommendations,
                    context
                )
            
            # Get user embedding
            user_embedding = self._get_user_embedding(user_id)
            
            # Get candidate items
            candidate_items = self._get_candidate_items(user_id)
            
            # Calculate scores
            scores = []
            for item_id in candidate_items:
                item_embedding = self._get_item_embedding(item_id)
                content_score = self._calculate_content_similarity(
                    user_embedding,
                    self.item_features[item_id]
                )
                collaborative_score = self._calculate_collaborative_similarity(
                    user_embedding,
                    item_embedding
                )
                
                # Combine scores
                final_score = 0.7 * collaborative_score + 0.3 * content_score
                scores.append((item_id, final_score))
            
            # Sort and filter recommendations
            recommendations = sorted(scores, key=lambda x: x[1], reverse=True)
            recommendations = recommendations[:n_recommendations]
            
            # Enrich recommendations with metadata
            return self._enrich_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
            raise
            
    def _is_cold_start_user(self, user_id: str) -> bool:
        """Check if user has enough interactions"""
        return len(self.user_interaction_history.get(user_id, [])) < self.cold_start_threshold
        
    async def _get_cold_start_recommendations(
        self,
        user_id: str,
        n_recommendations: int,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get recommendations for cold start users"""
        try:
            if context and "user_preferences" in context:
                # Use content-based approach with user preferences
                return await self._get_content_based_recommendations(
                    user_id,
                    context["user_preferences"],
                    n_recommendations
                )
            else:
                # Use popularity-based approach
                return await self._get_popularity_based_recommendations(
                    n_recommendations
                )
        except Exception as e:
            logger.error(f"Error getting cold start recommendations: {str(e)}")
            raise
            
    def _calculate_content_similarity(
        self,
        user_embedding: np.ndarray,
        item_features: np.ndarray
    ) -> float:
        """Calculate content-based similarity score"""
        return cosine_similarity(
            user_embedding.reshape(1, -1),
            item_features.reshape(1, -1)
        )[0][0]
        
    def _calculate_collaborative_similarity(
        self,
        user_embedding: np.ndarray,
        item_embedding: np.ndarray
    ) -> float:
        """Calculate collaborative filtering similarity score"""
        return cosine_similarity(
            user_embedding.reshape(1, -1),
            item_embedding.reshape(1, -1)
        )[0][0]
        
    def _update_embeddings(self) -> None:
        """Update user and item embeddings cache"""
        self.model.eval()
        with torch.no_grad():
            self.user_embeddings = self.model.user_embeddings.weight.numpy()
            self.item_embeddings = self.model.item_embeddings.weight.numpy()
            
    def _get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get user embedding vector"""
        return self.user_embeddings[self._get_user_index(user_id)]
        
    def _get_item_embedding(self, item_id: str) -> np.ndarray:
        """Get item embedding vector"""
        return self.item_embeddings[self._get_item_index(item_id)]
        
    def _get_candidate_items(self, user_id: str) -> List[str]:
        """Get candidate items for recommendation"""
        # Exclude items user has already interacted with
        interacted_items = set(self.user_interaction_history.get(user_id, []))
        return [item_id for item_id in self.item_features.keys()
                if item_id not in interacted_items]
        
    def _enrich_recommendations(
        self,
        recommendations: List[tuple]
    ) -> List[Dict[str, Any]]:
        """Enrich recommendations with metadata"""
        enriched = []
        for item_id, score in recommendations:
            enriched.append({
                "item_id": item_id,
                "score": float(score),
                "features": self.item_features[item_id],
                "timestamp": datetime.now().isoformat()
            })
        return enriched
        
    def _get_user_index(self, user_id: str) -> int:
        """Convert user_id to index"""
        # Implementation depends on your user ID mapping
        pass
        
    def _get_item_index(self, item_id: str) -> int:
        """Convert item_id to index"""
        # Implementation depends on your item ID mapping
        pass 