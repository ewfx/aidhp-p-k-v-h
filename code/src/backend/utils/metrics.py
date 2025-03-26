from typing import Dict, List, Any
import numpy as np
from datetime import datetime
from collections import defaultdict
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    ndcg_score,
    mean_absolute_error,
    mean_squared_error
)

async def calculate_metrics(
    feedback_history: Dict[str, List[Dict[str, Any]]],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Calculate various recommendation system metrics"""
    try:
        # Convert dates
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Filter feedback by date range
        filtered_feedback = _filter_feedback_by_date(
            feedback_history,
            start,
            end
        )
        
        # Calculate different metrics
        engagement_metrics = calculate_engagement_metrics(filtered_feedback)
        accuracy_metrics = calculate_accuracy_metrics(filtered_feedback)
        diversity_metrics = calculate_diversity_metrics(filtered_feedback)
        coverage_metrics = calculate_coverage_metrics(filtered_feedback)
        temporal_metrics = calculate_temporal_metrics(filtered_feedback)
        
        # Combine all metrics
        return {
            "engagement": engagement_metrics,
            "accuracy": accuracy_metrics,
            "diversity": diversity_metrics,
            "coverage": coverage_metrics,
            "temporal": temporal_metrics
        }
        
    except Exception as e:
        raise Exception(f"Error calculating metrics: {str(e)}")

def calculate_engagement_metrics(
    feedback: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Calculate engagement-related metrics"""
    total_interactions = 0
    click_through_rate = 0
    conversion_rate = 0
    avg_session_length = 0
    
    for user_feedback in feedback.values():
        # Count interactions
        total_interactions += len(user_feedback)
        
        # Calculate CTR
        clicks = sum(1 for f in user_feedback if f["interaction_type"] == "click")
        impressions = sum(1 for f in user_feedback if f["interaction_type"] == "impression")
        if impressions > 0:
            click_through_rate += clicks / impressions
            
        # Calculate conversion rate
        conversions = sum(1 for f in user_feedback if f["interaction_type"] == "purchase")
        if clicks > 0:
            conversion_rate += conversions / clicks
            
        # Calculate average session length
        sessions = _group_interactions_into_sessions(user_feedback)
        if sessions:
            avg_session_length += sum(len(s) for s in sessions) / len(sessions)
    
    n_users = len(feedback)
    if n_users > 0:
        click_through_rate /= n_users
        conversion_rate /= n_users
        avg_session_length /= n_users
    
    return {
        "total_interactions": total_interactions,
        "click_through_rate": click_through_rate,
        "conversion_rate": conversion_rate,
        "avg_session_length": avg_session_length
    }

def calculate_accuracy_metrics(
    feedback: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Calculate accuracy-related metrics"""
    y_true = []
    y_pred = []
    ratings_true = []
    ratings_pred = []
    
    for user_feedback in feedback.values():
        for interaction in user_feedback:
            if "rating" in interaction:
                ratings_true.append(interaction["rating"])
                ratings_pred.append(interaction.get("predicted_rating", 0))
            
            # For binary metrics (e.g., click/no-click)
            y_true.append(1 if interaction["interaction_type"] == "click" else 0)
            y_pred.append(1 if interaction.get("predicted_score", 0) > 0.5 else 0)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ratings_true = np.array(ratings_true)
    ratings_pred = np.array(ratings_pred)
    
    metrics = {}
    
    # Binary metrics
    if len(y_true) > 0:
        metrics.update({
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred)
        })
    
    # Rating metrics
    if len(ratings_true) > 0:
        metrics.update({
            "mae": mean_absolute_error(ratings_true, ratings_pred),
            "rmse": np.sqrt(mean_squared_error(ratings_true, ratings_pred))
        })
    
    return metrics

def calculate_diversity_metrics(
    feedback: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Calculate diversity-related metrics"""
    user_items = defaultdict(set)
    item_categories = defaultdict(set)
    
    # Collect items and categories per user
    for user_id, user_feedback in feedback.items():
        for interaction in user_feedback:
            item_id = interaction["item_id"]
            user_items[user_id].add(item_id)
            
            # Collect item categories if available
            if "category" in interaction:
                item_categories[item_id].add(interaction["category"])
    
    # Calculate intra-list diversity
    category_diversity = 0
    item_diversity = 0
    n_users = len(user_items)
    
    if n_users > 0:
        # Category diversity
        for items in user_items.values():
            categories = set()
            for item in items:
                categories.update(item_categories[item])
            if len(items) > 0:
                category_diversity += len(categories) / len(items)
        category_diversity /= n_users
        
        # Item diversity (using Jaccard similarity)
        for user1 in user_items:
            for user2 in user_items:
                if user1 < user2:  # Avoid counting pairs twice
                    items1 = user_items[user1]
                    items2 = user_items[user2]
                    if items1 and items2:
                        similarity = len(items1 & items2) / len(items1 | items2)
                        item_diversity += 1 - similarity
        
        # Normalize item diversity
        n_pairs = (n_users * (n_users - 1)) / 2
        if n_pairs > 0:
            item_diversity /= n_pairs
    
    return {
        "category_diversity": category_diversity,
        "item_diversity": item_diversity
    }

def calculate_coverage_metrics(
    feedback: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Calculate coverage-related metrics"""
    all_items = set()
    recommended_items = set()
    user_coverage = set()
    
    # Collect items and users
    for user_id, user_feedback in feedback.items():
        user_coverage.add(user_id)
        for interaction in user_feedback:
            item_id = interaction["item_id"]
            all_items.add(item_id)
            if interaction.get("is_recommended", False):
                recommended_items.add(item_id)
    
    # Calculate metrics
    item_coverage = len(recommended_items) / len(all_items) if all_items else 0
    
    return {
        "item_coverage": item_coverage,
        "user_coverage": len(user_coverage)
    }

def calculate_temporal_metrics(
    feedback: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Calculate temporal/novelty metrics"""
    item_timestamps = defaultdict(list)
    user_timestamps = defaultdict(list)
    
    # Collect timestamps
    for user_id, user_feedback in feedback.items():
        for interaction in user_feedback:
            timestamp = datetime.fromisoformat(interaction["timestamp"])
            item_id = interaction["item_id"]
            
            item_timestamps[item_id].append(timestamp)
            user_timestamps[user_id].append(timestamp)
    
    # Calculate metrics
    avg_item_freshness = 0
    n_items = len(item_timestamps)
    
    if n_items > 0:
        for timestamps in item_timestamps.values():
            if timestamps:
                latest = max(timestamps)
                first = min(timestamps)
                freshness = (latest - first).total_seconds() / 86400  # Convert to days
                avg_item_freshness += freshness
        avg_item_freshness /= n_items
    
    # Calculate user engagement frequency
    avg_user_frequency = 0
    n_users = len(user_timestamps)
    
    if n_users > 0:
        for timestamps in user_timestamps.values():
            if len(timestamps) > 1:
                timestamps.sort()
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 86400
                           for i in range(len(timestamps)-1)]
                avg_user_frequency += sum(intervals) / len(intervals)
        avg_user_frequency /= n_users
    
    return {
        "avg_item_freshness_days": avg_item_freshness,
        "avg_user_frequency_days": avg_user_frequency
    }

def _filter_feedback_by_date(
    feedback: Dict[str, List[Dict[str, Any]]],
    start: datetime,
    end: datetime
) -> Dict[str, List[Dict[str, Any]]]:
    """Filter feedback history by date range"""
    filtered = defaultdict(list)
    
    for user_id, user_feedback in feedback.items():
        for interaction in user_feedback:
            timestamp = datetime.fromisoformat(interaction["timestamp"])
            if start <= timestamp <= end:
                filtered[user_id].append(interaction)
                
    return filtered

def _group_interactions_into_sessions(
    interactions: List[Dict[str, Any]],
    session_timeout: int = 1800  # 30 minutes in seconds
) -> List[List[Dict[str, Any]]]:
    """Group interactions into sessions based on time differences"""
    if not interactions:
        return []
    
    # Sort interactions by timestamp
    sorted_interactions = sorted(
        interactions,
        key=lambda x: datetime.fromisoformat(x["timestamp"])
    )
    
    sessions = [[sorted_interactions[0]]]
    
    for interaction in sorted_interactions[1:]:
        current_time = datetime.fromisoformat(interaction["timestamp"])
        last_time = datetime.fromisoformat(sessions[-1][-1]["timestamp"])
        
        if (current_time - last_time).total_seconds() > session_timeout:
            # Start new session
            sessions.append([])
        
        sessions[-1].append(interaction)
    
    return sessions 