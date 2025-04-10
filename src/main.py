from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from cache_service import recommendation_cache
from data_processing import get_all_genres as get_all_genres_data
from recommendations import get_top_workers_by_genre
from schemas import RecommendationResponse, TrainingResponse, CacheStatusResponse
from typing import List
from models_training import train_svd, train_knn
import logging
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

logger = logging.getLogger(__name__)

app = FastAPI()
# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    genre_name: str = Query(..., description="Genre to get recommendations for"),
    use_cache: bool = Query(True, description="Whether to use cached results")
):
    """Get worker recommendations for a specific genre."""
    logger.info(f"Received recommendation request for genre: {genre_name}")
    
    if use_cache and recommendation_cache.is_cache_fresh():
        cached = recommendation_cache.get_cached_recommendations(genre_name)
        if cached:
            logger.info("Returning cached results")
            return cached
    
    # Fall back to live generation
    logger.info("Generating fresh recommendations")
    result = get_top_workers_by_genre(genre_name)
    
    # Update cache if needed
    if use_cache:
        recommendation_cache.cache_data[genre_name] = result.model_dump()
        recommendation_cache._save_cache()
    
    return result

@app.get("/genres", response_model=List[str])
async def get_all_genres():
    """Get all available genres."""
    return get_all_genres_data()

@app.post("/update-cache", response_model=TrainingResponse)
async def update_cache(background_tasks: BackgroundTasks):
    """
    Trigger cache update in the background.
    Call this endpoint manually when you want to refresh the cache.
    """
    logger.info("Manual cache update triggered via API")
    background_tasks.add_task(update_all_recommendations)
    return TrainingResponse(
        message="Cache update started in background",
        status="started"
    )

async def update_all_recommendations():
    """Function to update all recommendations in the cache."""
    try:
        logger.info("Running manual cache update")
        count = recommendation_cache.update_all_recommendations()
        logger.info(f"Cache updated with {count} genres")
        return True
    except Exception as e:
        logger.error(f"Error in manual cache update: {e}")
        return False

@app.get("/cache-status", response_model=CacheStatusResponse)
async def get_cache_status():
    """Get information about the cache state."""
    return {
        "last_updated": datetime.fromisoformat(recommendation_cache.last_updated) if recommendation_cache.last_updated else None,
        "genres_cached": len(recommendation_cache.cache_data),
        "is_fresh": recommendation_cache.is_cache_fresh(),
        "message": "Call POST /update-cache to refresh recommendations"
    }

@app.post("/train-models", response_model=TrainingResponse)
async def train_models(background_tasks: BackgroundTasks):
    """
    Trigger background model training.
    
    Returns:
        TrainingResponse: Status message about the training process
    """
    logger.info("Training triggered via API...")
    
    # Run training in background
    background_tasks.add_task(run_training)
    
    return TrainingResponse(
        message="Model training started in background",
        status="started"
    )

def run_training():
    """Run both models in sequence (background task)."""
    try:
        logger.info("Starting model training...")
        train_svd()
        train_knn()
        logger.info("Model training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise