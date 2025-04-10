import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config import RECOMMENDATIONS_CACHE_FILE, CACHE_FOLDER
from data_processing import get_all_genres
from recommendations import get_top_workers_by_genre
from schemas import RecommendationResponse
import logging

logger = logging.getLogger(__name__)

class RecommendationCache:
    def __init__(self):
        self.cache_file = RECOMMENDATIONS_CACHE_FILE
        self.last_updated = None
        self.cache_data = {}
        self._ensure_cache_folder_exists()
        self._load_cache()

    def _ensure_cache_folder_exists(self):
        """Ensure the cache folder exists."""
        if not os.path.exists(CACHE_FOLDER):
            os.makedirs(CACHE_FOLDER)

    def _load_cache(self):
        """Load cache from file if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache_data = data.get('recommendations', {})
                    self.last_updated = data.get('last_updated')
                    logger.info("Cache loaded successfully")
            else:
                logger.info("No cache file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.cache_data = {}

    def _save_cache(self):
        """Save current cache to file."""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'recommendations': self.cache_data
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def update_all_recommendations(self):
        """Generate and cache recommendations for all genres."""
        all_genres = get_all_genres()
        logger.info(f"Updating recommendations for {len(all_genres)} genres")
        
        updated_cache = {}
        for genre in all_genres:
            try:
                result = get_top_workers_by_genre(genre)
                # Ensure proper serialization
                updated_cache[genre] = {
                    "recommendations": [
                        {
                            "workerId": rec.workerId,
                            "name": rec.name,
                            "score": float(rec.score)
                        }
                        for rec in result.recommendations
                    ]
                }
            except Exception as e:
                logger.error(f"Error generating recommendations for {genre}: {e}")
                continue
        
        self.cache_data = updated_cache
        self._save_cache()
        return len(all_genres)

    def get_cached_recommendations(self, genre: str) -> Optional[RecommendationResponse]:
        """Get cached recommendations for a genre."""
        cached = self.cache_data.get(genre)
        if cached:
            try:
                # Convert the cached dictionary to a RecommendationResponse object
                return RecommendationResponse(**cached)
            except Exception as e:
                logger.error(f"Error parsing cached recommendations for {genre}: {e}")
                return None
        return None

    def is_cache_fresh(self, hours: int = 12) -> bool:
        """Check if cache is fresh (updated within specified hours)."""
        if not self.last_updated:
            return False
        try:
            last_updated = datetime.fromisoformat(self.last_updated)
            return datetime.now() - last_updated < timedelta(hours=hours)
        except:
            return False

# Create a global cache instance
recommendation_cache = RecommendationCache()