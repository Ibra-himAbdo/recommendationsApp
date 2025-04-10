import os

# File paths
DATA_FOLDER = os.path.join(os.getcwd(), "data")
WORKERS_FILE = os.path.join(DATA_FOLDER, "movies_snowflake.csv")
RATINGS_FILE = os.path.join(DATA_FOLDER, "ratings_snowflake.csv")
FINAL_FILE = os.path.join(DATA_FOLDER, "movies_updated_final.csv")


# PKL files
PKL_FOLDER = os.path.join(os.getcwd(), "pkl_objects")
SVD_MODEL_FILE = os.path.join(PKL_FOLDER, "svd_model.pkl")
KNN_MODEL_FILE = os.path.join(PKL_FOLDER, "knn_model.pkl")
FINAL_DATASET_FILE = os.path.join(PKL_FOLDER, "final_dataset.pkl")

# Cache files
CACHE_FOLDER = os.path.join(os.getcwd(), "cache")
RECOMMENDATIONS_CACHE_FILE = os.path.join(CACHE_FOLDER, "recommendations_cache.json")

# KNN & SVD Weights
WEIGHT_KNN = 0.4
WEIGHT_SVD = 0.6

# SVD Configuration
RATING_SCALE = (1, 5)
TEST_SIZE = 0.2  # 20% test split

