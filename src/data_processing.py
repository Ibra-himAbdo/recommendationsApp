import pandas as pd
from config import WORKERS_FILE, RATINGS_FILE, FINAL_FILE

def load_raw_workers_data():
    """Loads workers dataset."""
    workers_df = pd.read_csv(WORKERS_FILE)
    workers_df.rename(columns={'title': 'names', 'movieId': 'workerId'}, inplace=True)
    return workers_df

def load_ratings_data():
    """Loads ratings dataset."""
    ratings_df = pd.read_csv(RATINGS_FILE)
    ratings_df.rename(columns={'movieId': 'workerId'}, inplace=True)
    return ratings_df

def load_final_data():
    """Loads final dataset."""
    return pd.read_csv(FINAL_FILE)

def preprocess_data(ratings_df: pd.DataFrame):
    """Prepares dataset for matrix factorization and KNN."""
    final_dataset = ratings_df.pivot(index="workerId", columns="userId", values="rating")
    final_dataset.fillna(0, inplace=True)
    final_dataset.reset_index(inplace=True)  # 🔥 Add this line

    return final_dataset

def get_all_genres() -> list[str]:
    """Get all unique genres from the dataset."""
    final_df = load_final_data()
    # Split genres (assuming they're separated by '|')
    all_genres = set()
    for genres in final_df['genres'].str.split('|'):
        if isinstance(genres, list):
            all_genres.update(genres)
    return sorted(all_genres)

if __name__ == '__main__':
    #print(load_ratings_data().head())
    print(get_all_genres())