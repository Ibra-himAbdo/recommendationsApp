import pickle
from dataclasses import dataclass
from data_processing import load_final_data
from scipy.sparse import csr_matrix
from config import WEIGHT_KNN, WEIGHT_SVD, KNN_MODEL_FILE, SVD_MODEL_FILE, FINAL_DATASET_FILE
import service_pb2

@dataclass
class WorkerRecommendation:
    workerId: int
    name: str
    score: float

def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Load models and data
knn = load_model(KNN_MODEL_FILE)
svd = load_model(SVD_MODEL_FILE)
final_dataset = load_model(FINAL_DATASET_FILE)


worker_df = load_final_data()
csr_data = csr_matrix(final_dataset.values)

def get_top_workers_by_genre(genre_name, user_id=1, weight_knn=0.4, weight_svd=0.6, top_n=8):

    # Find workers in the specified genre
    workers_in_genre = worker_df[worker_df['genres'].str.contains(genre_name, case=False, na=False)]

    if not workers_in_genre.empty:
        # Ensure workerId is treated as an integer
        genre_worker_ids = workers_in_genre['workerId'].astype(int).unique()

        # Get KNN similarities
        distances, indices = knn.kneighbors(csr_data, n_neighbors=11)

        worker_scores = {}

        for worker_id in genre_worker_ids:
            worker_id = int(worker_id)  # Ensure workerId is an integer

            if worker_id in final_dataset['workerId'].values:
                worker_idx = final_dataset[final_dataset['workerId'] == worker_id].index[0]

                # KNN Similarity (convert distance to similarity)
                for i, idx in enumerate(indices[worker_idx]):
                    neighbor_id = int(final_dataset.iloc[idx]['workerId'])  # Force integer conversion
                    similarity = 1 - distances[worker_idx][i]  # Higher similarity is better
                    worker_scores[neighbor_id] = worker_scores.get(neighbor_id, 0) + (weight_knn * similarity)

        # Get SVD Predicted Ratings
        for worker_id in list(worker_scores.keys()):  # Ensure we iterate safely
            worker_id = int(worker_id)  # Force int before prediction
            predicted_rating = svd.predict(uid=int(user_id), iid=worker_id).est  # Force int for SVD
            worker_scores[worker_id] += weight_svd * predicted_rating  # Ensure int key

        # Rank workers by final score
        ranked_workers = sorted(worker_scores.items(), key=lambda x: -x[1])

        # Display the top N workers
        print(f"Top {top_n} workers in '{genre_name}' using Hybrid (KNN + SVD):")
        recommendations = []
        for i, (worker_id, score) in enumerate(ranked_workers[:top_n], 1):
            worker_id = int(worker_id)  # Ensure correct integer format

            # 🔥 **Fix: Check if worker exists before accessing**
            worker_row = worker_df.loc[worker_df['workerId'] == worker_id, 'names']
            if not worker_row.empty:
                worker_name = worker_row.values[0]  # Get name safely
            else:
                worker_name = "Unknown Worker"  # Fallback name

            print(f"{i}. {worker_name} (WorkerID: {worker_id}) - Final Score: {score:.2f}")
            
            recommendations.append(service_pb2.WorkerRecommendation(
                workerId=worker_id,
                name=worker_name,
                score=score
            ))
            
        return service_pb2.RecommendationResponse(recommendations=recommendations)

    else:
        print(f"No workers found for the genre '{genre_name}'.")
        return service_pb2.RecommendationResponse(recommendations=[])



def get_top_workers_by_genre2(genre_name:str, user_id=1, top_n=8):

    # Find workers in the specified genre
    workers_in_genre = worker_df[worker_df['genres'].str.contains(genre_name, case=False, na=False)]

    if workers_in_genre.empty:
        return []

    # Ensure workerId is treated as an integer
    genre_worker_ids = workers_in_genre['workerId'].astype(int).unique()

    # Get KNN similarities
    distances, indices = knn.kneighbors(csr_data, n_neighbors=11)

    worker_scores = {}

    for worker_id in genre_worker_ids:
        worker_id = int(worker_id)

        if worker_id in final_dataset['workerId'].values:
            worker_idx = final_dataset[final_dataset['workerId'] == worker_id].index[0]

            # KNN Similarity (convert distance to similarity)
            for i, idx in enumerate(indices[worker_idx]):
                neighbor_id = int(final_dataset.iloc[idx]['workerId'])
                similarity = 1 - distances[worker_idx][i]  # Higher similarity is better
                worker_scores[neighbor_id] = worker_scores.get(neighbor_id, 0) + (WEIGHT_KNN * similarity)

    # Get SVD Predicted Ratings
    for worker_id in list(worker_scores.keys()):
        worker_id = int(worker_id)
        predicted_rating = svd.predict(uid=int(user_id), iid=worker_id).est
        worker_scores[worker_id] += WEIGHT_SVD * predicted_rating

    # Rank workers by final score
    ranked_workers = sorted(worker_scores.items(), key=lambda x: -x[1])[:top_n]

    # Build the result list
    recommendations = []
    for worker_id, score in ranked_workers:
        worker_id = int(worker_id)

        # Check if worker exists
        worker_row = worker_df.loc[worker_df['workerId'] == worker_id, 'names']
        worker_name = worker_row.values[0] if not worker_row.empty else "Unknown Worker"

        # Create a recommendation object
        recommendations.append(WorkerRecommendation(workerId=worker_id, name=worker_name, score=score))

    return recommendations