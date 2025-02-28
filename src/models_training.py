from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
from data_processing import load_ratings_data, preprocess_data
from config import RATING_SCALE, TEST_SIZE, SVD_MODEL_FILE, KNN_MODEL_FILE, FINAL_DATASET_FILE


def save_model(model, filename):
    """ Save the model using pickle. """
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def train_svd():
    """Trains and returns an SVD model."""
    ratings_df = load_ratings_data()

    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(ratings_df[['userId', 'workerId', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=TEST_SIZE, random_state=42)

    svd = SVD()
    svd.fit(trainset)

    # Evaluate SVD
    test_predictions = svd.test(testset)
    rmse = accuracy.rmse(test_predictions)
    mae = accuracy.mae(test_predictions)

    # Save trained model
    save_model(svd, SVD_MODEL_FILE)

    print('SVD model trained and saved.')

    return svd, rmse, mae

def train_knn():
    """Trains and returns a KNN model."""
    ratings_df = load_ratings_data()
    final_dataset = preprocess_data(ratings_df)

    csr_data = csr_matrix(final_dataset.values)

    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)

    # Save trained model
    save_model(knn, KNN_MODEL_FILE)
    save_model(final_dataset, FINAL_DATASET_FILE)

    print('KNN model trained and saved.')

    return knn


if __name__ == '__main__':
    x,y,z = train_svd()
    i = train_knn()