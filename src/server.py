import grpc
from concurrent import futures
import threading
import service_pb2 as pb2
import service_pb2_grpc as pb2_grpc
from models_training import train_svd, train_knn
from recommendations import get_top_workers_by_genre_grpc

class RecommendationService(pb2_grpc.LongServiceServicer):
    def GetWorkerRecommendations(self, request, context):
        """Handle worker recommendations."""
        genre_name = request.query
        print(f"Received recommendation request for genre: {genre_name}")

        response = get_top_workers_by_genre_grpc(genre_name)
        return response

    def RunModelTraining(self, request, context):
        """Trigger background model training."""
        print("Training triggered via gRPC...")

        # Run training in a separate thread
        threading.Thread(target=self.do_training).start()

        return pb2.Empty()

    def do_training(self):
        """Run both models in sequence."""
        try:
            train_svd()
            train_knn()
            print("Model training completed!")
        except Exception as e:
            print(f"Error during training: {e}")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_LongServiceServicer_to_server(RecommendationService(), server)
    
    server.add_insecure_port('[::]:50051')
    print("Server is running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
