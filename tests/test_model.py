import unittest
import mlflow
import dagshub
import os

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("HOUSE_PRICE_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("HOUSE_PRICE_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "nitinbdkt777"
        repo_name = "house-price-predictor"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        cls.client = mlflow.MlflowClient()

        # Define the registered model name
        cls.model_name = "Xgboost"

    def test_model_loaded_from_production(self):
        prod_version = self.client.get_latest_versions(self.model_name, stages=['Production'])
        self.assertTrue(len(prod_version) > 0, "No model in Production stage")

if __name__ == "__main__":
    unittest.main()
