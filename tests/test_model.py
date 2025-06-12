import os
import mlflow
import dagshub
import unittest

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mlflow.set_tracking_uri("https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow")

        # Use token-based authentication (non-interactive)
        os.environ["CAPSTONE_TEST"] = os.getenv("CAPSTONE_TEST")  # already set in GitHub Actions
        dagshub.init(repo_owner="nitinbdkt777", repo_name="house-price-predictor")

        client = mlflow.MlflowClient()
        prod_version = client.get_latest_versions("Xgboost", stages=["Production"])
        print(prod_version)

    def test_model_loaded_from_production(self):
        self.assertTrue(True)  # Add meaningful assertions here

if __name__ == "__main__":
    unittest.main()
