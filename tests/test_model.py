import os
import mlflow
import dagshub
import unittest

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mlflow.set_tracking_uri("https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow")

        token = os.getenv("CAPSTONE_TEST")
        if not token:
            raise EnvironmentError("CAPSTONE_TEST is not set in environment.")
    
        os.environ["CAPSTONE_TEST"] = token
        dagshub.init(repo_owner="nitinbdkt777", repo_name="house-price-predictor")

        client = mlflow.MlflowClient()
        prod_version = client.get_latest_versions("Xgboost", stages=["Production"])
        print(prod_version)

    def test_model_loaded_from_production(self):
        self.assertTrue(True)  # Add meaningful assertions here

if __name__ == "__main__":
    unittest.main()
