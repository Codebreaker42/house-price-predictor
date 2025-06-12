import unittest
import mlflow
import dagshub

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mlflow.set_tracking_uri('https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow')
        dagshub.init(repo_owner="nitinbdkt777", repo_name="house-price-predictor")
        cls.client = mlflow.MlflowClient()
        cls.model_name = "Xgboost"

    def test_model_loaded_from_production(self):
        prod_version = self.client.get_latest_versions(self.model_name, stages=['Production'])
        self.assertTrue(len(prod_version) > 0, "No model in Production stage")

if __name__ == "__main__":
    unittest.main()
