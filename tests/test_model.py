import unittest
import mlflow
import os
import dagshub 

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mlflow.set_tracking_uri('https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow')
        dagshub.init(repo_owner= "nitinbdkt777", repo_name="house-price-predictor")

        model_name= "Xgboost"
        # load the new model from MLflow model registry 
        client = mlflow.MlflowClient()

        prod_version = client.get_latest_versions(model_name, stages=['Production'])
        print(prod_version)

if __name__ == "__main__":
    unittest.main()
        
