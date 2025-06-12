# tests/test_model.py

import mlflow
import os
import dagshub

def test_model_loading():
    mlflow.set_tracking_uri('https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow')
    dagshub.init(repo_owner="nitinbdkt777", repo_name="house-price-predictor")

    model_name = "Xgboost"
    client = mlflow.MlflowClient()
    prod_version = client.get_latest_versions(model_name, stages=['Production'])

    assert prod_version is not None
