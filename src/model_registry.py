import os 
import sys 
import json 
import dagshub
import mlflow
import mlflow.tracking
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger

logger= Logger("model_registry.log")

# below code for loacal use 
mlflow.set_tracking_uri('https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow')
dagshub.init(repo_owner= "nitinbdkt777", repo_name="house-price-predictor")
logger.debug("dagshub local setup is successfully done")

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path,"r") as file:
            model_info= json.load(file)
        logger.debug(f"model info loaded from {file_path}") 
        return model_info
    
    except FileNotFoundError:
        logger.debug(f"file not found {file_path}")
        raise

    except Exception as e:
        logger.debug(f"error while loading model : {e}")
        raise

def register_model(model_name: str, model_info: dict) ->None:
    try:
        model_uri= f"runs:/{model_info['run_id']}/ {model_info['model_path']}"

        # register the model 
        model_version = mlflow.register_model(model_uri, model_name)

        # transition the model to staging stage (you can put any stage)
        client= mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name= model_name,
            version= model_version.version,
            stage="Staging"
        )

        logger.debug(f"Model {model_name} version {model_version.version} registered and transitioned to staging")

    except Exception as e:
        logger.debug("error during registering the model")

def main()-> None:
    try:
        model_info_path= 'model_reports/model_info.json'
        model_info= load_model_info(model_info_path)

        model_name= "Xgboost"
        register_model(model_name , model_info)
    except Exception as e:
        logger.debug(f"error while model registry : {e}")

if __name__ == "__main__":
    main()