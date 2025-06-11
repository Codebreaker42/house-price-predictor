import os
import mlflow 
import sys 
import dagshub

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger

logger= Logger("model_promotion.log")

# below code for loacal use 
mlflow.set_tracking_uri('https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow')
dagshub.init(repo_owner= "nitinbdkt777", repo_name="house-price-predictor")
logger.debug("dagshub local setup is successfully done")

def promote_model():
    """ Its promote the latest model in production to the previous
      production model and make it archives
    """
    model_name= "Xgboost"

    client= mlflow.MlflowClient()

    # get the latest version in staging 
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version
    logger.debug("successfully load the latest version")

    # archives the current production model 
    prod_versions= client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version= version.version,
            stage="Archived"
        )
    logger.debug("Successfully Archives the previous production version")

    # promote the new model in production 
    client.transition_model_version_stage(
        name= model_name,
        version= version.version,
        stage="Production"
    )
    logger.debug(f"Model version {latest_version_staging} promoted to production")

def main():
    promote_model()

if __name__ == "__main__":
    main()
