import os
import mlflow 
import sys 
import dagshub

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger

logger= Logger("model_promotion.log")

# Below code block is for production use
# -------------------------------------------------------------------------------------
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


# below code for loacal use 
# mlflow.set_tracking_uri('https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow')
# dagshub.init(repo_owner= "nitinbdkt777", repo_name="house-price-predictor")
# logger.debug("dagshub local setup is successfully done")

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
        version= latest_version_staging,
        stage="Production"
    )
    logger.debug(f"Model version {latest_version_staging} promoted to production")

def main():
    promote_model()

if __name__ == "__main__":
    main()
