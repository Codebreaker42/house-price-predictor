import os 
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import json
import yaml
from dvclive import Live
import mlflow 
import dagshub 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger 
from utils.Train_test_split import train_test_split_and_save
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 

logger= Logger('model_evaluation.log')

logger.debug(f"Step-5 : Model Evaluation Logging Starts Here")

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow')
dagshub.init(repo_owner='nitinbdkt777', repo_name='house-price-predictor', mlflow=True)

logger.debug("dagshub setup in local successfully done")

def load_params(params_path : str ) -> dict:
    try:
        with open("params.yaml",'r') as file:
            params= yaml.safe_load(file)
        logger.debug(f"params file successfully opened from {params_path}")
        return params

    except FileNotFoundError:
        logger.debug(f"params.yaml not found in {params_path}")
        raise

    except Exception as e:
        logger.debug(f"error while loading params : {e}")
        raise 

def load_model(file_path: str):
    try:
        with open(file_path , 'rb') as f:
            model = pickle.load(f)
            
        logger.debug(f"model is successfully loaded from {file_path}")
        return model 
    
    except FileNotFoundError as e:
        logger.debug(f"file is not found : {e}")
        raise

    except Exception as e:
        logger.debug(f"error while loading model : {e}")
        raise

def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred= model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = r2_score(y_test, y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'mse': mse,
            'mae': mae,
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict

    except Exception as e:
        logger.debug(f"error while evaluation model performence : {e}")
        raise

def save_metric(metric : dict, file_path : str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metric, file, indent= 4)

        logger.debug(f"Metrics saved to {file_path} successfully.")

    except Exception as e:
        logger.debug(f"error occur while saving metrics in json : {e}")
        raise 

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """ saving current model id to future use of model registry and fetching best model to achieve automation """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_info= {"run_id": run_id, "model_path": model_path}
        with open(file_path,'w') as file:
            json.dump(model_info, file, indent=4)
        
    except Exception as e:
        logger.debug(f"error while saving model info : {e}")
    
def main() -> None:
    mlflow.set_experiment("my-dvc-model")
    with mlflow.start_run() as run:
        try:
            # loading params 
            params= load_params(params_path = 'params.yaml')
            logger.debug("loading the model")
            model_file_path= 'model/model.pkl'
            model = load_model(model_file_path)
            logger.debug(f"model successfully loaded from {model_file_path}")

            # log model in mlflow 
            mlflow.sklearn.log_model(model, "model")
            logger.debug("model is successfully logged in mlflow")

            df= pd.read_csv("dataset/feature_engineering/main_data.csv")
            logger.debug("dataset successfully loaded")

            # train test split 
            test_size= params['model_evaluation']['test_size']
            random_state= params['model_evaluation']['random_state']
            x_train, x_test, y_train , y_test = train_test_split_and_save(df, test_size, random_state)

            logger.debug(f"evaluating model performence")
            metrics= evaluate_model(model, x_test, y_test)
            print(metrics['accuracy'])
            print(metrics['mse'])
            print(metrics['mae'])

            # log metric to mlflow 
            for metric_name , metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            logger.debug("metric are successfully logged in mlflow ")

            # saving metric in json 
            save_metric(metrics, 'model_reports/metrics.json')
            logger.debug("metrics is saved successfully")

            # logging model parameter to mlflow
            if hasattr(model, 'get_params'):
                param= model.get_params()
                for param_name , param_value in param.items():
                    mlflow.log_param(param_name, param_value)

            # Log the metrics file to MLflow
            mlflow.log_artifact('model_reports/metrics.json')
            logger.debug("artifacts logged in mlflow successfully")

            # saving model info 
            save_model_info(run.info.run_id, "model", "model_reports/model_info.json")
            logger.debug("Model info saved successfully in model_reports/model_info.json")

            # experiment tracking using dvclive (to track params and metrics)
            with Live(save_dvc_exp= True) as live:
                logger.debug("tracking metrics")
                live.log_metric('accuracy', metrics['accuracy'])
                live.log_metric("mean squared error", metrics['mse'])
                live.log_metric("mean absolute error", metrics['mae'])

                logger.debug("Tracking Params")
                live.log_params(params)
            logger.debug("experiment tracking successfully done")

        except Exception as e:
            logger.debug(f"Unexpected error while Model Evaluation : {e}")

if __name__ == '__main__':
    main()