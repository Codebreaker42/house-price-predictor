import os 
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import json
import yaml

log_dir= "logs"
os.makedirs(log_dir, exist_ok= True)

# setting up logger 
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# make handler
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler 
log_file_path= os.path.join(log_dir, 'model_evaluation.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# defining formatter 
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s ')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handler 
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug(f"Step-5 : Model Evaluation Logging Starts Here")

def load_params(params_path : str ) -> dict:
    try:
        with open("params.yaml",'r') as file:
            params= yaml.safe_load(file)
        logger.debug("params file successfully opened from {params_path}")
        return params

    except FileNotFoundError:
        logger.debug(f"params.yaml not found in {params_path}")
        raise

    except Exception as e:
        logger.debug(f"error while loading params : {e}")
        raise 

def train_test_split_and_save(df: pd.DataFrame, test_size : float , random_state= int)  :
    try:
        # spliting into two parts
        df['price'].dtype
        x=df.drop(columns=['price'])
        y=np.log(df['price'])
        logger.debug("successfully spilited into two parts")

        # train test split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= test_size,random_state =random_state)

        # reshaping y column
        y_train=np.array(y_train).reshape(-1,1)
        y_test=np.array(y_test).reshape(-1,1)

        # print(x_train.shape)
        # print(x_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        return x_train, x_test, y_train, y_test

    except Exception as e:
        logger.debug(f"error occured while train test spliting")
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

        logger.debug(f"Metrics saved to {file_path} successfully in json format")

    except Exception as e:
        logger.debug(f"error occur while saving metrics in json : {e}")
        raise 
    
def main() -> None:
    try:
        # loading params 
        params= load_params(params_path = 'params.yaml')
        logger.debug("loading the model")
        model_file_path= 'model/model.pkl'
        model = load_model(model_file_path)

        df= pd.read_csv("dataset/main_data.csv")
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

        # saving metric in json 
        save_metric(metrics, 'model_reports/metrics.json')

    except Exception as e:
        logger.debug(f"Unexpected error while Model Evaluation : {e}")

if __name__ == '__main__':
    main()