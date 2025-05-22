import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import yaml
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger 
from utils.Train_test_split import train_test_split_and_save

logger= Logger('model_training.log')

logger.debug("step4 - Model Training Logging starts here ")

def load_params(params_path : str ) -> dict:
    try:
        with open("params.yaml",'r') as file:
            params= yaml.safe_load(file)
        logger.debug(f"params file successfully opened from {params_path} ")
        return params

    except FileNotFoundError:
        logger.debug(f"params.yaml not found in {params_path}")
        raise

    except Exception as e:
        logger.debug(f"error while loading params : {e}")
        raise 

def train_model(x_train : np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingRegressor:
    try:
        lr= params['learning_rate']
        
        step1=ColumnTransformer([
            ('col_tnf',OneHotEncoder(sparse_output=False, drop='first'),[10,12])
        ],remainder='passthrough')

        step2=GradientBoostingRegressor(learning_rate=lr)
        pipe=Pipeline([
            ('step1',step1),
            ('step2',step2)
        ])

        pipe.fit(x_train,y_train)
        return pipe

    except Exception as e:
        logger.debug(f"unexpected error during model training : {e}")
        raise

def save_model(model, model_file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(model_file_path) , exist_ok=True)

        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.debug(f"Model is saved in {model_file_path} successfully")

    except FileNotFoundError as e:
        logger.debug(f"file path not found : {e}")
        raise

    except Exception as e:
        logger.debug(f"Unexpected error occur while saving model : {e}")
        raise

def main() -> None:
    try:
        params = load_params(params_path= 'params.yaml')
        # params = {'learning_rate': .6}
        df= pd.read_csv('dataset/feature_engineering/main_data.csv')
        logger.debug('csv file open successfully')

        # train test split 
        test_size= params['model_training']['test_size']
        random_state= params['model_training']['random_state']
        x_train,x_test, y_train, y_test = train_test_split_and_save(df, test_size, random_state)
        # print(x_train)
        logger.debug(f"train test split and save is successfully done")
        
        # model 
        lr = params['model_training']['lr']
        print(lr)
        prm= {'learning_rate': lr}
        model = train_model(x_train, y_train , prm)
        logger.debug("Model is trained successfully")

        # saving model 
        save_model_path= 'model/model.pkl'
        save_model(model, save_model_path)

    except Exception as e:
        logger.debug("Unexpected error occur while Model Training : {e}") 
        
if __name__ == "__main__":
    main()
