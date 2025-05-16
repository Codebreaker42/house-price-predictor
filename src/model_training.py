import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import yaml

import logging

# ensure the log directory exist
log_dir='logs'
os.makedirs(log_dir, exist_ok= True )

# setting up logger 
logger= logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# making handler 
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler 
log_file_path= os.path.join(log_dir, 'model_training.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# defining formatter 
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s ')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("step4 - Model Training Logging starts here ")

def load_params(params_path : str ) -> dict:
    try:
        with open("params.yaml",'r') as file:
            params= yaml.safe_load(file)
        logger.debug("params file successfully opened from {params_path} ")
        return params

    except FileNotFoundError:
        logger.debug(f"params.yaml not found in {params_path}")
        raise

    except Exception as e:
        logger.debug(f"error while loading params : {e}")
        raise 

def train_test_split_and_save(df: pd.DataFrame, test_size: float , random_state: int) :
    try:
        # spliting into two parts
        df['price'].dtype
        x=df.drop(columns=['price'])
        y=np.log(df['price'])
        logger.debug("successfully spilited into two parts")

        # train test split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)

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
        logging.debug("Unexpected error occur while Model Training : {e}")
        
if __name__ == "__main__":
    main()
