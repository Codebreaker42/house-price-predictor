import os 
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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

def train_test_split_and_save(df: pd.DataFrame) -> None:
    try:
        # spliting into two parts
        df['price'].dtype
        x=df.drop(columns=['price'])
        y=np.log(df['price'])
        logger.debug("successfully spilited into two parts")

        # train test split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=42)

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

def main() -> None:
    try:
        logger.debug("loading the model")
        model_file_path= 'model/model.pkl'
        model = load_model(model_file_path)

        df= pd.read_csv("dataset/main_data.csv")
        logger.debug("dataset successfully loaded")

        x_train, x_test, y_train , y_test = train_test_split_and_save(df)

        logger.debug(f"evaluating model performence")
        metrics= evaluate_model(model, x_test, y_test)
        print(metrics['accuracy'])
        print(metrics['mse'])
        print(metrics['mae'])

    except Exception as e:
        logger.debug(f"Unexpected error while Model Evaluation : {e}")

if __name__ == '__main__':
    main()