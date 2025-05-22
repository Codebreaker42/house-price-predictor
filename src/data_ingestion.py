import logging
import os 
import pandas as pd
import yaml
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger 

logger= Logger('data_ingestion.log')
logger.debug("step 1 - Data Ingestion logging starts here ")

def load_params(params_path : str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params= yaml.safe_load(file)
        logger.debug("params.yaml successfully opened")
        return params 

    except FileNotFoundError:
        logger.debug(f"file not found from {params_path}")
        raise

    except Exception as e:
        logger.debug(f"error while loading the parameters from .yaml {e}")
        raise 

def load_data(data_url: str) -> pd.DataFrame:
    """ load data from a csv file and change it into pd dataframe """
    try:
        df= pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url}')
        return df
    except pd.errors.ParseError as e:
        logger.error(f"failed to parse the csv file {e}")
        raise 
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the data : {e}')
        raise 

def preprocess_data(df: pd.DataFrame) -> None:
    """ removing the useless columns and saving csv file"""
    try:
        column= ['Unnamed: 0','additionalrooms','carpetarea','opensides','facing','totalfloor','overlooking','projectname','possesiondate','roadfaceing']
        df.drop(columns=column ,inplace=True)
        logger.debug(f"unwanted columns {column} are deleted successfully")
        subset= ['age','floor','ownership','amenitiesnot','amenitiesavailable','area','bathroom']
        df.dropna(subset=subset ,inplace=True )
        logger.debug(f"null values from {subset} is deleted successfully")
        data_path= 'dataset/data_ingestion'
        os.makedirs(data_path, exist_ok=True)
        dataset= os.path.join(data_path, 'main_data.csv')
        df.to_csv(dataset, index=False)
        logger.debug(f"dataset saved to {dataset} folder successfully")
    except Exception as e:
        logger.error(f'Unexpected error occur while saving the data {e}')
        raise

def main():
    try:
        params= load_params(params_path ='params.yaml')
        # data_path= 'https://raw.githubusercontent.com/Codebreaker42/house-price-predictor/refs/heads/master/Pune_property_data.csv'
        data_path= params['data_ingestion']['data_path']
        df= load_data(data_url= data_path)
        logger.debug("data loaded succesfully")
        preprocess_data(df)
        # print(df.info())
    except Exception as e:
        logger.error(f'Unexpected error occured while saving the file : {e}')



if __name__ == '__main__':
    main()