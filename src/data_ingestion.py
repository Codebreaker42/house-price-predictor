import logging
import os 
import pandas as pd

# log directory creation 
log_dir="logs"
os.makedirs(log_dir, exist_ok= True )

# logging configuratation 
logger= logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

# making handler 
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path= os.path.join(log_dir, 'data_ingestion.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# setting the format of logging message 
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("step 1 - Data Ingestion logging starts here")

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
        data_path= 'dataset'
        os.makedirs(data_path, exist_ok=True)
        dataset= os.path.join(data_path, 'main_data.csv')
        df.to_csv(dataset, index=False)
        logger.debug(f"dataset saved to {dataset} folder successfully")
    except Exception as e:
        logger.error(f'Unexpected error occur while saving the data {e}')
        raise

def main():
    try:
        data_path= 'https://raw.githubusercontent.com/Codebreaker42/house-price-predictor/refs/heads/master/Pune_property_data.csv'
        df= load_data(data_url= data_path)
        logger.debug("data loaded succesfully")
        preprocess_data(df)
        print(df.info())
    except Exception as e:
        logger.error(f'Unexpected error occured while saving the file : {e}')



if __name__ == '__main__':
    main()