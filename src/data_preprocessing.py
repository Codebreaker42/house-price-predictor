import os
import logging 
import pandas as pd 
import numpy as np

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
log_file_path= os.path.join(log_dir, 'data_preprocessing.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# defining formatter 
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s ')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("data Preprocessing Logging starts here ")

def preprocess_age(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['age']=df['age'].apply(str).str.replace('age','')
        df['age']=df['age'].apply(str).str.replace('years','')
        df['age']=df['age'].apply(str).str.replace('year','')
        df['age']=df['age'].apply(str).str.replace('-','')
        df['age']=df['age'].str.split(" ").str.slice(0,1).str.join('')
        df=df[df['age']!='']
        df['age']
        df['age']=df['age'].astype('int64')
        return df
    except Exception as e:
        logger.debug(f"unexpected error while preprocessing age {e}")
        raise

def main() -> None:
    try:
        df= pd.read_csv('dataset/main_data.csv')
        logger.debug('csv file open successfully') 
        df= preprocess_age(df)
        logger.debug("age preprocessing is successfully done")
        print(df.info())
    except Exception as e:
        logger.debug(f"Unexpected error : {e}")

if __name__ == "__main__":
    main()