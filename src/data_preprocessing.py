import os
import logging 
import numpy as np
import pandas as pd
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger 

logger= Logger('data_preprocessing.log')
logger.debug("step2 - Data Preprocessing Logging starts here ")

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
        logger.debug(f"unexpected error while preprocessing age: {e}")
        raise

def preprocess_area(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['area'].value_counts()
        df['area']=df['area'].apply(str).str.split(' ').str.slice(0,1).str.join('')
        df['area']=df['area'].apply(str).str.replace(',','')
        df['area']=df['area'].astype('int64')
        return df
    
    except Exception as e:
        logger.debug(f"unexpected error while preprocessing area: {e}")
        raise

def preprocess_bhk(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['bhk'].unique()
        df['bhk'].value_counts().plot(kind='bar')
        def find(text):
            if text=='1 BHK Apartment ' or text=='2 BHK Apartment ' or text=='3 BHK Apartment ' or text=='4 BHK Apartment ' or text=='5 BHK Apartment ':
                return text
            elif text=='1 BHK Villa ' or text=='2 BHK Villa ' or text=='3 BHK Villa ' or text=='4 BHK Villa ' or text=='5 BHK Villa ':
                return text
            else:
                return 'others'
        df['bhk']=df['bhk'].apply(find)
        df=df[df['bhk']!='others']
        return df

    except Exception as e:
        logger.debug(f"unexpected error while preprocessing bhk : {e}")
        raise 

def preprocess_floor(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['floor'].unique()
        df['floor']=df['floor'].apply(str).str.replace('Gr ','0')
        df['floor']=df['floor'].apply(str).str.replace('Gr','0')
        df['floor']=df['floor'].str.split(',').str.slice(0,1).str.join('')
        df['floor']=df['floor'].astype('int64')
        return df
    
    except Exception as e:
        logger.debug(f"unexpected error while preprocessing the floor ")
        raise

def preprocess_location(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # locality
        df.rename(columns={'locality':'place'},inplace=True)
        df['place'].unique()
        # Calculate value counts for Column1
        value_counts = df['place'].value_counts()

        # Identify categories with value counts less than 10
        categories_to_drop = value_counts[value_counts < 50].index

        # Drop rows with categories having value counts less than 10
        df = df[~df['place'].isin(categories_to_drop)]

        # Reset the index if needed
        df.reset_index(drop=True, inplace=True)
        df['place'].unique()
        # df['place'].value_counts() 
        return df

    except Exception as e:
        logger.debug(f"unexpected error while preprocessing the location")
        raise

def preprocess_price_per_square(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['pricepersquare'].value_counts()
        df.rename(columns={'pricepersquare': 'price_per_sq'},inplace=True)
        df['price_per_sq']=df['price_per_sq'].str.split(' ').str.slice(1,2).str.join('')
        df['price_per_sq']=df['price_per_sq'].apply(str).str.replace(',','')
        df['price_per_sq']=df['price_per_sq'].apply(str).str.replace('/','')
        df['price_per_sq']=df['price_per_sq'].astype('int64')
        return df
    except Exception as e:
        logger.debug(f"unexpected error while preprocess price per square: {e}")
        raise

def preprocess_status(df: pd.DataFrame)->pd.DataFrame:
    try:
        def extract(text):
            if text=='Ready to move,Unfurnished':
                return 'Unfurnished'
            elif text=='Ready to move,Semi-Furnished':
                return 'Semi-Furnished'
            elif text=='Ready to move,Furnished':
                return 'Furnished'
            else:
                return 'Ready to move'
        df['status']=df['status'].apply(extract) 
        return df
    
    except Exception as e:
        logger.debug(f"unexpected error while preprocess status: {e}")
        raise

def preprocess_balcony(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['balconies'].value_counts()
        # Calculate value counts for Column1
        value_counts = df['balconies'].value_counts()

        # Identify categories with value counts less than 10
        categories_to_drop = value_counts[value_counts < 135].index

        # Drop rows with categories having value counts less than 10
        df = df[~df['balconies'].isin(categories_to_drop)]

        # Reset the index if needed
        df.reset_index(drop=True, inplace=True)
        df['balconies']=df['balconies'].fillna(2)
        df['balconies'].value_counts()
        df['balconies']=df['balconies'].astype('int64')
        return df
    
    except Exception as e:
        logger.debug(f"unexpected error while preprocess balcony: {e}")
        raise
    
def preprocess_neworold(df: pd.DataFrame) -> pd.DataFrame:
    try:
        value_to_del= 'neworold'
        df= df[df['neworold'] != value_to_del]
        return df
    except Exception as e:
        logger.debug(f"unexpected error occur while preprocessing neworold: {e}")
        raise



def main() -> None:
    try:
        df= pd.read_csv('dataset/data_ingestion/main_data.csv')
        logger.debug('csv file open successfully') 
        
        # age 
        df= preprocess_age(df)
        logger.debug("age preprocessing is successfully done")

        # area 
        df=preprocess_area(df)
        logger.debug("area preprocessing is successfully done")

        #bhk
        df= preprocess_bhk(df)
        logger.debug("bhk preprocessing is successfully done")

        # floor 
        df= preprocess_floor(df)
        logger.debug("floor preprocessing is successfully done")

        # location 
        df= preprocess_location(df)
        logger.debug("location preprocessing is successfully done")

        # price per square 
        df= preprocess_price_per_square(df)
        logger.debug("price per square preprocessing is successfully done")

        # status 
        df= preprocess_status(df)
        logger.debug("balcony preprocessing is successfully done")

        # balcony 
        df= preprocess_balcony(df)
        logger.debug("balcony preprocessing is successfully done")

        # saving the dataset 
        data_path= 'dataset/data_preprocessing'
        os.makedirs(data_path, exist_ok=True)
        dataset= os.path.join(data_path,'main_data.csv')
        df.to_csv(dataset, index=False)
        logger.debug(f"dataset saved to {dataset} folder successfully")
    
    except Exception as e:
        logger.debug(f"Unexpected error : {e}")


if __name__ == "__main__":
    main()