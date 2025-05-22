import os
import logging 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.log_handler import Logger 

logger= Logger('feature_engineering.log')
logger.debug("ste3 - Feature Engineering Logging starts here ")

def feature_construction(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['amenitiesavailable'].unique()
        df['amenitiesavailable'].value_counts()
        df['lift_avail']=df['amenitiesavailable'].apply(lambda x: 'yes' if 'Lift Available' in x else 'no' )
        df['car_parking_avail']=df['amenitiesavailable'].apply(lambda x: 'yes' if 'Car Parking' in x else 'no')
        df['Gas_conn_avail']=df['amenitiesavailable'].apply(lambda x: 'yes' if 'Gas connection' in x else 'no' )
        df['lift_avail'].value_counts()
        df['car_parking_avail'].value_counts()
        df['Gas_conn_avail'].value_counts()
        df.drop(columns=['amenitiesavailable'],inplace=True)
        df.drop(columns=['amenitiesnot'],inplace=True)
        return df

    except Exception as e:
        logger.debug(f"unexpected error while feature construction : {e}")
        raise

def feature_transformation(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_original = df.copy()  # Keep a copy to access original column names
        ordinal_cols = ['neworold','ownership','status','lift_avail','car_parking_avail','Gas_conn_avail']
        categories = [
            ['Resale','New'],
            ['Leasehol','Power of Attorney','Co-Operative Society','Freehold'],
            ['Unfurnished','Semi-Furnished','Furnished','Ready to move'],
            ['no','yes'], 
            ['no','yes'], 
            ['no','yes']
        ]

        logger.debug("Column Transformer starts here")

        preprocessor = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(categories=categories), ordinal_cols)
            ],
            remainder='passthrough'
        )

        # Fit and transform
        df = preprocessor.fit_transform(df)
        logger.debug("Ordinal Encoding successfully completed")

        # Build clean column names
        remainder_cols = [col for col in df_original.columns if col not in ordinal_cols]
        final_columns = ordinal_cols + remainder_cols

        # Convert to DataFrame
        df = pd.DataFrame(df, columns=final_columns)
        return df

    except Exception as e:
        logger.debug(f"unexpected errror while feature transformation : {e}")
        raise

def change_datatype(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['neworold']=df['neworold'].astype('int64')
        df['ownership']=df['ownership'].astype('int64')
        df['status']=df['status'].astype('int64')
        df['lift_avail']=df['lift_avail'].astype('int64')
        df['car_parking_avail']=df['car_parking_avail'].astype('int64')
        df['Gas_conn_avail']=df['Gas_conn_avail'].astype('int64')
        df['age']= df['age'].astype('int64')
        df['area']= df['area'].astype('int64')
        df['balconies']= df['balconies'].astype('int64')
        df['bathroom']= df['bathroom'].astype('int64')
        df['floor']= df['floor'].astype('int64')
        df['price']= df['price'].astype('int64')
        df['price_per_sq']= df['price_per_sq'].astype('int64')
        return df
        
    except Exception as e:
        logger.debug(f"error while datatype changes : {e}")
        raise
        

def main() -> None:
    try:
        df= pd.read_csv('dataset/data_preprocessing/main_data.csv')
        logger.debug('csv file open successfully')

        # featrure construction 
        df= feature_construction(df)
        logger.debug("Feature construction successfully completed") 

        # feature transformation 
        df= feature_transformation(df)
        logger.debug(f"Feature Transformation successfully completed")

        df= change_datatype(df)
        logger.debug(f"columns successfully changes in specific datatype")
        # print(df.info())

        # saving the dataset 
        data_path= 'dataset/feature_engineering'
        os.makedirs(data_path, exist_ok=True)
        dataset= os.path.join(data_path, 'main_data.csv')
        df.to_csv(dataset, index=False)
        logger.debug(f"dataset saved to {dataset} folder successfully") 
        
    except Exception as e:
        logger.debug(f"Unexpected occur {e}")

if __name__ == "__main__" :
    main()

