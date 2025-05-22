import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_and_save(df: pd.DataFrame, test_size: float , random_state: int) :
    try:
        # spliting into two parts
        df['price'].dtype
        x=df.drop(columns=['price'])
        y=np.log(df['price'])

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
        raise
    