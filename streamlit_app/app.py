import streamlit as st
import numpy as np
import pickle as pkl
import mlflow 
import dagshub 
import os

# pipe=pkl.load(open('app/pipe.pkl','rb'))
df=pkl.load(open('app/df.pkl','rb'))

st.title('Pune House Price Prediction')

# # Below code block is for production use
# # -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("HOUSE_PRICE_TOKEN")
# if not dagshub_token:
#     raise EnvironmentError("HOUSE_PRICE_TOKEN environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "nitinbdkt777"
# repo_name = "house-price-predictor"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# dagshub locally 
mlflow.set_tracking_uri("https://dagshub.com/nitinbdkt777/house-price-predictor.mlflow")
dagshub.init(repo_owner="nitinbdkt777" , repo_name= "house-price-predictor", mlflow=True)

def load_latest_version_model(model_name):
    client = mlflow.MlflowClient()
    latest_version= client.get_latest_versions(model_name, stages=['Production'])

    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name= "Xgboost"
model_version= load_latest_version_model(model_name)
model_uri= f"models:/{model_name}/{model_version}"

@st.cache_resource
def load_model():
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise 
    
model= load_model()
    
def app():
    # age
    age=st.number_input('How Old House',value=0, step=1)

    # total area
    area=st.number_input('Total Area(In square feet))', value=100, step=10 )

    # balconies
    balconies=st.selectbox('Balcony', [1,2,3])

    #bathroom 
    bathroom= st.selectbox('Bathroom',[1,2,3,4,5])

    # bhk
    bhk= st.selectbox('BHK(Apartment or Villa)',df['bhk'].unique())

    # floor
    floor=st.number_input('House Floor', value=1, step=1 )

    # place
    place=st.selectbox('Place',df['place'].unique())

    # price_per_square
    price_per_sq=st.number_input('Price Per Square(in Feet)', value=100, step=10)

    # new or old
    neworold= st.selectbox('New or Resale', ['New','Resale'])

    # ownership
    ownership= st.selectbox('Ownership', ['Leasehol','Power of Attorney','Co-Operative Society','Freehold'])

    # status
    status= st.selectbox('Ownership', ['Unfurnished','Semi-Furnished','Furnished','Ready to move'])

    # lift_avail
    lift_avail=st.selectbox('Lift Availble?',['No','yes'])

    # car_parking_avail
    car_parking_avail= st.selectbox('Car Parking Availble?',['No','yes'])

    # gas_connection_availble
    Gas_conn_avail=st.selectbox('Gas Connection Availble?',['No','yes'])

    if st.button('Predict'):
        # new or old
        if neworold=="Resale":
            neworold=0
        else:
            neworold=1

        #ownership
        if ownership== "Leasehol":
            ownership=0
        elif ownership=="Power of Attorney":
            ownership=1
        elif ownership=='Co-Operative Society':
            ownership=2
        else:
            ownership=3

        # status
        if status=="Unfurnished":
            status=0
        elif status=="Semi-Furnished":
            status=1
        elif status=="Furnished":
            status=2
        else:
            status=3
        
        # lift_avail
        if lift_avail=='No':
            lift_avail=0
        else:
            lift_avail=1

        # car_parking
        if car_parking_avail=='No':
            car_parking_avail=0
        else:
            car_parking_avail=1

        # Gas_conn_avail
        if Gas_conn_avail=='No':
            Gas_conn_avail=0
        else:
            Gas_conn_avail=1
        
        query=np.array([age,area,balconies,bathroom,floor,status,price_per_sq,neworold,ownership,lift_avail,place,car_parking_avail,bhk,Gas_conn_avail])
        query= query.reshape(1,14)
        st.title("This House Price Is: " + str(int(np.exp(model.predict(query)[0]))) + " \u20B9 ")

if __name__ == "__main__":
    app()