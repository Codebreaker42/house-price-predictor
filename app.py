import streamlit as st
import numpy as np
st.title('Pune House Price Prediction')

import pickle as pkl
pipe=pkl.load(open('pipe.pkl','rb'))
df=pkl.load(open('df.pkl','rb'))

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
    
    query=np.array([age,area,balconies,bathroom,bhk,floor,place,price_per_sq,neworold,ownership,status,lift_avail,car_parking_avail,Gas_conn_avail])
    query= query.reshape(1,14)
    st.title("This House Price Is: " + str(int(np.exp(pipe.predict(query)[0]))) + " \u20B9 ")
