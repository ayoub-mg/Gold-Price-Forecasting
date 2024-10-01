# Import the required libraries for the machine learning application.
import joblib
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

st.title('Predicting the Adjusted closing price of Gold ETF ')

Low = st.number_input('Insert the Low price of Gold')
GDX_Adj_Close = st.number_input('Insert Gold Miners ETF (GDX_Adj Close) price')
SF_Open = st.number_input('Insert the Silver Futures open(SF_open) price')
EG_Open = st.number_input('Insert the Eldorado Gold Corporation (EG_open) price')
PLT_Open = st.number_input('Insert the Platinum Price (PLT_open)')
USDI_Open = st.number_input('Insert the US Dollar Index (USDI_Open) price')
OF_Price = st.number_input('Insert the Brent Crude Oil Futures (OF_Price) price')
SF_Volume = st.number_input('Insert the Silver Futures volume (SF_Volume) price')

# If button is pressed.
if st.button("Submit"):

   # Unpickle classifier.
   model = joblib.load('./model/gold_forecast_model.pkl')


   # Store inputs into dataframe.
   X = [Low, GDX_Adj_Close, SF_Open, EG_Open, PLT_Open, USDI_Open, OF_Price, SF_Volume]
  
   X = np.array(X).reshape(1,-1)

   # Predition.
   prediction = model.predict(X)
    
   # Output prediction.
   st.text(f"The model predicts a future gold price of ${round(prediction[0], 2)}.")
  