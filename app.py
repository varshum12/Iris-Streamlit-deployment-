# Write your code here
import streamlit as st
import pandas as pd
from utils import predict_species
import joblib

model  =  joblib.load("notebook/iris_model.joblib")

st.set_page_config(page_title= "Iris end to end project deployment" , layout  = 'wide')
st.title("Iris Project Deployment")
st.subheader("By Varsha  Mhetre" )

sep_len  =  st.number_input("sepal_length"  , min_value = 0.00 , step= 0.01  )
sep_wid  =  st.number_input("sepal_width"  , min_value = 0.00 , step= 0.01  )
pet_len  =  st.number_input("petal_length"  , min_value = 0.00 , step= 0.01  )
pet_wid  =  st.number_input("petal_width ", min_value = 0.00 , step= 0.01  )



button  =  st.button("Prediction" , type  = "primary")

if button:
    st.subheader("Result")
    
    

    pred ,  prob  = predict_species( model ,  sep_len  , sep_wid  ,  pet_len  ,  pet_wid)
    st.subheader("prediction")
    st.subheader(f"Prediction : {pred}")
    st.subheader("probabilities")
  
    st.dataframe(prob)

