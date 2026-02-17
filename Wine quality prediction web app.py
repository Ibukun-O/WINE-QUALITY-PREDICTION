# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 22:14:51 2026

@author: USER
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Documents/trained_model.sav', 'rb'))


# creating a function for prediction
def wine_quality_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):
      return 'Good Quality Wine'
    else:
      return 'Bad Quality Wine'
  
    
def main():
    
    # giving it a title
    st.title('Wine Quality Prediction Web App')
    
    
    # getting input data from the user
   
    fixed_acidity = st.text_input('Fixed acidity value')
    volatile_acidity = st.text_input('Volatile acidity value')
    citric_acid = st.text_input('Citric acid value')
    residual_sugar = st.text_input('Residual sugar value')
    chlorides = st.text_input('Chlorides value')
    free_sulfur_dioxide = st.text_input('Free sulphur dioxide value')
    total_sulfur_dioxide = st.text_input('Total sulfur dioxide value')
    density = st.text_input('Density value')
    pH = st.text_input('pH value')
    sulphates = st.text_input('Sulphates value')
    alcohol = st.text_input('alcohol')
    
    # code for prediction
    quality_check = ''

    # creating a button for prediction
    
    if st.button('quality_check_results'):
        quality_check = wine_quality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
    
    st.success(quality_check)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
