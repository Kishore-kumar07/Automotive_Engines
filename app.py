import joblib
import streamlit as st
import numpy as np

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example: Predict function in Streamlit
def predict_engine_condition(input_data):
    # Scale the input data using the scaler
    scaled_data = scaler.transform([input_data])
    
    # Predict using the trained model
    prediction = model.predict(scaled_data)
    
    return prediction

# Streamlit interface
st.title("Engine Condition Prediction")

# Collect user inputs for each feature
engine_rpm = st.number_input('Engine rpm', min_value=0)
lub_oil_pressure = st.number_input('Lub oil pressure', min_value=0.0)
fuel_pressure = st.number_input('Fuel pressure', min_value=0.0)
coolant_pressure = st.number_input('Coolant pressure', min_value=0.0)
lub_oil_temp = st.number_input('Lub oil temp', min_value=0.0)
coolant_temp = st.number_input('Coolant temp', min_value=0.0)

# Input list to be passed for prediction
input_features = [engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]

# Predict button
if st.button('Predict Engine Condition'):
    result = predict_engine_condition(input_features)
    st.write(f"Predicted Engine Condition: {result[0]}")

