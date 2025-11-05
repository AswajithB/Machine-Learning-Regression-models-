import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# ------------------------------
# Load Saved Files
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
le_fuel = pickle.load(open("Fuel_Type.pkl", "rb"))
le_trans = pickle.load(open("Transmission.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

# ------------------------------
# App Title
# ------------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Prediction App")
st.markdown("Predict the **selling price** of your car based on its features.")

st.write("---")

# ------------------------------
# Two Columns for Feature Inputs
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.1, max_value=50.0, step=0.1)
    kms_driven = st.number_input("Kms Driven", min_value=100, max_value=200000, step=100)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col2:
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    year = st.slider("Year of Purchase", 2000, datetime.now().year, 2015)
    car_age = datetime.now().year - year

# ------------------------------
# Predict Button
# ------------------------------
st.write("---")
if st.button("🔍 Predict Selling Price"):

    # Prepare Input Data
    input_data = pd.DataFrame({
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [le_fuel.transform([fuel_type])[0]],
        "Seller_Type_Individual": [1 if seller_type == "Individual" else 0],
        "Transmission": [le_trans.transform([transmission])[0]],
        "Owner": [owner],
        "Car_Age": [car_age]
    })

    # Align with training columns
    expected_cols = list(scaler.feature_names_in_)
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_cols]

    # Scale and Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")

# ---------------------------