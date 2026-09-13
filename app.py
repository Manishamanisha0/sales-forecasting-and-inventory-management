import streamlit as st
import pandas as pd
import joblib
import json
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Based Sales Forecasting and Inventory Management System")
st.write("Predict sales using a trained machine learning model.")

# -----------------------------
# File Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "sales_forecasting_model.pkl"
)

features_path = os.path.join(
    BASE_DIR,
    "forecast_features.json"
)

# -----------------------------
# Check Files
# -----------------------------
if not os.path.exists(model_path):
    st.error("sales_forecasting_model.pkl not found.")
    st.stop()

if not os.path.exists(features_path):
    st.error("forecast_features.json not found.")
    st.stop()

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load(model_path)

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Load Features from JSON
# -----------------------------
try:
    with open(features_path, "r") as f:
        forecast_features = json.load(f)

except Exception as e:
    st.error(f"Error loading features: {e}")
    st.stop()

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Input Values")

input_data = {}

for feature in forecast_features:
    input_data[feature] = st.number_input(
        f"Enter {feature}",
        value=0.0
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Sales"):

    input_df = pd.DataFrame(
        [input_data],
        columns=forecast_features
    )

    try:
        prediction = model.predict(input_df)

        st.success(
            f"Predicted Sales: {prediction[0]:.2f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
