import os
import json
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "sales_forecasting_model.pkl"
)

features_path = os.path.join(
    BASE_DIR,
    "forecast_features.json"
)

if not os.path.exists(model_path):
    st.error("sales_forecasting_model.pkl not found.")
    st.stop()

if not os.path.exists(features_path):
    st.error("forecast_features.json not found.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Load features
with open(features_path, "r") as f:
    forecast_features = json.load(f)

st.success("Model and features loaded successfully!")
