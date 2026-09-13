import streamlit as st
import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "sales_forecasting_model.pkl"
)

features_path = os.path.join(
    BASE_DIR,
    "forecast_features.json"
)

# Check model
if not os.path.isfile(model_path):
    st.error("sales_forecasting_model.pkl not found.")
    st.stop()

# Check features
if not os.path.isfile(features_path):
    st.error("forecast_features.json not found.")
    st.stop()

# Load model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {repr(e)}")
    st.stop()

# Load features
try:
    with open(features_path, "r", encoding="utf-8") as f:
        forecast_features = json.load(f)
except Exception as e:
    st.error(f"Error loading features: {repr(e)}")
    st.stop()

st.success("Model and features loaded successfully!")
