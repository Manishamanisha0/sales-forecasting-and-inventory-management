import streamlit as st
import joblib
import json
import os
from pathlib import Path

# ---------------------------------
# Project directory
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "sales_forecasting_model.pkl"
features_path = BASE_DIR / "forecast_features.json"

# ---------------------------------
# Check model file
# ---------------------------------
if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.write("Files available:", os.listdir(BASE_DIR))
    st.stop()

if not model_path.is_file():
    st.error("sales_forecasting_model.pkl is not a file.")
    st.stop()

# ---------------------------------
# Check feature file
# ---------------------------------
if not features_path.exists():
    st.error(f"Feature file not found: {features_path}")
    st.stop()

# ---------------------------------
# Load model
# ---------------------------------
try:
    st.write("Loading model...")

    model = joblib.load(str(model_path))

    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {repr(e)}")
    st.write("Model path:", str(model_path))
    st.write("Model size:", model_path.stat().st_size, "bytes")
    st.stop()

# ---------------------------------
# Load features
# ---------------------------------
try:
    with open(features_path, "r", encoding="utf-8") as f:
        forecast_features = json.load(f)

    st.success("Features loaded successfully!")

except Exception as e:
    st.error(f"Error loading features: {repr(e)}")
    st.stop()
