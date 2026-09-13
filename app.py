import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "sales_forecasting_model_compressed.pkl"
)

features_path = os.path.join(
    BASE_DIR,
    "forecast_features.pkl"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found: {model_path}"
    )

if not os.path.exists(features_path):
    raise FileNotFoundError(
        f"Features file not found: {features_path}"
    )

model = joblib.load(model_path)
forecast_features = joblib.load(features_path)



