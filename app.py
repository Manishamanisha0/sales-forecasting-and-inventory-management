import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and features
try:
    model = joblib.load("sales_forecasting_model.pkl")
    features = joblib.load("forecast_features.pkl")
except FileNotFoundError:
    st.error("Model or features file not found. Please ensure 'sales_forecasting_model.pkl' and 'forecast_features.pkl' are in the directory.")
    st.stop()

# Load the cleaned sales data for the get_product_forecast function
try:
    df_cleaned = pd.read_csv("cleaned_sales_data.csv")
    df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])
except FileNotFoundError:
    st.error("cleaned_sales_data.csv not found. Please ensure it's in the same directory.")
    st.stop()

# Load inventory recommendations
try:
    inventory_report = pd.read_csv("inventory_recommendations.csv")
except FileNotFoundError:
    st.error("inventory_recommendations.csv not found. Please ensure it's in the same directory.")
    st.stop()

def get_product_forecast_streamlit(product_id):
    # Ensure df_cleaned is available
    if df_cleaned.empty:
        return "Error: Cleaned data not loaded."

    product_data = df_cleaned[df_cleaned["Product_ID"] == product_id].copy()

    if product_data.empty:
        return "Product not found in historical data."

    latest_data = product_data.iloc[-1]

    # Create input_data for prediction, ensuring all 'features' are present
    input_data_dict = {}
    for feature in features:
        if feature in latest_data:
            input_data_dict[feature] = [latest_data[feature]]
        else:
            # This case should ideally not be reached if features are consistent
            st.warning(f"Feature '{feature}' not found in latest product data for prediction.")
            input_data_dict[feature] = [0] # Use a placeholder if missing

    input_data = pd.DataFrame(input_data_dict)

    prediction = model.predict(input_data)[0]

    return round(max(prediction, 0), 2)

# Streamlit App
st.title("Sales Forecasting and Inventory Management Dashboard")

st.header("Sales Forecast")
product_id_options = df_cleaned["Product_ID"].unique()
if product_id_options.size > 0:
    product_id_input = st.selectbox(
        "Select Product ID for Sales Forecast:",
        product_id_options
    )

    if product_id_input:
        forecast = get_product_forecast_streamlit(product_id_input)
        if isinstance(forecast, str):
            st.write(forecast)
        else:
            st.success(f"Predicted Sales for {product_id_input} for the next period: {forecast} units")
else:
    st.warning("No product data available for forecasting.")

st.header("Inventory Status Overview")

# Display inventory report
st.dataframe(inventory_report)

# Filter for reorder required products
st.subheader("Products Requiring Reorder")
reorder_needed = inventory_report[
    inventory_report["Inventory_Status"] == "Reorder Required"
]

if not reorder_needed.empty:
    st.dataframe(reorder_needed)
else:
    st.info("No products currently require reordering.")
