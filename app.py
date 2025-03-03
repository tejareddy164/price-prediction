import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained models and label encoders
try:
    model_price = joblib.load("price_model.pkl")
    model_discount = joblib.load("discount_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
except FileNotFoundError:
    st.error("Model files not found! Ensure 'price_model.pkl', 'discount_model.pkl', and 'label_encoders.pkl' are in the project directory.")
    st.stop()

# Streamlit UI
st.title("🛒 Shop Price & Discount Prediction")
st.write("Enter product details to predict the selling price and discount.")

# User inputs
item_name = st.text_input("Enter Item Name")
stock = st.number_input("Stock Available", min_value=1, step=1)
seasonal_demand = st.number_input("Seasonal Demand", min_value=0, step=1)

# Predict button
if st.button("Predict Price & Discount"):
    try:
        # Encode item name
        item_encoded = encoders["Item"].transform([item_name])[0]

        # Prepare input data
        input_features = np.array([[item_encoded, stock, seasonal_demand]])

        # Predict price & discount
        predicted_price = model_price.predict(input_features)[0]
        predicted_discount = model_discount.predict(input_features)[0]

        # Display result
        st.success(f"Predicted Selling Price: ₹{predicted_price:.2f}")
        st.success(f"Predicted Discount: ₹{predicted_discount:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Footer
st.write("Developed by Your Name 🚀")
