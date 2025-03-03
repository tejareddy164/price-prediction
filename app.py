import streamlit as st
import pandas as pd
import joblib
import os

# Check if model files exist
if not os.path.exists("price_model.pkl") or not os.path.exists("label_encoders.pkl"):
    st.error("Model files not found! Ensure 'price_model.pkl' and 'label_encoders.pkl' are in the project directory.")
    st.stop()

# Load the trained model and encoders
model = joblib.load("price_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Streamlit UI
st.title("🛒 Shop Price Prediction App")
st.write("Enter product details to predict the price.")

# User input fields
item_name = st.text_input("Item Name")
category = st.selectbox("Category", ["Grocery", "Clothing", "Electronics", "Winter", "Summer", "Spring", "Fall"])

# Predict price button
if st.button("Predict Price"):
    try:
        # Encode categorical data
        category_encoded = encoders["Category"].transform([category])[0]

        # Create input DataFrame
        input_data = pd.DataFrame([[category_encoded]], columns=["Category"])

        # Predict price
        predicted_price = model.predict(input_data)[0]

        # Display result
        st.success(f"Estimated Price: ₹{predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Footer
st.write("Developed by Your Name 🚀")
