import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoders
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
        st.error("Error in prediction. Please check input values.")

# Footer
st.write("Developed by Your Name 🚀")
