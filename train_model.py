import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("shop_dataset_large.csv")

# Encode categorical column
label_encoders = {}
for column in ["Item"]:  # Add other categorical columns if needed
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Features and targets
X = df[["Item", "Stock", "Seasonal_Demand"]]
y_price = df["Base_Price"]
y_discount = df["Discount"]

# Train-test split
X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)
_, _, y_train_discount, y_test_discount = train_test_split(X, y_discount, test_size=0.2, random_state=42)

# Train XGBoost models
model_price = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model_discount = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

model_price.fit(X_train, y_train_price)
model_discount.fit(X_train, y_train_discount)

# Save models
joblib.dump(model_price, "price_model.pkl")
joblib.dump(model_discount, "discount_model.pkl")

# Test predictions
y_pred_price = model_price.predict(X_test)
y_pred_discount = model_discount.predict(X_test)

# Print accuracy
print("Price Model Error:", mean_absolute_error(y_test_price, y_pred_price))
print("Discount Model Error:", mean_absolute_error(y_test_discount, y_pred_discount))
