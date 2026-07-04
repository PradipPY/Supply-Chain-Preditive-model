import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("--- Loading and Preprocessing Data ---")
df = pd.read_csv('DataCoSupplyChainDatasetRefined.csv', encoding='latin1')

keep_columns = [
    'late_delivery_risk', 'type', 'days_for_shipment_scheduled',
    'customer_segment', 'department_name', 'market',
    'order_region', 'order_item_quantity', 'order_item_product_price', 'shipping_mode'
]
df_clean = df[keep_columns].copy()

X = df_clean.drop(columns=['late_delivery_risk'])
y = df_clean['late_delivery_risk']

# One-hot encoding text features
X_encoded = pd.get_dummies(X, columns=['type', 'customer_segment', 'department_name',
                           'market', 'order_region', 'shipping_mode'], drop_first=True)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)

print("\n--- Training Logistic Regression Model ---")

# 1. Initialize the model
# We increase max_iter so the model has enough time to find the best mathematical solution
model = LogisticRegression(max_iter=1000)

# 2. Train the model using our training sets
model.fit(X_train, y_train)
print("Model training complete!")

# 3. Predict the delivery outcomes on the hidden test set
y_pred = model.predict(X_test)

# 4. Evaluate how well our model performed
print("\n--- Model Evaluation Results ---")
print(f"Overall Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
