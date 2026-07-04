import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data
df = pd.read_csv('DataCoSupplyChainDatasetRefined.csv', encoding='latin1')

keep_columns = [
    'late_delivery_risk', 'type', 'days_for_shipment_scheduled',
    'customer_segment', 'department_name', 'market',
    'order_region', 'order_item_quantity', 'order_item_product_price', 'shipping_mode'
]
df_clean = df[keep_columns].copy()

# Keep X completely RAW (no pd.get_dummies here!)
X = df_clean.drop(columns=['late_delivery_risk'])
y = df_clean['late_delivery_risk']

# Identify our text columns vs our numerical columns
categorical_features = ['type', 'customer_segment',
                        'department_name', 'market', 'order_region', 'shipping_mode']
numerical_features = ['days_for_shipment_scheduled',
                      'order_item_quantity', 'order_item_product_price']

# Split into train and test sets using the raw data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("--- Building the Production Pipeline ---")

# Step A: Create a preprocessor that only applies One-Hot Encoding to the text columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore',
         drop='first'), categorical_features)
    ],
    remainder='passthrough'  # Leave numerical columns alone
)

# Step B: Glue the preprocessor and the Decision Tree together into a single Pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])

# 2. Train the ENTIRE pipeline at once
# This trains both the encoder rules AND the decision tree equations simultaneously
pipeline.fit(X_train, y_train)
print("Pipeline training complete!")

# 3. Evaluate the pipeline
y_pred = pipeline.predict(X_test)
print(
    f"\nPipeline Accuracy on Raw Test Data: {accuracy_score(y_test, y_pred):.4f}")

# =========================================================
# 🔮 PREDICTING A COMPLETELY RAW NEW ORDER
# =========================================================
print("\n--- Testing Pipeline with a Mock Live Order ---")

# This looks exactly like data coming fresh out of a checkout database!
live_order = pd.DataFrame([{
    'type': 'TRANSFER',
    'days_for_shipment_scheduled': 3,
    'customer_segment': 'Consumer',
    'department_name': 'Apparel',
    'market': 'LATAM',
    'order_region': 'South America',
    'order_item_quantity': 2,
    'order_item_product_price': 55.00,
    'shipping_mode': 'Standard Class'
}])

# Feed the RAW dataframe directly into the pipeline!
prediction = pipeline.predict(live_order)
probabilities = pipeline.predict_proba(live_order)

print(f"Prediction Result: Class {prediction[0]}")
print(
    f"Confidence: {probabilities[0][0]*100:.1f}% On-Time | {probabilities[0][1]*100:.1f}% Late")
