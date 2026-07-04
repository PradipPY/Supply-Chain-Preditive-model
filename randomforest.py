import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data
df = pd.read_csv('DataCoSupplyChainDatasetRefined.csv', encoding='latin1')

keep_columns = [
    'late_delivery_risk', 'type', 'days_for_shipment_scheduled',
    'customer_segment', 'department_name', 'market',
    'order_region', 'order_item_quantity', 'order_item_product_price', 'shipping_mode'
]
df_clean = df[keep_columns].copy()

X = df_clean.drop(columns=['late_delivery_risk'])
y = df_clean['late_delivery_risk']

categorical_features = ['type', 'customer_segment',
                        'department_name', 'market', 'order_region', 'shipping_mode']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("--- Building Random Forest Pipeline ---")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore',
         drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

pipeline_rf = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
])

# Train
print("\n--- Training Random Forest (100 Trees Voting) ---")
pipeline_rf.fit(X_train, y_train)
print("Random Forest training complete!")

# Evaluate (ADDED BACK THE METRICS)
y_pred_rf = pipeline_rf.predict(X_test)
print(
    f"\nRandom Forest Accuracy on Raw Test Data: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rf))

# =========================================================
# 🔮 PREDICTING A COMPLETELY RAW NEW ORDER (RANDOM FOREST)
# =========================================================
print("\n--- Testing Random Forest Pipeline with a Mock Live Order ---")

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

prediction = pipeline_rf.predict(live_order)
probabilities = pipeline_rf.predict_proba(live_order)

print(f"Prediction Result: Class {prediction[0]}")
if prediction[0] == 1:
    print("🚨 ACTION REQUIRED: Random Forest flags this order as HIGH RISK for delay!")
else:
    print("✅ SAFE: Random Forest predicts this order will arrive on time.")

print(
    f"Forest Voting Breakdown: {probabilities[0][0]*100:.1f}% of trees voted On-Time | {probabilities[0][1]*100:.1f}% of trees voted Late")
