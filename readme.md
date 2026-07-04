# Enterprise Supply Chain: Late Delivery Risk Prediction Pipeline

An end-to-end machine learning project built to proactively calculate and classify operational risks and shipment delays using the comprehensive DataCo Supply Chain enterprise dataset.

## Executive Summary & Core Objective

In global logistics, a late delivery is not just an inconvenience—it translates directly to contractual financial penalties, diminished customer retention, and fractured supply chains.

The core objective of this engineering architecture is to build an intelligent, deployable binary classifier capable of calculating the risk of a delivery being delayed at the exact millisecond an order is placed. This gives fulfillment centers a predictive window to prioritize high-risk packages before they even leave the loading dock.

## Production Repository Architecture

To maintain enterprise-grade reproducibility and isolate dependencies, the repository is structured as a single, multi-model command center:

```
Supply-Chain-Predictive-model/
│
├── .venv/                      # Isolated virtual environment sandbox
├── .gitignore                  # Prevents tracking raw heavy data and hidden dependencies
├── DataCoSupplyChainDatasetRefined.csv # Core dataset (180,519 records)
│
├── logistic_regression.py      # Script: Baseline Linear Classifier
├── decision_tree.py            # Script: Production Pipeline Tree Classifier
├── randomforest.py             # Script: Ensemble Voting Production Pipeline
│
└── README.md                   # Master Technical Dashboard & Reference Guide
```

## Step-by-Step Data Engineering & Pipeline Design

### 1. The Real-World Data Availability Constraint

A common trap in data science is training a model on data it won't have access to in production. To prevent this, I strictly removed all post-delivery features (e.g., actual days to ship, delivery status updates). The models are trained only on features known at checkout:

**Categorical Inputs:** Payment Type, Customer Segment, Department Name, Market, Order Region, Shipping Mode.

**Numerical Inputs:** Scheduled Shipment Days, Item Quantity, Product Price.

### 2. Preventing Data Leakage via Scikit-Learn Pipelines

Instead of applying transformations globally using `pd.get_dummies()`, which leaks statistical properties from the test set into the training set, this repository upgrades the codebase to Scikit-Learn Pipeline and ColumnTransformer modules.

The data conveyor belt functions as follows:

```
Raw Live DataFrame ──> [ColumnTransformer: OneHotEncoder] ──> [Trained Classifier] ──> Live Prediction
```

`OneHotEncoder(handle_unknown='ignore', drop='first')`: Automatically transforms text categories into a mathematical binary space (expanding inputs out to 47 features) while safely ignoring unseen categories in production without crashing.

**Pipeline Integration:** Binds the encoder rules and model mathematical weights into a single compiled object that can accept a raw string-based Python dictionary instantly.

## Data Breakdown & Ingestion Split

**Total Volume:** 180,519 operational rows.

**Cleanliness:** 0 missing values across target parameters.

**Target Balance (late_delivery_risk):**
- Class 1 (Late Delivery): 54.8%
- Class 0 (On-Time Delivery): 45.2%

**Data Splitting Strategy:** The dataset was separated using an 80/20 split locked with a deterministic seed (random_state=42) yielding:
- Training Set: 144,415 rows (used to fit structural weights).
- Testing Set: 36,104 rows (held out strictly to simulate completely unknown data).

## Algorithmic Journey & Leaderboard

I iteratively built and evaluated three distinct families of machine learning algorithms to map the complexity of the supply chain network.

| Architecture | Test Accuracy | Late (1) Precision | Late (1) Recall | Technical Analysis & Core Takeaway |
|---|---|---|---|---|
| **Model 1: Logistic Regression** | 69.17% | 84% | 54% | Linear Baseline: Highly conservative. High precision means it rarely gives false alarms, but its linear boundary misses 46% of actual late shipments. |
| **Model 2: Decision Tree (depth=10)** | 69.01% | 83% | 54% | Non-Linear Flowchart: I discovered that an unconstrained tree overfits wildly to a depth of 45 layers. Limiting it to a depth of 10 stabilized the metrics but hit an identical performance wall. |
| **Model 3: Random Forest (100 Trees)** | 69.32% | 82% | 57% | Ensemble Voting: Leverages 100 micro-trees voting concurrently. Brought a slight 3% lift to target recall, confirming a persistent statistical plateau. |

## Crucial Data Science Finding: The Informational Ceiling

When three radically different algorithms (Linear, Single-Tree Flowchart, and Multi-Tree Forest Ensemble) all hit the exact same ceiling (~69% accuracy, ~54-57% recall), the problem is no longer the model; it is the data. Because the checkout dataset lacks real-time operational context (such as severe weather events, customs clearance backlogs, or warehouse under-staffing), different packages with the exact same checkout profile end up with conflicting historical outcomes. The models are playing the statistical averages perfectly, proving they have extracted 100% of the available signal from the current features.

## Live Production Operational Inference

Both `decision_tree.py` and `randomforest.py` terminate with a live deployment simulation engine. Rather than testing on pre-processed metrics, they accept a raw, unstructured string dataframe mirroring a live API call from an e-commerce platform:

```python
# Simulated real-time order payload
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

# Single execution line handles encoding transformation and matrix multiplication
prediction = pipeline_rf.predict(live_order)
probabilities = pipeline_rf.predict_proba(live_order)
```

### The Tie-Breaker Phenomenon

During testing, the Random Forest ensemble demonstrated the boundary limits of the dataset by returning a perfect 50.0% On-Time / 50.0% Late tie on the mock order. In binary classification ties, Scikit-Learn defaults to the first index, classifying it as Class 0 (SAFE). This highlights the importance of checking probability confidence thresholds rather than just looking at the final hard classification label.

## Environment Initialization & Execution Guide

To stand up this project locally and replicate the terminal metrics:

**Activate the pre-configured virtual environment:**

```bash
.venv\Scripts\activate
```

**Execute the Baseline Linear Classifier:**

```bash
python logistic_regression.py
```

**Execute the Pipeline-driven Decision Tree Classifier:**

```bash
python decision_tree.py
```

**Execute the 100-Tree Ensemble Majority Voting Pipeline:**

```bash
python randomforest.py
```
