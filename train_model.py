"""
Credit Scoring and Financial Fraud Classification Model Training Pipeline.
Trains a Logistic Regression / Random Forest baseline on synthetic transaction data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def generate_synthetic_data(n_samples: int = 2000):
    np.random.seed(42)
    transaction_amount = np.random.exponential(scale=50, size=n_samples)
    user_risk_score = np.random.uniform(300, 850, size=n_samples)
    distance_from_home_km = np.random.exponential(scale=15, size=n_samples)
    velocity_tx_1h = np.random.poisson(lam=2, size=n_samples)

    # Fraud probability heuristic
    logits = (
        0.02 * transaction_amount
        - 0.005 * user_risk_score
        + 0.05 * distance_from_home_km
        + 0.4 * velocity_tx_1h
        - 1.5
    )
    probs = 1 / (1 + np.exp(-logits))
    is_fraud = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        "transaction_amount": transaction_amount,
        "user_credit_score": user_risk_score,
        "distance_from_home_km": distance_from_home_km,
        "velocity_1h": velocity_tx_1h,
        "is_fraud": is_fraud
    })
    return df


def train_and_evaluate():
    print("[FraudD] Generating dataset and training models...")
    df = generate_synthetic_data()
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"[FraudD] Logistic Regression ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    train_and_evaluate()
