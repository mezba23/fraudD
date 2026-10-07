"""
Flask Inference Service for Real-time Transaction Fraud Prediction.
"""

from flask import Flask, request, jsonify
import numpy as np
from train_model import train_and_evaluate

app = Flask(__name__)
model = train_and_evaluate()


@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "FraudD ML Scoring Engine"})


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    try:
        tx_amount = float(data.get("transaction_amount", 50.0))
        credit_score = float(data.get("user_credit_score", 700.0))
        dist_km = float(data.get("distance_from_home_km", 5.0))
        velocity = float(data.get("velocity_1h", 1.0))

        features = np.array([[tx_amount, credit_score, dist_km, velocity]])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        return jsonify({
            "is_fraud": bool(pred == 1),
            "fraud_probability": round(float(prob), 4),
            "risk_verdict": "HIGH RISK" if prob > 0.60 else ("MEDIUM RISK" if prob > 0.30 else "LOW RISK")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5001, debug=True)
