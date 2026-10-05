"""
app.py
------
Flask server for FutureWealth Predictor (INR Currency & Milestone Breakdown).
"""

import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
import joblib

app = Flask(__name__, static_folder="static")

MODEL_PATH = "model/futurewealth_model.joblib"
METRICS_PATH = "model/metrics.json"
model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

NUMERIC_FEATURES = ["salary", "experience_years", "upskill_spend",
                     "invest_pct", "risk_tolerance", "dti_ratio"]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def simulate(salary0, growth_rate, invest_pct, return_rate, dti):
    """Compound a 10-year salary + net worth path with 3, 5, 10 year milestones."""
    dti_drag = clamp(1 - (dti / 100) * 0.8, 0.2, 1)
    monthly_return = return_rate / 100 / 12

    salary = salary0
    net_worth = 0.0
    years = [0]
    salary_series = [salary0]
    nw_series = [0.0]

    milestones = {}

    for year in range(1, 11):
        year_growth = growth_rate * (0.985 ** (year - 1))
        salary = salary * (1 + year_growth / 100)

        monthly_contribution = (salary / 12) * (invest_pct / 100) * dti_drag
        for _ in range(12):
            net_worth = net_worth * (1 + monthly_return) + monthly_contribution

        years.append(year)
        salary_series.append(salary)
        nw_series.append(net_worth)

        if year in [3, 5, 10]:
            milestones[f"year_{year}"] = {
                "salary": round(salary, 2),
                "net_worth": round(net_worth, 2)
            }

    return {
        "years": years,
        "salary": salary_series,
        "net_worth": nw_series,
        "milestones": milestones
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    try:
        row = {
            "salary": float(payload["salary"]),
            "experience_years": float(payload["experience_years"]),
            "upskill_spend": float(payload["upskill_spend"]),
            "invest_pct": float(payload["invest_pct"]),
            "risk_tolerance": float(payload["risk_tolerance"]),
            "dti_ratio": float(payload["dti_ratio"]),
            "industry": str(payload["industry"]),
        }
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    X = pd.DataFrame([row])[NUMERIC_FEATURES + ["industry"]]
    growth_rate, return_rate, recommended_invest_pct = model.predict(X)[0]
    growth_rate = float(np.clip(growth_rate, 1.0, 18.0))
    return_rate = float(np.clip(return_rate, 4.0, 15.0))
    recommended_invest_pct = float(np.clip(recommended_invest_pct, 10, 50))

    current_run = simulate(row["salary"], growth_rate, row["invest_pct"], return_rate, row["dti_ratio"])
    recommended_run = simulate(row["salary"], growth_rate, recommended_invest_pct, return_rate, row["dti_ratio"])

    return jsonify({
        "prediction": {
            "growth_rate_pct": growth_rate,
            "return_rate_pct": return_rate,
            "recommended_invest_pct": recommended_invest_pct,
        },
        "current_run": current_run,
        "recommended_run": recommended_run,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
