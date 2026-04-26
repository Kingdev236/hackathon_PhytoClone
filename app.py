from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

readings_a = []  # Plant A — source
readings_b = []  # Plant B — receiver

# ── Load ML model ─────────────────────────────────────
MODEL_PATH = "stress_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("ML model loaded from stress_model.pkl")
    USE_ML = True
else:
    print("No model found — using simple rules. Run train_model.py first.")
    USE_ML = False

# ── Stress detection ──────────────────────────────────
def get_stress(voltage_mv):
    if USE_ML:
        reading = np.array([[voltage_mv]])
        result  = model.predict(reading)
        score   = model.score_samples(reading)[0]
        if result[0] == 1:
            return "Healthy", round(score, 3)
        else:
            return ("High Stress" if score < -0.6 else "Mild Stress"), round(score, 3)
    else:
        if voltage_mv > 1.5:  return "Healthy", 0
        if voltage_mv > 0.5:  return "Mild Stress", 0
        return "High Stress", 0

# ── Receive Plant A data ──────────────────────────────
@app.route("/data/a", methods=["POST"])
def receive_a():
    d = request.json
    d["timestamp"] = time.time()
    stress, score  = get_stress(d.get("voltage_mv", 0))
    d["stress"]        = stress
    d["anomaly_score"] = score
    readings_a.append(d)
    if len(readings_a) > 100: readings_a.pop(0)
    print(f"[Plant A] {d.get('voltage_mv')}mV → {stress}")
    return jsonify({"status": "ok"})

# ── Receive Plant B data ──────────────────────────────
@app.route("/data/b", methods=["POST"])
def receive_b():
    d = request.json
    d["timestamp"] = time.time()
    stress, score  = get_stress(d.get("voltage_mv", 0))
    d["stress"]        = stress
    d["anomaly_score"] = score
    readings_b.append(d)
    if len(readings_b) > 100: readings_b.pop(0)
    print(f"[Plant B] {d.get('voltage_mv')}mV → {stress}")
    return jsonify({"status": "ok"})

# ── Send data to website ──────────────────────────────
@app.route("/readings/a", methods=["GET"])
def get_readings_a():
    return jsonify(readings_a[-50:])

@app.route("/readings/b", methods=["GET"])
def get_readings_b():
    return jsonify(readings_b[-50:])

# ── Retrain ───────────────────────────────────────────
@app.route("/retrain", methods=["POST"])
def retrain():
    if len(readings_a) < 10:
        return jsonify({"error": "Need at least 10 readings to retrain"})
    X = np.array([[r["voltage_mv"]] for r in readings_a[-50:]])
    from sklearn.ensemble import IsolationForest
    new_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    new_model.fit(X)
    global model, USE_ML
    model  = new_model
    USE_ML = True
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model retrained on {len(X)} readings")
    return jsonify({"status": "retrained", "samples": len(X)})

# ── Health check ──────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":   "running",
        "ml_model": USE_ML,
        "plant_a":  len(readings_a),
        "plant_b":  len(readings_b)
    })

if __name__ == "__main__":
    print("PhytoClone backend running on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)