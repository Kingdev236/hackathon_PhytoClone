from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

readings_a = []
readings_b = []

# ── Load ML model ─────────────────────────────────────
# loads the trained Isolation Forest from disk
# if no model exists yet, falls back to simple rules
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
def get_stress(moisture, temp, light):
    if USE_ML:
        # use isolation forest to detect anomalies
        reading = np.array([[moisture, temp, light]])
        result  = model.predict(reading)      # 1 = normal, -1 = anomaly
        score   = model.score_samples(reading)[0]  # how anomalous it is

        if result[0] == 1:
            return "Healthy", round(score, 3)
        else:
            # score more negative = more stressed
            if score < -0.6:
                return "High Stress", round(score, 3)
            else:
                return "Mild Stress", round(score, 3)
    else:
        # fallback rules if no model trained yet
        score = 0
        if moisture < 30: score += 2
        if moisture > 80: score += 1
        if temp > 35:     score += 2
        if temp < 10:     score += 1
        if light < 20:    score += 1
        if score == 0:    return "Healthy", 0
        if score <= 2:    return "Mild Stress", 0
        return "High Stress", 0

# ── Receive Plant A data ──────────────────────────────
@app.route("/data/a", methods=["POST"])
def receive_a():
    d = request.json
    d["timestamp"] = time.time()

    stress, score = get_stress(
        d.get("moisture", 50),
        d.get("temp", 25),
        d.get("light", 50)
    )
    d["stress"]       = stress  # "Healthy" / "Mild Stress" / "High Stress"
    d["anomaly_score"] = score  # how weird the reading is

    readings_a.append(d)
    if len(readings_a) > 100: readings_a.pop(0)
    print(f"[Plant A] mv:{d.get('moisture')} temp:{d.get('temp')} → {stress} (score:{score})")
    return jsonify({"status": "ok"})

# ── Receive Plant B data ──────────────────────────────
@app.route("/data/b", methods=["POST"])
def receive_b():
    d = request.json
    d["timestamp"] = time.time()

    stress, score = get_stress(
        d.get("moisture", 50),
        d.get("temp", 25),
        d.get("light", 50)
    )
    d["stress"]        = stress
    d["anomaly_score"] = score

    readings_b.append(d)
    if len(readings_b) > 100: readings_b.pop(0)
    print(f"[Plant B] mv:{d.get('moisture')} temp:{d.get('temp')} → {stress} (score:{score})")
    return jsonify({"status": "ok"})

# ── Send Plant A data to website ──────────────────────
@app.route("/readings/a", methods=["GET"])
def get_readings_a():
    return jsonify(readings_a[-20:])

# ── Send Plant B data to website ──────────────────────
@app.route("/readings/b", methods=["GET"])
def get_readings_b():
    return jsonify(readings_b[-20:])

# ── Retrain endpoint ──────────────────────────────────
# POST to /retrain to retrain the model with current healthy readings
@app.route("/retrain", methods=["POST"])
def retrain():
    if len(readings_a) < 10:
        return jsonify({"error": "Need at least 10 readings to retrain"})

    # use the last 50 readings as "healthy" training data
    X = np.array([[r["moisture"], r["temp"], r["light"]] for r in readings_a[-50:]])

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
    print("PhotoClone backend running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)