# train_model.py
# Run this once to train the model on healthy plant data
# It saves the model to a file so app.py can use it

import numpy as np
import pickle
from sklearn.ensemble import IsolationForest

# ── Simulated healthy plant readings ──────────────────
# Replace these with REAL readings from your ESP32
# Format: [millivolts, temperature, humidity]
# These should all be from a HEALTHY plant

healthy_readings = [
    [2.1, 22, 61],
    [2.2, 22, 62],
    [2.0, 23, 60],
    [2.3, 22, 63],
    [2.1, 23, 61],
    [2.2, 24, 60],
    [1.9, 22, 64],
    [2.0, 23, 62],
    [2.1, 22, 61],
    [2.2, 23, 60],
    [2.3, 22, 63],
    [2.1, 24, 61],
    [2.0, 23, 62],
    [2.2, 22, 60],
    [2.1, 23, 61],
    [2.3, 22, 62],
    [2.0, 24, 63],
    [2.1, 23, 61],
    [2.2, 22, 60],
    [2.1, 23, 62],
]

X = np.array(healthy_readings)

# ── Train Isolation Forest ────────────────────────────
# contamination = how much of your training data you expect
# to already be slightly weird. 0.05 = 5%
model = IsolationForest(
    n_estimators=100,      # number of trees in the forest
    contamination=0.05,    # expect 5% of training data might be slightly off
    random_state=42        # makes results repeatable
)

model.fit(X)
print("Model trained on", len(X), "healthy readings")

# ── Test it with a fake anomaly ───────────────────────
test_normal  = np.array([[2.1, 23, 61]])   # should be normal
test_anomaly = np.array([[0.1, 40, 20]])   # should be anomaly (hot, dry, low signal)

def check(reading, label):
    result = model.predict(reading)        # returns 1 = normal, -1 = anomaly
    score  = model.score_samples(reading)  # more negative = more anomalous
    status = "NORMAL" if result[0] == 1 else "ANOMALY"
    print(f"{label}: {status} (score: {score[0]:.3f})")

check(test_normal,  "Normal reading")
check(test_anomaly, "Anomaly reading")

# ── Save the model ────────────────────────────────────
with open("stress_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to stress_model.pkl")
print("Now restart app.py to use it")