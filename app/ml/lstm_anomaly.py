import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# Load model + scaler
_lstm_model = load_model(MODEL_DIR / "lstm_autoencoder_model.keras")
_scaler = joblib.load(MODEL_DIR / "minmax_scaler.joblib")

TIME_STEPS = 50

# You MUST match training columns
FEATURE_COLUMNS = None  # will infer dynamically


def create_sequences(data, time_steps=50):
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:(i + time_steps)])
    return np.array(sequences)


def compute_reconstruction_error(X, X_pred):
    return np.mean(np.abs(X - X_pred), axis=(1, 2))


def detect_anomaly(data: list[dict]):
    """
    data = [
      {"rpm": ..., "speed": ..., "throttle": ...},
      ...
    ]
    """

    df = pd.DataFrame(data)

    # Scale using same scaler
    scaled = _scaler.transform(df)

    # Create sequences
    X = create_sequences(scaled, TIME_STEPS)

    if len(X) == 0:
        return {"error": "Not enough data for sequence"}

    # Predict reconstruction
    X_pred = _lstm_model.predict(X)

    # Compute error
    errors = compute_reconstruction_error(X, X_pred)

    # Threshold (you can tune this!)
    threshold = np.mean(errors) + 2 * np.std(errors)

    anomaly_flags = errors > threshold

    return {
        "anomaly_detected": bool(np.any(anomaly_flags)),
        "anomaly_count": int(np.sum(anomaly_flags)),
        "reconstruction_error": errors.tolist(),
        "threshold": float(threshold)
    }
