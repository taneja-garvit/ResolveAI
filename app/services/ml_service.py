import os
import pickle
import numpy as np

MODEL_PATH = os.path.join("app", "models", "ml_model.pkl")
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ML model not found at {MODEL_PATH}. Train or place the model file before starting the service."
        )

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    return _model


def predict_confidence(similarity_score: float, query_length: int):
    """
    Predict confidence using trained ML model
    """
    model = load_model()
    features = np.array([[similarity_score, query_length]])
    prob = model.predict_proba(features)

    # probability of class 1 (confident)
    return float(prob[0][1])