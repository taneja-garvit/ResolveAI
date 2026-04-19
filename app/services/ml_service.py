import pickle
import numpy as np

# load trained model once
model = pickle.load(open("app/models/ml_model.pkl", "rb"))

def predict_confidence(similarity_score: float, query_length: int):
    """
    Predict confidence using trained ML model
    """
    features = np.array([[similarity_score, query_length]])
    prob = model.predict_proba(features)

    # probability of class 1 (confident)
    return float(prob[0][1])