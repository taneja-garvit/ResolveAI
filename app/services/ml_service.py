import pickle
import numpy as np

# simple dummy confidence model
def predict_confidence(similarity_score: float):
    # fake logic (you can replace with real ML model later)
    if similarity_score > 0.7:
        return 0.9
    return 0.4

