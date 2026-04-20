import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join("app", "models", "ml_model.pkl")

# Bootstrap training data for the confidence calibration model.
# Feature 1: normalized retrieval similarity from FAISS search.
# Feature 2: user query length in characters.
X = np.array(
    [
        [0.96, 18],
        [0.94, 25],
        [0.92, 38],
        [0.90, 42],
        [0.88, 50],
        [0.86, 65],
        [0.84, 30],
        [0.82, 44],
        [0.80, 58],
        [0.78, 72],
        [0.76, 80],
        [0.74, 34],
        [0.72, 60],
        [0.70, 68],
        [0.68, 88],
        [0.66, 96],
        [0.64, 105],
        [0.62, 48],
        [0.60, 74],
        [0.58, 90],
        [0.56, 120],
        [0.54, 132],
        [0.50, 95],
        [0.48, 140],
        [0.44, 110],
        [0.40, 125],
        [0.36, 145],
        [0.32, 160],
        [0.28, 180],
        [0.24, 210],
    ],
    dtype=float,
)

y = np.array(
    [
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    dtype=int,
)


def train_and_save_model(model_path: str = MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


if __name__ == "__main__":
    saved_path = train_and_save_model()
    print(f"Confidence model trained and saved at {saved_path}")