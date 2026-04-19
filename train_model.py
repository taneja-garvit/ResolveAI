import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import os

os.makedirs("app/models", exist_ok=True)

X = np.array([
    [0.9,10],
    [0.85,12],
    [0.8, 15],
    [0.3, 40],
    [0.25, 50],
    [0.2, 60],
])

y= np.array([1,1,1,0,0,0])

model = LogisticRegression()
model.fit(X,y)

with open("app/models/ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved at app/models/ml_model.pkl")