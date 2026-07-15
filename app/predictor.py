"""
==============================================================
AI Exposure Predictor - Prediction Pipeline

This module loads the trained model and preprocessing objects
and predicts AI Exposure Score for new task descriptions.

Training Pipeline
-----------------
1. TF-IDF Vectorization
2. OneHotEncoder (Task Type)
3. StandardScaler (Incumbents Responding)
4. hstack() Feature Combination
5. Random Forest Regressor

Author : Divya Nehete
==============================================================
"""

# ==========================================================
# Imports
# ==========================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

DATA_DIR = BASE_DIR / "data" / "processed"


# ==========================================================
# Load Saved Objects
# ==========================================================

# ⚠️ Change this filename if your saved model has another name
MODEL_PATH = MODEL_DIR / "final_random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

tfidf = joblib.load(DATA_DIR / "tfidf_vectorizer.pkl")

encoder = joblib.load(DATA_DIR / "tasktype_encoder.pkl")

scaler = joblib.load(DATA_DIR / "scaler.pkl")


# ==========================================================
# Default Numerical Feature
# ==========================================================

# Replace with actual dataset mean later
DEFAULT_INCUMBENTS = 8.0


# ==========================================================
# Prediction Function
# ==========================================================

def predict_ai_exposure(task_description, task_type):

    # -----------------------------------------
    # TF-IDF Features
    # -----------------------------------------

    X_text = tfidf.transform([task_description])

    # -----------------------------------------
    # OneHot Encoding
    # -----------------------------------------

    X_tasktype = encoder.transform(
        pd.DataFrame({
            "Task Type": [task_type]
        })
    )

    # -----------------------------------------
    # Numerical Feature
    # -----------------------------------------

    X_num = scaler.transform(
        np.array([[DEFAULT_INCUMBENTS]])
    )

    # -----------------------------------------
    # Combine Features
    # -----------------------------------------

    X = hstack([
        X_text,
        X_tasktype,
        X_num
    ])

    # -----------------------------------------
    # Predict
    # -----------------------------------------

    prediction = model.predict(X)

    return float(prediction[0])


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI Exposure Predictor Test")
    print("=" * 60)

    print("\nAvailable Task Types:")

    print(encoder.categories_[0])

    sample_task = "Analyze financial reports using Excel."

    sample_type = encoder.categories_[0][0]

    score = predict_ai_exposure(
        sample_task,
        sample_type
    )

    print("\nPrediction Successful!")

    print(f"\nTask Description : {sample_task}")

    print(f"Task Type        : {sample_type}")

    print(f"Predicted Score  : {score:.3f}")

    print("=" * 60)