import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocessing import TARGET
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import joblib
from sklearn.metrics import confusion_matrix



BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "models"
MODEL_PATH = OUTPUT_DIR / "diabetes_model.joblib"

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, cm

X_train = pd.read_csv("data/output/train_transformed.csv")
y_train = X_train.pop(TARGET)

X_test = pd.read_csv("data/output/test_transformed.csv")
y_test = X_test.pop(TARGET)

booster=XGBClassifier(enable_categorical=True, random_state=42)
booster.fit(X_train,y_train)
y_predboost=booster.predict(X_test)
metrics=classification_report(y_test, y_predboost)
print(metrics)

joblib.dump(booster, MODEL_PATH)
print(f"\nSaved Machine Learning models in {OUTPUT_DIR}")
