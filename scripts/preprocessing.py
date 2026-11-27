"""
Preprocessing for the Kaggle "Diabetes prediction dataset" (iammustafatz).

Columns: gender, age, hypertension, heart_disease, smoking_history,
         bmi, HbA1c_level, blood_glucose_level, diabetes (target)

How to run (VS Code terminal, inside the folder that holds this file):
    pip install pandas scikit-learn joblib
    python preprocessing.py

Put diabetes_prediction_dataset.csv next to this file or in a data/ folder next to it.
Results are saved in data/output/ in the project folder.

What it does:
    1. Load and inspect the data
    2. Clean it (duplicates, text values, impossible values)
    3. Split into train / test (stratified)
    4. Build the preprocessing pipeline (impute + scale + one-hot encode)
    5. Fit it on the training set only, transform both sets
    6. Save the fitted preprocessor and the cleaned splits
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_NAME = "diabetes_prediction_dataset.csv"
OUTPUT_DIR = BASE_DIR.parent / "data" / "output"
PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.joblib"


def find_dataset() -> Path:
    """Look for the CSV next to this file, then in a data/ folder next to it."""
    for candidate in (BASE_DIR / CSV_NAME, BASE_DIR / "data" / CSV_NAME,
                      BASE_DIR.parent / "data" / CSV_NAME):
        if candidate.exists():
            return candidate
    sys.exit(f"Could not find {CSV_NAME}.\n"
             f"Put it in {BASE_DIR} or in {BASE_DIR.parent / 'data'} and run again.")

TARGET = "diabetes"
NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_history"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

GENDERS = {"Female", "Male", "Other"}
SMOKING = {"never", "former", "current", "not current", "ever", "No Info"}

# Values outside these ranges are treated as data-entry errors (set to NaN,
# then filled by the imputer). Wide on purpose: they only catch impossible values.
VALID_RANGES = {
    "age": (0, 120),
    "bmi": (10, 100),
    "HbA1c_level": (3, 20),
    "blood_glucose_level": (40, 600),
}


# ---------------------------------------------------------------- 1. load
def load_data() -> pd.DataFrame:
    path = find_dataset()
    print(f"Loading {path}")
    df = pd.read_csv(path)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing columns: {sorted(missing_cols)}")
    return df


def inspect(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:\n", df.dtypes.to_string())
    print("\nMissing values per column:\n", df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nTarget balance (diabetes):")
    counts = df[TARGET].value_counts()
    for label, n in counts.items():
        print(f"  {label}: {n} ({n / len(df):.1%})")
    for col in CATEGORICAL:
        print(f"\n{col} values:\n", df[col].value_counts(dropna=False).to_string())
    print("\nNumeric summary:\n", df[NUMERIC].describe().round(2).to_string())
    print("=" * 60)


# ---------------------------------------------------------------- 2. clean
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start = len(df)

    # Duplicate rows add no information and inflate test scores.
    df = df.drop_duplicates()
    print(f"Dropped {start - len(df)} duplicate rows")

    # Tidy text columns: trim spaces, standard capitalisation.
    df["gender"] = df["gender"].astype(str).str.strip().str.capitalize()
    smoking = df["smoking_history"].astype(str).str.strip()
    df["smoking_history"] = np.where(
        smoking.str.lower() == "no info", "No Info", smoking.str.lower())

    # Unknown category values become NaN (the imputer fills them later).
    df.loc[~df["gender"].isin(GENDERS), "gender"] = np.nan
    df.loc[~df["smoking_history"].isin(SMOKING), "smoking_history"] = np.nan

    # Binary columns must be 0 or 1.
    for col in BINARY:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~df[col].isin([0, 1]), col] = np.nan

    # Numbers: force numeric, then blank out impossible values.
    for col, (low, high) in VALID_RANGES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = ~df[col].between(low, high) & df[col].notna()
        if bad.any():
            print(f"{col}: {bad.sum()} values outside {low}-{high} set to NaN")
        df.loc[bad, col] = np.nan

    # Rows without a valid target cannot be used for training.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])]
    df[TARGET] = df[TARGET].astype(int)

    print(f"Rows after cleaning: {len(df)}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 3. split
def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df[FEATURES]
    y = df[TARGET]
    # stratify=y keeps the same share of diabetic rows in train and test.
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ---------------------------------------------------------------- 4. pipeline
def build_preprocessor() -> ColumnTransformer:
    """Same steps at training time and at prediction time (inside the saved model)."""
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary_steps = SimpleImputer(strategy="most_frequent")
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERIC),
        ("bin", binary_steps, BINARY),
        ("cat", categorical_steps, CATEGORICAL),
    ])


# ---------------------------------------------------------------- 5 + 6. run
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    inspect(df)
    df = clean(df)

    X_train, X_test, y_train, y_test = split(df)
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Diabetic share - train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")

    preprocessor = build_preprocessor()
    # Fit on TRAIN only, so no information from the test set leaks in.
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    print(f"\nFeatures after preprocessing ({len(feature_names)}):")
    for name in feature_names:
        print("  ", name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cleaned (not yet scaled) splits: handy for the notebook and for checking.
    X_train.assign(**{TARGET: y_train}).to_csv(f"{OUTPUT_DIR}/train_clean.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(f"{OUTPUT_DIR}/test_clean.csv", index=False)

    # Fully transformed splits: what the model actually sees.
    pd.DataFrame(X_train_t, columns=feature_names).assign(**{TARGET: y_train.values}) \
        .to_csv(f"{OUTPUT_DIR}/train_transformed.csv", index=False)
    pd.DataFrame(X_test_t, columns=feature_names).assign(**{TARGET: y_test.values}) \
        .to_csv(f"{OUTPUT_DIR}/test_transformed.csv", index=False)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"\nSaved preprocessor.joblib and 4 CSV files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

