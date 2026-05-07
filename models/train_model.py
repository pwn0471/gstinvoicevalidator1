import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_or_create_dataset(data_path: Path) -> pd.DataFrame:
    if data_path.exists():
        return pd.read_csv(data_path)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from data.create_dataset import create_invoice_dataset

    return create_invoice_dataset(data_path, n_samples=1000)


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def train_and_save_model():
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "dataset.csv"
    model_path = repo_root / "models" / "gst_invoice_validator.joblib"
    report_path = repo_root / "models" / "model_report.json"

    df = load_or_create_dataset(data_path)
    df = df.dropna()

    X = df[["invoice_amount", "gst_rate", "total_amount", "vendor_frequency"]]
    y = np.where(df["label"] == "Valid", 0, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = {
        "accuracy": float(accuracy),
        "classification_report": classification_report(y_test, y_pred, target_names=["Valid", "Suspicious"], output_dict=True),
        "confusion_matrix": cm,
        "feature_names": ["invoice_amount", "gst_rate", "total_amount", "vendor_frequency"],
        "labels": ["Valid", "Suspicious"],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Model training complete")
    print(f"Saved model to: {model_path}")
    print(f"Saved report to: {report_path}")
    print(f"Accuracy: {accuracy:.2%}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    train_and_save_model()
