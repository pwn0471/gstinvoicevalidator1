import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class InvoiceInput(BaseModel):
    invoice_amount: float = Field(..., gt=0, description="The base invoice amount")
    gst_rate: float = Field(..., ge=0, description="GST rate as a percentage")
    total_amount: float = Field(..., gt=0, description="Total amount charged")
    vendor_frequency: Optional[int] = Field(1, ge=1, description="How many invoices vendor has submitted")


app = FastAPI(title="GST Invoice Validator API")
model: Optional[object] = None
model_path = Path(__file__).resolve().parents[1] / "models" / "gst_invoice_validator.joblib"
report_path = Path(__file__).resolve().parents[1] / "models" / "model_report.json"


def load_model():
    global model
    if model is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found. Run `python models/train_model.py` first. Expected: {model_path}"
            )
        model = joblib.load(model_path)
    return model


@app.get("/")
def root():
    return {"message": "GST Invoice Validator API is running."}


@app.post("/predict")
def predict(invoice: InvoiceInput):
    clf = load_model()
    data = np.array(
        [
            [
                invoice.invoice_amount,
                invoice.gst_rate,
                invoice.total_amount,
                invoice.vendor_frequency or 1,
            ]
        ]
    )
    try:
        prediction_idx = clf.predict(data)[0]
        prediction = "Valid" if prediction_idx == 0 else "Suspicious"
        confidence = float(clf.predict_proba(data).max())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    response = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "input": invoice.dict(),
    }

    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            response["model_report"] = json.load(f)

    return response
