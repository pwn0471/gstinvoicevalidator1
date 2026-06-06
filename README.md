# GST Invoice Validator (ML-Based) (POD - 64)

A beginner-friendly Python project that trains a machine learning model to classify GST invoices as `Valid` or `Suspicious`, serves predictions through a FastAPI backend, and provides an interactive Streamlit frontend.

## Folder Structure

- `data/`
  - `create_dataset.py` — synthetic invoice dataset creation and saving
- `models/`
  - `train_model.py` — data loading, preprocessing, model training, evaluation, and model saving
- `api/`
  - `app.py` — FastAPI backend with `/predict` endpoint
- `frontend/`
  - `app.py` — Streamlit UI for invoice validation and visualization
- `requirements.txt` — Python dependencies
- `README.md` — project overview and setup instructions

## Setup

1. Create a virtual environment and activate it:

```powershell
cd c:\Users\bgtsa\Desktop\gstinvoicevalidator
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model:

```powershell
python models\train_model.py
```

   This generates:
   - `models/gst_invoice_validator.joblib`
   - `models/model_report.json`

4. Run the backend API:

```powershell
uvicorn api.app:app --reload
```

5. Run the Streamlit frontend:

```powershell
streamlit run frontend\app.py
```

## Usage

### Sample API request

```json
{
  "invoice_amount": 1200.0,
  "gst_rate": 18.0,
  "total_amount": 1416.0,
  "vendor_frequency": 12
}
```

### Streamlit UI

Open the UI and enter:
- Invoice Amount
- GST Rate
- Total Amount
- Vendor Frequency

Then click **Validate Invoice**.

## Notes

- The model uses features:
  - `invoice_amount`
  - `gst_rate`
  - `total_amount`
  - `vendor_frequency`
- The backend returns a prediction plus confidence score.
- The frontend displays accuracy and a confusion matrix visualization.
