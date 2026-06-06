# 🚀 GST Invoice Validator (ML-Based) | POD - 64

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)

## 📌 Overview

**GST Invoice Validator** is a Machine Learning-powered application that automatically classifies GST invoices as **Valid** or **Suspicious**.

The project combines Machine Learning, FastAPI, and Streamlit to provide a complete end-to-end solution for invoice validation and fraud detection.

---

## ✨ Features

- 🤖 Machine Learning-based GST invoice classification
- ⚡ FastAPI REST API for real-time predictions
- 🎨 Interactive Streamlit dashboard
- 📊 Confidence score generation
- 📈 Confusion Matrix visualization
- 📋 Model performance reporting
- 🚀 Beginner-friendly project structure

---

## 🏗️ System Architecture

```text
User Input
     │
     ▼
Streamlit Frontend
     │
     ▼
FastAPI Backend
     │
     ▼
Machine Learning Model
     │
     ▼
Prediction:
Valid / Suspicious
```

---

## 📂 Project Structure

```text
GST-Invoice-Validator/
│
├── data/
│   └── create_dataset.py
│
├── models/
│   ├── train_model.py
│   ├── gst_invoice_validator.joblib
│   └── model_report.json
│
├── api/
│   └── app.py
│
├── frontend/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

## 🧠 Machine Learning Features

The model is trained using the following invoice attributes:

| Feature | Description |
|----------|------------|
| invoice_amount | Invoice amount before GST |
| gst_rate | GST percentage applied |
| total_amount | Final invoice amount |
| vendor_frequency | Number of transactions with vendor |

### Prediction Classes

- ✅ Valid Invoice
- ⚠️ Suspicious Invoice

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gst-invoice-validator.git

cd gst-invoice-validator
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Train the Model

```bash
python models/train_model.py
```

Generated files:

```text
models/
├── gst_invoice_validator.joblib
└── model_report.json
```

---

## 🚀 Run FastAPI Backend

```bash
uvicorn api.app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger API Docs:

```text
http://127.0.0.1:8000/docs
```

---

## 🎨 Run Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## 📡 API Example

### Request

```json
{
  "invoice_amount": 1200.0,
  "gst_rate": 18.0,
  "total_amount": 1416.0,
  "vendor_frequency": 12
}
```

### Response

```json
{
  "prediction": "Valid",
  "confidence": 0.96
}
```

---

## 📊 Model Evaluation

The application provides:

- Accuracy Score
- Classification Report
- Confusion Matrix
- Prediction Confidence

These metrics help assess the reliability and effectiveness of the invoice validation model.

---

## 🔄 Application Workflow

1. User enters invoice details.
2. Streamlit sends data to FastAPI.
3. FastAPI loads the trained model.
4. Model predicts invoice status.
5. Result and confidence score are displayed.
6. Performance metrics are visualized.

---

## 📸 Screenshots

### Home Page

Add your Streamlit homepage screenshot here.

```markdown
![Home Page](screenshots/home.png)
```

### Prediction Result

Add your prediction result screenshot here.

```markdown
![Prediction](screenshots/result.png)
```

---

## 🔮 Future Enhancements

- 📂 CSV Invoice Upload
- 📄 PDF Invoice Analysis
- ☁️ Cloud Deployment
- 🗄️ Database Integration
- 📊 Admin Dashboard
- 🧠 Advanced Fraud Detection Models
- 🔔 Real-Time Alerts

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Core Programming |
| Scikit-Learn | Machine Learning |
| FastAPI | Backend API |
| Streamlit | Frontend UI |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |
| Joblib | Model Serialization |

---

## 👨‍💻 Author

### Pawan Kumar

B.Tech in Information Technology

Interested in:
- Machine Learning
- Full Stack Development
- AI Applications
- Software Engineering

---

## ⭐ Show Your Support

If you found this project useful:

- ⭐ Star this repository
- 🍴 Fork the project
- 🛠️ Contribute to improve it
- 📢 Share it with others

---

### 🚀 Built with Machine Learning, FastAPI, and Streamlit
