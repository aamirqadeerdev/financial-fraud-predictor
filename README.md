

# Financial Fraud Detector

A real-time financial fraud detection system built with Machine Learning, FastAPI, and Streamlit. The system analyzes credit card transactions and predicts whether they are fraudulent or legitimate with high accuracy.

## Live Demo
[Click here to try the app]() ← We will add this after deployment

## API Documentation
[Click here to view API docs]() ← We will add this after deployment

## What It Does

Enter any transaction details and the system instantly predicts whether it is fraud or legitimate. The app provides a fraud probability score, confidence level, risk categorization, and recommended actions following Canadian financial compliance standards including FINTRAC reporting requirements.

## How It Works — ML Pipeline

**Step 1 — Model Training**
The Random Forest model is trained on 284,807 real credit card transactions from a European bank. The dataset contains 492 confirmed fraud cases representing a 0.17% fraud rate — a realistic class imbalance handled using balanced class weights.

**Step 2 — Real Time Prediction**
Transaction features are scaled using the same StandardScaler used during training and passed to the Random Forest model which returns a fraud probability score in milliseconds.

**Step 3 — Risk Categorization**
Transactions are categorized as HIGH RISK, MEDIUM RISK, or LOW RISK based on fraud probability with specific recommended actions for each category following FINTRAC compliance guidelines.

## Model Performance

- AUC-ROC Score: 0.9580
- Algorithm: Random Forest Classifier
- Training Dataset: 227,845 transactions
- Testing Dataset: 56,962 transactions
- Fraud Threshold: 0.30

## Tech Stack

- **Scikit-learn** — Random Forest ML model training
- **FastAPI** — real-time REST API for predictions
- **Streamlit** — interactive web demo interface
- **Pandas & NumPy** — data processing pipeline
- **Joblib** — model persistence
- **Conda** — environment management
- **Python 3.10** — core language

## Project Structure
```
Financial Fraud Predictor/
├── train_model.py     # ML model training pipeline
├── schemas.py         # Pydantic data validation
├── main.py            # FastAPI REST API
├── app.py             # Streamlit demo interface
├── requirements.txt   # Python dependencies
├── environment.yml    # Conda environment
└── .gitignore         # Git ignore rules
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API welcome and status |
| GET | /health | Health check |
| POST | /predict | Single transaction prediction |
| GET | /model/info | Model metadata and performance |

## How to Run Locally

**1. Clone the repository**
```
git clone https://github.com/aamirqadeerdev/financial-fraud-predictor.git
cd financial-fraud-predictor
```

**2. Create Conda environment**
```
conda env create -f environment.yml
conda activate fraud-detection
```

**3. Download dataset**
Download creditcard.csv from Kaggle and save to data/ folder:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**4. Train the model**
```
python train_model.py
```

**5. Start FastAPI server**
```
uvicorn main:app --reload
```

**6. Run Streamlit demo**
```
streamlit run app.py
```

## Compliance Documentation

This project includes professional compliance documentation:

- **Fraud Detection Development Guidelines** — coding standards and ML best practices
- **Fraud Detection Compliance Checklist** — PCI DSS, FINTRAC, OSFI, EU AI Act, SR 11-7, PIPEDA

## Business Application

This proof of concept demonstrates fraud detection capability using a public dataset. For production deployment the model would be retrained on the client's proprietary transaction data with customized features, thresholds, and integration with existing banking systems.

## Author

Aamir Qadeer — Full Stack Developer and AI Engineer
- Available for Canadian remote opportunities
- Open to relocation to Canada
