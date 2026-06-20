# Diabetes Prediction API

A FastAPI-based machine learning API for early diabetes risk prediction using a Stacking Ensemble Learning framework. The system combines Random Forest, Gradient Boosting, AdaBoost, and XGBoost models through a Logistic Regression meta-learner to improve predictive performance.

## Features

* FastAPI REST API
* Stacking Ensemble Architecture
* Random Forest, Gradient Boosting, AdaBoost, and XGBoost base models
* Logistic Regression meta-model
* Automated feature engineering
* Feature scaling using StandardScaler
* Threshold optimization using F1-score
* Probability-based diabetes risk prediction
* Swagger API documentation

## Model Workflow

Input Data
→ Feature Engineering
→ Feature Reordering
→ Feature Scaling
→ Base Models (RF, GBDT, AdaBoost, XGBoost)
→ Meta Learner (Logistic Regression)
→ Threshold Optimization
→ Final Diabetes Prediction

## Input Features

| Feature             | Description         |
| ------------------- | ------------------- |
| gender              | Male/Female/Other   |
| age                 | Age in years        |
| hypertension        | 0 or 1              |
| heart_disease       | 0 or 1              |
| smoking_history     | Smoking category    |
| bmi                 | Body Mass Index     |
| HbA1c_level         | HbA1c measurement   |
| blood_glucose_level | Blood glucose level |

## API Endpoint

### Predict Diabetes

**POST** `/predict`

Example Request:

```json
{
  "gender": "Female",
  "age": 80,
  "hypertension": 0,
  "heart_disease": 1,
  "smoking_history": "never",
  "bmi": 25.19,
  "HbA1c_level": 6.6,
  "blood_glucose_level": 140
}
```

Example Response:

```json
{
  "prediction": 1,
  "diabetes": true,
  "probability": 0.9986,
  "threshold": 0.89
}
```

## Installation

```bash
git clone https://github.com/PriyamChakrabarty/diabetes-prediction-api.git
cd diabetes-prediction-api
pip install -r backend/requirements.txt
```

## Run Locally

```bash
uvicorn backend.api:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

## Technologies Used

* Python
* FastAPI
* Scikit-Learn
* XGBoost
* Pandas
* NumPy
* Joblib
* Uvicorn

## Model Performance

The proposed stacking ensemble model was developed to enhance early diabetes detection by combining multiple machine learning algorithms and optimizing the classification threshold using F1-score.

## Author

Priyam Chakrabarty

M.E. Information Technology
Jadavpur University
