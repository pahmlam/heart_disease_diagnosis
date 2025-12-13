
---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple)

# ❤️ Cardiovascular Disease Risk Prediction System

A **clinical decision support system (CDSS)** for predicting the risk of **cardiovascular disease** based on clinical medical indicators.
The project leverages **Machine Learning (Stacking Ensemble)** combined with **Feature Engineering** to maximize predictive performance and is deployed as a **RESTful Web API** with a user-friendly frontend interface.

---

## 📑 Table of Contents

* [Introduction](#introduction)
* [Theory & Methodology](#theory--methodology)

  * [1. Dataset](#1-dataset)
  * [2. Feature Engineering](#2-feature-engineering)
  * [3. Stacking Ensemble Model](#3-stacking-ensemble-model)
* [Project Structure](#project-structure)
* [Installation & Usage](#installation--usage)
* [API Documentation](#api-documentation)
* [Technologies Used](#technologies-used)

---

## Introduction

Cardiovascular disease is one of the leading causes of mortality worldwide. Early diagnosis plays a crucial role in effective treatment and prevention.
This project aims to build a **machine learning–based decision support system** that enables clinicians or individual users to quickly assess heart disease risk using key medical attributes such as age, cholesterol level, blood pressure, and more.

---

## Theory & Methodology

### 1. Dataset

The project uses the **Cleveland Heart Disease Dataset** from the **UCI Machine Learning Repository**.

* **Size:** 303 samples
* **Features:** 13 clinical attributes
  (Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Rest ECG, Max Heart Rate, Exercise-Induced Angina, Oldpeak, Slope, CA, Thal)
* **Target Variable:**

  * `0` — No heart disease
  * `1` — Presence of heart disease

---

### 2. Feature Engineering

Instead of relying solely on raw clinical data, this project applies **Feature Engineering** to create additional informative features that enhance signal extraction and model learning capacity.

Newly engineered features include:

* **Cholesterol per Age (`chol_per_age`)**
  Ratio of cholesterol level to age, reflecting relative lipid accumulation with aging.
* **Blood Pressure per Age (`bps_per_age`)**
  Ratio of resting systolic blood pressure to age.
* **Heart Rate Ratio (`hr_ratio`)**
  Ratio of maximum heart rate to age.
* **Age Binning**
  Discretization of age into groups to better capture non-linear trends.

📈 **Experimental results** show that Feature Engineering improves test accuracy significantly—from approximately **84%** (raw features) to **90–93%** after enhancement.

---

### 3. Stacking Ensemble Model

To achieve optimal performance and robustness, the system employs **Ensemble Learning** using the **Stacking** strategy.
Stacking combines multiple base learners to reduce bias and variance while improving generalization.

#### Model Architecture

**Level-0 (Base Learners):**

* **K-Nearest Neighbors (KNN)**
  Distance-based classifier with optimal `k ≈ 11`, selected via cross-validation.
* **Decision Tree (DT)**
  Depth-limited decision tree to prevent overfitting.
* **Naive Bayes (NB)**
  Probabilistic classifier based on Bayes’ theorem with feature independence assumptions.

**Level-1 (Meta Learner):**

* **KNN Classifier**
  Aggregates probability outputs from base learners to produce the final prediction.

---

## Project Structure

```bash
heart_disease_project/
├── data/
│   └── cleveland.csv        # Raw dataset
├── model/
│   └── heart_model.pkl      # Trained model pipeline (incl. preprocessing)
├── main.py                  # FastAPI backend
├── train.py                 # Model training & feature engineering
├── index.html               # Frontend UI
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## Installation & Usage

### Step 1: Clone Repository & Setup Environment

```bash
git clone <your-repo-url>
cd heart_disease_project

# Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Train the Model

Before running the application, train the model to generate `heart_model.pkl`.
This process includes **data preprocessing, feature engineering, and model stacking**.

```bash
python train.py
```

Expected output:

```
Test Accuracy: 0.9xxx
Model saved successfully.
```

---

### Step 3: Start the Backend Server

```bash
python main.py
```

The API server will run at:
`http://127.0.0.1:8000`

---

### Step 4: Use the Frontend Interface

Open `index.html` in any modern browser (Chrome, Firefox, Edge).
Enter clinical indicators and click **“Predict Now”** to obtain results.

---

## API Documentation

### Endpoint: `/predict`

* **Method:** `POST`
* **Description:** Receives clinical data and returns heart disease risk prediction with confidence score.

### Request Body (JSON)

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

### Response (JSON)

```json
{
  "prediction": 1,
  "result_text": "High risk of heart disease",
  "confidence": 85.5,
  "features_engineered": {
    "chol_per_age": 3.698,
    "bps_per_age": 2.301,
    "hr_ratio": 2.381
  }
}
```

---

## Technologies Used

* **Language:** Python 3.9+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Pipeline, StackingClassifier, Imputer)
* **Backend:** FastAPI, Uvicorn
* **Frontend:** HTML5, Bootstrap 5, JavaScript (Fetch API)

---
