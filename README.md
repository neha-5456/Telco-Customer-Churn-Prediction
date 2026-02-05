<div align="center">

# 🔮 ChurnSense — Telco Customer Churn Prediction

### Predict customer churn using Machine Learning & serve it through a beautiful FastAPI web app

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

<img src="https://img.shields.io/badge/STATUS-COMPLETE-success?style=flat-square" alt="status"/>
&nbsp;
<img src="https://img.shields.io/badge/ML%20MODEL-Random%20Forest-blue?style=flat-square" alt="model"/>
&nbsp;
<img src="https://img.shields.io/badge/ROC--AUC-0.84+-purple?style=flat-square" alt="auc"/>

---

**[📊 View Notebook](#-data-cleaning--eda-notebook)** · **[🚀 Run the App](#-quick-start)** · **[🔌 API Docs](#-api-reference)** · **[📸 Screenshots](#-screenshots)**

</div>

<br/>

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Key Findings from EDA](#-key-findings-from-eda)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Data Cleaning & EDA Notebook](#-data-cleaning--eda-notebook)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)
- [What I Learned](#-what-i-learned)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 About the Project

Customer churn is one of the most critical challenges in the telecom industry. Acquiring a new customer costs **5-7x more** than retaining an existing one. This project builds an end-to-end ML pipeline that:

1. **Cleans & explores** the Telco Customer Churn dataset (7,043 customers, 21 features)
2. **Trains a Random Forest model** to predict which customers are likely to leave
3. **Serves predictions** through a production-ready FastAPI web application with a beautiful dark-themed UI

### The Problem

> Given a telecom customer's demographics, account details, and service subscriptions, predict whether they will **churn** (leave the company) or **stay**.

### Dataset

| Property | Value |
|----------|-------|
| Source | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 (demographics, services, billing) |
| Target | `Churn` (Yes/No) |
| Class Balance | 73.5% No / 26.5% Yes (moderate imbalance) |

---

## 🔍 Key Findings from EDA

| Insight | Detail |
|---------|--------|
| 📉 **New customers churn the most** | 0-12 month tenure → ~47% churn rate |
| 📝 **Month-to-month contracts are risky** | ~42% churn vs ~3% for two-year contracts |
| 💳 **Electronic check = highest churn** | ~45% churn rate vs ~15% for auto-pay methods |
| 🌐 **Fiber optic users churn more** | ~42% churn vs ~19% for DSL |
| 💰 **Higher monthly charges → more churn** | Churned customers pay a median of $79.65 vs $64.43 |

> **Key takeaway:** The strongest churn predictors are **contract type**, **tenure**, **payment method**, and **internet service type**.

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data Analysis** | Pandas, NumPy, Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (RandomForestClassifier, StandardScaler) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML5, CSS3 (custom dark theme), Vanilla JavaScript |
| **Templating** | Jinja2 |
| **Serialization** | Pickle |

---

## 📁 Project Structure

```
telco-churn-prediction/
│
├── 📓 notebooks/
│   └── 01_data_cleaning.ipynb        # Data cleaning, EDA & feature engineering
│
├── 📊 data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── 🌐 churn-app/
│   ├── main.py                        # FastAPI application (routes + prediction)
│   ├── train_model.py                 # Model training script
│   ├── requirements.txt               # Python dependencies
│   ├── templates/
│   │   └── index.html                 # Beautiful dark-themed frontend
│   ├── static/                        # Static assets
│   └── model/                         # Auto-generated after training
│       ├── churn_model.pkl            # Trained Random Forest model
│       ├── scaler.pkl                 # Fitted StandardScaler
│       └── feature_columns.pkl        # Feature column names
│
├── README.md                          # ← You are here
└── LICENSE
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
cd telco-churn-prediction
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
cd churn-app
pip install -r requirements.txt
```

**4. Download the dataset**

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the CSV file at:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

**5. Train the model**

```bash
python train_model.py
```

Expected output:

```
Loading data...
  Shape: (7043, 21)
Cleaning data...
Encoding features...
Splitting data...
Scaling numerical features...
Training Random Forest model...

==================================================
MODEL EVALUATION
==================================================
              precision    recall  f1-score   support

   No Churn       0.84      0.90      0.87      1033
      Churn       0.66      0.53      0.59       373

    accuracy                           0.80      1406
   macro avg       0.75      0.72      0.73      1406
weighted avg       0.79      0.80      0.80      1406

ROC-AUC Score: 0.84+

Model artifacts saved to ./model/
```

**6. Start the server**

```bash
uvicorn main:app --reload
```

**7. Open the app**

Visit 👉 **http://127.0.0.1:8000** in your browser.

---

## 📊 Data Cleaning & EDA Notebook

The Jupyter notebook (`notebooks/01_data_cleaning.ipynb`) covers the complete data pipeline:

### Step 1 — Data Inspection
- Loaded 7,043 rows × 21 columns
- Identified `TotalCharges` as `object` type (should be `float64`)
- Checked for duplicates (0 found)

### Step 2 — Data Cleaning
- Found 11 rows where `TotalCharges` contained blank spaces `" "`
- All 11 rows had `tenure = 0` (brand new customers, never billed)
- Converted to numeric with `pd.to_numeric(errors='coerce')`
- **Dropped the 11 NaN rows** (only 0.16% of data)

### Step 3 — EDA Highlights
- Churn distribution analysis (26.5% churn rate → moderate class imbalance)
- Contract type vs churn (month-to-month = highest risk)
- Tenure grouping analysis (new customers churn 5x more than loyal ones)
- Monthly charges comparison (churned customers pay ~$15 more on median)
- Internet service analysis (fiber optic = highest churn)
- Payment method analysis (electronic check = 45% churn rate)
- Correlation heatmap of numerical features

### Step 4 — Feature Engineering
- Dropped `customerID` (identifier, not a feature)
- Encoded target variable `Churn` → 0/1
- One-hot encoded 15 categorical columns with `drop_first=True`

### Step 5 — Train-Test Split (Before Scaling!)
- 80/20 split with `stratify=y` to preserve class distribution
- **Scaling done AFTER split** to prevent data leakage

### ⚠️ Data Leakage Prevention

A critical concept implemented in this project:

```python
# ✅ CORRECT — fit scaler on train only
scaler.fit_transform(X_train[num_cols])    # fit + transform on train
scaler.transform(X_test[num_cols])          # transform only on test

# ❌ WRONG — causes data leakage
scaler.fit_transform(data[num_cols])        # fits on ALL data including test
train_test_split(...)                        # then splits — too late!
```

> If the scaler sees test data during `fit()`, the model indirectly learns test set statistics, leading to overly optimistic metrics.

---

## 📈 Model Performance

| Metric | No Churn (0) | Churn (1) | Overall |
|--------|:------------:|:---------:|:-------:|
| **Precision** | 0.84 | 0.66 | 0.79 (weighted) |
| **Recall** | 0.90 | 0.53 | 0.80 (weighted) |
| **F1-Score** | 0.87 | 0.59 | 0.80 (weighted) |
| **ROC-AUC** | — | — | **0.84+** |

### Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| n_estimators | 200 |
| max_depth | 10 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| Scaling | StandardScaler (tenure, MonthlyCharges, TotalCharges) |

---

## 🔌 API Reference

### `GET /` — Home Page

Serves the frontend UI.

### `GET /health` — Health Check

```json
{
  "status": "healthy",
  "model_features": 30
}
```

### `GET /docs` — Swagger UI

Interactive API documentation (auto-generated by FastAPI).

### `POST /predict` — Predict Churn

**Request Body:**

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.70,
  "TotalCharges": 151.65
}
```

**Response:**

```json
{
  "success": true,
  "prediction": 1,
  "churn_probability": 78.5,
  "stay_probability": 21.5,
  "risk_level": "Critical",
  "risk_color": "#ef4444",
  "message": "This customer has a 78.5% chance of churning."
}
```

**Risk Levels:**

| Churn Probability | Risk Level | Color |
|:-----------------:|:----------:|:-----:|
| 0% – 19% | 🟢 Low | Green |
| 20% – 39% | 🟡 Medium | Yellow |
| 40% – 69% | 🟠 High | Orange |
| 70% – 100% | 🔴 Critical | Red |

### cURL Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 85.50
  }'
```

---

## 📸 Screenshots

> Add your own screenshots here after running the app.

| Home Page | Prediction Result |
|:---------:|:-----------------:|
| *Form with dark theme UI* | *Animated gauge + risk level* |

To take screenshots:
1. Run the app (`uvicorn main:app --reload`)
2. Open http://127.0.0.1:8000
3. Take a screenshot of the form
4. Fill in details, click predict, and screenshot the result
5. Save images to a `screenshots/` folder and update the paths above

---

## 📚 What I Learned

This project helped me understand several important concepts:

- **Data Leakage** — Why you must split BEFORE scaling, and how fitting the scaler on all data creates optimistically biased metrics
- **Handling Hidden Missing Values** — TotalCharges had blank spaces `" "` instead of NaN, which wouldn't be caught by `.isnull()`. Always inspect data types.
- **Class Imbalance** — With ~26.5% churn rate, accuracy alone is misleading. F1-score and ROC-AUC give a better picture.
- **Feature Encoding** — Using `drop_first=True` in `get_dummies` to avoid the dummy variable trap (multicollinearity)
- **Stratified Splitting** — Using `stratify=y` ensures both train and test sets maintain the same churn ratio
- **API Design** — Building a prediction endpoint with Pydantic validation and proper error handling
- **Full-Stack ML** — Connecting a trained model to a web interface end-to-end

---

## 🔮 Future Improvements

- [ ] Add **SMOTE** or class weights to handle class imbalance
- [ ] Try **XGBoost** and **LightGBM** for comparison
- [ ] Add **feature importance** visualization on the frontend
- [ ] Implement **batch prediction** (upload CSV → get predictions)
- [ ] Add **SHAP explanations** for individual predictions
- [ ] Deploy to **Render** / **Railway** / **AWS**
- [ ] Add **user authentication** for production use
- [ ] Create a **Dockerized** version for easy deployment
- [ ] Add **unit tests** for the API endpoints

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) on Kaggle
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

<div align="center">

**⭐ If you found this project helpful, give it a star!**

Made with ❤️ and Python

</div>
