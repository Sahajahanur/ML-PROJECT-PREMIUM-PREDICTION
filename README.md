# 🌟 Health Insurance Premium Prediction
End-to-End Machine Learning Project with Deployment
<p align="center"> <img src="https://img.shields.io/badge/ML-End--to--End-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/Streamlit-Deployed-brightgreen?style=for-the-badge"/> <img src="https://img.shields.io/badge/Status-Production-success?style=for-the-badge"/> </p>

## 🧠 Project Description

This project predicts an individual's annual health insurance cost based on risk factors such as age, income, lifestyle, medical history, BMI, and smoking habits.

It includes:
* ✔️ Data preprocessing
* ✔️ Designing risk score system
* ✔️ Training two ML models for different age groups
* ✔️ Saving models & scalers using Joblib
* ✔️ Creating a beautiful Streamlit web app
* ✔️ Publishing the app on Streamlit Cloud
* ✔️ Full Git + GitHub version control
  
A complete production-ready ML pipeline.  

# ✅  FLOW DIAGRAM

                   ┌──────────────────────────────┐
                   │        User Input (UI)        │
                   │  Age, Income, BMI, Region,    │
                   │  Smoking, Medical History,    │
                   │  Dependants, Plan, Gender     │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │    Preprocessing Layer    │
                    │ ─────────────────────────  │
                    │  - One-hot encoding        │
                    │  - Ordinal encoding        │
                    │  - Numerical scaling       │
                    │  - Disease risk scoring    │
                    │  - Feature alignment       │
                    └───────────────┬────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │   Medical Risk Score Engine   │
                    │ ───────────────────────────── │
                    │  ("Disease & Heart disease")  │
                    │       ↓ split diseases        │
                    │  Assign risk → Normalize 0–1  │
                    └───────────────┬──────────────┘
                                    │
                                    ▼
            ┌──────────────────────────────────────────────────┐
            │              Age-Based Model Routing              │
            │──────────────────────────────────────────────────│
            │   IF age ≤ 25  → use   model_young.joblib        │
            │   ELSE         → use   model_rest.joblib         │
            └───────────────┬──────────────────────────────────┘
                            │
                            ▼
              ┌────────────────────────────────┐
              │        ML Model (Regression)    │
              │  (Trained with scikit-learn)    │
              └───────────────┬────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │     Predicted Premium (₹)       │
              └────────────────────────────────┘


## 🧠 Project Overview

Insurance companies determine premium amounts based on multiple risk factors.
This ML project predicts the approximate annual health insurance premium using:

* Age
* Number of dependants
* Income
* Genetic risk
* Medical history
* Region
* BMI category
* Smoking status
* Gender
* Employment status
* Insurance plan type

A custom logic is also implemented to compute the normalized medical risk score.  

## ⚙️ Outputs
* 💰 Predicted yearly insurance premium
* 🏥 Normalized health risk score (custom algorithm)

## 🏗️ Architecture
    Raw Input → Preprocessing → Risk Score → Feature Encoding →
    Model Selection (Young/Rest) → Prediction → Streamlit UI Output

The model uses a dual-model approach:

🟦 Age ≤ 25 → Young Model

🟪 Age > 25 → Rest Model

This dual-model approach improves accuracy.

## 📁 Project Structure

ML-PROJECT-PREMIUM-PREDICTION/

│

├── artifacts/

│   ├── model_young.joblib

│   ├── model_rest.joblib

│   ├── scaler_young.joblib

│   └── scaler_rest.joblib

│
├── main.py                  # Streamlit UI

├── prediction_helper.py     # ML prediction + preprocessing logic

├── requirements.txt         # App dependencies

└── README.md                # Project documentation

## 🏗️ Tech Stack

| Area                     | Tools / Libraries     |
| ------------------------ | --------------------- |
| **Programming Language** | Python                |
| **ML Libraries**         | scikit-learn, XGBoost |
| **Utility Libraries**    | pandas, numpy, joblib |
| **Deployment**           | Streamlit Cloud       |
| **Version Control**      | Git + GitHub          |


## 🧮 2. Normalized Medical Risk Score

Medical history is split (e.g., "Diabetes & Heart disease")

Each condition is assigned a risk:

| Disease       | Score |
| ------------- | ----- |
| Diabetes      | 6     |
| Heart disease | 8     |
| High BP       | 6     |
| Thyroid       | 5     |
| No Disease    | 0     |

## 🎨 Streamlit UI Features
* ✔ Modern dark theme
* ✔ Dropdowns for categorical variables
* ✔ Number inputs for numeric variables
* ✔ Clean layout (3 × 4 grid)
* ✔ Live prediction display
* ✔ Works perfectly on desktop & mobile

## 🚀 How to Run Locally

1. Clone the repo
   
       git clone https://github.com/Sahajahanur/ML-PROJECT-PREMIUM-PREDICTION.git
2. Navigate into project

       cd ML-PROJECT-PREMIUM-PREDICTION
3. Install dependencies

       pip install -r requirements.txt
   
4. Run the Streamlit app

       streamlit run main.py

## 🚀 Live App

### 👉 Try the App:

🔗 https://codebasics-ml-project-premium-prediction-srl.streamlit.app/

<img width="951" height="690" alt="image" src="https://github.com/user-attachments/assets/6aeb5c88-9fcf-4ff0-811c-a59b53855292" />

## 📜 Requirements

pandas==2.2.3

numpy==2.2.6

joblib==1.5.2

streamlit==1.48.0

scikit-learn==1.7.2

xgboost==3.1.1

## 💡 What I Learned
* Handling multi-condition medical data
* Creating normalized risk scores
* Encoding categoricals carefully
* Working with multiple ML models
* Streamlit UI design
* GitHub version control
* Deploying ML apps on Streamlit Cloud

##  📬 Contacts  

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:connectingsrl@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahajahanur-laskar/)

I’m always open to discussing Data Analytics, Machine Learning, Streamlit Apps, and End-to-End Projects!  





