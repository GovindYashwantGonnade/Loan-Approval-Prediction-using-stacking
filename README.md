# 🏦 End-to-End Loan Approval Prediction System

This project is an **end-to-end Machine Learning pipeline** for predicting loan approval status based on applicant and property information.  
The goal is to build a robust and explainable model that can assist financial institutions in **automating loan approval decisions** with high accuracy and fairness.

---

## 📌 Project Overview

- **Problem:** Predict whether a loan application will be approved or not based on applicant details.  
- **Dataset:** Real-world structured loan dataset (`loan.xlsx`).  
- **Goal:** Build a predictive model with good generalization, balanced performance, and a clean, automated ML workflow.

---

## 🚀 Key Features

- ✅ Automated data preprocessing using `ColumnTransformer` and `Pipeline`  
- 🧮 Feature engineering including log transformation, scaling, and encoding  
- 📊 Multiple ML algorithms — Logistic Regression, Random Forest, XGBoost, LightGBM  
- 🧠 Stacking ensemble to improve model performance  
- ⚖️ SMOTE applied to handle class imbalance effectively  
- 📈 Evaluation with F1-score for balanced precision and recall

---

## 🧰 Tech Stack

- **Languages:** Python  
- **Libraries:**  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `scikit-learn` (ColumnTransformer, Pipelines, Models, Metrics)  
  - `xgboost`, `lightgbm`  
  - `imblearn` (SMOTE for imbalance handling)

---

## 🧪 Model Selection & Metrics

| Model                  | F1-score | ROC-AUC |
|-------------------------|-----------|----------|
| Logistic Regression     | ~0.80     | ~0.84    |
| Random Forest           | ~0.85     | ~0.86    |
| XGBoost                 | ~0.86     | ~0.87    |
| LightGBM                | ~0.86     | ~0.87    |
| **Stacking Ensemble**   | **0.87**  | **0.88** |

- **Why F1-score?**  
  - The dataset is imbalanced.  
  - F1-score balances **precision** and **recall**, avoiding misleading high accuracy.  
  - It ensures both false positives and false negatives are minimized.

---

## ⚙️ Workflow

1. **Data Loading & Exploration**  
   - Import dataset and analyze data distribution.

2. **Preprocessing**  
   - Identify numeric and categorical columns.  
   - Apply log transform to skewed numeric columns.  
   - Impute missing values, scale numeric features, and one-hot encode categorical features using `ColumnTransformer`.

3. **Class Imbalance Handling**  
   - Apply **SMOTE** (Synthetic Minority Oversampling Technique) to balance the target variable.

4. **Model Training & Stacking**  
   - Train baseline and advanced ML models.  
   - Use StackingClassifier with Logistic Regression as final estimator to combine model strengths.

5. **Evaluation**  
   - Evaluate models using F1-score, ROC-AUC, and classification reports.

---

## 📊 Results

- Final Stacking Ensemble Model achieved:
  - ✅ **F1-score:** 0.87  
  - ✅ Balanced precision and recall across classes  
  - ✅ Improved performance over individual models

---

## 🧭 Future Enhancements

- Deploy the model with Streamlit or Flask for real-time predictions.  
- Add explainability techniques (e.g., SHAP) for better interpretability.  
- Explore hyperparameter tuning and feature selection for further optimization.

---


