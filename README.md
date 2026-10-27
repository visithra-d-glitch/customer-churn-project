# 📊 Customer Churn Prediction & Retention Analysis
An End-to-End Machine Learning and Exploratory Data Analysis (EDA) project to predict customer churn risk for a telecommunications provider and help businesses improve customer retention strategies.

## Executive Summary
Customer churn occurs when customers stop doing business with a service provider. Acquiring new customers costs significantly more than retaining existing ones. 
This project analyzes historical customer data, identifies key operational drivers of customer attrition, builds predictive Machine Learning models to identify high-risk customers before they leave, and provides actionable business insights.

## Key Features & Insights
- **Data Preprocessing & Cleaning:** Managed missing values, encoded categorical features, and performed feature scaling.
- **Exploratory Data Analysis (EDA):** Uncovered business insights (e.g., month-to-month contract holders and fiber optic internet subscribers show higher churn rates).
- **Class Imbalance Management:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance.
- **Predictive Modeling:** Trained and evaluated multiple ML algorithms (Logistic Regression, Random Forest, XGBoost).
- **Business-Centric Metrics:** Evaluated models using **ROC-AUC** and **Recall** to minimize false negatives (missing actual churners).
  
## Tech Stack & Tools
- **Programming Language:** Python 3.
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Interactive UI / Web App:** `streamlit` (Optional)
- **Environment:** Jupyter Notebook / Visual Studio Code

## 📂 Project Structure

```text
customer-churn-prediction/
├── data/
│   ├── raw_churn_data.csv          # Original Dataset
│   └── processed_data.csv        # Processed Dataset
├── notebooks/
│   ├── 01_eda_and_visualization.ipynb
│   └── 02_model_building_and_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py     # Data cleaning functions
│   └── model_pipeline.py         # Model training script
├── app.py                         # Streamlit web application script
├── requirements.txt               # Required Python packages
├── README.md                      # Project documentation
└── .gitignore
