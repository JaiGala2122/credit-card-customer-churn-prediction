# Customer Churn Prediction for Credit Card Company

Predictive analytics project to identify customers at risk of churning using machine learning techniques. Built with Python, Scikit-learn, and Gradient Boosting to achieve 86.9% accuracy in predicting customer attrition.

## Project Overview

This project analyzes customer data from a European credit card company to predict which customers are likely to churn. Using advanced machine learning algorithms and feature engineering, the model identifies at-risk customers, enabling proactive retention strategies.

## Key Results

- **Best Model**: Gradient Boosting Classifier
- **Accuracy**: 86.9%
- **ROC-AUC Score**: 0.870
- **Cross-validation Score**: 0.8636 (± 0.0087)
- **Business Impact**: Model can identify 87% of at-risk customers

## Dataset

- **Source**: European Credit Card Customer Data
- **Size**: 10,000 customers
- **Features**: 14 attributes including demographics, account information, and product usage
- **Target Variable**: Exited (1 = Churned, 0 = Stayed)
- **Overall Churn Rate**: 20.37%

### Features Used
- CreditScore
- Geography (France, Spain, Germany)
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

## Technologies Used

**Programming Language**: Python 3.8+

**Libraries**:
- Data Analysis: Pandas, NumPy
- Machine Learning: Scikit-learn
- Visualization: Matplotlib, Seaborn
- Model Persistence: Joblib

**Models Implemented**:
- Gradient Boosting Classifier
- Random Forest Classifier
- Logistic Regression

## Installation
```bash
## Project Workflow

### 1. Data Loading & Exploration
- Loaded 10,000 customer records with 14 features
- Analyzed data structure and distributions
- Verified data quality (no missing values detected)

### 2. Data Preprocessing
- Removed irrelevant columns (RowNumber, CustomerId, Surname)
- Encoded categorical variables using Label Encoding and One-Hot Encoding
  - Gender: Male=1, Female=0
  - Geography: One-hot encoded (Germany, Spain, France)
- Converted data types for model compatibility

### 3. Feature Engineering
Created new features to enhance model performance:
- **BalanceToSalary**: Ratio of account balance to estimated salary
- **TenureAgeRatio**: Ratio of tenure to age
- **IsZeroBalance**: Binary indicator for zero balance accounts
- **AgeGroup**: Categorical age bins (Young, Middle, Senior, Elderly)

### 4. Exploratory Data Analysis
- Visualized churn distribution (20.37% overall churn rate)
- Analyzed numerical feature distributions (CreditScore, Age, Tenure, Balance, EstimatedSalary)
- Created correlation matrix to identify feature relationships
- Examined churn patterns by:
  - Gender (Female: 25.1%, Male: 16.5%)
  - Geography (Germany shows highest churn at 32%)
  - Age groups (50+ age group shows 44.6% churn)

### 5. Model Training & Evaluation
- Split data into 80% training and 20% testing sets
- Standardized features using StandardScaler
- Trained three classification models:
  - Logistic Regression (baseline)
  - Random Forest Classifier
  - Gradient Boosting Classifier
- Evaluated models using:
  - Accuracy
  - ROC-AUC Score
  - 5-fold Cross-validation
  - Confusion Matrix
  - Classification Report

### 6. Feature Importance Analysis
- Extracted feature importances from best model (Gradient Boosting)
- Identified top 10 predictive features
- Generated actionable business insights

## Model Performance Comparison

| Model | Accuracy | ROC-AUC | CV Score |
|-------|----------|---------|----------|
| **Gradient Boosting** | **86.9%** | **0.870** | **0.8636** |
| Random Forest | 86.2% | 0.852 | 0.8599 |
| Logistic Regression | 82.7% | 0.790 | 0.8226 |

### Gradient Boosting - Detailed Metrics

**Classification Report**:
