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


