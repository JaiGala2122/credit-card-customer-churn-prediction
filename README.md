# Customer Churn Prediction - Machine Learning Project

A model that predicts which credit card customers are likely to leave, so a retention team knows who to contact first.

**Full write-up, results and charts:** [galajainam.com](https://www.galajainam.com/projects/customer-churn/index.html)

## 🎯 Project Overview

This project shows data analysis and machine learning skills through:
- Cleaning and preparing customer data with pandas
- Creating new features from balance, salary, age and tenure
- Comparing three models and choosing the best one
- Tuning the decision cutoff to catch more customers who leave

## 📦 Dataset

European credit card customer data - 10,000 customers, 14 columns. 20.37% of customers left (`Exited = 1`).

## ⚙️ How It Works

1. Clean the data - drop ID columns, encode gender and country
2. Add features - balance-to-salary, tenure-to-age, zero-balance flag, age groups
3. Train and compare - logistic regression, random forest and gradient boosting on an 80/20 split
4. Pick the best model - gradient boosting, chosen by ROC-AUC
5. Tune the cutoff - check how many leavers are caught at different cutoffs and how well the model ranks customers by risk

## 🛠️ Skills Demonstrated

- Data cleaning and encoding with pandas
- Feature engineering
- Model comparison with scikit-learn
- Choosing the right metric for imbalanced data (ROC-AUC, recall, precision)
- Threshold tuning and lift analysis
- Charts with matplotlib and seaborn

## 🚀 Quick Start

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_analysis.ipynb
```

Run all cells from top to bottom. The notebook reads `data/Europe Credit Card Data.csv` and saves the model to `models/`.

## 📁 Files

```
customer_churn_analysis.ipynb   Full analysis and models
data/                           Dataset
models/                         Saved model and scaler
requirements.txt                Python packages
```
