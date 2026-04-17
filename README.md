# Credit Risk Prediction with Explainable AI (SHAP)

## Overview

This project builds a machine learning model to predict whether a customer is likely to default on a loan within the next two years. In addition to prediction, the project focuses on explaining model decisions using SHAP, making the system more transparent and suitable for real-world financial use.

The dataset used is the *Give Me Some Credit* dataset from Kaggle.

---

## Problem Statement

Credit risk assessment is a critical task for financial institutions. While machine learning models can achieve strong predictive performance, they are often difficult to interpret.

This project aims to balance both aspects by building a reliable prediction system and providing clear explanations behind each decision.

---

## Dataset

The dataset contains 150,000 records with numerical features related to customer financial behavior, such as credit utilization, income, debt ratio, and payment history.

The target variable indicates whether a person experienced serious delinquency within two years. The data is highly imbalanced, with a much smaller proportion of default cases.

---

## Approach

The workflow begins with cleaning and preparing the data. Missing values are handled using median imputation, and features are scaled for better model performance.

Since the dataset is imbalanced, SMOTE is applied to improve the model’s ability to learn patterns from default cases.

An XGBoost classifier is then trained on the processed data. The model is evaluated using ROC-AUC, confusion matrix, and classification metrics to understand both overall performance and class-wise behavior.

---

## Results

The model achieves a ROC-AUC score of 0.846 with an overall accuracy of around 91%.

It performs well in identifying non-default cases, while recall for default cases is moderate, which is expected given the class imbalance. Overall, the model demonstrates strong generalization on unseen data.

---

## Model Explainability (SHAP)

To make the model interpretable, SHAP is used to explain predictions.

At a global level, features such as credit utilization, number of open credit lines, and past due payments are found to have the highest impact on predictions.

At a local level, individual predictions are broken down to show how each feature contributes to the final outcome. This helps in understanding why a particular customer is classified as high or low risk.

---

## Fairness Analysis

The model is also analyzed across different groups based on age, income, and debt levels.

The analysis shows that younger individuals and lower-income groups tend to have higher default rates. Group-wise evaluation metrics are computed to ensure that the model does not behave unfairly across different segments.

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, and Imbalanced-learn.

---

## Studies / References

This project is based on concepts and techniques from:

- Kaggle: *Give Me Some Credit* dataset  
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System  
- Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions  
- Chawla et al. (2002) — SMOTE: Synthetic Minority Over-sampling Technique  
- Scikit-learn documentation (model evaluation and preprocessing)  

---

## Conclusion

This project demonstrates how machine learning can be applied to credit risk prediction while maintaining transparency through explainability techniques. It highlights the importance of handling imbalanced data, interpreting model outputs, and evaluating fairness in decision-making systems.

---

## Future Work

Future improvements could include better hyperparameter tuning, deployment as a web application, and more advanced fairness-aware modeling techniques.
