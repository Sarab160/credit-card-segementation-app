# 🏦 Customer Credit Card Segmentation with DBSCAN & Decision Tree

This project applies unsupervised learning (clustering using DBSCAN) followed by supervised learning (classification using Decision Tree) to analyze credit card customer behavior. The project is built with Streamlit to provide an interactive dashboard for model evaluation, confusion matrix visualization, and real-time prediction.

## 📌 Project Overview

Dataset: Credit Card Dataset (CC GENERAL.csv)

Unsupervised Learning: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Supervised Learning: Decision Tree Classifier

Visualization: Seaborn, Matplotlib, Streamlit

App Features:

Show model scores (Precision, Recall, F1, Train/Test Score)

Show clustering performance (Silhouette Score, Number of Clusters)

Confusion Matrix visualization

User inputs to predict cluster (Low Spender / High Spender / Outlier, etc.)

## 📊 Machine Learning Workflow
### 1. Data Preprocessing

Handle missing values (CREDIT_LIMIT, MINIMUM_PAYMENTS)

Standardize data using StandardScaler

### 2. Unsupervised Learning (Clustering)

Apply DBSCAN to group customers

Identify outliers (-1) and spending clusters (0, 1, 2 …)

Evaluate clustering with Silhouette Score

### 3. Supervised Learning (Classification)

Split data into train/test sets

Train Decision Tree Classifier on cluster labels

Evaluate with:

Train Score

Test Score

Precision, Recall, F1 (macro-average for multiclass)

Confusion Matrix

## 🖥️ Streamlit App Features
### 📌 Sidebar

Train/Test Score

Precision, Recall, F1 Score

Silhouette Score

Cluster details (labels & count)

### 📌 Main Page

Confusion Matrix (shown on button click)

Prediction Section:

Input customer details

Get predicted cluster label (e.g., "High Spender", "Low Spender", "Outlier")
### 🔮 Cluster Interpretation
Cluster Label	Meaning
-1	Noise / Outlier
0	Low Spender
1	High Spender
2	Moderate Spender

## 📈 Results

Example model results:

Train Score: 0.98

Test Score: 0.96

Precision (macro): 0.85

Recall (macro): 0.66

F1 Score (macro): 0.74

Silhouette Score: 0.49
## 🛠️ Technologies Used

Python

Pandas, Numpy

Scikit-learn

Seaborn, Matplotlib

Streamli
