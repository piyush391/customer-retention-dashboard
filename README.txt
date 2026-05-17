# Customer Retention Intelligence Platform

## Overview

Customer Retention Intelligence is an AI-powered banking analytics platform designed to analyze customer engagement, churn behavior, relationship strength, product utilization, and premium customer risk.

The project combines:

* Machine Learning
* Predictive Analytics
* Behavioral Intelligence
* Interactive Data Visualization
* Customer Segmentation
* Executive Decision Support

The dashboard helps identify high-risk customers, analyze engagement patterns, and support proactive retention strategies using predictive intelligence.

---

# Project Objectives

The primary objectives of this project are:

* Predict customer churn probability
* Analyze customer engagement behavior
* Detect high-risk premium customers
* Evaluate product utilization impact on retention
* Build proactive retention intelligence
* Visualize customer behavioral clusters
* Support executive-level decision making

---

# Key Features

## 1. Predictive Churn Analytics

Uses a trained machine learning model to:

* estimate customer churn probability
* identify high-risk customer segments
* support proactive retention campaigns

---

## 2. Relationship Strength Index (RSI)

Custom engagement metric that evaluates:

* customer activity
* product adoption
* balance strength
* behavioral engagement

Higher RSI values indicate:

* stronger loyalty
* deeper banking relationships
* lower churn vulnerability

---

## 3. Product Utilization Intelligence

Analyzes how banking product adoption impacts:

* customer loyalty
* churn probability
* retention stability

Includes:

* product count analysis
* credit card stickiness analysis
* cross-sell opportunity identification

---

## 4. Premium Customer Risk Detection

Detects financially valuable customers showing:

* low engagement
* elevated churn risk
* silent disengagement behavior

Supports:

* VIP retention strategy
* premium customer intervention
* high-value segmentation

---

## 5. Retention Recommendation Engine

Automatically categorizes customers into:

* Immediate Retention
* Cross-Sell
* Reactivation
* Loyalty Program

This transforms the project from analytics-only into a decision-support platform.

---

## 6. Behavioral Network Intelligence

Advanced customer similarity analysis using:

* NetworkX
* community detection
* behavioral clustering
* 3D network visualization

Helps identify:

* isolated customers
* hidden customer segments
* vulnerable engagement clusters

---

## 7. Executive Insights Dashboard

Provides executive-level interpretation of:

* churn trends
* engagement quality
* product depth
* customer relationship health
* retention performance

---

# Technologies Used

## Frontend & Dashboard

* Streamlit
* Plotly
* HTML/CSS
* Vanta.js

## Data Processing

* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* Pickle

## Network Intelligence

* NetworkX

---

# Dashboard Modules

| Module               | Purpose                        |
| -------------------- | ------------------------------ |
| KPI Intelligence     | Executive retention metrics    |
| Engagement Analytics | Activity vs churn analysis     |
| Product Utilization  | Product depth intelligence     |
| Premium Risk         | High-value customer monitoring |
| Retention Engine     | Recommendation generation      |
| Network Intelligence | Behavioral clustering          |
| Executive Insights   | Strategic interpretation       |

---

# KPI Metrics Included

## Customers

Total customer population under current filter conditions.

## Churn Rate

Percentage of customers who exited the bank.

## Average RSI

Average relationship strength and engagement score.

## Average Churn Risk

Machine learning predicted churn probability.

## Premium Customers

High-value customers requiring strategic retention focus.

---

# Dataset

Dataset Used:

European Banking Customer Dataset

The dataset includes:

* customer demographics
* account balance
* salary
* product ownership
* activity behavior
* churn labels

---

# Machine Learning Workflow

## Data Preparation

* preprocessing
* feature engineering
* missing value handling
* encoding

## Model Training

The churn prediction model is trained using:

* classification algorithms
* probability prediction
* churn probability estimation

## Prediction Output

The model generates:

* churn probability
* customer risk segmentation

---

# Network Intelligence Architecture

The behavioral network model connects customers based on:

* relationship strength similarity
* churn probability similarity
* product utilization similarity

Community detection is used to identify:

* behavioral clusters
* isolated customers
* hidden engagement structures

---

# Project Architecture

```text
customer-retention-intelligence/
│
├── app.py
├── preprocess.py
├── model.py
├── metrics.py
├── requirements.txt
├── README.md
├── model.py
├── European_Bank.csv
│
└── assets/
    └── screenshots/
```

---

# Installation

## Clone Repository

```bash
git clone <repository-link>
cd customer-retention-intelligence
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

---

# Future Improvements

Planned advanced enhancements:

* SHAP explainability
* ROC curve analysis
* threshold tuning
* downloadable reports
* geographic churn heatmaps
* advanced segmentation
* real-time prediction APIs

---

# Business Value

This platform helps banking organizations:

* reduce churn
* improve retention strategy
* identify vulnerable customers
* prioritize premium customer protection
* strengthen product engagement
* improve customer lifetime value

---

# Author

Piyush Solanki

Customer Retention Intelligence Platform

---

# License

This project is intended for educational, portfolio, and analytical demonstration purposes.
