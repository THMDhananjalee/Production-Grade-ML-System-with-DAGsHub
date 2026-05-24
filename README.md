# 🚀 Telecom Customer Churn Prediction with End-to-End MLOps Pipeline

## 📌 Project Overview

Customer retention is one of the most critical financial levers in the telecom industry. Research indicates that reducing churn by just 5% can boost company profits by 25% to 95%.

This repository hosts a production-ready, fully automated machine learning system that predicts telecom customer churn based on account metrics, service usage, and billing data. Going beyond passive classification, this system actively closes the loop by triggering the OpenAI GPT API to automatically generate personalized retention offers for high-risk customers before they defect.

---

## 🛠️ System Architecture & Automated Workflow

The entire machine learning life cycle is engineered to run seamlessly with zero manual intervention.

1. Data Ingestion — Automated ingestion of raw data streams into the pipeline environment.
2. Data Validation — Verifies data integrity, checking for schema drift, missing fields, or corrupted types.
3. Feature Engineering — Handles categorical encoding, scales numerical variables, and prepares data for modeling.
4. Model Training & Benchmarking — Trains Logistic Regression, Random Forest, and XGBoost concurrently to identify the strongest model.
5. Evaluation & LLM Action — Selects the champion model based on ROC-AUC. If a customer is flagged as high-risk, the pipeline passes their profile to OpenAI GPT-3 to generate a targeted retention offer.
6. Model Registration & Deployment — Registers the winning model and pushes it live to a containerized REST API for real-time inference.

---

## 🏗️ My MLOps Contribution

My primary responsibility was architecting and implementing the complete MLOps backbone of this system to ensure full reproducibility, rigorous experiment tracking, and clean pipeline orchestration.

- Data & Artifact Versioning (DVC) — Integrated DVC to track large datasets and model weights, guaranteeing absolute reproducibility across environments.
- Experiment Tracking (MLflow) — Built out MLflow integration to programmatically log hyperparameters, metrics, confusion matrices, and ROC curves for every training run.
- Centralized Hub (DAGsHub) — Linked our Git repository, DVC storage, and MLflow tracking into a single transparent dashboard for collaborative governance.
- Pipeline Orchestration (Apache Airflow) — Authored the DAGs to automate the entire 6-stage lifecycle on a daily operational schedule with zero manual steps.

---

## 📊 Dataset & Model Performance

Dataset: IBM Telco Customer Churn — 7,043 customers, 21 features including demographics, services, and billing information.

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.80 | 0.84 |
| Random Forest | 0.79 | 0.83 |
| XGBoost (Champion) | 0.80 | 0.84 |

XGBoost was selected as the final production model for its superior performance and optimal precision-recall balance.

---

## 🧰 Tech Stack

Python, XGBoost, scikit-learn, MLflow, DVC, DAGsHub, Apache Airflow, FastAPI, Docker, OpenAI GPT-3

---

## 💡 Key Takeaway

Building this pipeline reinforced a fundamental reality of production-grade AI — training a model is only a tiny fraction of the challenge. True engineering value lies in building robust pipelines that make models reproducible, auditable, fully automated, and safely deployable. That is the essence of MLOps.
