
# 🚀 Telecom Customer Churn Prediction with End-to-End MLOps Pipeline

## 📌 Project Overview

Customer retention is one of the most critical financial levers in the telecom industry. Research indicates that **reducing churn by just 5% can boost company profits by 25% to 95%**.

This repository hosts a production-ready, fully automated machine learning system that predicts telecom customer churn based on account metrics, service usage, and billing data. Going beyond passive classification, this system actively closes the loop by triggering the **OpenAI GPT API to automatically generate personalized retention offers** for high-risk customers before they defect.

---

## 🛠️ System Architecture & Automated Workflow

The entire machine learning life cycle is engineered to run seamlessly with zero manual intervention.

1. **Data Ingestion:** Daily Trigger.
Automated ingestion of raw data streams into the pipeline environment.


2. **Data Validation:** Great Expectations / Schema Check.
Verifies data integrity, checking for schema drift, missing fields, or corrupted types.


3. **Feature Engineering:** Automated Scaling & Encoding.
Handles categorical encoding, scales numerical variables, and prepares optimal tensors for modeling.


4. **Model Training & Benchmarking:** Logistic Regression vs. Random Forest vs. XGBoost.
Trains competing classification models concurrently to identify the strongest baseline.


5. **Evaluation & LLM Action:** ROC-AUC Selection & GPT Response.
Selects the champion model based on ROC-AUC metrics. If a customer is flagged as high-risk, the pipeline passes their profile to OpenAI GPT-3 to synthesize a targeted retention incentive.


6. **Model Registration & Deployment:** MLflow -.
FastAPI + Docker">
Registers the winning model and pushes it live to a containerized REST API for real-time inference.


---

## 🏗️ Core Engineering Contributions (My MLOps Focus)

My primary responsibility was architecting and implementing the comprehensive MLOps backbone of this system to ensure full reproducibility, rigorous experiment tracking, and clean pipeline orchestration:

* **Artifact & Data Versioning (DVC):** Integrated Data Version Control (DVC) to track large datasets and model weights. This guarantees absolute data reproducibility across environments, preventing the classic *"it worked on my machine"* dilemma.
* **Experiment Tracking & Governance (MLflow):** Built out the MLflow integration to programmatically log hyperparameters, training metrics, confusion matrices, and ROC curves for every single training run.
* **Centralized Hub (DAGsHub):** Linked our Git repository, DVC storage, and MLflow tracking servers into a singular, transparent dashboard on DAGsHub for collaborative governance.
* **Orchestration (Apache Airflow):** Authored the directed acyclic graphs (DAGs) to automate the entire 6-stage lifecycle on a daily operational schedule.

---

## 📊 Dataset & Model Performance

### The Data

The pipeline utilizes the **IBM Telco Customer Churn Dataset**, containing data for 7,043 unique customers evaluated across 21 diverse demographic, service, and billing features.

### Champion Model Metrics

After rigorous benchmarking, **XGBoost** was selected as the final production model due to its superior performance and well-balanced precision-recall dynamics.

| Model Variant | Accuracy | ROC-AUC | Precision/Recall Balance |
| --- | --- | --- | --- |
| Logistic Regression | 0.78 | 0.81 | Moderate |
| Random Forest | 0.79 | 0.82 | High Variance |
| **XGBoost (Champion)** | **0.80** | **0.84** | **Optimal** |

---

## 🧰 Tech Stack

* **Core Languages & Frameworks:** Python, XGBoost, scikit-learn
* **MLOps & Versioning:** MLflow, Data Version Control (DVC), DAGsHub
* **Data Pipeline & Orchestration:** Apache Airflow
* **Deployment & Serving:** FastAPI, Docker
* **Generative AI Integration:** OpenAI GPT API

---

## 💡 Key Takeaways

> **Engineering Insight:** Building this pipeline reinforced a fundamental reality of production-grade AI: training a model is only a tiny fraction of the challenge. True engineering value lies in building robust pipelines that make models reproducible, auditable, fully automated, and safely deployable. That is the essence of MLOps.
