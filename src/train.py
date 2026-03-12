import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_and_log(model, model_name, params, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc
        })

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

        # ROC curve plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig('roc_curve.png')
        mlflow.log_artifact('roc_curve.png')
        plt.close()

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally for later use
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f'models/{model_name}.pkl')

def main():
    X_train = pd.read_pickle('data/processed/X_train.pkl')
    X_test = pd.read_pickle('data/processed/X_test.pkl')
    y_train = pd.read_pickle('data/processed/y_train.pkl')
    y_test = pd.read_pickle('data/processed/y_test.pkl')

    models = [
        (LogisticRegression(max_iter=1000), 'LogisticRegression', {'C': 1.0}),
        (RandomForestClassifier(n_estimators=100, random_state=42), 'RandomForest', {'n_estimators': 100}),
        (XGBClassifier(eval_metric='logloss', random_state=42), 'XGBoost', {'n_estimators': 100, 'learning_rate': 0.1})
    ]

    for model, name, params in models:
        train_and_log(model, name, params, X_train, y_train, X_test, y_test)

    # For API, copy the best model (XGBoost) to api/model
    import shutil
    os.makedirs('api/model', exist_ok=True)
    shutil.copy('models/XGBoost.pkl', 'api/model/best_model.pkl')
    shutil.copy('data/processed/scaler.pkl', 'api/model/scaler.pkl')
    shutil.copy('data/processed/feature_columns.pkl', 'api/model/feature_columns.pkl')

if __name__ == "__main__":
    main()