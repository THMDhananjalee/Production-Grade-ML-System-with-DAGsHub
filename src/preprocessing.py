import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    # Use the exact filename with spaces
    df = pd.read_csv('data/raw/Churn Prediction DataSet .csv')
    df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric, coerce errors (blanks become NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save processed data and preprocessor objects
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_pickle('data/processed/X_train.pkl')
    X_test.to_pickle('data/processed/X_test.pkl')
    y_train.to_pickle('data/processed/y_train.pkl')
    y_test.to_pickle('data/processed/y_test.pkl')
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'data/processed/feature_columns.pkl')
    print("Preprocessing completed.")

if __name__ == "__main__":
    main() 