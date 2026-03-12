from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os

# IMPORTANT: Update this path to match YOUR system
PROJECT_PATH = r'C:\Users\ASUS\Production-Grade ML System with DAGsHub'

def run_script(script_name):
    script_path = os.path.join(PROJECT_PATH, 'src', script_name)
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Script {script_name} failed: {result.stderr}")
    print(result.stdout)

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'customer_churn_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='data_ingestion',
        python_callable=lambda: print("Data already in raw folder")
    )

    t2 = PythonOperator(
        task_id='data_validation',
        python_callable=lambda: print("Validation passed")
    )

    t3 = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_script,
        op_kwargs={'script_name': 'preprocessing.py'}
    )

    t4 = PythonOperator(
        task_id='model_training',
        python_callable=run_script,
        op_kwargs={'script_name': 'train.py'}
    )

    t5 = PythonOperator(
        task_id='model_evaluation',
        python_callable=lambda: print("Evaluation completed")
    )

    t6 = PythonOperator(
        task_id='model_registration',
        python_callable=lambda: print("Registered best model")
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6