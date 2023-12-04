from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def check_sklearn():
    try:
        import sklearn
        print("scikit-learn is installed. Version:", sklearn.__version__)
    except ImportError:
        print("scikit-learn is not installed.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 12, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'check',
    default_args=default_args,
    description='DAG to check scikit-learn installation',
    schedule_interval=timedelta(days=1),  # Adjust as needed
)

check_task = PythonOperator(
    task_id='check_sklearn_task',
    python_callable=check_sklearn,
    dag=dag,
)

check_task  # Set up task dependencies as needed
