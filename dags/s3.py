from datetime import datetime
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.S3_hook import S3Hook


def upload_to_s3(filename: str, key: str, bucket_name: str) -> None:
    hook = S3Hook('S3_conn')
    hook.load_file(filename=filename, key=key, bucket_name=bucket_name)


with DAG(
    dag_id='s3_dag',
    schedule_interval='@daily',
    start_date=datetime(2023, 12, 27),
    catchup=False
) as dag:

    # Upload the file
    task_upload_to_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
        op_kwargs={
            'filename': 'Ml_Pipeline_Airflow/data/training_data/train_data.csv',
            'key': 'train_data.csv',
            'bucket_name': 'airflow'
        }
    )