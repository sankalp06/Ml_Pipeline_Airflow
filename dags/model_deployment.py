from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
import os


scripts_folder_path = "/opt/airflow/ml_pipeline_scripts" 
data_folder_path1 = "/opt/airflow/data" 
pipeline_folder_path2 = "/opt/airflow/model_pipelines" 


os.chdir(scripts_folder_path)
os.chdir(data_folder_path1)
os.chdir(pipeline_folder_path2)

exec(open('/opt/airflow/ml_pipeline_scripts/ml_pipeline.py').read())
exec(open('/opt/airflow/ml_pipeline_scripts/ml_ops.py').read())



def predict_save_csv(**kwargs):
    directory = kwargs['directory']
    model_filename = find_best_model(directory)
    new_data_filename = kwargs['new_data_filename']
    output_filename = kwargs['output_filename']
    # Load the pre-trained pipeline
    loaded_pipeline = load_pipeline(model_filename)
    # Read the new data
    new_data = pd.read_csv(new_data_filename)
    # Predict with the loaded pipeline
    predictions = predict_with_pipeline(loaded_pipeline, new_data)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Label'])
    # Save predictions to a CSV file
    predictions_df.to_csv(output_filename, index=False)
    print("prediction is done")

default_args = {
    'owner': 'sankalp',
    'depends_on_past': False,
    'start_date': datetime(2023, 12, 4),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_deployment',
    default_args=default_args,
    description='prediction using best model',
    schedule_interval=timedelta(days=1),
)



predict = PythonOperator(
    task_id='predict_save_csv',
    python_callable=predict_save_csv,
    op_kwargs ={
        #'model_filename':'/opt/airflow/model_pipelines/Anti_Benchmark_Models/after_tuned/RandomForestClassifier_20231207053800/model_20231207053800.joblib',
        'directory': '/opt/airflow/model_pipelines/Anti_Benchmark_Models/after_tuned', 
        'new_data_filename':'/opt/airflow/data/validation_data/test_data.csv', 
        'output_filename':'/opt/airflow/data/predictions/predictions_after_tuned.csv',
    },
    dag=dag,
)


predict
