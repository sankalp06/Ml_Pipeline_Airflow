from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os


scripts_folder_path = "/opt/airflow/ml_pipeline_scripts" 
data_folder_path1 = "/opt/airflow/data" 
pipeline_folder_path2 = "/opt/airflow/model_pipelines" 

os.chdir(scripts_folder_path)
os.chdir(data_folder_path1)
os.chdir(pipeline_folder_path2)

exec(open('/opt/airflow/ml_pipeline_scripts/ml_pipeline.py').read())

def load(**kwargs):
    id = kwargs['id']
    output_filename = kwargs['output_filename']
    # Fetch dataset
    dataset = fetch_ucirepo(id=id) 
    # Extract features and targets
    X_ = dataset.data.features 
    y_ = dataset.data.targets 
    # Merge X and y into a single DataFrame
    df = pd.concat([X_, y_], axis=1)
    # Save the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)


def train_and_evaluate_model(**kwargs):
    file_path = kwargs['file_path']
    target_column = kwargs['target_column']
    classifier = kwargs['classifier']
    df = pd.read_csv(file_path)
    df = handle_duplicates(df)
    X, y = separate_features_target(df, target_column)
    numerical_features, categorical_features = identify_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = create_preprocessor(numerical_features,categorical_features)
    if classifier is None:
        classifier = RandomForestClassifier()
    pipeline = create_pipeline(preprocessor, classifier)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_pipeline(pipeline, X_train, y_train)
    
    accuracy, report, conf_matrix, mae, mse, r_squared = evaluate_model(pipeline, X_test, y_test)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    folder_path = f"/opt/airflow/model_pipelines/Base_model/{classifier.__class__.__name__}_{timestamp}"
    os.makedirs(folder_path, exist_ok=True)
    save_model_and_metrics(pipeline, folder_path, accuracy, report, conf_matrix, mae, mse, r_squared)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    return accuracy

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
    'ml_pipeline',
    default_args=default_args,
    description='creating ml pipeline using airfloe',
    schedule_interval=timedelta(days=1),
)

load_data = PythonOperator(
    task_id='loading_data',
    python_callable=load,
     op_kwargs={
        'id':890, 
        'output_filename':'/opt/airflow/data/training_data/train_data.csv',
    },
    dag=dag,
)

train_save_model_1 = PythonOperator(
    task_id='train_save_model',
    python_callable=train_and_evaluate_model,
    op_kwargs={
        'file_path':'/opt/airflow/data/training_data/train_data.csv', 
        'target_column':'cid', 
        'classifier':RandomForestClassifier(),
    },
    dag=dag,
)

train_save_model_2 = PythonOperator(
    task_id='train_save_model_1',
    python_callable=train_and_evaluate_model,
    op_kwargs={
        'file_path':'/opt/airflow/data/training_data/train_data.csv', 
        'target_column':'cid', 
        'classifier':SVC(),
    },
    dag=dag,
)

load_data >> [train_save_model_1,train_save_model_2]
