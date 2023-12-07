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
exec(open('/opt/airflow/ml_pipeline_scripts/hyper_tuning.py').read())

def train_and_evaluate_with_tuning(**kwargs):
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    file_path=kwargs['file_path'] 
    target_column=kwargs['target_column']
    classifier=kwargs['classifier']

    df = pd.read_csv(file_path)
    df = handle_duplicates(df)
    # Separate features and target variable
    X, y = separate_features_target(df, target_column)
    # Identify numerical and categorical features
    numerical_features, categorical_features = identify_features(X)
    # Create preprocessor
    preprocessor = create_preprocessor(numerical_features,categorical_features)
    # Use the specified classifier or default to RandomForestClassifier
    if classifier is None:
        classifier = RandomForestClassifier()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Tune hyperparameters
    grid_search = tune_hyperparameters(X_train, y_train, preprocessor, classifier, param_grid)
    # Train the best pipeline on the training data
    best_pipeline = train_pipeline_tunning(grid_search, X_train, y_train)

    accuracy, report, conf_matrix, mae, mse, r_squared = evaluate_model(best_pipeline, X_test, y_test)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    folder_path = f"/opt/airflow/model_pipelines/Tuned_model/{classifier.__class__.__name__}_{timestamp}"
    os.makedirs(folder_path, exist_ok=True)
    save_model_and_metrics(best_pipeline, folder_path, accuracy, report, conf_matrix, mae, mse, r_squared)

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
    'start_date': datetime(2023, 12, 5),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'hyper_tuning_ml_model',
    default_args=default_args,
    description='Tuning ML model',
    schedule_interval=timedelta(days=1),
)


train_save_model = PythonOperator(
    task_id='tune_model',
    python_callable=train_and_evaluate_with_tuning,
    op_kwargs={
        'file_path':'/opt/airflow/data/training_data/train_data.csv', 
        'target_column':'cid', 
        'classifier':RandomForestClassifier(),
    },
    dag=dag,
)

train_save_model
