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
    file_path=kwargs['file_path'] 
    target_column=kwargs['target_column']
    save_model_filename= kwargs['save_model_filename']
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
    # Create the pipeline
    pipeline = create_pipeline(preprocessor, classifier)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the pipeline on the training data
    train_pipeline(pipeline, X_train, y_train)
    # Evaluate the model on the testing data
    accuracy = evaluate_pipeline(pipeline, X_test, y_test)
    # Save the trained pipeline to a file
    save_pipeline(pipeline, save_model_filename) 
    return accuracy


def predict_save_csv(**kwargs):
    model_filename = kwargs['model_filename']
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
    'simple_ml_pipeline',
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

train_save_model = PythonOperator(
    task_id='train_save_model',
    python_callable=train_and_evaluate_model,
    op_kwargs={
        'file_path':'/opt/airflow/data/training_data/train_data.csv', 
        'target_column':'cid', 
        'save_model_filename':'/opt/airflow/model_pipelines/trained_pipeline.joblib', 
        'classifier':RandomForestClassifier(),
    },
    dag=dag,
)

predict = PythonOperator(
    task_id='predict_save_csv',
    python_callable=predict_save_csv,
    op_kwargs ={
        'model_filename':'/opt/airflow/model_pipelines/trained_pipeline.joblib', 
        'new_data_filename':'/opt/airflow/data/validation_data/test_data.csv', 
        'output_filename':'/opt/airflow/data/predictions/predictions.csv',
    },
    dag=dag,
)

load_data >> train_save_model >> predict 
