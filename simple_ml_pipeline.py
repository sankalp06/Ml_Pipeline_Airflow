import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from ucimlrepo import fetch_ucirepo
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator


def separate_features_target(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def identify_features(X):
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    return numerical_features, categorical_features

def handle_duplicates(df):
    df_no_duplicates = df.drop_duplicates()
    return df_no_duplicates

def create_preprocessor(numerical_features,categorical_features):
    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def create_classifier():
    return RandomForestClassifier()

def create_pipeline(preprocessor, classifier):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)

def evaluate_pipeline(pipeline, X_test, y_test):
    accuracy = pipeline.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy}')

def save_pipeline(pipeline, file_path):
    joblib.dump(pipeline, file_path)

def load_pipeline(file_path):
    return joblib.load(file_path)

def predict_with_pipeline(pipeline, new_data):
    return pipeline.predict(new_data)


def load(id = 890, output_filename= './dags/train_data.csv'):
    # Fetch dataset
    dataset = fetch_ucirepo(id=id) 
    # Extract features and targets
    X_ = dataset.data.features 
    y_ = dataset.data.targets 
    # Merge X and y into a single DataFrame
    df = pd.concat([X_, y_], axis=1)
    # Save the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)

def train_and_evaluate_model(file_path='./dags/train_data.csv', target_column='cid', save_model_filename= './dags/trained_pipeline.joblib', classifier=RandomForestClassifier()):
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


def predict_save_csv(model_filename = './dags/trained_pipeline.joblib', new_data_filename = './dags/test_data.csv', output_filename= './dags/predictions.csv'):
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
    'start_date': datetime(2023, 12, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'simple_ml_pipeline',
    default_args=default_args,
    description='creating ml pipeline using airfloe',
    schedule_interval=timedelta(days=1),  # Adjust as needed
)

load_data = PythonOperator(
    task_id='loading_data',
    python_callable=load,
    dag=dag,
)

train_save_model = PythonOperator(
    task_id='train_save_model',
    python_callable=train_and_evaluate_model,
    dag=dag,
)

predict = PythonOperator(
    task_id='predict_save_csv',
    python_callable=predict_save_csv,
    dag=dag,
)

load_data >> train_save_model >> predict # Set up task dependencies as needed
