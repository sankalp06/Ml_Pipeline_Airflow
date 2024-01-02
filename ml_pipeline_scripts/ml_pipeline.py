
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo
import pandas as pd
import joblib
import os
import json

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
    return accuracy

def save_pipeline(pipeline, file_path):
    joblib.dump(pipeline, file_path)

def load_pipeline(file_path):
    return joblib.load(file_path)

def predict_with_pipeline(pipeline, new_data):
    return pipeline.predict(new_data)


def save_metrics(metrics, folder_path):
    with open(os.path.join(folder_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    accuracy = pipeline.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    return accuracy, report, conf_matrix, mae, mse, r_squared

def save_model_and_metrics(pipeline, folder_path, accuracy, report, conf_matrix, mae, mse, r_squared):
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    save_model_filename = os.path.join(folder_path, f"model_{timestamp}.joblib")
    save_pipeline(pipeline, save_model_filename)
    metrics_dict = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
        'r_squared': r_squared
    }
    save_metrics(metrics_dict, folder_path)

