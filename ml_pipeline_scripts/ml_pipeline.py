
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
import pandas as pd
import joblib

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

