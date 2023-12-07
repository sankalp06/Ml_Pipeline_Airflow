
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_hyperparameters(X_train, y_train, preprocessor, classifier, param_grid):
    # Create the pipeline
    pipeline = create_pipeline(preprocessor, classifier)
    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Return the grid_search object
    return grid_search

def train_pipeline_tunning(grid_search, X_train, y_train):
    # Access the best_estimator_ from grid_search
    best_pipeline = grid_search.best_estimator_
    
    # Train the best pipeline on the training data
    best_pipeline.fit(X_train, y_train)
  
    # Print the best hyperparameters and their corresponding accuracy
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)
    
    # Return the best trained pipeline
    return best_pipeline
