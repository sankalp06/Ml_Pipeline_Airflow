import os 
import json

def find_best_model(directory):
    best_error = float('inf')  # Initialize with a high value
    best_model_path = None

    # Iterate over all folders in the specified directory
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        # Check if the current item is a directory
        if os.path.isdir(folder_path):
            metrics_file_path = os.path.join(folder_path, 'metrics.json')

            # Check if metrics.json file exists in the current folder
            if os.path.exists(metrics_file_path):
                # Read metrics from JSON file
                with open(metrics_file_path, 'r') as metrics_file:
                    metrics_data = json.load(metrics_file)

                    # Check if "mean_absolute_error" is present in the metrics data
                    if 'mean_absolute_error' in metrics_data:
                        current_error = metrics_data['mean_absolute_error']

                        # Update best model if the current error is lower
                        if current_error < best_error:
                            # Search for the model file dynamically
                            model_files = [f for f in os.listdir(folder_path) if f.endswith('.joblib')]
                            if model_files:
                                best_error = current_error
                                model_file_path = os.path.join(folder_path, model_files[0])
                                best_model_path = model_file_path

    # Replace backslashes with forward slashes
    best_model_path = best_model_path.replace('\\', '/')
    print(type(best_model_path))
    

    return best_model_path


