import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.io import arff  # for handling .arff files

# Function to load .arff datasets
def load_arff_dataset(filepath):
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    return df

# Train-test split
def split_data(data):
    label_column = 'outlier'
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    feature_columns = data.columns.drop([label_column, 'id'], errors='ignore')
    if data[label_column].dtype in ['object', 'bytes']:
        data[label_column] = data[label_column].apply(lambda x: 1 if x in [b"yes", "yes", 'yes', b'y', 'y'] else 0)
    else:
        data[label_column] = data[label_column].astype(int)

    X = data[feature_columns].values
    y = data[label_column].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate models
def evaluate_models(X, models, dataset_name):
    """
    Evaluate models on the given data.
    Differentiates between algorithms with fit_predict and those with fit + predict.
    """
    results = []

    for model_name, model in models.items():
        try:
            print(f"Running {model_name} on {dataset_name}")

            # Check if the algorithm has a fit_predict method
            if hasattr(model, "fit_predict"):
                y_pred = model.fit_predict(X)
                y_pred = np.where(y_pred == 1, 0, 1)  # Adjust predictions for anomaly detection
            else:
                # Fit the model
                model.fit(X)
                y_pred = model.predict(X)
                y_pred = np.where(y_pred == 1, 0, 1)  # Adjust predictions for anomaly detection

            # Collect metrics (labels are assumed to be available)
            accuracy = None
            if "y_true" in locals():
                accuracy = accuracy_score(y_true, y_pred)

            # Metrics
            metrics = {
                "Algorithm": model_name,
                "Dataset": dataset_name,
                "Accuracy": accuracy,
                "Precision": precision_score(y_true, y_pred) if "y_true" in locals() else None,
                "Recall": recall_score(y_true, y_pred) if "y_true" in locals() else None,
                "F1 Score": f1_score(y_true, y_pred) if "y_true" in locals() else None
            }
            results.append(metrics)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue

    return results


# Save intermediate results
def save_intermediate_results(results, file_path):
    results_df = pd.DataFrame(results)
    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, index=False)

# Process datasets
def process_datasets(parent_folder, results_file_path):
    """
    Process all datasets in the given parent folder and evaluate models.
    """
    all_results = []

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            for file in os.listdir(folder_path):
                if file.endswith('.arff'):
                    file_path = os.path.join(folder_path, file)
                    print(f"Processing file: {file_path}")
                    try:
                        # Load dataset
                        data = load_arff_dataset(file_path)
                        X, _, y_true, _ = split_data(data)  # Only use X and y_true

                        # Define models
                        models = {
                            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
                            "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
                            "Gaussian Mixture Models": GaussianMixture(n_components=2, covariance_type="full"),
                            "K-means": KMeans(n_clusters=2, random_state=42),
                            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
                            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
                            "Spectral Clustering": SpectralClustering(n_clusters=2, random_state=42),
                            "Mean Shift": MeanShift(),
                            "Parallel Convex Hull": ParallelCHoutsideConvexHullAnomalyDetector()
                        }

                        # Evaluate models
                        results = evaluate_models(X, models, file_path)

                        # Save intermediate results
                        save_intermediate_results(results, results_file_path)
                        all_results.extend(results)

                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue

    return all_results


# Aggregate results
def aggregate_results(results_file_path, output_file_path):
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
        numeric_columns = results_df.select_dtypes(include='number').columns
        avg_results = results_df.groupby("Algorithm")[numeric_columns].mean().reset_index()
        avg_results.to_csv(output_file_path, index=False)
        print(f"Aggregated results saved to: {output_file_path}")

if __name__ == "__main__":
    datasets_folder = "/kaggle/input/ch-data/literature"
    results_file_path = "/kaggle/working/results_per_dataset.csv"
    avg_results_file_path = "/kaggle/working/average_results.csv"

    # Process datasets and save intermediate results
    print(f"Processing datasets in: {datasets_folder}")
    process_datasets(datasets_folder, results_file_path)

    # Aggregate and save final results
    aggregate_results(results_file_path, avg_results_file_path)

    print("Processing complete! Check the 'Output' section for saved files.")
