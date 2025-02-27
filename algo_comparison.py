import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,balanced_accuracy_score

# Define models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture


from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from scipy.spatial import ConvexHull
from scipy.io import arff  # for handling .arff files
from ConvexHullAnomalyDetectorClass import *

# Function to load .arff datasets
from scipy.io import arff

# Updated function to load .arff datasets
def load_arff_dataset(filepath):
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    return df


# Train-test split
def split_data(data):
    # tag column
    label_column = 'outlier'

    # check if tag column exist
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    # ignore id column
    feature_columns = data.columns.drop([label_column, 'id'], errors='ignore')

    # proccess tags
    if data[label_column].dtype == 'object' or data[label_column].dtype.name == 'bytes':

        data[label_column] = data[label_column].apply(lambda x: 1 if x in [b"yes", "yes", 'yes', b'y', 'y'] else 0)
    elif data[label_column].dtype in ['int64', 'float64']:

        unique_labels = data[label_column].unique()
        if len(unique_labels) > 2:
            print(f"Warning: Label column '{label_column}' has more than two unique values.")
    else:
        # make sure its numeric
        data[label_column] = data[label_column].astype(int)

    # data and tags seperation
    X = data[feature_columns].values
    y = data[label_column].values

    #Train/Test
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Stopping criteria
def elbow_point_stopping(prev_f, current_f, threshold=1e-3):
    return abs(current_f - prev_f) < threshold


# Evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, models):
    results = []
    for model_name, model in models.items():
        model.fit(X_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == 1, 0, 1)  # Adjust predictions for anomaly detection
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Recall": recall,
            "Precision": precision
        })
    return pd.DataFrame(results)


# Optimize hyperparameters
def optimize_hyperparameters(model, param_grid, X_train, y_train):
    opt = BayesSearchCV(model, param_grid, n_iter=30, cv=3, random_state=42)
    opt.fit(X_train, y_train)
    return opt.best_estimator_

# # Main function
# if __name__ == "__main__":
#     # Path to the parent directory containing folders of datasets
#     parent_folder = "datasets"
#
#     # Placeholder for results
#     results = []
#
#     # Iterate through all folders and subfolders
#     for folder in os.listdir(parent_folder):
#         folder_path = os.path.join(parent_folder, folder)
#         if os.path.isdir(folder_path):
#             print(f"Processing folder: {folder}")
#
#             for file in os.listdir(folder_path):
#                 if file.endswith('.arff'):
#                     file_path = os.path.join(folder_path, file)
#                     print(f"Processing file: {file_path}")
#
#                     # Load dataset
#                     data = load_arff_dataset(file_path)
#                     try:
#                         X_train, X_test, y_train, y_test = split_data(data)
#                     except Exception as e:
#                         print(f"Error processing {file_path}: {e}")
#                         continue
#
#                     num_zeros = np.sum(y_test == 0)  # Count 0s
#                     num_ones = np.sum(y_test == 1)  # Count 1s
#
#                     print(f"Number of 0s: {num_zeros}")
#                     print(f"Number of 1s: {num_ones}")
#
#                     # Define models, including the custom Convex Hull algorithm
#                     models = {
#                         # "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
#                         # "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
#                         # "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
#                         # "Gaussian Mixture Models": GaussianMixture(n_components=2, covariance_type="full"),
#                         # "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
#                         # "K-Means": KMeans(n_clusters=2, random_state=42),
#                         "Convex Hull": PcaConvexHullAnomalyDetector()  # Your custom algorithm
#                     }
#
#                     # Evaluate models on the current dataset
#                     for algo_name, algo in models.items():
#                         if algo_name=="Convex Hull":
#                             print()
#                         try:
#                             algo.fit(X_train)
#
#                             # Compute fit accuracy
#                             algo.compute_fit_accuracy(X_train, y_train)
#
#                             #algo.plot_pca_with_hull(X_train, y_train)
#
#
#                             y_pred = algo.predict(X_test)
#                             y_pred = np.where(y_pred == 1, 0, 1)  # Adjust predictions for anomaly detection
#
#                             # Calculate metrics
#                             metrics = {
#                                 "Accuracy": accuracy_score(y_test, y_pred),
#                                 "F1 Score": f1_score(y_test, y_pred),
#                                 "Recall": recall_score(y_test, y_pred),
#                                 "Precision": precision_score(y_test, y_pred)
#                             }
#
#                             # Append results
#                             results.append({
#                                 "Dataset": file_path,
#                                 "Algorithm": algo_name,
#                                 **metrics
#                             })
#
#                         except Exception as e:
#                             print(f"Error with {algo_name} on {file_path}: {e}")
#                             continue
#         print('')
#     # Convert results to a DataFrame
#     results_df = pd.DataFrame(results)
#
#     # Save results to CSV
#     results_df.to_csv("results_table.csv", index=False)
#     print("Results saved to results_table.csv")


def evaluate_models_on_dataset(models, X_train, y_train, X_test, y_test,file_path):
    """
    Evaluate all models on the given train and test data.
    """
    results = []

    for algo_name, algo in models.items():
        try:
            print(f'running {algo_name} on {file_path}')
            # Train the model
            algo.fit(X_train)

            # Train set predictions
            y_train_pred = algo.predict(X_train)
            y_train_pred = np.where(y_train_pred == 1, 0, 1)  # Adjust for anomaly detection

            # Test set predictions
            y_test_pred = algo.predict(X_test)
            y_test_pred = np.where(y_test_pred == 1, 0, 1)  # Adjust for anomaly detection

            # Metrics for train
            train_metrics = {
                "Algorithm": algo_name,
                "Dataset": "Train",
                "Accuracy": accuracy_score(y_train, y_train_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_train, y_train_pred),
                "F1 Score": f1_score(y_train, y_train_pred),
                "Recall": recall_score(y_train, y_train_pred),
                "Precision": precision_score(y_train, y_train_pred)
            }

            # Metrics for test
            test_metrics = {
                "Algorithm": algo_name,
                "Dataset": "Test",
                "Accuracy": accuracy_score(y_test, y_test_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred),
                "F1 Score": f1_score(y_test, y_test_pred),
                "Recall": recall_score(y_test, y_test_pred),
                "Precision": precision_score(y_test, y_test_pred)
            }

            results.append(train_metrics)
            results.append(test_metrics)

        except Exception as e:
            print(f"Error with {algo_name}: {e}")
            continue

    return results


def process_datasets(parent_folder):
    """
    Process all datasets in the given parent folder and evaluate models.
    """
    all_results = []
    break_at=3
    count=0
    for folder in os.listdir(parent_folder):
        if count==break_at:
            break
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder}")

            for file in os.listdir(folder_path):
                if file.endswith('.arff'):
                    file_path = os.path.join(folder_path, file)
                    print(f"Processing file: {file_path}")

                    try:
                        # Load and split dataset
                        data = load_arff_dataset(file_path)
                        X_train, X_test, y_train, y_test = split_data(data)

                        # Models dictionary
                        models = {
                            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
                            "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
                            # "Gaussian Mixture Models": GaussianMixture(n_components=2, covariance_type="full"),
                            # "K-means": KMeans(n_clusters=2, random_state=42),
                            # "Parallel Convex Hull": ParallelCHoutsideConvexHullAnomalyDetector(),
                        }

                        # Evaluate  models
                        results = evaluate_models_on_dataset(models, X_train, y_train, X_test, y_test,file_path)
                        all_results.extend(results)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
    return all_results


def aggregate_results(results_df):
    """
    Aggregate results across all datasets to compute averages.
    """
    avg_results = results_df.groupby(["Algorithm", "Dataset"]).mean().reset_index()
    return avg_results


if __name__ == "__main__":
    parent_folder = "datasets"

    # Process all datasets and evaluate models
    results = process_datasets(parent_folder)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save per-dataset results
    results_df.to_csv("results_per_dataset.csv", index=False)
    print("Results saved to results_per_dataset.csv")

    # Compute averages
    avg_results = aggregate_results(results_df)
    avg_results.to_csv("average_results.csv", index=False)
    print("Average results saved to average_results.csv")

    # Print average results for review
    print("Average Results:")
    print(avg_results)
#
# if __name__ == "__main__":
#     from google.colab import drive
#
#     drive.mount('/content/drive')
#
#     parent_folder = "/content/drive/MyDrive/datasets"  # Adjust to your folder
#
#     # Process all datasets and evaluate models
#     results = process_datasets(parent_folder)
#
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
#
#     # Save per-dataset results
#     results_df.to_csv("/content/drive/MyDrive/results_per_dataset.csv", index=False)
#     print("Results saved to /content/drive/MyDrive/results_per_dataset.csv")
#
#     # Compute averages
#     avg_results = aggregate_results(results_df)
#     avg_results.to_csv("/content/drive/MyDrive/average_results.csv", index=False)
#     print("Average results saved to /content/drive/MyDrive/average_results.csv")
#
#     # Print average results for review
#     print("Average Results:")
#     print(avg_results)