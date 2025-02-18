import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score
# Import models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.io import arff  # for handling .arff files
from ConvexHullAnomalyDetectorClass import ParallelCHoutsideConvexHullAnomalyDetector,ConfigurableConvexHullAnomalyDetector
import logging

# For convex hull anomaly detection (replace with your custom class import if needed)

#path="C:/Users/Asael/PycharmProjects/convexhull"
path='/home/someliejonson'
# Dimensionality reduction
from sklearn.decomposition import PCA
logging.basicConfig(
    filename=f"{path}/process_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Script started")

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
    return X,y


# Evaluate models
def evaluate_models(X, y_true, models, dataset_name):
    """
    Evaluate models on the given data (handles fit_predict and fit + predict).
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

            # Collect metrics only if ground truth labels are available
            if y_true is not None and len(y_true) > 0:
                accuracy = accuracy_score(y_true, y_pred)
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            else:
                accuracy = precision = recall = f1 = "N/A"

            # Append results
            metrics = {
                "Algorithm": model_name,
                "Dataset": dataset_name,
                "Accuracy": accuracy,
                "BalancedAccuracy": balanced_accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
            results.append(metrics)

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            logging.error(f"Error with {model_name}: {e}")
            continue

    return results


# Save intermediate results
def save_intermediate_results(results, file_path):
    results_df = pd.DataFrame(results)
    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
        print('file exist saved intermediate results to file')
    else:
        results_df.to_csv(file_path, index=False)
        print('saved intermediate results to file')

def get_processed_files(results_file_path):
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
        return results_df["Dataset"].unique()
    return []




def sensetivity_eval(parent_folder, results_file_path):
    """
    Processes each .arff dataset found in 'parent_folder' subdirectories.

    For each dataset, we demonstrate four sweeps:
      1) Lambda sweep (0..10 by 0.1).
      2) Row-count sweep (e.g. [100, 500, 1000]).
      3) Column-partition sweep (fraction=0.25, consecutive column blocks).
      4) Row-partition sweep (fraction=0.25, consecutive row blocks).

    You can remove whichever sweeps you don't need.

    Prerequisites (already defined in your code):
      - get_processed_files(results_file_path)
      - load_arff_dataset(file_path)
      - split_data(df)
      - evaluate_models(X, y, models, dataset_name)
      - save_intermediate_results(results, results_file_path)
      - ConfigurableConvexHullAnomalyDetector

    Also uses partition_features_by_fraction (for columns) and
    partition_rows_by_fraction (for rows).
    """

    processed_files = get_processed_files(results_file_path)
    all_results = []

    # Example sweeps:
    lambda_values = np.arange(0.0, 10, 1)  # 0, 0.1, ..., 1.0
    row_counts = [100, 500, 1000]
    column_fraction = 0.25
    row_fraction = 0.25

    # Default lambda for row-count, column-fraction, row-fraction sweeps
    default_lambda = 0.5

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # only process .arff not in processed_files
                if file.endswith('.arff') and file_path not in processed_files:
                    print(f"Processing file: {file_path}")

                    try:
                        data = load_arff_dataset(file_path)
                        X_full, _, y_full, _ = split_data(data)
                        n_samples, n_features = X_full.shape

                        # =========================================
                        # SWEEP #1: LAMBDA (all rows & columns)
                        # =========================================
                        for lam in lambda_values:
                            model_name = f"CH_elbowPCA_lam={lam}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=lam
                                )
                            }
                            results = evaluate_models(X_full, y_full, models,
                                                      dataset_name=f"{file_path}-lam={lam}")
                            for r in results:
                                r["SweepType"] = "lambda"
                                r["Lam"] = lam
                                # r["RowCount"] = n_samples
                                # r["ColCount"] = n_features
                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        # =========================================
                        # SWEEP #2: ROW-COUNT (fixed lambda)
                        # =========================================
                        for rc in row_counts:
                            actual_rc = min(rc, n_samples)
                            idx = np.random.choice(n_samples, size=actual_rc, replace=False)
                            X_subset = X_full[idx, :]
                            y_subset = y_full[idx]

                            model_name = f"CH_elbowPCA_rows={rc}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=default_lambda
                                )
                            }
                            results = evaluate_models(X_subset, y_subset, models,
                                                      dataset_name=f"{file_path}-rows={rc}")
                            for r in results:
                                r["SweepType"] = "row_count"
                                r["Lam"] = default_lambda
                                # r["RowCount"] = actual_rc
                                # r["ColCount"] = n_features
                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        # =========================================
                        # SWEEP #3: COLUMN PARTITION (fraction=0.25)
                        #   consecutive chunk slices of columns
                        # =========================================
                        partitions = partition_features_by_fraction(
                            n_features, fraction=column_fraction
                        )
                        # Example: if n_features=9 => fraction=0.25 => 4 slices:
                        #   [0,1], [2,3], [4,5], [6,7,8]
                        for i, chunk in enumerate(partitions):
                            chosen_cols = chunk
                            X_colsubset = X_full[:, chosen_cols]
                            y_colsubset = y_full

                            model_name = f"CH_elbowPCA_colSlice={i}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=default_lambda
                                )
                            }
                            results = evaluate_models(X_colsubset, y_colsubset, models,
                                                      dataset_name=f"{file_path}-colSlice={i}")
                            for r in results:
                                r["SweepType"] = "col_partition"
                                # r["PartitionIdx"] = i
                                # r["Fraction"] = column_fraction
                                r["Lam"] = default_lambda
                                # r["RowCount"] = n_samples
                                # r["ColCount"] = len(chunk)

                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        # =========================================
                        # SWEEP #4: ROW PARTITION (fraction=0.25)
                        #   consecutive chunk slices of rows
                        # =========================================
                        row_partitions = partition_rows_by_fraction(n_samples, row_fraction)
                        # Example: if n_samples=9 => fraction=0.25 => 4 slices:
                        #   [0,1], [2,3], [4,5], [6,7,8]
                        for i, row_chunk in enumerate(row_partitions):
                            X_rowsubset = X_full[row_chunk, :]
                            y_rowsubset = y_full[row_chunk]

                            model_name = f"CH_elbowPCA_rowSlice={i}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=default_lambda
                                )
                            }
                            results = evaluate_models(X_rowsubset, y_rowsubset, models,
                                                      dataset_name=f"{file_path}-rowSlice={i}")
                            for r in results:
                                r["SweepType"] = "row_partition"
                                # r["PartitionIdx"] = i
                                # r["Fraction"] = row_fraction
                                r["Lam"] = default_lambda
                                # r["RowCount"] = len(row_chunk)
                                # r["ColCount"] = n_features

                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        print(f"Finished sweeps for: {file_path}")

                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue

    return all_results

# -------------- New Helper for Unique Dataset Names --------------
import os

def make_dataset_sweep_name(file_path, sweep_type, sweep_value):
    """
    Creates a unique name for each (file, sweep_type, value), e.g.:
      file_path = "/home/.../mydata.arff"
      sweep_type = "lam"
      sweep_value = 3
    => "mydata-lam=3"

    This is what we'll store in the 'Dataset' column,
    so that we can skip re-processing if it already exists.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return f"{base_name}-{sweep_type}={sweep_value}"

# -------------- Fixed sensetivity_eval Function --------------
def sensetivity_eval_2(parent_folder, results_file_path):
    """
    Processes each .arff dataset in 'parent_folder' subdirectories.

    Sweeps:
      1) Lambda sweep: lam in [1..10, step=1].
      2) Row-count sweep: row_counts = [100, 500, 1000].
      3) Column-partition sweep (fraction=0.25).
      4) Row-partition sweep (fraction=0.25).

    Changes from original code:
      - lambda range changed to 'range(1, 11)'
      - each sweep now uses a unique dataset name (via 'make_dataset_sweep_name'),
        so we can check 'if sweep_dataset_name in processed_files' to skip duplicates.
      - added more logging calls to help debug and follow progress.
    """
    processed_files = get_processed_files(results_file_path)
    all_results = []

    # SWEEP parameters
    lambda_values = range(1, 11)  # now 1..10 inclusive
    row_counts = [100, 500, 1000]
    column_fraction = 0.25
    row_fraction = 0.25
    default_lambda = 0.5

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Only proceed if .arff
                if file.endswith('.arff'):
                    logging.info(f"Starting sensitivity sweeps for file: {file_path}")
                    try:
                        data = load_arff_dataset(file_path)
                        X_full , y_full  = split_data(data)
                        n_samples, n_features = X_full.shape

                        # =======================
                        # 1) Lambda Sweep
                        # =======================
                        for lam in lambda_values:
                            # Construct a unique name for the CSV
                            sweep_dataset_name = make_dataset_sweep_name(file_path, "lam", lam)
                            if sweep_dataset_name in processed_files:
                                logging.info(f"Skipping already-processed lam sweep: {sweep_dataset_name}")
                                continue

                            model_name = f"CH_elbowPCA_lam={lam}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=lam
                                )
                            }
                            results = evaluate_models(X_full, y_full, models, dataset_name=sweep_dataset_name)
                            for r in results:
                                r["SweepType"] = "lambda"
                                r["Lam"] = lam

                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)
                            processed_files.add(sweep_dataset_name)  # mark done

                        # =======================
                        # 2) Row-Count Sweep
                        # =======================
                        for rc in row_counts:
                            sweep_dataset_name = make_dataset_sweep_name(file_path, "rows", rc)
                            if sweep_dataset_name in processed_files:
                                logging.info(f"Skipping already-processed row_count sweep: {sweep_dataset_name}")
                                continue

                            actual_rc = min(rc, n_samples)
                            idx = np.random.choice(n_samples, size=actual_rc, replace=False)
                            X_subset = X_full[idx, :]
                            y_subset = y_full[idx]

                            model_name = f"CH_elbowPCA_rows={rc}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=default_lambda
                                )
                            }
                            results = evaluate_models(X_subset, y_subset, models, dataset_name=sweep_dataset_name)
                            for r in results:
                                r["SweepType"] = "row_count"
                                r["Lam"] = default_lambda
                                r["RowCount"] = actual_rc

                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)
                            processed_files.add(sweep_dataset_name)  # mark done

                        # =======================
                        # 3) Column Partition
                        # =======================
                        partitions = partition_features_by_fraction(n_features, fraction=column_fraction)
                        for i, chunk in enumerate(partitions):
                            sweep_dataset_name = make_dataset_sweep_name(file_path, "colSlice", i)
                            if sweep_dataset_name in processed_files:
                                logging.info(
                                    f"Skipping already-processed col_partition sweep: {sweep_dataset_name}")
                                continue

                            X_colsubset = X_full[:, chunk]
                            y_colsubset = y_full

                            model_name = f"CH_elbowPCA_colSlice={i}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=default_lambda
                                )
                            }
                            results = evaluate_models(X_colsubset, y_colsubset, models,
                                                      dataset_name=sweep_dataset_name)
                            for r in results:
                                r["SweepType"] = "col_partition"
                                r["Lam"] = default_lambda
                                r["PartitionIdx"] = i
                                r["ColCount"] = len(chunk)

                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)
                            processed_files.add(sweep_dataset_name)

                        # =======================
                        # 4) Row Partition
                        # =======================
                        row_partitions = partition_rows_by_fraction(n_samples, row_fraction)
                        for i, row_chunk in enumerate(row_partitions):
                            sweep_dataset_name = make_dataset_sweep_name(file_path, "rowSlice", i)
                            if sweep_dataset_name in processed_files:
                                logging.info(
                                    f"Skipping already-processed row_partition sweep: {sweep_dataset_name}")
                                continue

                            X_rowsubset = X_full[row_chunk, :]
                            y_rowsubset = y_full[row_chunk]

                            model_name = f"CH_elbowPCA_rowSlice={i}"
                            models = {
                                model_name: ConfigurableConvexHullAnomalyDetector(
                                    method="pca",
                                    stopping_criteria="elbow",
                                    n_components=2,
                                    lam=default_lambda
                                )
                            }
                            results = evaluate_models(X_rowsubset, y_rowsubset, models,
                                                      dataset_name=sweep_dataset_name)
                            for r in results:
                                r["SweepType"] = "row_partition"
                                r["Lam"] = default_lambda
                                r["PartitionIdx"] = i
                                r["RowCount"] = len(row_chunk)

                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)
                            processed_files.add(sweep_dataset_name)

                        logging.info(f"Finished all sweeps for file: {file_path}")

                    except Exception as e:
                        logging.error(f"Error processing file {file_path}: {e}")
                        continue

    return all_results
    #  -- If you haven't defined partition_features_by_fraction somewhere, here's a quick example:
def partition_features_by_fraction(n_features, fraction=0.25):
    """
    Partition 'n_features' consecutive indices into '1/fraction' slices.
    Each slice has size = floor(fraction * n_features),
    except the last slice which gets any leftover.
    """
    import math
    lumps = int(round(1.0 / fraction))
    chunk_size = math.floor(fraction * n_features)
    if chunk_size < 1:
        chunk_size = 1

    col_indices = list(range(n_features))
    partitions = []
    current_start = 0
    for i in range(lumps):
        if i < lumps - 1:
            end = current_start + chunk_size
            partitions.append(col_indices[current_start:end])
            current_start = end
        else:
            partitions.append(col_indices[current_start:])
    return partitions

import math

def partition_rows_by_fraction(n_rows, fraction=0.25):
    """
    Partition 'n_rows' consecutive indices into '1/fraction' slices.
    Each slice has size = floor(fraction * n_rows),
    except the last slice, which gets any leftover.

    Example:
      n_rows=9, fraction=0.25 => lumps=4
      chunk_size = floor(0.25*9)=2
      => partitions: [0,1], [2,3], [4,5], [6,7,8]
    """

    lumps = int(round(1.0 / fraction))
    chunk_size = math.floor(fraction * n_rows)
    if chunk_size < 1:
        chunk_size = 1  # ensure at least 1 row

    row_indices = list(range(n_rows))
    partitions = []

    current_start = 0
    for i in range(lumps):
        if i < lumps - 1:
            end = current_start + chunk_size
            partitions.append(row_indices[current_start:end])
            current_start = end
        else:
            # last slice => leftover
            partitions.append(row_indices[current_start:])

    return partitions

# Process datasets
def process_datasets(parent_folder, results_file_path):
    processed_files = get_processed_files(results_file_path)
    all_results = []

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.arff') and file_path not in processed_files:  # Compare full path

                    print(f"Processing file: {file_path}")
                    # Processing code...
                    try:
                        # Load dataset
                        data = load_arff_dataset(file_path)
                        X, _, y_true, _ = split_data(data)  # Ensure y_true is extracted

                        # Define models
                        models = {
                            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
                            "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
                            "Gaussian Mixture Models": GaussianMixture(n_components=2, covariance_type="full"),
                            "K-means": KMeans(n_clusters=2, random_state=42),
                            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
                            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
                            #"Spectral Clustering": SpectralClustering(n_clusters=2, random_state=42),
                            "Mean Shift": MeanShift(),

                            "Convex Hull Naive + PCA": ConfigurableConvexHullAnomalyDetector(method="pca",
                                                                                             stopping_criteria="naive",
                                                                                             n_components=2),
                            "Convex Hull Elbow + PCA": ConfigurableConvexHullAnomalyDetector(method="pca",
                                                                                             stopping_criteria="elbow",
                                                                                             n_components=2),
                            "Convex Hull Optimal + PCA": ConfigurableConvexHullAnomalyDetector(method="pca",
                                                                                               stopping_criteria="optimal",
                                                                                               n_components=2),
                            "Convex Hull Naive + t-SNE": ConfigurableConvexHullAnomalyDetector(method="tsne",
                                                                                               stopping_criteria="naive",
                                                                                               n_components=2),
                            "Convex Hull Elbow + t-SNE": ConfigurableConvexHullAnomalyDetector(method="tsne",
                                                                                               stopping_criteria="elbow",
                                                                                               n_components=2),
                            "Convex Hull Optimal + t-SNE": ConfigurableConvexHullAnomalyDetector(method="tsne",
                                                                                                 stopping_criteria="optimal",
                                                                                                 n_components=2),
                        }

                        # Evaluate models
                        results = evaluate_models(X, y_true, models, file_path)

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
        logging.info(f"Aggregated results saved to: {output_file_path}")


def test_run():
    datasets_folder = "/home/convexhull1/literature"
    results_file_path = "/home/convexhull1/results_per_dataset.csv"
    avg_results_file_path = "/home/convexhull1/average_results.csv"

    # Select one folder for testing
    test_folder = "WDBC"  # Change to any folder name you want to test
    folder_path = os.path.join(datasets_folder, test_folder)

    if os.path.isdir(folder_path):
        print(f"Testing on folder: {test_folder}")
        count = 0

        for file in os.listdir(folder_path):
            if count >= 2:  # Process only two files
                break

            if file.endswith('.arff'):
                file_path = os.path.join(folder_path, file)
                print(f"Processing file: {file_path}")
                try:
                    # Call your processing logic
                    data = load_arff_dataset(file_path)
                    X, _, y_true, _ = split_data(data)

                    # Define models for testing
                    models = {
                        "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
                        "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
                    }

                    # Evaluate models
                    results = evaluate_models(X, y_true, models, file_path)

                    # Save intermediate results
                    save_intermediate_results(results, results_file_path)
                    count += 1
                    print(f"Processed and saved results for: {file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Aggregate results for the two files
    if os.path.exists(results_file_path):
        aggregate_results(results_file_path, avg_results_file_path)
        print(f"Aggregated results saved to: {avg_results_file_path}")

    return


def test_lam_effects():
    """
    Demonstrates how different lambda values can lead to different 'Sp' sets
    and different final hull volumes, using a small 2D dataset.
    """

    # 1) A small 2D dataset of 10 points
    X = np.array([
        [0,0], [1,0], [2,0], [3,0],
        [0,1], [1.5,0.5], [2,1],
        [1,2], [2,2], [3,2]
    ], dtype=float)
    y = np.zeros(X.shape[0], dtype=int)  # Not really used unless "optimal" stopping

    # We'll import your refactored or original class
    # (Below, I'm assuming it's named 'ConvexHullAnomalyDetectorLambda')
    # from your_module import ConvexHullAnomalyDetectorLambda  # example

    # Let's define some lambda values to test:
    lam_values = [0.0, 0.2, 0.5, 1.0]

    for lam in lam_values:
        print(f"\n========================")
        print(f" Testing lambda = {lam}")
        print(f"========================")

        # 2) Instantiate the detector
        model = ConfigurableConvexHullAnomalyDetector(
            method="pca",               # we do PCA(2), though it's already 2D
            stopping_criteria="naive",  # so we see the effect of lam in naive removal
            n_components=2,
            lam=lam,
            max_iter=50                 # let it try up to 50 removals
        )

        # 3) Fit
        model.fit(X, y)  # y not strictly needed for "naive"

        # 4) See how many points remain
        final_sp = model.Sp  # The set of "surviving" points in reduced space
        print(f"  Final # of points in Sp: {len(final_sp)}")

        # 5) Let's figure out which points got removed
        #    We know which survived because model.Sp are the 2D reduced points
        #    But if we're using PCA(2) on a 2D dataset => effectively the same shape
        #    We can compare original X_reduced_ vs final Sp
        survived_points = set(final_sp)         # set of (x, y) in reduced space
        original_points = set(map(tuple, model.X_reduced_))
        removed_points  = original_points - survived_points

        # 6) Print removed points
        if len(removed_points) == 0:
            print("  No points were removed.")
        else:
            print(f"  Removed {len(removed_points)} points:")
            for pt in removed_points:
                print(f"    {pt}")

        # 7) Print final hull volume
        #    If hull_equations_ is None, that means we couldn't build a hull
        if model.hull_equations_ is None:
            print("  Final hull_equations_ is None => no hull built.")
        else:
            # Let's quickly compute the final volume for clarity
            arr_sp = np.array(list(model.Sp))
            if arr_sp.shape[0] >= 3:  # at least 3 points for a 2D hull
                from scipy.spatial import ConvexHull
                final_hull = ConvexHull(arr_sp)
                print(f"  Final hull volume: {final_hull.volume:.4f}")
            else:
                print("  Final hull: Not enough points to measure volume > 0.")

if __name__ == "__main__":

    #test_run()
    #test_lam_effects()
    #print()

    try :

        # datasets_folder  = f"{path}/datasets"  # Update with your VM path
        # results_file_path = f"{path}/results_per_dataset_sensetivity.csv"
        # avg_results_file_path = f"{path}/average_results_sensetivity.csv"

        datasets_folder =f"{path}/literature"  # Update with your VM path
        results_file_path = f"{path}/results_per_dataset_sensativity_nomaxiter.csv"
        avg_results_file_path = f"{path}/average_results_sensativity_nomaxiter.csv"

        # Process datasets and save intermediate results
        print(f"Processing datasets in: {datasets_folder}")
        logging.info(f"Processing datasets in: {datasets_folder}")
        sensetivity_eval_2(datasets_folder, results_file_path)

        print("finished running models")
        logging.info("finished running models")

        print("aggregating files")
        logging.info("aggregating files")
        # Aggregate and save final results
        aggregate_results(results_file_path, avg_results_file_path)

        print("Processing complete! Check the results in your specified paths.")
        logging.info("Processing complete! Check the results in your specified paths.")
    except Exception as e:
        print(f"Error running script because: {e}")
        logging.error(f"Error running script because: {e}")

     # Optional: Upload results to Google Cloud Storage
    # try:
    #     from google.cloud import storage
    #
    #     def upload_to_bucket(blob_name, file_path, bucket_name):
    #         client = storage.Client()
    #         bucket = client.bucket(bucket_name)
    #         blob = bucket.blob(blob_name)
    #         blob.upload_from_filename(file_path)
    #         print(f"File {file_path} uploaded to {bucket_name}/{blob_name}")
    #
    #     # Upload files to Google Cloud Storage bucket
    #     bucket_name = "your-bucket-name"  # Replace with your bucket name
    #     upload_to_bucket("results_per_dataset.csv", results_file_path, bucket_name)
    #     upload_to_bucket("average_results.csv", avg_results_file_path, bucket_name)
    # except Exception as e:
    #     print(f"Error uploading to Google Cloud Storage: {e}")
    #     logging.error(f"Error uploading to Google Cloud Storage: {e}")

    # Stop the VM instance programmatically
    print("Stopping the instance to save costs...")
    os.system("gcloud compute instances stop <your-instance-name> --zone=<your-instance-zone>")

