import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import f1_score
from ConvexHullAnomalyDetectorClass import ConfigurableConvexHullAnomalyDetector

# Configure logging
path = '/home/someliejonson'
logging.basicConfig(
    filename=f"{path}/process_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Sensitivity analysis script started")


# Helper functions from your code
def load_arff_dataset(filepath):
    from scipy.io import arff
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    return df


def split_data(data):
    from sklearn.model_selection import train_test_split
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


def evaluate_models(X, y_true, models, dataset_name):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
    results = []
    for model_name, model in models.items():
        try:
            logging.info(f"Running {model_name} on {dataset_name}")
            print(f"Running {model_name} on {dataset_name}")
            if hasattr(model, "fit_predict"):
                y_pred = model.fit_predict(X)
                y_pred = np.where(y_pred == 1, 0, 1)
            else:
                model.fit(X)
                y_pred = model.predict(X)
                y_pred = np.where(y_pred == 1, 0, 1)
            if y_true is not None and len(y_true) > 0:
                accuracy = accuracy_score(y_true, y_pred)
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=1)
                recall = recall_score(y_true, y_pred, zero_division=1)
                f1 = f1_score(y_true, y_pred, zero_division=1)
            else:
                accuracy = balanced_accuracy = precision = recall = f1 = "N/A"
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
            logging.info(f"Completed {model_name} on {dataset_name}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            logging.error(f"Error with {model_name} on {dataset_name}: {e}")
            continue
    return results


def save_intermediate_results(results, file_path):
    results_df = pd.DataFrame(results)
    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
        print('File exists – saved intermediate results to file')
        logging.info(f"Appended results to {file_path}")
    else:
        results_df.to_csv(file_path, index=False)
        print('Saved intermediate results to file')
        logging.info(f"Saved results to new file {file_path}")


def get_processed_files(results_file_path):
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
        processed = results_df["Dataset"].unique()
        logging.info(f"Found {len(processed)} processed files.")
        return processed
    return []


# Original sensitivity evaluation function (unchanged except for added logging)
def sensetivity_eval(parent_folder, results_file_path):
    processed_files = get_processed_files(results_file_path)
    all_results = []
    logging.info("Starting sensitivity evaluation sweeps.")

    # Sweep parameters as originally defined
    lambda_values = np.arange(0.0, 1.1, 0.1)  # lambda from 0 to 1 by 0.1
    row_counts = [100, 500, 1000]
    column_fraction = 0.25
    row_fraction = 0.25
    default_lambda = 0.5

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            logging.info(f"Processing folder: {folder_name}")
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.arff') and file_path not in processed_files:
                    logging.info(f"Processing file: {file_path}")
                    print(f"Processing file: {file_path}")
                    try:
                        data = load_arff_dataset(file_path)
                        X_full, _, y_full, _ = split_data(data)
                        n_samples, n_features = X_full.shape
                        logging.info(f"{file_path} has {n_samples} samples and {n_features} features.")

                        # Sweep #1: Lambda values
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
                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        # Sweep #2: Row-count
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
                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        # Sweep #3: Column-partition
                        partitions = partition_features_by_fraction(n_features, fraction=column_fraction)
                        for i, chunk in enumerate(partitions):
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
                                                      dataset_name=f"{file_path}-colSlice={i}")
                            for r in results:
                                r["SweepType"] = "col_partition"
                                r["Lam"] = default_lambda
                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        # Sweep #4: Row-partition
                        row_partitions = partition_rows_by_fraction(n_samples, row_fraction)
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
                                r["Lam"] = default_lambda
                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        logging.info(f"Finished sweeps for: {file_path}")
                        print(f"Finished sweeps for: {file_path}")

                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        logging.error(f"Error processing file {file_path}: {e}")
                        continue
    return all_results


# Helper: partition features by fraction
def partition_features_by_fraction(n_features, fraction=0.25):
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


def partition_rows_by_fraction(n_rows, fraction=0.25):
    import math
    lumps = int(round(1.0 / fraction))
    chunk_size = math.floor(fraction * n_rows)
    if chunk_size < 1:
        chunk_size = 1
    row_indices = list(range(n_rows))
    partitions = []
    current_start = 0
    for i in range(lumps):
        if i < lumps - 1:
            end = current_start + chunk_size
            partitions.append(row_indices[current_start:end])
            current_start = end
        else:
            partitions.append(row_indices[current_start:])
    return partitions


# Test function: process only a small sample from each dataset
def test_sensitivity_eval_small(parent_folder, results_file_path, n_rows=10):
    """
    A reduced version of sensetivity_eval that:
      1) Only grabs 'n_rows' rows from each dataset.
      2) Uses a very small lambda sweep.
      3) Skips or simplifies the row/column partition sweeps.
    This helps quickly verify whether code is working on small samples.
    """

    # Very small lam sweep
    lambda_values = [0.0, 1.0]  # Instead of np.arange(0.0, 10, 1)
    processed_files = get_processed_files(results_file_path)
    all_results = []

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Only process .arff not in processed_files
                # (Adjust your logic here if you append suffixes in `dataset_name`)
                if file.endswith('.arff') and file_path not in processed_files:
                    print(f"[TEST] Processing file (small-sample): {file_path}")
                    try:
                        data = load_arff_dataset(file_path)

                        # Grab only n_rows randomly (or fewer if dataset is smaller)
                        total_rows = data.shape[0]
                        sample_size = min(n_rows, total_rows)
                        data_sample = data.sample(sample_size, random_state=42)

                        # Quick train/test split
                        X_sample, X_test, y_sample, y_test = split_data(data_sample)

                        # Keep it very simple: only do a few lambda sweeps
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

                            # Evaluate
                            results = evaluate_models(X_sample, y_sample, models,
                                                     dataset_name=f"{file_path}-TEST-lam={lam}")
                            # Tag the results as test runs
                            for r in results:
                                r["SweepType"] = "test_lambda"
                                r["Lam"] = lam

                            # Save partial results
                            save_intermediate_results(results, results_file_path)
                            all_results.extend(results)

                        print(f"[TEST] Finished small-sample sweeps for: {file_path}")

                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        logging.error(f"Error in test_sensitivity_eval_small for file {file_path}: {e}")
                        continue

    return all_results


if __name__ == "__main__":
    try:
        parent_folder = f"{path}/literature"
        results_file_path = f"{path}/results_smalltest.csv"

        # Use the new small test function
        print("Running small-sample sensitivity test...")
        test_sensitivity_eval_small(parent_folder, results_file_path, n_rows=10)

        # Then optionally aggregate
        aggregate_results(results_file_path, f"{path}/average_results_smalltest.csv")

        print("Small-sample test complete!")
    except Exception as e:
        print(f"Error: {e}")