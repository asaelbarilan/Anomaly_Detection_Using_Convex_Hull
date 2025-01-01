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
from ConvexHullAnomalyDetectorClass import ParallelCHoutsideConvexHullAnomalyDetector
import logging

# For convex hull anomaly detection (replace with your custom class import if needed)

#path="C:/Users/Asael/PycharmProjects/convexhull"
path='/home/convexhull1'
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
    return train_test_split(X, y, test_size=0.2, random_state=42)


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
                            "Convex Hull": ParallelCHoutsideConvexHullAnomalyDetector(),
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
if __name__ == "__main__":

    #test_run()



    try :

        # datasets_folder  = f"{path}/datasets"  # Update with your VM path
        # results_file_path = f"{path}/results_per_dataset.csv"
        # avg_results_file_path = f"{path}/average_results.csv"

        datasets_folder = datasets_folder = "/home/convexhull1/literature"  # Update with your VM path
        results_file_path = "/home/convexhull1/results_per_dataset.csv"
        avg_results_file_path = "/home/convexhull1/average_results.csv"

        # Process datasets and save intermediate results
        print(f"Processing datasets in: {datasets_folder}")
        logging.info(f"Processing datasets in: {datasets_folder}")
        process_datasets(datasets_folder, results_file_path)

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

