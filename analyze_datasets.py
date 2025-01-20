import os
import pandas as pd
import numpy as np

# Function to get processed files
def get_processed_files(results_file_path):
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
        return results_df['Dataset'].unique()
    return []

# Function to save intermediate results
def save_intermediate_results(results, file_path):
    results_df = pd.DataFrame(results)
    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, index=False)

# Aggregation per dataset
def aggregate_per_dataset(results_file_path, output_dataset_aggregation_path):
    if not os.path.exists(results_file_path):
        print(f"Results file {results_file_path} does not exist.")
        return

    # Load the results file
    results_df = pd.read_csv(results_file_path)

    # Extract dataset (folder) information from the dataset paths
    results_df['DatasetFolder'] = results_df['Dataset'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    # Numeric columns to aggregate
    numeric_columns = ['Accuracy', 'BalancedAccuracy', 'Precision', 'Recall', 'F1 Score']

    # Group by dataset folder and algorithm, then calculate the mean
    dataset_aggregated = results_df.groupby(['DatasetFolder', 'Algorithm'])[numeric_columns].mean().reset_index()

    # Save the aggregated results to a new file
    dataset_aggregated.to_csv(output_dataset_aggregation_path, index=False)
    print(f"Dataset aggregated results saved to {output_dataset_aggregation_path}")

# Main processing function
def process_datasets(parent_folder, results_file_path):
    processed_files = get_processed_files(results_file_path)
    all_results = []

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            for file in os.listdir(folder_path):
                if file.endswith('.arff') and file not in processed_files:
                    file_path = os.path.join(folder_path, file)
                    print(f"Processing file: {file_path}")
                    # Simulate processing (replace with your actual processing logic)
                    try:
                        # Simulating metrics for various algorithms
                        results = [
                            {
                                "Algorithm": algorithm,
                                "Dataset": file_path,
                                "Accuracy": np.random.random(),
                                "BalancedAccuracy": np.random.random(),
                                "Precision": np.random.random(),
                                "Recall": np.random.random(),
                                "F1 Score": np.random.random(),
                            }
                            for algorithm in [
                                "Isolation Forest", "One-Class SVM", "K-Means", "DBSCAN", "Convex Hull"
                            ]
                        ]
                        save_intermediate_results(results, results_file_path)
                        all_results.extend(results)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue

    return all_results


import pandas as pd

def find_best_algorithm_per_dataset(input_csv_path, output_csv_path, metric="Accuracy"):
    """
    Finds the best algorithm for each dataset based on the specified metric.

    Parameters:
    - input_csv_path: Path to the CSV file containing the results.
    - output_csv_path: Path to save the results with the best algorithm per dataset.
    - metric: The metric to use for determining the best algorithm (default: "Accuracy").
    """
    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Group by dataset and find the algorithm with the highest metric score
    df['DatasetForAlgorithm'] = df['Dataset'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    best_algorithms = df.loc[df.groupby("DatasetForAlgorithm")[metric].idxmax()]

    # Save the result to a CSV
    best_algorithms.to_csv(output_csv_path, index=False)

    print(f"Best algorithms per dataset saved to {output_csv_path}")


def find_algorithm_per_dataset_by_rank(
    input_csv_path,
    output_csv_path,
    metric="Accuracy",
    rank_position=2
):
    """
    Finds the algorithm(s) that are in 'rank_position' place for each dataset,
    based on the specified metric (descending order).

    For example, rank_position=1 => best algorithm,
                 rank_position=2 => second-best algorithm, etc.

    Parameters:
    -----------
    input_csv_path : str
        Path to the CSV file containing the results (with 'Dataset' column, 'Algorithm', etc.).
    output_csv_path : str
        Path to save the resulting CSV.
    metric : str
        Which metric to use for ranking (e.g., "Accuracy").
    rank_position : int
        Which rank to extract (e.g., 2 => second place).
    """
    # 1) Load the CSV
    df = pd.read_csv(input_csv_path)

    # 2) Derive a 'DatasetForAlgorithm' or similar column if needed
    #    (Assumes your 'Dataset' column has a path that includes the dataset folder)
    df['DatasetForAlgorithm'] = df['Dataset'].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )

    # 3) Rank each dataset's rows by the metric in descending order
    #    "dense" ranking means if there's a tie, they share the same rank
    df["RankInDataset"] = df.groupby("DatasetForAlgorithm")[metric]\
                            .rank(method="dense", ascending=False)

    # 4) Filter for only the desired rank_position
    #    e.g. rank_position=2 => second place
    subset = df[df["RankInDataset"] == rank_position]

    # 5) Save to CSV
    subset.to_csv(output_csv_path, index=False)
    print(f"Algorithms in {rank_position}-place per dataset saved to {output_csv_path}")


def second_best_algorithm_per_dataset(
    input_csv_path,
    output_csv_path,
    metric="Accuracy"
):
    """
    Finds the second-best algorithm (by distinct score) for each dataset
    based on the specified metric, ensuring only one row per dataset is returned.

    How it works:
      1) For each dataset group, sort descending by 'metric'.
      2) Drop duplicates on 'metric' so all ties for 1st place are removed from the top row.
      3) Pick the next row (row index=1) as second-best if it exists.

    Parameters:
    -----------
    input_csv_path : str
        Path to the CSV file containing all results (with columns including 'Dataset', 'Algorithm', etc.).
    output_csv_path : str
        Where to save the CSV of second-best algorithms per dataset.
    metric : str, optional
        The metric to rank on (default = "Accuracy").
    """

    # 1) Load the CSV
    df = pd.read_csv(input_csv_path)

    # 2) Extract the dataset name from the 'Dataset' path
    #    e.g. if Dataset="/path/to/KDDCup99/something.arff",
    #    then DatasetForAlgorithm="KDDCup99"
    df["DatasetForAlgorithm"] = df["Dataset"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )

    # 3) For each dataset, sort by the chosen metric descending,
    #    then drop duplicates on that metric, so ties for 1st place
    #    all collapse into a single row. Then take row index=1 if available.
    second_best_rows = []
    for dataset_name, group in df.groupby("DatasetForAlgorithm"):
        # Sort descending on the metric
        group_sorted = group.sort_values(by=metric, ascending=False)

        # Drop duplicates on 'metric' so that if multiple algorithms share top score,
        # they are considered the same "first place" score.
        group_no_dups = group_sorted.drop_duplicates(subset=[metric], keep='first')

        # Now if we have >= 2 distinct scores, row index=1 is "second best"
        if len(group_no_dups) >= 2:
            row_second_best = group_no_dups.iloc[1]
            second_best_rows.append(row_second_best)
        else:
            # If there's only 1 distinct score in this dataset,
            # there's no second-best distinct score.
            # (You can decide to skip it or store something else.)
            pass

    # 4) Combine all second-best rows into a DataFrame
    df_second_best = pd.DataFrame(second_best_rows)

    # 5) Save to CSV
    df_second_best.to_csv(output_csv_path, index=False)
    print(f"Second-best algorithms (by distinct {metric}) saved to {output_csv_path}")

def aggregate_results(results_file_path, output_file_path):
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
        numeric_columns = results_df.select_dtypes(include='number').columns
        avg_results = results_df.groupby("Algorithm")[numeric_columns].mean().reset_index()
        avg_results.to_csv(output_file_path, index=False)
        print(f"Aggregated results saved to: {output_file_path}")

    ['KDDCup99', 'WPBC', 'Waveform']



def aggregate_results_for_subset(
    results_file_path,
    output_file_path,
    dataset_subset=["KDDCup99", "WPBC", "Waveform"]
):
    """
    Aggregates results by 'Algorithm' but only for the datasets in 'dataset_subset'.
    Expects a 'Dataset' column in the CSV that includes the dataset path/folder.

    Parameters
    ----------
    results_file_path : str
        Path to the CSV file with results (including columns like 'Dataset', 'Algorithm', etc.)
    output_file_path : str
        Where to save the aggregated CSV.
    dataset_subset : list of str
        Which dataset names to include when aggregating, e.g. ["KDDCup99", "WPBC", "Waveform"].
    """
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)

        # 1) Extract the dataset folder name from the 'Dataset' path:
        #    e.g. if 'Dataset' is something like ".../KDDCup99/somefile.arff",
        #    then we take "KDDCup99" as the dataset name:
        results_df["DatasetName"] = results_df["Dataset"].apply(
            lambda x: os.path.basename(os.path.dirname(x))
        )

        # 2) Filter to only include the specified subset of dataset names
        filtered_df = results_df[results_df["DatasetName"].isin(dataset_subset)]

        # 3) Select numeric columns and group by "Algorithm"
        numeric_columns = filtered_df.select_dtypes(include='number').columns
        if len(filtered_df) == 0:
            print("No rows match the chosen dataset subset. Nothing to aggregate.")
            return

        avg_results = (
            filtered_df
            .groupby("Algorithm")[numeric_columns]
            .mean()
            .reset_index()
        )

        # 4) Save to output CSV
        avg_results.to_csv(output_file_path, index=False)
        print(f"Aggregated results (only for {dataset_subset}) saved to: {output_file_path}")
    else:
        print(f"File {results_file_path} does not exist. Nothing aggregated.")

# Entry point
if __name__ == "__main__":
    # #File paths
    # datasets_folder = "./literature"
    # results_file_path = "./results_per_datasetfinal.csv"
    # output_dataset_aggregation_path = "./dataset_aggregated_results.csv"
    #
    # # Process datasets
    # print(f"Processing datasets in: {datasets_folder}")
    # #process_datasets(datasets_folder, results_file_path)
    #
    # # Aggregate results per dataset
    # #aggregate_per_dataset(results_file_path, output_dataset_aggregation_path)
    #
    # # Example usage:
    # input_csv_path = "results_per_datasetfinal.csv"  # Replace with your input file path
    # output_csv_path = "best_algorithms_per_dataset_with_naives.csv"  # Replace with your desired output file path
    # metric = "F1 Score"  # You can choose Accuracy, F1 Score, or any other metric
    #
    # find_best_algorithm_per_dataset(input_csv_path, output_csv_path, metric)



    path = "C:/Users/Asael/PycharmProjects/convexhull"
    datasets_folder  = f"{path}/datasets"  # Update with your VM path
    results_file_path = f"{path}/results_per_datasetfinal.csv"
    avg_results_file_path = f"{path}/average_results_subset.csv"



    aggregate_results_for_subset(results_file_path, avg_results_file_path)
    print('')