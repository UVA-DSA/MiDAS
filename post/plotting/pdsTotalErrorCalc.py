import os
import pandas as pd

# Function to process individual trial CSV files and return stats per pedal
def process_trial_csv_by_pedal(file_path):
    df = pd.read_csv(file_path)

    pedal_stats = {}

    for index, row in df.iterrows():
        pedal = row['Pedal']
        fp = row['False Positives']
        fn = row['False Negatives']
        tp = row['True Positives']
        tn = row['True Negatives']
        total = fp + fn + tp + tn

        if total == 0:
            continue  # Skip if no data to avoid divide by zero

        err = (fp + fn) / total * 100
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fp_pct = fp / total * 100
        fn_pct = fn / total * 100

        pedal_stats[pedal] = {
            'Error %': err,
            'Precision': precision,
            'Recall': recall,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp,
            'True Negatives': tn,
            'Total Samples': total,
            'FP %': fp_pct,
            'FN %': fn_pct
        }

    return pedal_stats


def countErrorsByPedal(path, t):
    all_stats = {}

    # Loop through each folder (trial)
    for trial_folder in os.listdir(path):
        trial_folder_path = os.path.join(path, f"{trial_folder}/PDSError/")

        if os.path.isdir(trial_folder_path):
            # Loop through CSV files in the trial folder
            for file_name in os.listdir(trial_folder_path):
                if file_name.endswith(f"{t}.csv"):
                    file_path = os.path.join(trial_folder_path, file_name)
                    print(f"Processing: {file_path}")

                    pedal_stats = process_trial_csv_by_pedal(file_path)

                    for pedal, stats in pedal_stats.items():
                        if pedal not in all_stats:
                            all_stats[pedal] = {
                                'False Positives': 0,
                                'False Negatives': 0,
                                'True Positives': 0,
                                'True Negatives': 0,
                                'Total Samples': 0
                            }

                        all_stats[pedal]['False Positives'] += stats['False Positives']
                        all_stats[pedal]['False Negatives'] += stats['False Negatives']
                        all_stats[pedal]['True Positives'] += stats['True Positives']
                        all_stats[pedal]['True Negatives'] += stats['True Negatives']
                        all_stats[pedal]['Total Samples'] += stats['Total Samples']

    # Prepare final results per pedal
    results = []
    for pedal, stats in all_stats.items():
        fp = stats['False Positives']
        fn = stats['False Negatives']
        tp = stats['True Positives']
        tn = stats['True Negatives']
        total = stats['Total Samples']

        err = (fp + fn) / total * 100 if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fp_pct = fp / total * 100 if total > 0 else 0
        fn_pct = fn / total * 100 if total > 0 else 0

        results.append({
            "Threshold": t,
            "Pedal": pedal,
            "Error %": err,
            "Precision": precision,
            "Recall": recall,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp,
            "True Negatives": tn,
            "Total Samples": total,
            "FP %": fp_pct,
            "FN %": fn_pct
        })

    # Create a DataFrame and append/save to CSV
    results_df = pd.DataFrame(results)
    csv_file_path = './data/Stats on PDS Error.csv'

    if os.path.exists(csv_file_path):
        results_df.to_csv(csv_file_path, mode='a', header=False, index=False)
        print(f"\nResults appended to '{csv_file_path}'.")
    else:
        results_df.to_csv(csv_file_path, mode='w', header=True, index=False)
        print(f"\nNew file created and results saved to '{csv_file_path}'.")


if __name__ == "__main__":
    path = "./data/"
    thresholds = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
    for t in thresholds:
        countErrorsByPedal(path, t)
