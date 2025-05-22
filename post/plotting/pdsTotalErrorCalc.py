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
            continue

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
    trial_results_by_trial = {}

    for trial_folder in os.listdir(path):
        trial_folder_path = os.path.join(path, trial_folder)
        pds_error_path = os.path.join(trial_folder_path, "PDSError")

        if os.path.isdir(pds_error_path):
            trial_stats = {}

            for file_name in os.listdir(pds_error_path):
                if file_name.endswith(f"{t}.csv"):
                    file_path = os.path.join(pds_error_path, file_name)
                    print(f"Processing: {file_path}")

                    pedal_stats = process_trial_csv_by_pedal(file_path)

                    for pedal, stats in pedal_stats.items():
                        # Collect global stats
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

                        # Collect per-trial stats
                        if pedal not in trial_stats:
                            trial_stats[pedal] = {
                                'False Positives': 0,
                                'False Negatives': 0,
                                'True Positives': 0,
                                'True Negatives': 0,
                                'Total Samples': 0
                            }

                        trial_stats[pedal]['False Positives'] += stats['False Positives']
                        trial_stats[pedal]['False Negatives'] += stats['False Negatives']
                        trial_stats[pedal]['True Positives'] += stats['True Positives']
                        trial_stats[pedal]['True Negatives'] += stats['True Negatives']
                        trial_stats[pedal]['Total Samples'] += stats['Total Samples']

            # Add per-trial results for this threshold
            trial_rows = []
            for pedal, stats in trial_stats.items():
                fp, fn, tp, tn, total = stats.values()
                err = (fp + fn) / total * 100 if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                fp_pct = fp / total * 100 if total > 0 else 0
                fn_pct = fn / total * 100 if total > 0 else 0

                trial_rows.append({
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

            trial_df = pd.DataFrame(trial_rows)
            output_file = os.path.join(trial_folder_path, f"Stats_on_PDS_Error_by_threshold.csv")
            if os.path.exists(output_file):
                trial_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                trial_df.to_csv(output_file, mode='w', header=True, index=False)

    # Save global combined results
    combined_results = []
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

        combined_results.append({
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

    global_results_df = pd.DataFrame(combined_results)
    global_csv_path = f'{path}/Stats on PDS Error.csv'
    if os.path.exists(global_csv_path):
        global_results_df.to_csv(global_csv_path, mode='a', header=False, index=False)
    else:
        global_results_df.to_csv(global_csv_path, mode='w', header=True, index=False)

def extract_best_thresholds_by_trial(path, output_csv):
    all_rows = []

    for trial_folder in os.listdir(path):
        trial_folder_path = os.path.join(path, trial_folder)
        stats_file = os.path.join(trial_folder_path, "Stats_on_PDS_Error_by_threshold.csv")

        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file)

            for pedal in df["Pedal"].unique():
                pedal_df = df[df["Pedal"] == pedal]

                if not pedal_df.empty:
                    best_precision_row = pedal_df.loc[pedal_df["Precision"].idxmax()].copy()
                    best_precision_row["Metric"] = "Best Precision"

                    best_recall_row = pedal_df.loc[pedal_df["Recall"].idxmax()].copy()
                    best_recall_row["Metric"] = "Best Recall"

                    best_precision_row["Trial"] = trial_folder
                    best_recall_row["Trial"] = trial_folder

                    all_rows.append(best_precision_row)
                    all_rows.append(best_recall_row)

    if all_rows:
        final_df = pd.DataFrame(all_rows)
        output_path = os.path.join(path, "Per Trial Best Thresholds.csv")
        final_df.to_csv(output_path, index=False)
        print(f"\nCreated summary: {output_path}")
    else:
        print("\nNo data found to generate best threshold summary.")

#RUN FOR PEDAL DATA ON ALL TRIALS IN 'path' TO FIND BEST THRESHOLDS
if __name__ == "__main__":
    path = "./data/"
    thresholds = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
    for t in thresholds:
        countErrorsByPedal(path, t)

    extract_best_thresholds_by_trial(path, "Per Trial Best Thresholds.csv")