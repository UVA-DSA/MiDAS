import pandas as pd
import glob
import os
def comparePDSExpectedExperimental(path, threshold = ""):

    # Load CSV files
    expected_df = pd.read_csv(f"{path}/obs/pedal_detection.csv")
    experimental_df = pd.read_csv(f"{path}/Synched Data/PDS_sync.csv")

    # Mapping from expected column to experimental column
    pedal_mapping = {
        'Upper Left': 'Pedal 1 Pressed',
        'Upper Right': 'Pedal 2 Pressed',
        'Lower Left': 'Pedal 3 Pressed',
        'Lower Right': 'Pedal 4 Pressed'
    }

    results = []

    # Totals
    total_fp = 0
    total_fn = 0
    total_tp = 0
    total_tn = 0

    # Compare each pedal
    for expected_col, experimental_col in pedal_mapping.items():
        expected_vals = expected_df[expected_col]
        experimental_vals = experimental_df[experimental_col]

        false_positives = ((expected_vals == 0) & (experimental_vals == 1)).sum()
        false_negatives = ((expected_vals == 1) & (experimental_vals == 0)).sum()
        true_positives  = ((expected_vals == 1) & (experimental_vals == 1)).sum()
        true_negatives  = ((expected_vals == 0) & (experimental_vals == 0)).sum()

        # Add to totals
        total_fp += false_positives
        total_fn += false_negatives
        total_tp += true_positives
        total_tn += true_negatives

        results.append({
            'Pedal': expected_col,
            'False Positives': false_positives,
            'False Negatives': false_negatives,
            'True Positives': true_positives,
            'True Negatives': true_negatives
        })

    # Add total row
    results.append({
        'Pedal': 'Total',
        'False Positives': total_fp,
        'False Negatives': total_fn,
        'True Positives': total_tp,
        'True Negatives': total_tn
    })


    # Create results DataFrame
    results_df = pd.DataFrame(results)
    try:
        os.mkdir(f"{path}/PDSError/")
        print(f"Directory created successfully.")
    except FileExistsError:
        print(f"Directory already exists.")
    # Save to CSV
    results_df.to_csv(f"{path}/PDSError/PDS_error{threshold}.csv", index=False)

    print(f"Comparison complete. Results with total saved to '{path}/PDSError/PDS_error{threshold}'.csv")

#RUN FOR CSVs
if __name__ == '__main__':
    matchingPath = glob.glob("./data/*")
    for path in matchingPath:
        print(path)
        try:
            comparePDSExpectedExperimental(path)
        except:
            print(path)