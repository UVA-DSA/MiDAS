import pandas as pd
from pathlib import Path

def combine_csvs(annotation_csv, data_csv):
    # Combine the two csv files into one based on the specified columns
    annotation_csv_id_column = "frame_number"
    data_csv_id_column = "obs_frame_idx"
    combined = pd.merge(
        annotation_csv,
        data_csv,
        left_on=[annotation_csv_id_column],
        right_on=[data_csv_id_column],
        how="left"
    )

    # drop the first row only
    combined = combined.drop(index=0)
    return combined


if __name__ == "__main__":
    trial_name = "bt5"

    base_dir = Path(r"/standard/UVA-DSA/MIDAS/Organized/final_data") / trial_name / "synched_data"

    annotation_csv_path = base_dir / f"peg_{trial_name}_final_annotation_frames.csv"
    data_csv_path = base_dir / f"synched_all_{trial_name}.csv"
    output_csv_path = base_dir / f"final_annotation_{trial_name}.csv"

    # Load CSVs
    annotation_csv = pd.read_csv(annotation_csv_path)
    data_csv = pd.read_csv(data_csv_path)

    print("columns in annotation csv:", annotation_csv.columns)
    print("columns in data csv:", data_csv.columns)

    # Merge
    combined_csv = combine_csvs(annotation_csv, data_csv)

    # Save without scientific notation
    combined_csv.to_csv(
        output_csv_path,
        index=False,
        float_format="%.0f"   # force plain integer/decimal formatting
    )

    print(f"✅ Combined CSV saved to: {output_csv_path}")
