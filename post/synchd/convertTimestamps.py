import pandas as pd

# Load the CSV file
file_path = 'post\\data\\PegT_S221_T1_2024-07-19\\PDS.csv'
df = pd.read_csv(file_path)

# Assuming the column with epoch timestamps is named 'timestamp'
# Convert timestamps to milliseconds
df['timestamp'] = df['timestamp'] // 1000000

# Save the modified DataFrame back to a CSV file
output_file_path = 'post\\data\\PegT_S221_T1_2024-07-19\\modifiedPDS.csv'
df.to_csv(output_file_path, index=False)

print(f"Timestamps have been converted to milliseconds and saved to {output_file_path}")
