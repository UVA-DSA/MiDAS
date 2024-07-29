import pandas as pd

# Load the CSV file
file_path = 'post\data\PegT_S221_T1_2024-07-19\obs\OBStimestamp_PegT_S221_T1_2024-07-19.csv'
df = pd.read_csv(file_path)

# Assuming the column with epoch timestamps is named 'timestamp'
# Convert timestamps to milliseconds
#df["Epoch"] = df['Epoch'] * 1000000

for i in range(0,15871):
    if(i % 3 == 0):
        try:
            df.iloc[i,0] = df.iloc[i,0] + 333333
        except Exception:
            print("ouch")
            pass
    elif(i%3 == 1):
        try:
            df.iloc[i,0] = df.iloc[i,0] + 666666
        except Exception:
            print("ouch")

            pass
    else:
        try:
            df.iloc[i,0] = df.iloc[i,0] + 0
        except Exception:
            print("ouch")

            pass

# Save the modified DataFrame back to a CSV file
output_file_path = 'post\data\PegT_S221_T1_2024-07-19\obs\OBStimestamp_PegT_S221_T1_2024-07-19.csv'
df.to_csv(output_file_path, index=False)

print(f"Timestamps have been converted to nano and saved to {output_file_path}")
