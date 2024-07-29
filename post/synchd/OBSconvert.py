import pandas as pd

def OBSConvert(path):
    # Load the CSV file
    file_path = f'{path}\obs\OBStimestamp_PegT_S221_T1_2024-07-19.csv'
    df = pd.read_csv(file_path)
    df.iloc[0,0] = "OBS Stamps"

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
    df.to_csv(file_path, index=False)
    df.to_csv(f"{path}\Synched Data\final_sync.csv")

    print(f"Timestamps have been converted to nano and saved to {file_path}")
