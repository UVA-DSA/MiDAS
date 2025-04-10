import pandas as pd
import glob
from OBSstamps import createOBSStamps

chunkList = []

def OBSConvert(path):
    # Load the CSV file
    filePath = f"{path}/OBS*.csv"
    matchingPath = glob.glob(filePath)
    if(len(matchingPath) == 0):
        createOBSStamps(path)
        matchingPath = glob.glob(filePath)
    print(matchingPath)
    
    # Read just the first row
    with open(matchingPath[0], 'r') as f:
        first_line = f.readline().strip()

    # Try to convert first line to an integer — if it fails, it's probably a header
    try:
        int(first_line)
        has_header = False
    except ValueError:
        has_header = True

    if not has_header:
        # Read whole file with no header, add column name, and save
        df = pd.read_csv(matchingPath[0], header=None, names=["OBS Stamps"])
        df.to_csv("timestamps_with_header.csv", index=False)
        print("Header added.")
    else:
        print("File already has a header.")

    # Save the modified DataFrame back to a CSV file
    df.to_csv(matchingPath[0], index=False)
    dataLists = ["final_sync", "PDS_sync", "trakstar_sync", "sw_L_acc", "sw_L_gyro", "sw_R_acc", "sw_R_gyro", "zed_sync"]
    for name in dataLists:
        try:
            df.to_csv(f"{path}/Synched Data/{name}.csv", index=False)
        except:
            continue
    print(f"Timestamps have been converted to nano and saved to {filePath}")

if __name__ == "__main__":
    path = "./data/Bowel_S216_T1_2024-07-18"
    OBSConvert(path)
