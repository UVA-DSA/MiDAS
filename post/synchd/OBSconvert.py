import pandas as pd
import glob

def OBSConvert(path):
    # Load the CSV file
    filePath = f"{path}/OBS*.csv"
    matchingPath = glob.glob(filePath)
    df = pd.read_csv(matchingPath[0], header=None, index_col=False)
    df.columns = ['OBS Stamps']
    # Convert timestamps to milliseconds
    df["OBS Stamps"] = df['OBS Stamps'] * 1000000

    for i in range(1,df.shape[0]):
        if(i % 3 == 1):
            try:
                df.iloc[i,0] = df.iloc[i,0] + 333333
            except Exception:
                print("ouch")
                pass
        elif(i%3 == 2):
            try:
                df.iloc[i,0] = df.iloc[i,0] + 666666
            except Exception:
                print("ouch")

                pass


    # Save the modified DataFrame back to a CSV file
    df.to_csv(matchingPath[0], index=False)
    dataLists = ["final_sync", "PDS_sync", "trakStar_sync", "sw_L_acc", "sw_L_gyro", "sw_R_acc", "sw_R_gyro"]
    for name in dataLists:
        df.to_csv(f"{path}/Synched Data/{name}.csv", index=False)
    print(f"Timestamps have been converted to nano and saved to {filePath}")

if __name__ == "__main__":
    path = "./data/INGUINAL_S113_T3_2024-07-18"
    OBSConvert(path)
