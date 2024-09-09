import pandas as pd
import glob

chunkList = []

def OBSConvert(path):
    # Load the CSV file
    filePath = f"{path}/OBS*.csv"
    matchingPath = glob.glob(filePath)
    print(matchingPath)
    for df in pd.read_csv(matchingPath[0], header=None, index_col=False, chunksize = 1000):
        df.columns = ['OBS Stamps']
        # Convert timestamps to nanoseconds
        df.iloc[:,0] = df.iloc[:,0] * 1000000
        for i in range(1,df.shape[0]):
            if(i % 3 == 1):
                try:
                    df.iloc[i,0] = df.iloc[i,0] + 333333
                except Exception as e:
                    print(e)
                    pass
            elif(i%3 == 2):
                try:
                    df.iloc[i,0] = df.iloc[i,0] + 666666
                except Exception as e:
                    print(e)

                    pass
        chunkList.append(df)
    df = pd.concat(chunkList, axis = 0)


    # Save the modified DataFrame back to a CSV file
    df.to_csv(matchingPath[0], index=False)
    dataLists = ["final_sync", "PDS_sync", "trakstar_sync", "sw_L_acc", "sw_L_gyro", "sw_R_acc", "sw_R_gyro", "zed_sync"]
    for name in dataLists:
        df.to_csv(f"{path}/Synched Data/{name}.csv", index=False)
    print(f"Timestamps have been converted to nano and saved to {filePath}")

if __name__ == "__main__":
    path = "D:/Data MIDAS/VentralH_S218_T2_2024-07-19/"
    OBSConvert(path)
