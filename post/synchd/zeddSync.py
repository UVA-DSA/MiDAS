import pandas as pd
import glob
import csv

def ZedSync(path):
    zeddDF = pd.read_csv(f'{path}/camera/zed/zed_frames.csv')
    obsPath = f"{path}/OBS*.csv"
    matchingPath = glob.glob(obsPath)
    obsDF = pd.read_csv(matchingPath[0], index_col=False)

    zeddStampList = zeddDF.iloc[:,0].tolist()
    obsStampList = obsDF.iloc[:,0].tolist()
    finalDF = pd.read_csv(f'{path}/Synched Data/zed_sync.csv')
    zeddMatched = pd.DataFrame()
    zero_row = pd.DataFrame([[0]*2], columns=zeddDF.columns)


    iterator = 0
    for i in obsStampList:
        if iterator % 1000 < 5:
            print(f"Zedd: {iterator}")
        while iterator < len(zeddStampList)-2:
            try:
                if i >= zeddStampList[iterator] and i < zeddStampList[iterator+1]:
                    zeddMatched = pd.concat([zeddMatched, zeddDF.iloc[[iterator]]], ignore_index=True)
                    break
                if i < zeddStampList[iterator]:
                    trakMatched = pd.concat([trakMatched, zero_row])
                    break
            except Exception as e:
                print(f"Zedd Sync Error: {e}")
                break
            iterator+=1

    finalDF = pd.concat([finalDF, zeddMatched], axis = 1)
    finalDF.to_csv(f'{path}/Synched Data/zed_sync.csv' , index=False)


if __name__ == "__main__":
    path = "D:/Data MIDAS/VentralH_S218_T2_2024-07-19/"
    ZedSync(path)

