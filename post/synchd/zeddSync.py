import pandas as pd
import numpy as np
import glob
import csv

def ZedSync(path):
    zeddDF = pd.read_csv(f'{path}/camera/Zed/timestamps.csv')
    zeddDF['File Number'] = np.NAN
    zeddDF.columns = ["Zed Stamps", "ID", "File Number"]
    obsPath = f"{path}/OBS*.csv"
    matchingPath = glob.glob(obsPath)
    obsDF = pd.read_csv(matchingPath[0], index_col=False)

    zeddStampList = zeddDF.iloc[:,0].tolist()
    obsStampList = obsDF.iloc[:,0].tolist()
    # finalDF = pd.read_csv(f'{path}/Synched Data/zed_sync.csv')
    finalDF = pd.read_csv(matchingPath[0], index_col = False)
    zedMatched = pd.DataFrame()
    zero_row = pd.DataFrame([[0]*3], columns=zeddDF.columns)

    fileNum = 0
    iterator = 0
    for i in obsStampList:
        if iterator % 10000 < 3:
            print(f"Zedd: {iterator}")
        while iterator < len(zeddStampList)-2:
            try:
                if i >= zeddStampList[iterator] and i < zeddStampList[iterator+1]:
                    if iterator > 0:
                        if zeddDF.loc[iterator, 'ID'] < zeddDF.loc[iterator-1, "ID"]:
                            fileNum += 1    
                    zeddDF.loc[iterator, 'File Number'] = fileNum
                    zedMatched = pd.concat([zedMatched, zeddDF.iloc[[iterator]]], ignore_index=True)
                    break
                if i < zeddStampList[iterator]:
                    zedMatched = pd.concat([zedMatched, zero_row])
                    break
            except Exception as e:
                print(f"Zedd Sync Error: {e}")
                break
            iterator+=1
            

    finalDF = pd.concat([finalDF, zedMatched], axis = 1)
    finalDF.to_csv(f'{path}/Synched Data/zed_sync.csv' , index=False)


if __name__ == "__main__":
    path = "D:/Data MIDAS/Bowel_S216_T1_2024-07-18/"
    ZedSync(path)

