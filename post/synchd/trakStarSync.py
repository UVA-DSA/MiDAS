import glob
import pandas as pd
from trakStarCSVSynthesis import trakStarCSVSynthesis

def trakStarCSVSync(path):
    trakStarCSVSynthesis(path)

    trakDF = pd.read_csv(f"{path}/Trakstar/trakstar_organized.csv")
    trakStampList = trakDF['TrackStar time'].toList()
    
    obsDF = pd.read_csv(f"{path}/obs/*.csv")
    obsStampList = obsDF.iloc[:,0].tolist()

    finalDF = obsDF.copy()
    trakCols = trakDF.columns
    for col in trakCols:
        finalDF[col] = pd.NA
    iterator = 0
    for i in obsStampList:
        while True:
            try:
                if i >= trakStampList[iterator] and i < trakStampList[iterator+1]:
                    finalDF = pd.concat([finalDF, trakDF.iloc[[iterator]]], ignore_index=True)
                    break
            except Exception:
                print("trakSync Error")
                break
            iterator+=1

    finalDF.to_csv(f'{path}/new_trackStar_synced.csv', index =False)




