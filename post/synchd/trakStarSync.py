import glob
import pandas as pd
from trakStarCSVSynthesis import trakStarCSVSynthesis

def trakStarSync(path):
    trakStarCSVSynthesis(path)

    trakDF = pd.read_csv(f"{path}/Trakstar/trakstar_organized.csv")
    trakDecimalList = trakDF.iloc[:,0].tolist()
    trakStampList = []
    for num in trakDecimalList:
        try:
            trakStampList.append(int(num))
        except Exception:
            pass
    
    obsPath = f"{path}/obs/*.csv"
    matchingPath = glob.glob(obsPath)
    obsDF = pd.read_csv(matchingPath[0], index_col=False)
    obsStampList = obsDF.iloc[:,0].tolist()
    print(trakStampList[1], "  ", obsStampList[1])
    print(trakStampList[1] - obsStampList[1])
    finalDF = pd.read_csv(f'{path}/Synched Data/trakstar_sync.csv')
    trakMatched = pd.DataFrame(columns=trakDF.columns)
    zero_row = pd.DataFrame([[0]*29], columns=trakDF.columns)


    iterator = 0
    for i in obsStampList:
        if iterator % 1000 < 10:
            print(f"trak: {iterator}")
        while iterator < len(trakStampList) - 1:
            try:
                if i >= trakStampList[iterator] and i < trakStampList[iterator+1]:
                    trakMatched = pd.concat([trakMatched, trakDF.iloc[[iterator]]], ignore_index=True)
                    break
                if i < trakStampList[iterator]:
                    trakMatched = pd.concat([trakMatched, zero_row])
                    break
            except Exception as e:
                print(f"trakSync Error: {e}")
                break
            iterator+=1
    finalDF = pd.concat([finalDF, trakMatched], axis = 1)
    finalDF.to_csv(f'{path}/Synched Data/trakStar_sync.csv', index =False)


if __name__ == "__main__":
    path = "./data/Inguinal_S113_T3_2024-07-18"
    trakStarSync(path)

