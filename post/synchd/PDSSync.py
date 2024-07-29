import pandas as pd

def PDSSync(path):
    pdsDF = pd.read_csv(f'{path}\PDS.csv')
    obsDF = pd.read_csv(f"{path}\obs\OBStimestamp_PegT_S221_T1_2024-07-19.csv")

    pdsStampList = pdsDF['timestamp'].tolist()
    obsStampList = obsDF.iloc[:,0].tolist()
    
    finalDF = obsDF.copy()
    PDS_cols = pdsDF.columns
    for col in PDS_cols:
        finalDF[col] = pd.NA

    iterator = 0
    for i in obsStampList:
        print(i)
        while True:
            try:
                if i >= pdsStampList[iterator] and i < pdsStampList[iterator+1]:
                    finalDF = pd.concat([finalDF, pdsDF.iloc[[iterator]]], ignore_index=True)
                    break
            except Exception:
                break
            iterator+=1

    finalDF.to_csv(f'{path}/Synched Data/new_synced.csv' , index=False)
