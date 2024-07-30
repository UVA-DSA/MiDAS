import pandas as pd
import glob

def PDSSync(path):
    pdsDF = pd.read_csv(f'{path}/PDS.csv')
    pdsDF.drop(index=pdsDF.index[0])
    obsPath = f"{path}/obs/*.csv"
    matchingPath = glob.glob(obsPath)
    obsDF = pd.read_csv(matchingPath[0], index_col=False)

    pdsDecimalList = pdsDF.iloc[:,-1].tolist()
    pdsStampList = []
    for num in pdsDecimalList:
        try:
            pdsStampList.append(int(num))
        except Exception:
            pass
    obsStampList = obsDF.iloc[:,0].tolist()
    
    finalDF = pd.read_csv(f'{path}/Synched Data/PDS_sync.csv')
    PDS_matched = pd.DataFrame()
    # print(PDS_cols)
    # for col in PDS_cols:
        # finalDF[col] = pd.NA

    iterator = 0
    for i in obsStampList:
        print(f"PDS: {iterator}")
        while iterator < len(pdsStampList)-2:
            try:
                if i >= pdsStampList[iterator] and i < pdsStampList[iterator+1]:
                    PDS_matched = pd.concat([PDS_matched, pdsDF.iloc[[iterator]]], ignore_index=True)
                    break
            except Exception as e:
                print(f"PDS Sync Error: {e}")
                break
            iterator+=1

    finalDF = pd.concat([finalDF, PDS_matched], axis = 1)
    finalDF = finalDF.drop(columns=['Computer Time'])
    finalDF.to_csv(f'{path}/Synched Data/PDS_sync.csv' , index=False)


if __name__ == "__main__":
    path = "./data/Inguinal_S113_T3_2024-07-18"
    PDSSync(path)

