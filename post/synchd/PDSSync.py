import pandas as pd
import glob

def PDSSync(path):
    pdsDF = pd.read_csv(f'{path}/PDS.csv')
    pdsDF.columns = ["Pedal 1 Pressure","Pedal 2 Pressure","Pedal 3 Pressure","Pedal 4 Pressure","Pedal 5 Pressure","Pedal 6 Pressure","Pedal 7 Pressure","Pedal 1 Pressed","Pedal 2 Pressed","Pedal 3 Pressed","Pedal 4 Pressed","Pedal 5 Pressesd","Pedal 6 Pressed","Pedal 7 Pressed","PDS Time"]
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
    cols = finalDF.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    finalDF = finalDF[cols]
    finalDF.to_csv(f'{path}/Synched Data/PDS_sync.csv' , index=False)


if __name__ == "__main__":
    path = "./data/Inguinal_S113_T3_2024-07-18"
    PDSSync(path)

