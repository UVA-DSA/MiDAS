import pandas as pd
import glob
import csv

def PDSSync(path):
    cols = ["Pedal 1 Pressure","Pedal 2 Pressure","Pedal 3 Pressure","Pedal 4 Pressure","Pedal 5 Pressure","Pedal 6 Pressure","Pedal 7 Pressure","Pedal 1 Pressed","Pedal 2 Pressed","Pedal 3 Pressed","Pedal 4 Pressed","Pedal 5 Pressed","Pedal 6 Pressed","Pedal 7 Pressed","PDS Time"]
    with open(f'{path}/PDS.csv', mode='r', newline='', errors='ignore') as infile:
        reader = csv.reader((line.replace('\0', '') for line in infile))
        original = list(reader)

    with open(f'{path}/PDS.csv', mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(cols)
        for row in original[1:]:
            writer.writerow(row)

    
    
    pdsDF = pd.read_csv(f'{path}/PDS.csv')
    obsPath = f"{path}/OBS*.csv"
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
        if iterator % 1000 < 5:
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
    cols = [cols[0], cols[-1]] + cols[1:-1]
    finalDF = finalDF[cols]
    finalDF.to_csv(f'{path}/Synched Data/PDS_sync.csv' , index=False)


if __name__ == "__main__":
    path = "./data/Inguinal_S113_T3_2024-07-18"
    PDSSync(path)

