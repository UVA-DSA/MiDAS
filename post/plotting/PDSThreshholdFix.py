import pandas as pd

if __name__ == "__main__":
    path = "D:\Data MIDAS\Bowel_S216_T1_2024-07-18\Synched Data\PDS_sync.csv"
    pdsDF = pd.read_csv(path)
    cols = pdsDF.columns.tolist()
    for j in range(1,8):
        pressureList = pdsDF[f'Pedal {j} Pressure'].tolist()
        binaryList = []
        for i in pressureList:
            if i > 1500:
                binaryList.append(1.0)
            else:
                binaryList.append(0.0)
        pdsDF[f'Pedal {j} Pressed'] = binaryList

    pdsDF.to_csv(path, index =False)



