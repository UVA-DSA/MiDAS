import pandas as pd

def thresholdValueChange(path, value):
    pdsDF = pd.read_csv(path)
    cols = pdsDF.columns.tolist()
    for j in range(1,8):
        pressureList = pdsDF[f'Pedal {j} Pressure'].tolist()
        binaryList = []
        for i in pressureList:
            if i > value:
                binaryList.append(1.0)
            else:
                binaryList.append(0.0)
        pdsDF[f'Pedal {j} Pressed'] = binaryList

    pdsDF.to_csv(path, index =False)

if __name__ == "__main__":
    path = "./data/Bowel_S207_T2_2024-07-18/Synched Data/PDS_sync.csv"
    thresholdValueChange(path, 1000)



