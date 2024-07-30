import glob
import pandas as pd
from trakStarCSVSynthesis import trakStarCSVSynthesis

def trakStarSync(path):
    trakStarCSVSynthesis(path)

    trakDF = pd.read_csv(f"{path}/Trakstar/trakstar_organized.csv")
    trakStampList = trakDF['TrackStar time'].toList()
    
    obsDF = pd.read_csv(f"{path}/obs/*.csv")
    obsStampList = obsDF.iloc[:,0].tolist()
organized_trakstar = pd.read_csv(combined_file)
df = pd.DataFrame(columns = ['OBS Stamps', 'TrakStar Time', 'SensorID_0','x_0','y_0','z_0','azimuth_0','elevation_0','roll_0','SensorID_1','x_1','y_1','z_1','azimuth_1','elevation_1','roll_1','SensorID_2','x_2','y_2','z_2','azimuth_2','elevation_2','roll_2','SensorID_3','x_3','y_3','z_3','azimuth_3','elevation_3','roll_3'])

trakstar_stamps = organized_trakstar['TrakStar time'].toList()

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


if __name__ == "__main__":
    trakStarSync("post/data/")

