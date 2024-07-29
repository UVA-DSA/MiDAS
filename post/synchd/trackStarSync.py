
import pandas as pd
from trackStarCSVSynthesis import subset_trackstar, combined_file
from post.synchd.PDSSync import obs, obs_stamps


organized_trakstar = pd.read_csv(combined_file)
df = pd.DataFrame(columns = ['OBS Stamps', 'TrakStar Time', 'SensorID_0','x_0','y_0','z_0','azimuth_0','elevation_0','roll_0','SensorID_1','x_1','y_1','z_1','azimuth_1','elevation_1','roll_1','SensorID_2','x_2','y_2','z_2','azimuth_2','elevation_2','roll_2','SensorID_3','x_3','y_3','z_3','azimuth_3','elevation_3','roll_3'])

trakstar_stamps = organized_trakstar['TrakStar time'].toList()


iterator = 0
for i in obs_stamps:
    while True:
        try:
            if i >= trakstar_stamps[iterator] and i < trakstar_stamps[iterator+1]:
                df = pd.concat([df, organized_trakstar.iloc[[iterator]]], ignore_index=True)
                break
        except Exception:
            break
        iterator+=1

df['OBS Stamps'] = obs['Epoch']

df.to_csv('post/data/PegT_S221_2024-07-19/new_trakStar_synced.csv', index =False)




