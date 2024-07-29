import pandas as pd 

original_file = 'data/PegT_S221_T1_2024-07-19/Trakstar/trakstar.csv'

original_df = pd.read_csv(original_file, usecols=['SensorID','x','y','z','azimuth','elevation','roll','TrakStar Time'])

combined_file = 'data/PegT_S221_T1_2024-07-19/Trakstar/trakstar_organized.csv'



df = pd.DataFrame(columns=['TrakStar time','SensorID_0','x_0','y_0','z_0','azimuth_0','elevation_0','roll_0','SensorID_1','x_1','y_1','z_1','azimuth_1','elevation_1','roll_1','SensorID_2','x_2','y_2','z_2','azimuth_2','elevation_2','roll_2','SensorID_3','x_3','y_3','z_3','azimuth_3','elevation_3','roll_3'])

subset_trackstar = original_df['TrakStar Time'].iloc[::4].reset_index(drop=True) * 1000000000

subset_0 = original_df[['SensorID','x','y','z','azimuth','elevation','roll']].iloc[::4].reset_index(drop=True)
subset_1 = original_df[['SensorID','x','y','z','azimuth','elevation','roll']].iloc[1::4].reset_index(drop=True)
subset_2 = original_df[['SensorID','x','y','z','azimuth','elevation','roll']].iloc[2::4].reset_index(drop=True)
subset_3 = original_df[['SensorID','x','y','z','azimuth','elevation','roll']].iloc[3::4].reset_index(drop=True)


df['TrakStar time'] = subset_trackstar
df[['SensorID_0','x_0', 'y_0', 'z_0', 'azimuth_0', 'elevation_0', 'roll_0']] = subset_0
df[['SensorID_1','x_1', 'y_1', 'z_1', 'azimuth_1', 'elevation_1', 'roll_1']] = subset_1
df[['SensorID_2','x_2', 'y_2', 'z_2', 'azimuth_2', 'elevation_2', 'roll_2']] = subset_2
df[['SensorID_3','x_3', 'y_3', 'z_3', 'azimuth_3', 'elevation_3', 'roll_3']] = subset_3

df.to_csv(combined_file, index=False)