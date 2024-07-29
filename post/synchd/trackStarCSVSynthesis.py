import pandas as pd 

original_file = pd.read_csv('data\**\Trakstar\trakstar.csv', usecols=['SensorID','x','y','z','azimuth','elevation','roll','TrakStar Time'])

combined_file = 'data\**\Trakstar\trakstar_organized.csv'



df = pd.DataFrame(columns=['SensorID', 'TrakStar time','x_0','y_0','z_0','azimuth_0','elevation_0','roll_0','x_1','y_1','z_1','azimuth_1','elevation_1','roll_1','x_2','y_2','z_2','azimuth_2','elevation_2','roll_2','x_3','y_3','z_3','azimuth_3','elevation_3','roll_3'])


df['x_0', 'y_0', 'z_0', 'azimuth_0', 'elevation_0', 'roll_0'] = original_file['x','y','z','azimuth','roll','elevation'].iloc[::3].reset_index(drop=True)

df.to_csv(combined_file)