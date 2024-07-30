import pandas as pd

sw_original = pd.read_csv('data/PegT_S221_T1_2024-07-19/smartwatch_data/sw_left/sw_data.csv')
sw_side = 'left'
#this can be either left or right, depending

sw_acc_file = 'data/PegT_S221_T1_2024-07-19/smartwatch_data/sw_left/sw_data_acc.csv'
sw_gyro_file = 'data/PegT_S221_T1_2024-07-19/smartwatch_data/sw_left/sw_data_gyro.csv'


string_convert = pd.DataFrame()
print(sw_original.columns.tolist())
string_convert['sensor_type'] = sw_original['sensor_type']
print(string_convert)
sw_acc_index = string_convert.str.contains("acc").idxmax()


sw_acc = pd.DataFrame()
sw_gyro = pd.DataFrame()

sw_acc = sw_original.iloc[sw_acc_index::2].reset_index(drop = True)
sw_gyro = sw_original.iloc[sw_acc_index+1::2].reset_index(drop = True)

sw_acc = sw_acc.drop(['sw_epoch_ms', 'sensor_type'])
sw_gyro = sw_gyro.drop(['sw_epoch_ms', 'sensor_type'])

sw_acc.to_csv(sw_acc_file, index = False)
sw_gyro.to_csv(sw_gyro_file, index = False)

