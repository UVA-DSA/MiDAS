import pandas as pd
import csv



def swCSVSynth(path):
    sides = ['right', 'left']
    for a in sides:
        print(a)
        cols =  ['sw_epoch_ms','wrist_position','sensor_type','value_X_Axis','value_Y_Axis','value_Z_Axis','seq','seq_num','Watch Time']
        dataPath = f'{path}/smartwatch_data/sw_{a}'
        file_path = f'{dataPath}/sw_data.csv'

        with open(file_path, mode='r', newline='') as infile:
            reader = csv.reader((line.replace('\0', '') for line in infile))
            original = list(reader)
        with open(file_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(cols)
            for row in original[1:]:
                writer.writerow(row)

        sw_original = pd.read_csv(file_path)
        sw_acc_file = f'{dataPath}/sw_data_acc.csv'
        sw_gyro_file = f'{dataPath}/sw_data_gyro.csv'

        sw_acc = pd.DataFrame()
        sw_gyro = pd.DataFrame()

        sw_acc = sw_original[sw_original['sensor_type'] == 'acc']
        sw_gyro = sw_original[sw_original['sensor_type'] == 'gyro']

        sw_acc = sw_acc.drop('sw_epoch_ms', axis = 1)
        sw_acc = sw_acc.drop('sensor_type', axis = 1)
        sw_gyro = sw_gyro.drop('sw_epoch_ms', axis = 1)
        sw_gyro = sw_gyro.drop('sensor_type', axis = 1)

        cols = sw_acc.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        sw_acc = sw_acc[cols]

        cols = sw_gyro.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        sw_gyro = sw_gyro[cols]

        sw_acc.to_csv(sw_acc_file, index = False)
        sw_gyro.to_csv(sw_gyro_file, index = False)

if __name__ == '__main__':
    swCSVSynth('./data/PegT_S221_T1_2024-07-19/')
