import pandas as pd
import csv



def swCSVSynth(path):
    cols =  ['sw_epoch_ms','wrist_position','sensor_type','value_X_Axis','value_Y_Axis','value_Z_Axis','seq','seq_num','server_epoch_ms']

    file_path = f'{path}/smartwatch_data/sw_right/sw_data.csv'

    with open(file_path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        original = list(reader)

    with open(file_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(cols)
        for row in original[1:]:
            writer.writerow(row)



    sw_original = pd.read_csv(file_path)
    sw_side = 'right'
    #this can be either left or right, depending

    sw_acc_file = 'data/PegT_S221_T1_2024-07-19/smartwatch_data/sw_right/sw_data_acc.csv'
    sw_gyro_file = 'data/PegT_S221_T1_2024-07-19/smartwatch_data/sw_right/sw_data_gyro.csv'




    cols =  ['sw_epoch_ms','wrist_position','sensor_type','value_X_Axis','value_Y_Axis','value_Z_Axis','seq','seq_num','server_epoch_ms']
    # sw_original.columns = ['sw_epoch_ms','wrist_position','sensor_type','value_X_Axis','value_Y_Axis','value_Z_Axis','seq','seq_num','server_epoch_ms']

    print(sw_original[0:100])

    sw_acc = pd.DataFrame()
    sw_gyro = pd.DataFrame()


    sw_acc = sw_original[sw_original['sensor_type'] == 'acc']
    sw_gyro = acc_rows = sw_original[sw_original['sensor_type'] == 'gyro']


    print(sw_acc, sw_gyro)

    sw_acc = sw_acc.drop('sw_epoch_ms', axis = 1)
    sw_acc = sw_acc.drop('sensor_type', axis = 1)
    sw_gyro = sw_gyro.drop('sw_epoch_ms', axis = 1)
    sw_gyro = sw_gyro.drop('sensor_type', axis = 1)

    sw_acc.to_csv(sw_acc_file, index = False)
    sw_gyro.to_csv(sw_gyro_file, index = False)

