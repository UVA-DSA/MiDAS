import matplotlib.pyplot as plt
import pandas as pd

original_data = pd.read_csv('data/PegT_S221_T1_2024-07-19/Trakstar/trakstar_organized.csv')
synced_data = pd.read_csv('data/PegT_S221_T1_2024-07-19/Trakstar/trakstar_synched.csv')


#need to switch this to sever time instead of file length
#x_synced = range(synced_data.iat[0, 0], synced_data.iat[-1, 0])
# x_original = range(len(original_data))


# print(x_original, x_synced)

figure, axis = plt.subplots(2, 2, figsize=(15, 15))

axis[0, 0].scatter(synced_data.index, synced_data['x_0'], color = "blue")
axis[0, 0].set_title('synced x_0 data vs frame')
axis[0, 1].scatter(original_data.index, original_data['x_0'])
axis[0,1].set_title('original x_0 data vs frame')

# axis[1,0].scatter(synced_data['server_time'], synced_data['smartwatch_1_value_Y_Axis'], color = 'red')
# axis[1,0].set_title('synced y-axis')
# axis[1,1].scatter(original_data['server_epoch_ms'], original_data['value_Y_Axis'], color = 'orange')
# axis[1,1].set_title('original y-axis')

plt.show()
