import matplotlib.pyplot as plt
import pandas as pd

synced_data = pd.read_csv('synced_data.csv')
original_data = pd.read_csv('sw_right_data.csv')

print(type(synced_data))

#need to switch this to sever time instead of file length
#x_synced = range(synced_data.iat[0, 0], synced_data.iat[-1, 0])
# x_original = range(len(original_data))


# print(x_original, x_synced)

figure, axis = plt.subplots(2, 2, figsize=(15, 15))

axis[0, 0].scatter(synced_data['server_time'], synced_data['smartwatch_1_value_X_Axis'], color = "blue")
axis[0, 0].set_title('synced x-axis')
axis[0, 1].scatter(original_data['server_epoch_ms'], original_data['value_X_Axis'])
axis[0,1].set_title('original x-axis')

axis[1,0].scatter(synced_data['server_time'], synced_data['smartwatch_1_value_Y_Axis'], color = 'red')
axis[1,0].set_title('synced y-axis')
axis[1,1].scatter(original_data['server_epoch_ms'], original_data['value_Y_Axis'], color = 'orange')
axis[1,1].set_title('original y-axis')

plt.show()
