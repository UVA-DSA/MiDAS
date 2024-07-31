import matplotlib.pyplot as plt
import pandas as pd

PDSSync = pd.read_csv('./data/Bowel_S216_T1_2024-07-18/Synched Data/PDS_sync.csv')
trakSync = pd.read_csv('./data/Bowel_S216_T1_2024-07-18/Synched Data/trakStar_sync.csv')

figure, axis = plt.subplots(2, 1, figsize=(15, 20))

axis[1].scatter(PDSSync.index, PDSSync['Pedal 1 Pressure'], color = "red", label = "Upper Left Pedal",s=10)
axis[1].scatter(PDSSync.index, PDSSync['Pedal 2 Pressure'], color = "orange", label = "Upper Right Pedal",s=10)
axis[1].scatter(PDSSync.index, PDSSync['Pedal 3 Pressure'], color = "yellow", label = "Lower Left Pedal",s=10)
axis[1].scatter(PDSSync.index, PDSSync['Pedal 4 Pressure'], color = "green", label = "Lower Right Pedal",s=10)
axis[1].scatter(PDSSync.index, PDSSync['Pedal 5 Pressure'], color = "blue", label = "Clutch",s=10)
axis[1].scatter(PDSSync.index, PDSSync['Pedal 6 Pressure'], color = "purple", label = "Camera",s=10)
axis[1].scatter(PDSSync.index, PDSSync['Pedal 7 Pressure'], color = "pink", label = "Long",s=10)
axis[1].set_title('Synced Pedal Data vs Frame')
axis[1].set_ylim(-50,14000)
axis[1].set_xlabel("Frame #")
axis[1].legend()

axis[0].scatter(trakSync.index, trakSync['x_0'],color = "orange", label = "x_0",s=5)
axis[0].scatter(trakSync.index, trakSync['y_0'], color = "green", label = "y_0",s=5)
axis[0].scatter(trakSync.index, trakSync['z_0'], color = "blue", label = "z_0",s=5)
axis[0].set_title('Synced trakStar Sensor 0 Data vs Frame')
axis[0].set_ylim(-400,1000)
axis[0].legend()


# axis[1,0].scatter(synced_data['server_time'], synced_data['smartwatch_1_value_Y_Axis'], color = 'red')
# axis[1,0].set_title('synced y-axis')
# axis[1,1].scatter(original_data['server_epoch_ms'], original_data['value_Y_Axis'], color = 'orange')
# axis[1,1].set_title('original y-axis')

plt.show()
