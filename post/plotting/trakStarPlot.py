import matplotlib.pyplot as plt
import pandas as pd

syncData = pd.read_csv('./data/Bowel_S216_T1_2024-07-18/Synched Data/trakStar_sync.csv')
originalData = pd.read_csv('./data/Bowel_S216_T1_2024-07-18/Trakstar/trakstar_organized.csv')


#need to switch this to sever time instead of file length
#x_synced = range(synced_data.iat[0, 0], synced_data.iat[-1, 0])
# x_original = range(len(original_data))


# print(x_original, x_synced)

figure, axis = plt.subplots(1, 2, figsize=(15, 30))
axis[1].scatter(syncData.index, syncData['x_0'], color = "blue", s=5)
axis[1].set_title('Synced x_0 Data vs Frame')
axis[1].set_ylim(400,900)
axis[1].set_xlabel("Frame #")
axis[0].scatter(originalData.index, originalData['x_0'], s=5,color = "red")
axis[0].set_title('Original x_0 Data vs Frame')
axis[0].set_ylim(400,900)
axis[0].set_xlabel("trakStar Data Index")


# axis[1,0].scatter(synced_data['server_time'], synced_data['smartwatch_1_value_Y_Axis'], color = 'red')
# axis[1,0].set_title('synced y-axis')
# axis[1,1].scatter(original_data['server_epoch_ms'], original_data['value_Y_Axis'], color = 'orange')
# axis[1,1].set_title('original y-axis')

plt.show()
