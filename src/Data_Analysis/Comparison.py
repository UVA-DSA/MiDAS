import matplotlib.pyplot as plt
import pandas as pd

synced_data = pd.read_csv('synced_data.csv')
original_data = pd.read_csv('sw_right_data.csv')



#need to switch this to sever time instead of file length
x_synced = range(len(synced_data))
x_original = range(len(original_data))


print(x_original, x_synced)

figure, axis = plt.subplots(2, 1, figsize=(15, 6))

axis[0].scatter(x_synced, synced_data['smartwatch_1_value_X_Axis'])
axis[1].scatter(x_original, original_data['value_X_Axis'])

plt.show()
