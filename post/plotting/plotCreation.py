import matplotlib.pyplot as plt
import pandas as pd

PDSSync = pd.read_csv('./plotting/PDS_sync.csv')
PDSOG = pd.read_csv('./data/Bowel_S216_T1_2024-07-18/PDS.csv')
trakSync = pd.read_csv('./data/Bowel_S216_T1_2024-07-18/Synched Data/trakStar_sync.csv')

figure, axis = plt.subplots(2, 1, figsize=(15, 20))

axis[1].plot(PDSSync.index, PDSSync['Pedal 1 Pressed'], color = "red", label = "Upper Left Pedal")
axis[1].plot(PDSSync.index, PDSSync['Pedal 2 Pressed'], color = "orange", label = "Upper Right Pedal")
axis[1].plot(PDSSync.index, PDSSync['Pedal 3 Pressed'], color = "yellow", label = "Lower Left Pedal")
axis[1].plot(PDSSync.index, PDSSync['Pedal 4 Pressed'], color = "green", label = "Lower Right Pedal")
axis[1].plot(PDSSync.index, PDSSync['Pedal 5 Pressed'], color = "blue", label = "Clutch")
axis[1].plot(PDSSync.index, PDSSync['Pedal 6 Pressed'], color = "purple", label = "Camera")
axis[1].plot(PDSSync.index, PDSSync['Pedal 7 Pressed'], color = "pink", label = "Long")
axis[1].set_title('Synced Pedal Data vs Frame')
axis[1].set_ylim(0,1.5)
axis[1].set_xlabel("Frame #")
axis[1].legend()

# axis[0].scatter(trakSync.index, trakSync['x_0'],color = "orange", label = "x_0",s=5)
# axis[0].scatter(trakSync.index, trakSync['y_0'], color = "green", label = "y_0",s=5)
# axis[0].scatter(trakSync.index, trakSync['z_0'], color = "blue", label = "z_0",s=5)

axis[0].plot(PDSOG.index, PDSOG['Pedal 1 Pressed'], color = "red", label = "Upper Left Pedal")
axis[0].plot(PDSOG.index, PDSOG['Pedal 2 Pressed'], color = "orange", label = "Upper Right Pedal")
axis[0].plot(PDSOG.index, PDSOG['Pedal 3 Pressed'], color = "yellow", label = "Lower Left Pedal")
axis[0].plot(PDSOG.index, PDSOG['Pedal 4 Pressed'], color = "green", label = "Lower Right Pedal")
axis[0].plot(PDSOG.index, PDSOG['Pedal 5 Pressed'], color = "blue", label = "Clutch")
axis[0].plot(PDSOG.index, PDSOG['Pedal 6 Pressed'], color = "purple", label = "Camera")
axis[0].plot(PDSOG.index, PDSOG['Pedal 7 Pressed'], color = "pink", label = "Long")

axis[0].set_title('Original Pedal Data vs Data Index')
axis[0].set_ylim(0,1.5)
axis[0].set_xlabel("Data Index")
axis[0].legend()

plt.show()
