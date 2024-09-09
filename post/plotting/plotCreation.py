import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

PDSSync = pd.DataFrame()
PDSOr = pd.DataFrame()
trakSync = pd.DataFrame()
trakOr = pd.DataFrame()

def PDSSyncComp(path):
    figure, axis = plt.subplots(2, 1, figsize=(20, 10))

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

    axis[0].plot(PDSOr.index, PDSOr['Pedal 1 Pressed'], color = "red", label = "Upper Left Pedal")
    axis[0].plot(PDSOr.index, PDSOr['Pedal 2 Pressed'], color = "orange", label = "Upper Right Pedal")
    axis[0].plot(PDSOr.index, PDSOr['Pedal 3 Pressed'], color = "yellow", label = "Lower Left Pedal")
    axis[0].plot(PDSOr.index, PDSOr['Pedal 4 Pressed'], color = "green", label = "Lower Right Pedal")
    axis[0].plot(PDSOr.index, PDSOr['Pedal 5 Pressed'], color = "blue", label = "Clutch")
    axis[0].plot(PDSOr.index, PDSOr['Pedal 6 Pressed'], color = "purple", label = "Camera")
    axis[0].plot(PDSOr.index, PDSOr['Pedal 7 Pressed'], color = "pink", label = "Long")

    axis[0].set_title('Original Pedal Data vs Data Index')
    axis[0].set_ylim(0,1.5)
    axis[0].set_xlabel("Data Index")
    axis[0].legend()

    plt.savefig(f'{path}/plots/PDS_sync.png',dpi=300, bbox_inches='tight', format='png')
    plt.close()

def trakSyncComp(path):
    figure, axis = plt.subplots(2, 1, figsize=(20, 10))

    axis[0].scatter(trakOr.index, trakOr['x_0'], color = "orange", label = "x_0",s=1)
    axis[0].scatter(trakOr.index, trakOr['y_0'], color = "green", label = "y_0",s=1)
    axis[0].scatter(trakOr.index, trakOr['z_0'], color = "blue", label = "z_0",s=1)
    axis[0].set_title('Original trakStar Data vs Data Index')
    axis[0].set_ylim(-400,1000)
    axis[0].set_xlabel("Data Index")
    axis[0].legend()
    
    axis[1].scatter(trakSync.index, trakSync['x_0'], color = "orange", label = "x_0",s=1)
    axis[1].scatter(trakSync.index, trakSync['y_0'], color = "green", label = "y_0",s=1)
    axis[1].scatter(trakSync.index, trakSync['z_0'], color = "blue", label = "z_0",s=1)
    axis[1].set_title('Synced trakStar Data vs Frame')
    axis[1].set_ylim(-400,1000)
    axis[1].set_xlabel("Frame #")
    axis[1].legend()

    plt.savefig(f'{path}/plots/trak_sync.png',dpi=300, bbox_inches='tight', format='png')
    plt.close()

def allShow(path):
    figure, axis = plt.subplots(2, 1, figsize=(20, 10))
    axis[0].scatter(trakSync.index, trakSync['x_0'], color = "orange", label = "x_0",s=3)
    axis[0].scatter(trakSync.index, trakSync['y_0'], color = "green", label = "y_0",s=3)
    axis[0].scatter(trakSync.index, trakSync['z_0'], color = "blue", label = "z_0",s=3)
    axis[0].set_title('Synced trakStar Data vs Frame')
    axis[0].set_ylim(-400,1000)
    axis[0].set_xlabel("Frame #")
    axis[0].legend()

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

    plt.savefig(f'{path}/plots/sync.png',dpi=300, bbox_inches='tight', format='png')
    plt.close()

if __name__ == '__main__':
    matchingPath = glob.glob("D:/Data MIDAS/*")
    for path in matchingPath:
        print(path)
        trakSync = pd.read_csv(f"{path}/Synched Data/trakstar_sync.csv")
        PDSSync = pd.read_csv(f"{path}/Synched Data/PDS_sync.csv")
        trakOr = pd.read_csv(f"{path}/Trakstar/trakstar_organized.csv")
        PDSOr = pd.read_csv(f"{path}/PDS.csv")
        try:
            os.mkdir(f"{path}/plots/")
            print(f"Directory created successfully.")
        except FileExistsError:
            print(f"Directory already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")
        PDSSyncComp(path)
        trakSyncComp(path)
        allShow(path)
        