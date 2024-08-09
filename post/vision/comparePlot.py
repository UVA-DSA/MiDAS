import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

PDSSync = pd.DataFrame()
pedalDetection = pd.DataFrame()

def plot(path):
    PDSSync = pd.read_csv(f"{path}/Synched Data/PDS_sync.csv")
    pedalDetection = pd.read_csv(f"{path}/obs/pedal_detection.csv")

    figure, axis = plt.subplots(2, 1, figsize=(20, 10))

    axis[0].plot(pedalDetection.index, pedalDetection['Upper Left'], color = "red", label = "Upper Left Pedal")
    axis[0].plot(pedalDetection.index, pedalDetection['Upper Right'], color = "yellow", label = "Upper Right Pedal")
    axis[0].plot(pedalDetection.index, pedalDetection['Lower Left'], color = "green", label = "Lower Left Pedal")
    axis[0].plot(pedalDetection.index, pedalDetection['Lower Right'], color = "blue", label = "Lower Right Pedal")

    axis[0].set_title('Pedal Indicators vs Frame')
    axis[0].set_ylim(0,1.5)
    axis[0].set_xlabel("Frame #")
    axis[0].legend()

    axis[1].plot(PDSSync.index, PDSSync['Pedal 1 Pressed'], color = "red", label = "Upper Left Pedal")
    axis[1].plot(PDSSync.index, PDSSync['Pedal 2 Pressed'], color = "yellow", label = "Upper Right Pedal")
    axis[1].plot(PDSSync.index, PDSSync['Pedal 3 Pressed'], color = "green", label = "Lower Left Pedal")
    axis[1].plot(PDSSync.index, PDSSync['Pedal 4 Pressed'], color = "blue", label = "Lower Right Pedal")

    axis[1].set_title('Synced Pedal Data vs Frame')
    axis[1].set_ylim(0,1.5)
    axis[1].set_xlabel("Frame #")
    axis[1].legend()

    plt.savefig(f'{path}/plots/pedal_detection_comp.png',dpi=300, bbox_inches='tight', format='png')
    plt.close()


if __name__ == '__main__':
    matchingPath = glob.glob("D:/Data MIDAS/B*")
    for path in matchingPath:
        print(path)
        PDSSync = pd.read_csv(f"{path}/Synched Data/PDS_sync.csv")
        detectionPath = glob.glob(f"{path}/obs/pedal_detection.csv")
        pedalDetection = pd.read_csv(detectionPath[0])
        try:
            os.mkdir(f"{path}/plots/")
            print(f"Directory created successfully.")
        except FileExistsError:
            print(f"Directory already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")
        plot(path)
        