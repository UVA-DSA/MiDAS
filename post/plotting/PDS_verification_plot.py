import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys
from PDSThreshholdFix import thresholdValueChange
from PDS_error_detection import comparePDSExpectedExperimental
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.pedalDetection import guiPedalExtraction

def PDSvsVid(path):
    guiPedalExtraction(path)
    PDSSync = pd.DataFrame()
    PDSVideo = pd.DataFrame()

    Threshold_Values_Test = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
    try:
        os.mkdir(f"{path}/plots/")
        print(f"Directory created successfully.")
    except FileExistsError:
        print(f"Directory already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
    for threshold in Threshold_Values_Test:
        thresholdValueChange(f"{path}/Synched Data/PDS_sync.csv", threshold)
        comparePDSExpectedExperimental(path, threshold)
        PDSSync = pd.read_csv(f"{path}/Synched Data/PDS_sync.csv")
        PDSVideo = pd.read_csv(f"{path}/obs/pedal_detection.csv")
            
        figure, axis = plt.subplots(2, 1, figsize=(20, 10))

        axis[1].plot(PDSSync.index, PDSSync['Pedal 1 Pressed'], color = "red", label = "Upper Left Pedal")
        axis[1].plot(PDSSync.index, PDSSync['Pedal 2 Pressed'], color = "orange", label = "Upper Right Pedal")
        axis[1].plot(PDSSync.index, PDSSync['Pedal 3 Pressed'], color = "green", label = "Lower Left Pedal")
        axis[1].plot(PDSSync.index, PDSSync['Pedal 4 Pressed'], color = "blue", label = "Lower Right Pedal")
        # axis[1].plot(PDSSync.index, PDSSync['Pedal 5 Pressed'], color = "blue", label = "Clutch")
        # axis[1].plot(PDSSync.index, PDSSync['Pedal 6 Pressed'], color = "purple", label = "Camera")
        # axis[1].plot(PDSSync.index, PDSSync['Pedal 7 Pressed'], color = "pink", label = "Long")
        axis[1].set_title('Synced Pedal Data vs Frame')
        axis[1].set_ylim(0,1.5)
        axis[1].set_xlabel("Frame #")
        axis[1].legend()

        axis[0].plot(PDSVideo.index, PDSVideo['Upper Left'], color = "red", label = "Upper Left Pedal")
        axis[0].plot(PDSVideo.index, PDSVideo['Upper Right'], color = "orange", label = "Upper Right Pedal")
        axis[0].plot(PDSVideo.index, PDSVideo['Lower Left'], color = "green", label = "Lower Left Pedal")
        axis[0].plot(PDSVideo.index, PDSVideo['Lower Right'], color = "blue", label = "Lower Right Pedal")
        # axis[0].plot(PDSVideo.index, PDSVideo['Pedal 5 Pressed'], color = "blue", label = "Clutch")
        # axis[0].plot(PDSVideo.index, PDSVideo['Pedal 6 Pressed'], color = "purple", label = "Camera")
        # axis[0].plot(PDSVideo.index, PDSVideo['Pedal 7 Pressed'], color = "pink", label = "Long")

        axis[0].set_title('Video Pedal Data vs Frame')
        axis[0].set_ylim(0,1.5)
        axis[0].set_xlabel("Frame #")
        axis[0].legend()

        plt.savefig(f'{path}/plots/PDS_vs_Vid_{threshold}.png',dpi=300, bbox_inches='tight', format='png')
        plt.close()
    print("Plots Created")

#RUN FOR GUI EXTRACTION, PLOTS, AND CSVs
if __name__ == '__main__':
    matchingPath = glob.glob("./data/*")
    for path in matchingPath:
        print(path)
        PDSvsVid(path)
        