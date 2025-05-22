# Run to get:
# - Corrected OBS timestamps
# - Synched data for all files in 'path'
# - PDS threshold data for the four pedals on the right (not including clutch/camera)
# - Stats on errors for the four pedals on the right

from OBS_stamps_generation import OBSCreate
from PDS_verification_plot import PDSvsVid
from pdsTotalErrorCalc import countErrorsByPedal, extract_best_thresholds_by_trial
from PDSSync import PDSSync
import pandas as pd
import glob
import os

root_dir = './testdata/'

if __name__ == "__main__":
    main_dir_glob = root_dir + '*'
    matchingPath = sorted(glob.glob(root_dir))
    print(matchingPath)
    file = open("log.txt", "a")
    for mainPath in matchingPath:    
        finalPath = f"{mainPath}/Synched Data/final_sync.csv"
        print("Path\t", mainPath)

        try:
            os.mkdir(f"{mainPath}/Synched Data/")
            print(f"Directory created successfully.")
        except FileExistsError:
            print(f"Directory already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            OBSCreate(mainPath)
        except Exception as e:
            print("OBS Error", e)
            file.write(f"{mainPath}, {e}\n")
            continue

        try:
            PDSSync(mainPath)
        except Exception as e:
            print("PDS Error", e)
            file.write(f"{mainPath}, {e}")

        try:
            PDSvsVid(mainPath)
        except Exception as e:
            print("Vid Extraction Error", e)
            file.write(f"{mainPath}, {e}\n")
            continue

    thresholds = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
    for t in thresholds:
        countErrorsByPedal(root_dir, t)

    extract_best_thresholds_by_trial(root_dir, "Per Trial Best Thresholds.csv")
      