import pandas as pd
import glob
import os
from OBSconvert import OBSConvert
from PDSSync import PDSSync
from trakStarSync import trakStarSync
from swSync import swSync
from zeddSync import ZedSync

sw_included = False #TOGGLE FOR PEG TRANSFER DATA, IF NOT PEG TRANSFER, SHOULD BE FALSE
root_dir = './data/*'

if __name__ == "__main__":
    matchingPath = sorted(glob.glob(root_dir))
    print(matchingPath)
    file = open("log.txt", "a")
    for mainPath in matchingPath:
        if mainPath[0] == "P":
            sw_included = True
        else:
            sw_included = False         
        finalPath = f"{mainPath}/Synched Data/final_sync.csv"
        try:
            os.mkdir(f"{mainPath}/Synched Data/")
            print(f"Directory created successfully.")
        except FileExistsError:
            print(f"Directory already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")
        print("Path", mainPath)
        try:
            OBSConvert(mainPath)
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
            trakStarSync(mainPath)
        except Exception as e:
            file.write(f"{mainPath}, {e}")
        try:
            if sw_included:
                swSync(mainPath)
        except Exception as e:
            file.write(f"{mainPath}, {e}")
        try:
            ZedSync(mainPath)
        except Exception as e: 
            file.write(f"{mainPath}, {e}")
        
        finalDF = pd.read_csv(finalPath)
        pdsDF = pd.read_csv(f"{mainPath}/Synched Data/PDS_sync.csv")
        swLADF = pd.read_csv(f"{mainPath}/Synched Data/sw_L_acc.csv")
        swLGDF = pd.read_csv(f"{mainPath}/Synched Data/sw_L_gyro.csv")
        swRADF = pd.read_csv(f"{mainPath}/Synched Data/sw_R_acc.csv")
        swRGDF = pd.read_csv(f"{mainPath}/Synched Data/sw_R_gyro.csv")
        trakstarDF = pd.read_csv(f"{mainPath}/Synched Data/trakstar_sync.csv")
        zeddDF = pd.read_csv(f"{mainPath}/Synched Data/zed_sync.csv")
        data = [zeddDF,pdsDF,trakstarDF]
        if sw_included:
            sw = [swLADF,swLGDF,swRADF,swRGDF]
            for i in sw:
                data.append(i)
        for s in data:
            s = s.drop('OBS Stamps', axis= 1)
            finalDF = pd.concat([finalDF, s], axis= 1)

        finalDF.to_csv(finalPath, index=False)    