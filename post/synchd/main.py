import pandas as pd
import glob
import os
from OBSconvert import OBSConvert
from PDSSync import PDSSync
from trakStarSync import trakStarSync
from swSync import swSync
from extract_zed_timestamps import main
from zeddSync import ZedSync

sw_included = False #TOGGLE FOR PEG TRANSFER DATA, IF NOT PEG TRANSFER, SHOULD BE FALSE

if __name__ == "__main__":
    matchingPath = glob.glob("D:/Data MIDAS/*")
    for mainPath in matchingPath:
        finalPath = f"{mainPath}/Synched Data/final_sync.csv"
        print(mainPath)
        try:
            os.mkdir(f"{mainPath}/Synched Data/")
            print(f"Directory created successfully.")
        except FileExistsError:
            print(f"Directory already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # OBSConvert(mainPath)
        # PDSSync(mainPath)
        # trakStarSync(mainPath)
        # if sw_included:
        #     swSync(mainPath)
        # main(mainPath)
        ZedSync(mainPath)

        finalDF = pd.read_csv(finalPath)
        # pdsDF = pd.read_csv(f"{mainPath}/Synched Data/PDS_sync.csv")
        # swLADF = pd.read_csv(f"{mainPath}/Synched Data/sw_L_acc.csv")
        # swLGDF = pd.read_csv(f"{mainPath}/Synched Data/sw_L_gyro.csv")
        # swRADF = pd.read_csv(f"{mainPath}/Synched Data/sw_R_acc.csv")
        # swRGDF = pd.read_csv(f"{mainPath}/Synched Data/sw_R_gyro.csv")
        # trakstarDF = pd.read_csv(f"{mainPath}/Synched Data/trakStar_sync.csv")
        zeddDF = pd.read_csv(f"{mainPath}/Synched Data/zedd_sync.csv")
#  pdsDF,trakstarDF,swLADF, swLGDF, swRADF, swRGDF
        data = [zeddDF]
        for s in data:
            s = s.drop('OBS Stamps', axis= 1)
            finalDF = pd.concat([finalDF, s], axis= 1)

        finalDF.to_csv(finalPath, index=False)    