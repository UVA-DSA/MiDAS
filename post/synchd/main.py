import pandas as pd
import glob
import os
from OBSconvert import OBSConvert
from PDSSync import PDSSync
from trakStarSync import trakStarSync

if __name__ == "__main__":
    mainPathCheck = './data/*'
    mainPath = glob.glob(mainPathCheck)[0]

    os.makedirs(f"{mainPath}/Synched Data/")
    finalPath = f"{mainPath}/Synched Data/final_sync.csv"

    OBSConvert(mainPath)
    PDSSync(mainPath)
    trakStarSync(mainPath)

    final_df = pd.read_csv(f"{mainPath}/Synched Data/final_sync.csv")
    pds_df = pd.read_csv(f"{mainPath}/Synched Data/PDS_sync.csv")
    #sw_df = pd.read_csv(f"{mainPath}/Synched Data/SW_sync.csv")
    trakstar_df = pd.read_csv(f"{mainPath}/Synched Data/trakstar_sync.csv")
    frames = [pds_df,trakstar_df]
    for fr in frames:
        fr = fr.drop('OBS Stamps', axis = 1)
        final_df = pd.concat([final_df, fr], axis = 1)
    final_df.to_csv(f'{mainPath}/Synched Data/final_sync.csv', index =False)






