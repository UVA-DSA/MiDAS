import pandas as pd
import glob
from OBSconvert import OBSConvert
from PDSSync import PDSSync
from trakStarSync import trakStarSync

mainPath = 'post/data/*'
finalPath = f"{mainPath}/Synched Data/final_sync.csv"

if __name__ == "__main__":

    #TODO: Create Dir for Synchd Data

    OBSConvert(mainPath)
    PDSSync(mainPath)
    trakStarSync(mainPath)

    pds_df = pd.read_csv(f"{mainPath}/Synched Data/PDS_sync.csv")
    sw_df = pd.read_csv(f"{mainPath}/Synched Data/SW_sync.csv")
    trakstar_df = pd.read_csv(f"{mainPath}/Synched Data/trakstar_sync.csv")

    df = pd.DataFrame(columns=['OBS time', 'pds time', ])





