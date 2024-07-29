import pandas as pd
import glob
from OBSconvert import OBSConvert
from PDSSync import PDSSync
from trakStarSync import trakstarSync

mainPath = 'post/data/*'
finalPath = f"{mainPath}/Synched Data/final_sync.csv"

if __name__ == "__main__":

    OBSConvert(mainPath)
    PDSSync(mainPath)
    trakstarSync(mainPath)

    pds_df = pd.read_csv(f"{mainPath}/Synched Data/PDS_sync.csv")
    sw_df = pd.read_csv(f"{mainPath}/Synched Data/SW_sync.csv")
    trakstar_df = pd.read_csv(f"{mainPath}/Synched Data/trakstar_sync.csv")

    df = pd.DataFrame(columns=['OBS time', 'pds time', ])





