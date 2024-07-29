import pandas as pd
import glob
from OBSconvert import OBSConvert
from PDSSync import PDSSync

mainPath = 'post\data\*'
finalPath = f"{mainPath}\Synched Data\final_sync.csv"
OBSConvert(mainPath)

PDSSync(mainPath)




