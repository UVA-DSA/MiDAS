import pandas as pd
import glob
from OBSconvert import OBSConvert
from PDSSync import PDSSync

mainPath = 'post\data\*'
OBSConvert(mainPath)
PDSSync(mainPath)




