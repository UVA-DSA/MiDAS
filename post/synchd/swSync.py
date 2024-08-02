import pandas as pd
import glob
from swCSVSynthesis import swCSVSynth

def swSync(path):

    swCSVSynth(path)
    obsPath = f"{path}/OBS*.csv"
    matchingPath = glob.glob(obsPath)
    obsDF = pd.read_csv(matchingPath[0], index_col=False)
    obsStampList = obsDF.iloc[:,0].tolist()


    original_LA_DF = pd.read_csv(f'{path}/smartwatch_data/sw_left/sw_data_acc.csv')
    original_LG_DF = pd.read_csv(f'{path}/smartwatch_data/sw_left/sw_data_gyro.csv')

    original_RA_DF = pd.read_csv(f'{path}/smartwatch_data/sw_right/sw_data_acc.csv')
    original_RG_DF = pd.read_csv(f'{path}/smartwatch_data/sw_right/sw_data_gyro.csv')

    final_LA_DF = pd.read_csv(f'{path}/Synched Data/sw_L_acc.csv')
    final_LG_DF = pd.read_csv(f'{path}/Synched Data/sw_L_gyro.csv')
    
    final_RA_DF = pd.read_csv(f'{path}/Synched Data/sw_R_acc.csv')
    final_RG_DF = pd.read_csv(f'{path}/Synched Data/sw_R_gyro.csv')

    files = [[original_LA_DF, final_LA_DF], [original_LG_DF, final_LG_DF], [original_RA_DF, final_RA_DF], [original_RG_DF, final_RG_DF]]
    zero_row = pd.DataFrame([[0]*7], columns=original_LA_DF.columns)


    for file in files:
        swStampList = file[0].iloc[:,0].tolist()
        iterator = 0
        matched = pd.DataFrame()
        for i in obsStampList:
            if iterator % 1000 < 5:
                print(f"sw:{iterator}")
            while iterator < len(file[0]) - 1:
                try:
                    if i >= swStampList[iterator] and i < swStampList[iterator+1]:
                        matched = pd.concat([matched, file[0].iloc[[iterator]]], ignore_index=True)
                        break
                    if i < swStampList[iterator]:
                        matched = pd.concat([matched, zero_row])
                        break
                except Exception as e:
                    print(f"swSync Error: {e}")
                    break
                iterator+=1
        file[1] = pd.concat([file[1], matched], axis = 1)
        
    files[0][1].to_csv(f'{path}/Synched Data/sw_L_acc.csv', index =False)
    files[1][1].to_csv(f'{path}/Synched Data/sw_L_gyro.csv', index = False)
    files[2][1].to_csv(f'{path}/Synched Data/sw_R_acc.csv', index = False)
    files[3][1].to_csv(f'{path}/Synched Data/sw_R_gyro.csv', index = False)


if __name__ == "__main__":
    path = "./data/PegT_S221_T1_2024-07-19"
    swSync(path)