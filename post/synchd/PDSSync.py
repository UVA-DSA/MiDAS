import pandas as pd

def PDSSync(path):
    pds = pd.read_csv(f'{path}\PDS.csv')
    obs = pd.read_csv(f"{path}\obs\OBStimestamp_PegT_S221_T1_2024-07-19.csv")
    new = pd.DataFrame(columns=pds.columns)

    pds_stamps = pds['timestamp'].tolist()
    obs_stamps = obs.iloc[:,0].tolist()
    iterator = 0
    for i in obs_stamps:
        print(i)
        while True:
            try:
                if i >= pds_stamps[iterator] and i < pds_stamps[iterator+1]:
                    new = pd.concat([new, pds.iloc[[iterator]]], ignore_index=True)
                    break
            except Exception:
                break
            iterator+=1

    new['OBS Stamps'] = obs['Epoch']


    new.to_csv('post\data\PegT_S221_T1_2024-07-19\\new_synced.csv' , index=False)
