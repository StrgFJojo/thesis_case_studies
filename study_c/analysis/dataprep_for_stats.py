# %% Imports
import flag
import numpy as np
import pandas as pd

from olympics.oly_modules import data_optimization as do

# %% Setup dataframes
synch_names = [
    'synchrony_Neck_to_R-Sho', 'synchrony_Neck_to_L-Sho',
    'synchrony_R-Sho_to_R-Elb', 'synchrony_R-Elb_to_R-Wr',
    'synchrony_L-Sho_to_L-Elb', 'synchrony_L-Elb_to_L-Wr',
    'synchrony_Neck_to_R-Hip', 'synchrony_R-Hip_to_R-Knee',
    'synchrony_R-Knee_to_R-Ank', 'synchrony_Neck_to_L-Hip',
    'synchrony_L-Hip_to_L-Knee', 'synchrony_L-Knee_to_L-Ank',
    'synchrony_Neck_to_Nose', 'synchrony_Nose_to_R-Eye',
    'synchrony_R-Eye_to_R-Ear', 'synchrony_Nose_to_L-Eye',
    'synchrony_L-Eye_to_L-Ear', 'normalized_distance']

data2022 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90(per_scene)/"
                          "per_scene_entanglement_beijing2022.pkl")
data2018 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90(per_scene)/"
                          "per_scene_entanglement_pyeongchang2018.pkl")
data2014 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90(per_scene)/"
                          "per_scene_entanglement_sochi2014.pkl")
data2010 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90(per_scene)/"
                          "per_scene_entanglement_vancouver2010.pkl")
data2006 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90(per_scene)/"
                          "per_scene_entanglement_torino2006.pkl")

data2018 = do.subsample_arr(data2018)
data2014 = do.subsample_arr(data2014)
data2010 = do.subsample_arr(data2010)

data = pd.concat([data2022, data2018, data2014, data2010, data2006])
data.Nat = data.Nat.map(flag.dflagize)
print(f'Created scene data frame. Total of {data.shape[0]} scenes'
      f' over {len(data.Season.unique())} events found.')
data = data.drop(['scene_id', 'ts_start', 'ts_end', 'frame_start',
                  'frame_end', 'FSTSS', 'TotalScore', ], axis=1)
data.iloc[:, 6:] = data.iloc[:, 6:].applymap(do.make_nan)
data = data.groupby(['Team', '#', 'Nat', 'SPTSS', 'Season', 'Event'])[
    synch_names].agg(np.hstack).reset_index()
data_2pax90 = data

data2022 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90_mirrored/entanglement_beijing2022.pkl")
data2018 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90_mirrored/entanglement_pyeongchang2018.pkl")
data2014 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90_mirrored/entanglement_sochi2014.pkl")
data2010 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90_mirrored/entanglement_vancouver2010.pkl")
data2006 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_90_mirrored/entanglement_torino2006.pkl")

data = pd.concat([data2022, data2018, data2014, data2010, data2006])
data.Nat = data.Nat.map(flag.dflagize)
data = data.drop(['ts_start', 'ts_end', 'frame_start', 'frame_end',
                  'FSTSS', 'TotalScore'], axis=1)
data_2pax90mirrored = data

data2022 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180/entanglement_beijing2022.pkl")
data2018 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180/entanglement_pyeongchang2018.pkl")
data2014 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180/entanglement_sochi2014.pkl")
data2010 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180/entanglement_vancouver2010.pkl")
data2006 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180/entanglement_torino2006.pkl")

data = pd.concat([data2022, data2018, data2014, data2010, data2006])
data.Nat = data.Nat.map(flag.dflagize)
data = data.drop(['ts_start', 'ts_end', 'frame_start', 'frame_end', 'FSTSS',
                  'TotalScore'], axis=1)
data_2pax180 = data

data2022 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180_mirrored/entanglement_beijing2022.pkl")
data2018 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180_mirrored/entanglement_pyeongchang2018.pkl")
data2014 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180_mirrored/entanglement_sochi2014.pkl")
data2010 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180_mirrored/entanglement_vancouver2010.pkl")
data2006 = pd.read_pickle("olympics/entanglement_per_competition/"
                          "2pax_180_mirrored/entanglement_torino2006.pkl")

data = pd.concat([data2022, data2018, data2014, data2010, data2006])
data.Nat = data.Nat.map(flag.dflagize)
data = data.drop(['ts_start', 'ts_end', 'frame_start', 'frame_end', 'FSTSS',
                  'TotalScore'], axis=1)
data_2pax180mirrored = data

data_2pax90.iloc[:, 6:] = data_2pax90.iloc[:, 6:].applymap(do.make_nan)
data_2pax90mirrored.iloc[:, 6:] = data_2pax90mirrored.iloc[:, 6:]\
    .applymap(do.make_nan)
data_2pax180.iloc[:, 6:] = data_2pax180.iloc[:, 6:]\
    .applymap(do.make_nan)
data_2pax180mirrored.iloc[:, 6:] = data_2pax180mirrored.iloc[:, 6:]\
    .applymap(do.make_nan)

data_2pax90.to_pickle('olympics/dataframes/'
                      'ts_data_2pax90.pkl')
data_2pax90mirrored.to_pickle('olympics/dataframes/'
                              'ts_data_2pax90mirrored.pkl')
data_2pax180.to_pickle('olympics/dataframes/'
                       'ts_data_2pax180.pkl')
data_2pax180mirrored.to_pickle('olympics/dataframes/'
                               'ts_data_2pax180mirrored.pkl')

var_2pax90 = data_2pax90.copy()
var_2pax90.iloc[:, 6:] = var_2pax90.iloc[:, 6:]\
    .applymap(np.nanvar)

var_2pax90mirrored = data_2pax90mirrored.copy()
var_2pax90mirrored.iloc[:, 6:] = var_2pax90mirrored.iloc[:, 6:]\
    .applymap(np.nanvar)

var_2pax180 = data_2pax180.copy()
var_2pax180.iloc[:, 6:] = var_2pax180.iloc[:, 6:]\
    .applymap(np.nanvar)

var_2pax180mirrored = data_2pax180mirrored.copy()
var_2pax180mirrored.iloc[:, 6:] = var_2pax180mirrored.iloc[:, 6:]\
    .applymap(np.nanvar)

var_2pax90.to_pickle('olympics/dataframes'
                     '/var_data_2pax90.pkl')
var_2pax90mirrored.to_pickle('olympics/dataframes'
                             '/var_data_2pax90mirrored.pkl')
var_2pax180.to_pickle('olympics/dataframes'
                      '/var_data_2pax180.pkl')
var_2pax180mirrored.to_pickle('olympics/dataframes'
                              '/var_data_2pax180mirrored.pkl')

avg_2pax90 = data_2pax90.copy()
avg_2pax90.iloc[:, 6:] = avg_2pax90.iloc[:, 6:].applymap(np.nanmean)

avg_2pax90mirrored = data_2pax90mirrored.copy()
avg_2pax90mirrored.iloc[:, 6:] = avg_2pax90mirrored.iloc[:, 6:]\
    .applymap(np.nanmean)

avg_2pax180 = data_2pax180.copy()
avg_2pax180.iloc[:, 6:] = avg_2pax180.iloc[:, 6:].applymap(np.nanmean)

avg_2pax180mirrored = data_2pax180mirrored.copy()
avg_2pax180mirrored.iloc[:, 6:] = avg_2pax180mirrored.iloc[:, 6:]\
    .applymap(np.nanmean)

avg_2pax90.to_pickle('olympics/dataframes'
                     '/avg_data_2pax90.pkl')
avg_2pax90mirrored.to_pickle('olympics/dataframes'
                             '/avg_data_2pax90mirrored.pkl')
avg_2pax180.to_pickle('olympics/dataframes'
                      '/avg_data_2pax180.pkl')
avg_2pax180mirrored.to_pickle('olympics/dataframes'
                              '/avg_data_2pax180mirrored.pkl')
