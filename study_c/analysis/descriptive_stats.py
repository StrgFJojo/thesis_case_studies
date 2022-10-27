import numpy as np
import pandas as pd

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
# %% From timeseries to predicors: average synch value of performance;
# variance of synch over performance

# Create IVs as timeseries
ts_2pax90 = pd.read_pickle('olympics/dataframes/ts_data_2pax90.pkl')
ts_2pax90mirrored = pd.read_pickle('olympics/dataframes/'
                                   'ts_data_2pax90mirrored.pkl')
ts_2pax180 = pd.read_pickle('olympics/dataframes/ts_data_2pax180.pkl')
ts_2pax180mirrored = pd.read_pickle('olympics/dataframes/'
                                    'ts_data_2pax180mirrored.pkl')

# Create IVs as avg over timeseries
avg_2pax90 = ts_2pax90.copy()
avg_2pax90.iloc[:, 6:] = avg_2pax90.iloc[:, 6:].applymap(np.nanmean)

avg_2pax90mirrored = ts_2pax90mirrored.copy()
avg_2pax90mirrored.iloc[:, 6:] = avg_2pax90mirrored.iloc[:, 6:]\
    .applymap(np.nanmean)

avg_2pax180 = ts_2pax180.copy()
avg_2pax180.iloc[:, 6:] = avg_2pax180.iloc[:, 6:].applymap(np.nanmean)

avg_2pax180mirrored = ts_2pax180mirrored.copy()
avg_2pax180mirrored.iloc[:, 6:] = avg_2pax180mirrored.iloc[:, 6:]\
    .applymap(np.nanmean)

# Create IVs as var over timeseries

var_2pax90 = ts_2pax90.copy()
var_2pax90.iloc[:, 6:] = var_2pax90.iloc[:, 6:].applymap(np.nanvar)

var_2pax90mirrored = ts_2pax90mirrored.copy()
var_2pax90mirrored.iloc[:, 6:] = var_2pax90mirrored.iloc[:, 6:]\
    .applymap(np.nanvar)

var_2pax180 = ts_2pax180.copy()
var_2pax180.iloc[:, 6:] = var_2pax180.iloc[:, 6:].applymap(np.nanvar)

var_2pax180mirrored = ts_2pax180mirrored.copy()
var_2pax180mirrored.iloc[:, 6:] = var_2pax180mirrored.iloc[:, 6:]\
    .applymap(np.nanvar)

# %% Location statistics

# synch as avg of timeseries

desc_avg2pax90 = avg_2pax90.loc[:, avg_2pax90.columns != '#'].describe()\
    .applymap(lambda x: f"{x:0.2f}")
desc_avg2pax90.columns = desc_avg2pax90.columns.str\
    .replace("synchrony_", "avg_synch_")

desc_avg2pax90mirrored = avg_2pax90mirrored\
    .loc[:, avg_2pax90mirrored.columns != '#']\
    .describe().applymap(lambda x: f"{x:0.2f}")
desc_avg2pax90mirrored.columns = desc_avg2pax90mirrored.columns\
    .str.replace("synchrony_", "avg_synch_")

desc_avg2pax180 = avg_2pax180\
    .loc[:, avg_2pax180.columns != '#']\
    .describe().applymap(lambda x: f"{x:0.2f}")
desc_avg2pax180.columns = desc_avg2pax180.columns.str\
    .replace("synchrony_", "avg_synch_")

desc_avg2pax180mirrored = avg_2pax180mirrored\
    .loc[:, avg_2pax180mirrored.columns != '#']\
    .describe().applymap(lambda x: f"{x:0.2f}")
desc_avg2pax180mirrored.columns = desc_avg2pax180mirrored\
    .columns.str.replace("synchrony_", "avg_synch_")

# synch as variance of timeseries

desc_var2pax90 = var_2pax90.loc[:, var_2pax90.columns != '#']\
    .describe().applymap(lambda x: f"{x:0.2f}")
desc_var2pax90.columns = desc_var2pax90.columns.str\
    .replace("synchrony_", "synch_var_")

desc_var2pax90mirrored = var_2pax90mirrored\
    .loc[:, var_2pax90mirrored.columns != '#']\
    .describe().applymap(lambda x: f"{x:0.2f}")
desc_var2pax90mirrored.columns = desc_var2pax90mirrored.columns.str\
    .replace("synchrony_", "synch_var_")

desc_var2pax180 = var_2pax180.loc[:, var_2pax180.columns != '#']\
    .describe().applymap(lambda x: f"{x:0.2f}")
desc_var2pax180.columns = desc_var2pax180.columns.str\
    .replace("synchrony_", "synch_var_")

desc_var2pax180mirrored = var_2pax180mirrored\
    .loc[:, var_2pax180mirrored.columns != '#']\
    .describe().applymap(lambda x: f"{x:0.2f}")
desc_var2pax180mirrored.columns = desc_var2pax180mirrored.columns.str\
    .replace("synchrony_", "synch_var_")
