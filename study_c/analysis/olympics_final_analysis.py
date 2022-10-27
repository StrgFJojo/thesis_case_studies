import importlib
import pandas as pd
from data_analysis_utilities import statistics_utilities as sh
from matplotlib import pyplot as plt

importlib.reload(sh)

data_manager_90 = sh.DataManager('olympics/dataframes/'
                                  'ts_data_2pax90.pkl',
                                  'synch_style_90', 'SPTSS')
data_manager_90mir = sh.DataManager('olympics/dataframes/'
                                     'ts_data_2pax90mirrored.pkl',
                                     'synch_style_90mirrored', 'SPTSS')


data_managers = [
                 data_manager_90,
                 data_manager_90mir]

reg_managers = []
for data_manager in data_managers:
    nats = data_manager.data.Nat.unique()
    seasons = data_manager.data.Season.unique()
    data_manager.data = pd.get_dummies(data_manager.data,
                                       prefix=['Season', 'Nat'],
                                       columns=['Season', 'Nat'],
                                       drop_first=True)
    nat_cols = [col for col in data_manager.data.columns if
               col.startswith('Nat')]
    season_cols = [col for col in data_manager.data.columns if
                col.startswith('Season')]
    for nat in nats:
        is_dummy = False
        for dummy_nat in nat_cols:
            if nat in dummy_nat:
                is_dummy = True
        if is_dummy == False:
            print('reference case for nat:', nat) # AT
            break
    for sea in seasons:
        is_dummy = False
        for dummy_sea in season_cols:
            if sea in dummy_sea:
                is_dummy = True
        if is_dummy == False:
            print('reference case for season:', sea) # 05/06
            break
    data_manager.data = data_manager.data.drop(columns=['Team', '#', 'Event'])
    data_manager.run()
    reg_managers.append(sh.RegModelManager(data=data_manager.med_data,
                                               dv_name='SPTSS',
                                               data_type='med',
                                               data_name=data_manager.name,
                                               avg_data=None
                                               ))
    drop_cols = [col for col in data_manager.med_data if (
                ('synch' not in col) and (col != 'normalized_distance') and (
                    col != data_manager.dv_name))]
    descr_data = data_manager.med_data.copy().drop(columns=drop_cols)
    descr = descr_data.describe().round(2).to_latex()

#%%
for reg_manager in reg_managers:
    corr, p_values = reg_manager.correlate()
    reg_manager.plot_cor_matrix(corr, p_values)
    plt.savefig(f"olympics/thesis_inputs/corr_{reg_manager.data_name}.png",
                bbox_inches='tight')

for reg_manager in reg_managers:
    reg_manager.run()  # create possible model setups and fit models
    for est in reg_manager.estimations:
        latex_result = est.summary().as_latex()
        latex_result_file = open(
            f"olympics/thesis_inputs/reg_{reg_manager.data_name}.txt", "wt")
        latex_result_file.write(latex_result)
        latex_result_file.close()

#%%
for data_manager in data_managers:
    synch_cols = [col for col in data_manager.med_data if col.startswith(
        'synch')]
    plt.figure()
    data_manager.med_data.boxplot(column=synch_cols,rot=90)
    plt.savefig(f"olympics/thesis_inputs/boxplot_{data_manager.name}.png",
                bbox_inches='tight')