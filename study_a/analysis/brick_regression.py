import importlib
from data_analysis_utilities import statistics_utilities as sh
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

importlib.reload(sh)

data_manager_90 = sh.DataManager('brick_experiments/data_for_regression/'
                                 'brick_experiments_2pax90.pkl',
                                 'synch_style_90','score')
data_manager_90mir = sh.DataManager('brick_experiments/data_for_regression/'
                                    'brick_experiments_2pax90mirrored.pkl',
                                    'synch_style_90mirrored','score')
data_manager_180 = sh.DataManager('brick_experiments/data_for_regression/'
                                  'brick_experiments_2pax180.pkl',
                                  'synch_style_180','score')
data_manager_180mir = sh.DataManager('brick_experiments/data_for_regression/'
                                     'brick_experiments_2pax180mirrored.pkl',
                                     'synch_style_180mirrored','score')

data_managers = [data_manager_90,
                 data_manager_90mir,
                 data_manager_180,
                 data_manager_180mir]

reg_managers = []
for data_manager in data_managers:
    data_manager.val_to_nan(-1)
    control_cols = [col for col in data_manager.data if col.startswith('avg') or
                    col.startswith('dif')]
    dif_cols = [col for col in data_manager.data if col.startswith('dif')]
    #drop = [col for col in data_manager.data if 'eye' in col]
    data_manager.data.loc[:,control_cols] = \
        data_manager.data.loc[:, control_cols].applymap(pd.to_numeric)
    #data_manager.data = data_manager.data.drop(columns=drop)
    data_manager.run()
    transformed_dfs = [data_manager.med_data]

    for idx, df in enumerate(transformed_dfs):
        data_type = 'avg' if idx == 0 \
            else 'med' if idx == 1 \
            else 'var' if idx == 2 \
            else 'bin'
        additional_data = data_manager.med_data if data_type == 'bin' else None
        reg_managers.append(sh.RegModelManager(data=df,
                                               dv_name='score',
                                               data_type=data_type,
                                               data_name=data_manager.name,
                                               avg_data=additional_data
                                               ))
for reg_manager in reg_managers:
    reg_manager.run()  # create possible model setups and fit models

plotter = sh.OverviewPlotter(reg_managers)
plotter.plot_all()
print('Best model:')
best_model, best_model_reg_manager = plotter.plot_best_r2()
vif = plotter.get_vif(best_model)
print(vif)