import importlib
import warnings

from matplotlib import pyplot as plt
from data_analysis_utilities import statistics_utilities as sh
import pandas as pd

importlib.reload(sh)
warnings.filterwarnings("ignore")

data_manager = sh.DataManager('ignacio_experiments/data_for_regression.pkl',
                                 'allpax','score')
data_manager.data =  pd.get_dummies(
    data_manager.data, prefix=['group'], columns=['group'], drop_first=True)
data_manager.data = data_manager.data.drop(
    columns=['id',
             'synchrony_r_hip_to_r_knee',
             'synchrony_l_hip_to_l_knee',
             'synchrony_r_knee_to_r_ank',
             'synchrony_l_knee_to_l_ank'])


reg_managers = []

data_manager.data.loc[:,'score'] = data_manager.data.loc[:,'score'].apply(
    pd.to_numeric)

data_manager.run()
transformed_dfs = [data_manager.avg_data,
                   data_manager.med_data,
                   data_manager.var_data,
                   data_manager.bin_data]



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
    corr, p_values = reg_manager.correlate()
    reg_manager.plot_cor_matrix(corr, p_values)
    plt.show()

for reg_manager in reg_managers:
    reg_manager.run()  # create possible model setups and fit models

print(len(reg_managers))
plotter = sh.OverviewPlotter(reg_managers)
plotter.plot_all()
best_model = plotter.plot_best_r2()
vif = plotter.get_vif(best_model)
print(vif)