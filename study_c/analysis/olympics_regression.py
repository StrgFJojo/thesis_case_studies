import importlib
import pandas as pd
from data_analysis_utilities import statistics_utilities as sh

importlib.reload(sh)

data_manager_90 = sh.DataManager('olympics/dataframes/'
                                 'ts_data_2pax90.pkl',
                                 'synch_style_90', 'SPTSS')
data_manager_90mir = sh.DataManager('olympics/dataframes/'
                                    'ts_data_2pax90mirrored.pkl',
                                    'synch_style_90mirrored', 'SPTSS')
data_manager_180 = sh.DataManager('olympics/dataframes/'
                                  'ts_data_2pax180.pkl',
                                  'synch_style_180', 'SPTSS')
data_manager_180mir = sh.DataManager('olympics/dataframes/'
                                     'ts_data_2pax180mirrored.pkl',
                                     'synch_style_180mirrored', 'SPTSS')


data_managers = [data_manager_90,
                 data_manager_90mir,
                 data_manager_180,
                 data_manager_180mir]

reg_managers = []
for data_manager in data_managers:
    data_manager.data = pd.get_dummies(
        data_manager.data, prefix=['Season', 'Nat'],
        columns=['Season', 'Nat'], drop_first=True)
    data_manager.data = data_manager.data.drop(columns=['Team', '#', 'Event'])

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
        additional_data = data_manager.avg_data if data_type == 'bin' else None
        reg_managers.append(sh.RegModelManager(data=df,
                                               dv_name='SPTSS',
                                               data_type=data_type,
                                               data_name=data_manager.name,
                                               avg_data=additional_data
                                               ))

for reg_manager in reg_managers:
    reg_manager.run()  # create possible model setups and fit models

plotter = sh.OverviewPlotter(reg_managers)
plotter.plot_all()

best_model,_ = plotter.plot_best_r2()
vif = plotter.get_vif(best_model)
print(vif)

best_result = best_model  # 90, bin
latex_result = best_result.summary().as_latex()
latex_result_file = open("olympics/latex_result.txt", "wt")
latex_result_file.write(latex_result)
latex_result_file.close()