import importlib
import pandas as pd
from data_analysis_utilities import statistics_utilities as sh
from matplotlib import pyplot as plt

importlib.reload(sh)

data_manager_90 = sh.DataManager('brick_experiments/data_for_regression/'
                                 'brick_experiments_2pax90.pkl',
                                  'synch_style_90','score')
data_manager_90mir = sh.DataManager('brick_experiments/data_for_regression/'
                                    'brick_experiments_2pax90mirrored.pkl',
                                     'synch_style_90mirrored','score')


data_managers = [
                 data_manager_90,
                 data_manager_90mir]

reg_managers = []
for data_manager in data_managers:
    data_manager.val_to_nan(-1)
    control_cols = [col for col in data_manager.data if col.startswith('avg')
                    or col.startswith('dif')]
    data_manager.data.loc[:,control_cols] \
        = data_manager.data.loc[:, control_cols].applymap(pd.to_numeric)
    data_manager.run()
    reg_managers.append(sh.RegModelManager(data=data_manager.med_data,
                                           dv_name='score',
                                           data_type='med',
                                           data_name=data_manager.name,
                                           avg_data=None
                                           ))
    descr = data_manager.med_data.describe().round(2).to_latex()
    latex_descr_file = open(f"brick_experiments/thesis_inputs/"
                            f"descr_{data_manager.name}.txt", "wt")
    latex_descr_file.write(descr)
    latex_descr_file.close()

#%%
for reg_manager in reg_managers:
    corr, p_values = reg_manager.correlate()
    reg_manager.plot_cor_matrix(corr, p_values)
    plt.savefig(f"brick_experiments/thesis_inputs/corr_"
                f"{reg_manager.data_name}.png", bbox_inches='tight')

for reg_manager in reg_managers:
    reg_manager.run()  # create possible model setups and fit models
    for est in reg_manager.estimations:
        latex_result = est.summary().as_latex()
        latex_result_file = open(f"brick_experiments/thesis_inputs/reg_"
                                 f"{reg_manager.data_name}.txt", "wt")
        latex_result_file.write(latex_result)
        latex_result_file.close()


#%%
for data_manager in data_managers:
    synch_cols = [col for col in data_manager.med_data if col.startswith(
        'synch') or col == 'normalized_distance']
    plt.figure()
    data_manager.med_data.boxplot(column=synch_cols,rot=90)
    plt.savefig(f"brick_experiments/thesis_inputs/boxplot_"
                f"{data_manager.name}.png", bbox_inches='tight')