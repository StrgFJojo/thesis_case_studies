from detector.pose_estimation import Pose, PoseEstimator
from data_analysis_utilities import statistics_utilities as sh
import pandas as pd
from functools import reduce
column_names = [
    f"synchrony_{Pose.kpt_names[keypoint_pair[0]]}"
    f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
    for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs
]

identifier_groups = [['leftpair_90','rightpair_90','opposingpair_90'],
                     ['leftpair_90mir','rightpair_90mir','opposingpair_90mir'],
                     ['leftpair_180', 'rightpair_180', 'opposingpair_180'],
                     ['leftpair_180mir','rightpair_180mir',
                      'opposingpair_180mir']]

identifiers = ['leftpair_90', 'leftpair_90mir', 'leftpair_180',
               'leftpair_180mir',
               'rightpair_90', 'rightpair_90mir', 'rightpair_180',
               'rightpair_180mir',
               'opposingpair_90', 'opposingpair_90mir', 'opposingpair_180',
               'opposingpair_180mir']


reg_managers = []
data_managers=[]
merged_dfs = []
final_dfs = []

# keep median and variance separate, but merge over pairs

med_data_per_identifier = []

for type in identifiers:
    data_manager = sh.DataManager(f'ignacio_experiments/data_for_regression_'
                                  f'{type}.pkl', type, 'score')
    data_manager.data = pd.get_dummies(data_manager.data, prefix=['group'],
                                       columns=['group'], drop_first=True)
    data_manager.data = data_manager.data.drop(
        columns=['synchrony_r_hip_to_r_knee',
                 'synchrony_l_hip_to_l_knee',
                 'synchrony_r_knee_to_r_ank',
                 'synchrony_l_knee_to_l_ank'])
    data_manager.data.loc[:, 'score'] = data_manager.data.loc[:, 'score'].apply(
        pd.to_numeric)

    data_manager.run()

    med_data_per_identifier.append(data_manager.med_data.add_prefix(type+"_"))

synch_style_dfs = []
for group in identifier_groups:
    per_group_med_data = [med_data_per_identifier[identifiers.index(
        type)] for type in group]

    for med_data in per_group_med_data:
        score_cols = [col for col in med_data.columns if 'score' in col]
        group_cols = [col for col in med_data.columns if 'group' in col]
        id_cols = [col for col in med_data.columns if 'id' in col]
        med_data.rename(
            columns={score_cols[0]: 'score',
                     group_cols[0]: 'group_treatment',
                     id_cols[0]: 'id'},
            inplace=True)
    group_df = reduce(
        lambda x, y: pd.merge(x, y, on=['id', 'score', 'group_treatment']),
        per_group_med_data)
    synch_style_df = group_df[['score', 'group_treatment']].copy()
    for bodypart_synch in column_names:
        bodypart_synch_cols =\
            [col for col in group_df.columns if bodypart_synch in col]
        synch_style_df[bodypart_synch] = group_df[bodypart_synch_cols].mean(
            axis=1)
    synch_style_df = synch_style_df.drop(columns=['synchrony_r_hip_to_r_knee',
                                                  'synchrony_l_hip_to_l_knee',
                                                  'synchrony_r_knee_to_r_ank',
                                                  'synchrony_l_knee_to_l_ank'])
    reg_managers.append(sh.RegModelManager(data=synch_style_df,
                                           dv_name='score',
                                           data_type='med',
                                           data_name=group[0].split('_')[1],
                                           avg_data=None
                                           ))

for reg_manager in reg_managers:
    reg_manager.run()  # create possible model setups and fit models

print(len(reg_managers))
plotter = sh.OverviewPlotter(reg_managers)
plotter.plot_all()
print ('\n\nBEST MODEL')
best_model ,_= plotter.plot_best_r2()
vif = plotter.get_vif(best_model)
print(vif)