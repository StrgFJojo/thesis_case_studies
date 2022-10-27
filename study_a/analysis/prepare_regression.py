from pathlib import Path
from detector.pose_estimation import Pose, PoseEstimator
import pandas as pd


column_names = [
            f"synchrony_{Pose.kpt_names[keypoint_pair[0]]}"
            f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
            for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs
        ]
column_names.append('normalized_distance')
scores = pd.read_csv('brick_experiments/brick_experiment_scores.csv',
                     sep=';')
controls = pd.read_csv('brick_experiments/brick_experiments_controls.csv',
                       sep=',')
controls.replace('na', '', inplace=True)

for style in ['2pax90', '2pax90mirrored', '2pax180', '2pax180mirrored']:
    files = Path(f'brick_experiments/output_{style}').glob('*')
    final_df = []
    for file in files:
        id = Path(str(file)).stem.replace('output', '')
        df = pd.read_csv(file)
        ts_list = []
        for i in range((df.shape[1])):
            cur_row = []
            for j in range(df.shape[0]):
                cur_row.append(df.iat[j, i])
            ts_list.append(cur_row)
        curr_dict = dict(zip(column_names, ts_list))
        curr_dict['id'] = id
        final_df.append(curr_dict)
    final_df = pd.DataFrame(final_df)
    final_df = final_df.merge(scores, on='id', how="left")
    final_df = final_df.merge(controls, on='id', how="left")
    final_df.drop('id', axis=1, inplace=True)
    final_df.to_pickle(f'brick_experiments/data_for_regression/'
                       f'brick_experiments_{style}.pkl')