from pathlib import Path
from detector.pose_estimation import Pose, PoseEstimator
import pandas as pd

column_names = [
    f"synchrony_{Pose.kpt_names[keypoint_pair[0]]}"
    f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
    for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs
]
scores = pd.read_csv('ignacio_experiments/team_metrics.csv', sep=',')
scores.rename(
    columns={'Team': 'id', 'Group': 'group', 'Score': 'score'},inplace=True)
files = Path('ignacio_experiments/ts_data').glob('*')
final_df = []
for file in files:
    id = Path(str(file)).stem.replace('entanglement_exp', '')
    if id == '.DS_Store':
        continue
    id = float(id)
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
final_df['id'] = pd.to_numeric(final_df['id'])
final_df = final_df.merge(scores, on='id', how="left")
#final_df.drop('id', axis=1, inplace=True)
final_df.to_pickle(f'ignacio_experiments/data_for_regression.pkl')
