import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import main
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from detector.pose_estimation import Pose, PoseEstimator

"""
This skript runs the detector that yields estimated synchronization scores.
"""

single_person_prominent_image_ids = pd.read_pickle(
    "testing/single_person_prominent_image_ids.pkl")
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
checkpoint_path = os.path.join(ROOT_DIR, 'models', 'checkpoint_iter_370000.pth')
net = PoseEstimationWithMobileNet()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
load_state(net, checkpoint)
net = net.eval()

labels = pd.read_pickle('testing/labels.pkl')
all_sq_errors = []
for index, row in tqdm(labels.iterrows()):

    # only if at least half body is visible
    if (sum(np.isnan(x) for x in row['synch_ground_truth']) < 13) or (
            row['image_id'] not in single_person_prominent_image_ids):
        continue

    img_sq_errors = [row['image_id']]

    if np.isnan(row['synch_ground_truth']).all():
        continue

    truth = row['synch_ground_truth']

    for duplicate_count in ['2', '4', '6', '8', '10']:
        img_file_name = f"testing/images_{duplicate_count}persons/" \
                        f"{str(row['image_id']).zfill(12)}.jpg"
        main.run(
            video=img_file_name,
            show_livestream=False,
            save_livestream=False,
            save_output_table=True,
            save_camera_input=False,
            synch_metric="allpax",
            cpu=True,
            net=net
        )

        results_path = os.path.join(ROOT_DIR, 'output_table.csv')

        results = pd.read_csv(results_path)
        results = results.values.tolist()
        results = results[0][0:-1]

        sq_errors = []

        for result_val, truth_val in zip(results, truth):
            if np.isnan([result_val, truth_val]).all():
                sq_errors.append('TN')
            elif np.isnan(result_val):
                sq_errors.append('FN')
            elif np.isnan(truth_val):
                sq_errors.append('FP')
            else:
                sq_errors.append((result_val - truth_val) ** 2)

        img_sq_errors.append(sq_errors)
    all_sq_errors.append(img_sq_errors)

column_names = [
    f"synchrony_{Pose.kpt_names[keypoint_pair[0]]}"
    f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
    for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs]

column_names_person_num = []
for person_num in ['2', '4', '6', '8', '10']:
    column_names_person_num.append(f'sq_errors_{person_num}persons')

column_names_person_num.insert(0, 'image_id')
df = pd.DataFrame(all_sq_errors, columns=column_names_person_num)

split_df = pd.DataFrame(df.iloc[:, 0].copy())
split_df = split_df.rename(columns={split_df.columns[0]: 'image_id'})

for col in df.columns:
    if col == 'image_id':
        continue
    col_names = [col_name + '_' + col for col_name in column_names]
    split_df = pd.concat(
        [split_df, pd.DataFrame(df[col].tolist(), columns=col_names)], axis=1)

split_df.to_csv('testing/test_squared_errors_prominent_full_body.csv')
