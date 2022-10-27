import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from detector.pose_estimation import Pose, PoseEstimator

"""
This skript creates the ground truth labels for testing.
"""

dataDir = '..'
dataType = 'val2017'
annFile = "/Users/josephinevandelden/PycharmProjects/" \
          "entanglement_detector/testing/annotation.json"

# skeleton as in openpose
skeleton_keypoint_pairs = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
]

column_names = [
    f"synchrony_{Pose.kpt_names[keypoint_pair[0]]}"
    f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
    for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs]
kpt_names = [
    "nose",
    "neck",
    "r_sho",
    "r_elb",
    "r_wri",
    "l_sho",
    "l_elb",
    "l_wri",
    "r_hip",
    "r_knee",
    "r_ank",
    "l_hip",
    "l_knee",
    "l_ank",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
]
# index of keypoints as in openpose
kp_mapping = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

s = json.load(open(annFile, 'r'))
kp_names = s['categories'][0].get("keypoints")
pose_pairs = s['categories'][0].get("skeleton")
all_ids = []
for im in s['images']:
    all_ids.append(im['id'])

visible_kps = []
visible_bps = []
for ann in s['annotations']:
    image_id = ann['image_id']
    kps = ann['keypoints']
    kps_visibility = kps[2::3]
    kps_visibility = [1 if x == 2 else 0 for x in kps_visibility]
    kps_visibility_openpose_order = [kps_visibility[int(x)] if x != -1 else 0
                                     for x in kp_mapping]
    bp_vis = []
    for pairs in skeleton_keypoint_pairs:
        kp1_visible = kps_visibility_openpose_order[pairs[0]]
        kp2_visible = kps_visibility_openpose_order[pairs[1]]
        if kp1_visible == 0 or kp2_visible == 0:
            bp_vis.append(np.nan)
        else:
            bp_vis.append(1)
    visible_bps.append([image_id, bp_vis])

synch_labels = pd.DataFrame(visible_bps,
                            columns=['image_id', 'synch_ground_truth'])
pd.to_pickle(synch_labels, "labels.pkl")
