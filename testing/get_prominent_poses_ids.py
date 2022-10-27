import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from detector.pose_estimation import Pose, PoseEstimator

"""
This skript filters image IDs for testing under normal conditions.

Normal conditions:
- poses make up at least 25% of image area
- Less than half of pose keypoints are occluded
"""

dataDir = '..'
dataType = 'val2017'
annFile = "/Users/josephinevandelden/PycharmProjects/entanglement_detector/" \
          "testing/annotation.json"

annotation_json_info = json.load(open(annFile, 'r'))

image_infos = annotation_json_info['images']
annotation_infos = annotation_json_info['annotations']

annotation_infos_by_image_id = {}
for annotation_info in annotation_infos:
    image_id = annotation_info['image_id']
    if image_id in annotation_infos_by_image_id:
        annotation_infos_by_image_id[image_id].append(annotation_info)
    else:
        annotation_infos_by_image_id[image_id] = [annotation_info]

image_ids = list(annotation_infos_by_image_id.keys())

image_id_to_image_info = {}
for image_info in image_infos:
    image_id_to_image_info[image_info['id']] = image_info

single_person_image_ids = list(
    filter(lambda image_id: len(annotation_infos_by_image_id[image_id]) == 1,
           image_ids))

single_person_prominent_image_ids = []
for im, ann in zip(image_infos, annotation_infos):
    if ann['image_id'] != im['id']:
        print('IDs not matching!')
        exit(1)
    if ann['image_id'] not in single_person_image_ids:
        print('skip id')
        continue
    image_size = im['height'] * im['width']
    pose_size = ann['bbox'][2] * ann['bbox'][3]
    if pose_size / image_size >= 0.25:
        single_person_prominent_image_ids.append(ann['image_id'])

pd.to_pickle(single_person_prominent_image_ids,
             "single_person_prominent_image_ids.pkl")
