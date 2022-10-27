import os

import cv2
import pandas as pd
import torch
from tqdm import tqdm

import main
from detector import input_handling
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules import pose_estimation_openpose as pose_estimation
from modules.load_state import load_state

synch_style = '2pax_180_mirrored'
comp_identifiers = ['beijing2022', 'pyeongchang2018',
                    'sochi2014', 'vancouver2010', 'torino2006']

for comp in comp_identifiers:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                             '../..'))
    video_path = os.path.join(ROOT_DIR, 'olympics/full_replays',
                              f'{comp}_fullreplay.mp4')
    checkpoint_path = os.path.join(ROOT_DIR, 'models',
                                   'checkpoint_iter_370000.pth')
    scenes_annotated_path = os.path.join(
        ROOT_DIR, 'olympics/competition_performances_with_results',
        f'performances-results_{comp}.csv')
    scenes_annotated = pd.read_csv(scenes_annotated_path)
    synchrony_allscenes = []
    normalized_distances_allscenes = []

    # set up model
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    for index, row in tqdm(scenes_annotated.iterrows(), total=scenes_annotated
                           .shape[0],
                           desc='Progress overall video scenes', leave=False):
        vid = cv2.VideoCapture(video_path)
        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))

        # initialize the FourCC and a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_scene_path = os.path.join(ROOT_DIR, 'olympics/video_scenes',
                                        'output%d.avi' % index)
        vid_writer = cv2.VideoWriter(video_scene_path, fourcc, fps,
                                     (frame_width, frame_height))

        # create video snippet from frames
        for i in tqdm(range(row.frame_start, row.frame_end, 10), leave=False,
                      desc='Progress current video scene (frames)'):
            vid.set(1, i)
            ret, frame = vid.read()
            if ret:
                vid_writer.write(frame)
            else:
                # release vid and vid_writer if ret is false
                vid.release()
                # note when releasing
                print("Released Video Resources")
                # Closes all the frames
                cv2.destroyAllWindows()
                break
        vid_writer.release()
        frame_provider = input_handling.VideoReader(video_scene_path)
        df, synchrony_totalvideo, normalized_distances_totalvideo = main \
            .run(net, frame_provider, show_livestream=False,
                 save_livestream=False, save_outputs=True,
                 synch_style=synch_style)

        synchrony_allscenes.append(synchrony_totalvideo)
        normalized_distances_allscenes.append(normalized_distances_totalvideo)

    # create dataframe
    col_names = list(("synchrony_" + pose_estimation
                      .keypointsMapping[pose_estimation.POSE_PAIRS[t][0]]
                      + "_to_" +
                      pose_estimation.keypointsMapping[pose_estimation
                                                       .POSE_PAIRS[t][1]])
                     for t in range(len(pose_estimation.POSE_PAIRS)))
    col_names = col_names[:-2]

    final_df = scenes_annotated.copy()
    final_df = final_df.iloc[:, 1:]  # drop unnamed column
    final_df[col_names] = ''

    final_df['normalized_distance'] = pd.Series(normalized_distances_allscenes)
    for i in range(len(synchrony_allscenes)):
        for j in range(17):
            final_df.at[i, col_names[j]] = synchrony_allscenes[i].T[j]

    final_df.to_pickle(
        f'olympics/entanglement_per_competition/{synch_style}/'
        f'entanglement_{comp}.pkl')
