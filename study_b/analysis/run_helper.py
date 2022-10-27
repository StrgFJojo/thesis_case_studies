import argparse
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time

from detector import (
    distance_calculation,
    height_calculation,
    input_handling,
    output_creation,
    pose_estimation,
    synchrony_detection,
    visualization,
)
from detector.pose_estimation import PoseEstimator, Pose
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

"""
manual pairwise analysis
"""
warnings.filterwarnings("ignore")


def run(
        video="",
        show_livestream=True,
        save_livestream=False,
        save_output_table=False,
        save_camera_input=False,
        synch_metric="2pax_90",
        cpu=True,
        net="",
):
    print(f"Style is {synch_metric}")
    # Check arguments
    synch_styles_excl_int_params = (
        synchrony_detection.SynchronyDetector.synch_styles_excl_int_params
    )
    if video == "":
        raise ValueError("-video has to be provided")
    if net == "":
        raise ValueError("-checkpoint-path has to be provided")
    if synch_metric not in synch_styles_excl_int_params:
        try:
            int(synch_metric)
        except ValueError:
            print(
                f"{synch_metric}is not a valid input for argument synch_style"
            )
            exit(1)

    height_size = 256
    stride = 8
    upsample_ratio = 4
    if not cpu and torch.cuda.is_available():
        net = net.cuda()
    previous_poses = []
    track = 1
    smooth = 1
    pose_estimator = pose_estimation.PoseEstimator(
        net, height_size, stride, upsample_ratio, cpu
    )

    """
    # Setup entanglement detection
    synch_detector = synchrony_detection.SynchronyDetector(synch_metric)
    distance_calculator = distance_calculation.DistanceCalculator()
    height_calculator = height_calculation.HeightCalculator()
    """

    # TODO delete
    # Setup entanglement detection
    synch_detector_90 = synchrony_detection.SynchronyDetector('2pax_90')
    synch_detector_90m = synchrony_detection.SynchronyDetector(
        '2pax_90_mirrored')
    synch_detector_180 = synchrony_detection.SynchronyDetector('2pax_180')
    synch_detector_180m = synchrony_detection.SynchronyDetector(
        '2pax_180_mirrored')

    output_handler_leftpair_90 = output_creation.OutputHandler(
        output_type="table", file_name="output_table_leftpair_90.csv")
    output_handler_leftpair_90m = output_creation.OutputHandler(
        output_type="table", file_name="output_table_leftpair_90mir.csv")
    output_handler_leftpair_180 = output_creation.OutputHandler(
        output_type="table", file_name="output_table_leftpair_180.csv")
    output_handler_leftpair_180m = output_creation.OutputHandler(
        output_type="table", file_name="output_table_leftpair_180mir.csv")

    output_handler_rightpair_90 = output_creation.OutputHandler(
        output_type="table", file_name="output_table_rightpair_90.csv")
    output_handler_rightpair_90m = output_creation.OutputHandler(
        output_type="table", file_name="output_table_rightpair_90mir.csv")
    output_handler_rightpair_180 = output_creation.OutputHandler(
        output_type="table", file_name="output_table_rightpair_180.csv")
    output_handler_rightpair_180m = output_creation.OutputHandler(
        output_type="table", file_name="output_table_rightpair_180mir.csv")

    output_handler_opposingpair_90 = output_creation.OutputHandler(
        output_type="table", file_name="output_table_opposingpair_90.csv")
    output_handler_opposingpair_90m = output_creation.OutputHandler(
        output_type="table", file_name="output_table_opposingpair_90mir.csv")
    output_handler_opposingpair_180 = output_creation.OutputHandler(
        output_type="table", file_name="output_table_opposingpair_180.csv")
    output_handler_opposingpair_180m = output_creation.OutputHandler(
        output_type="table", file_name="output_table_opposingpair_180mir.csv")

    distance_calculator = distance_calculation.DistanceCalculator()
    #######

    # Setup output generation
    delay = 1
    if save_livestream:
        output_handler_video = output_creation.OutputHandler(
            output_type="video", file_name="output_video.avi"
        )
    if save_camera_input:
        output_handler_video_raw = output_creation.OutputHandler(
            output_type="video", file_name="input_video.avi"
        )
    if save_output_table:
        output_handler_table = output_creation.OutputHandler(
            output_type="table", file_name="output_table.csv"
        )

    # Setup visualization
    visualizer = visualization.Visualizer()

    # Profile execution time per frame
    sec_per_frame = []

    # Iterate over video frames
    frame_provider = input_handling.VideoReader(video)

    #########
    def frame_run(previous_poses, delay):
        start_time = time.time()
        # Attach input frame to output video
        if save_camera_input:
            output_handler_video_raw.build_outputs(img)

        # Estimate poses
        all_poses = pose_estimator.img_to_poses(img)

        # Track poses between frames
        if track:
            pose_estimation.track_poses(
                previous_poses, all_poses, smooth=smooth
            )
            previous_poses = all_poses

        # TODO delete for push/commit

        if len(all_poses) != 3:
            column_names = [
                f"synchrony_{Pose.kpt_names[keypoint_pair[0]]}"
                f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
                for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs
            ]
            synch_scores = np.nan * np.empty(17)
            synch_dict = dict(zip(column_names, synch_scores))
            normalized_distance = dict({'normalized_distance': -1})
            output_handler_leftpair_90.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_leftpair_90m.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_leftpair_180.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_leftpair_180m.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_rightpair_90.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_rightpair_90m.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_rightpair_180.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_rightpair_180m.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_opposingpair_90.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_opposingpair_90m.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_opposingpair_180.build_outputs(
                {**synch_dict, **normalized_distance}
            )
            output_handler_opposingpair_180m.build_outputs(
                {**synch_dict, **normalized_distance}
            )
        else:
            # order poses by seating position
            poses_sorted = sorted(all_poses,
                                  key=lambda pose: (pose.keypoints[0])[0])
            for idx1, pose1 in enumerate(poses_sorted):
                for idx2, pose2 in enumerate(poses_sorted):
                    if idx2 > idx1:
                        poses_paired = [pose1, pose2]
                        synchrony90 = synch_detector_90.calculate_synchrony(
                            poses_paired)
                        synchrony90m = synch_detector_90m.calculate_synchrony(
                            poses_paired)
                        synchrony180 = synch_detector_180.calculate_synchrony(
                            poses_paired)
                        synchrony180m = synch_detector_180m.calculate_synchrony(
                            poses_paired)
                        normalized_distance = dict({'normalized_distance': -1})

                        if idx1 == 0 and idx2 == 1:  # left pair
                            output_handler_leftpair_90.build_outputs(
                                {**synchrony90, **normalized_distance}
                            )
                            output_handler_leftpair_90m.build_outputs(
                                {**synchrony90m, **normalized_distance}
                            )
                            output_handler_leftpair_180.build_outputs(
                                {**synchrony180, **normalized_distance}
                            )
                            output_handler_leftpair_180m.build_outputs(
                                {**synchrony180m, **normalized_distance}
                            )

                        elif idx1 == 1 and idx2 == 2:  # right pair
                            output_handler_rightpair_90.build_outputs(
                                {**synchrony90, **normalized_distance}
                            )
                            output_handler_rightpair_90m.build_outputs(
                                {**synchrony90m, **normalized_distance}
                            )
                            output_handler_rightpair_180.build_outputs(
                                {**synchrony180, **normalized_distance}
                            )
                            output_handler_rightpair_180m.build_outputs(
                                {**synchrony180m, **normalized_distance}
                            )

                        elif idx1 == 0 and idx2 == 2:  # opposing pair
                            output_handler_opposingpair_90.build_outputs(
                                {**synchrony90, **normalized_distance}
                            )
                            output_handler_opposingpair_90m.build_outputs(
                                {**synchrony90m, **normalized_distance}
                            )
                            output_handler_opposingpair_180.build_outputs(
                                {**synchrony180, **normalized_distance}
                            )
                            output_handler_opposingpair_180m.build_outputs(
                                {**synchrony180m, **normalized_distance}
                            )

        return previous_poses

    #########
    for frame_idx, img in enumerate(
            tqdm(frame_provider, desc="Frame processing")
    ):
        # for non-webcam input, only process every 30th frame
        if video != "0" and frame_idx % 30 != 0:
            continue
        previous_poses = frame_run(previous_poses, delay)

    profiler = pd.DataFrame(sec_per_frame)
    profiler.to_pickle('profiler.pkl')
    # Iteration over frames finished (No frames left or keyboard interrupt)
    # Release resources
    """
    if save_livestream:
        output_handler_video.release_outputs()
    if save_camera_input:
        output_handler_video_raw.release_outputs()
    if save_output_table:
        output_handler_table.release_outputs()
    if show_livestream:
        cv2.destroyAllWindows()
    """
    output_handler_leftpair_90.release_outputs()
    output_handler_leftpair_90m.release_outputs()
    output_handler_leftpair_180.release_outputs()
    output_handler_leftpair_180m.release_outputs()
    output_handler_rightpair_90.release_outputs()
    output_handler_rightpair_90m.release_outputs()
    output_handler_rightpair_180.release_outputs()
    output_handler_rightpair_180m.release_outputs()
    output_handler_opposingpair_90.release_outputs()
    output_handler_opposingpair_90m.release_outputs()
    output_handler_opposingpair_180.release_outputs()
    output_handler_opposingpair_180m.release_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--video", type=str, default="", help="path to video file or camera id"
    )
    parser.add_argument(
        "--show-livestream", default=True, help="show detection on stream"
    )
    parser.add_argument(
        "--save-livestream", default=False, help="save illustrated input video"
    )
    parser.add_argument(
        "--save-output-table", default=False, help="save entanglement as csv"
    )
    parser.add_argument(
        "--save-camera-input", default=False, help="save input from camera"
    )
    parser.add_argument(
        "--synch-metric", default="2pax_90", help="synchrony metric to be used"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="run inference on cpu"
    )
    parser.add_argument("--checkpoint-path", default="", type=str)
    args = parser.parse_args()

    # Setup pose estimation with OpenPose Lightweight
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    load_state(net, checkpoint)
    net = net.eval()
    run(
        args.video,
        args.show_livestream,
        args.save_livestream,
        args.save_output_table,
        args.save_camera_input,
        args.synch_metric,
        args.cpu,
        net,
    )

    exit()
