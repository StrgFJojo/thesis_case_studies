import os
from pathlib import Path

import torch

import run_helper
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from pathlib import Path
from tqdm import tqdm

files = Path('/Users/josephinevandelden/Desktop/ignacio_experiments').glob('*')
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))
checkpoint_path = os.path.join(ROOT_DIR, 'models', 'checkpoint_iter_370000.pth')
net = PoseEstimationWithMobileNet()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
load_state(net, checkpoint)
net = net.eval()

for file in tqdm(files, desc='File processing'):
    run_helper.run(
        video=str(file),
        show_livestream=False,
        save_livestream=False,
        save_output_table=True,
        save_camera_input=False,
        synch_metric="allpax",
        cpu=True,
        net=net
    )

    # Run pairs separately and averages in hindsight to enable
    # proper sanity checks

    os.rename(os.path.join(ROOT_DIR, 'output_table_leftpair_90.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_leftpair_90_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_leftpair_90mir.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_leftpair_90mir_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_leftpair_180.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_leftpair_180_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_leftpair_180mir.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_leftpair_180mir_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_rightpair_90.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_rightpair_90_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_rightpair_90mir.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_rightpair_90mir_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_rightpair_180.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_rightpair_180_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_rightpair_180mir.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_rightpair_180mir_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_opposingpair_90.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_opposingpair_90_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_opposingpair_90mir.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_opposingpair_90mir_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_opposingpair_180.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_opposingpair_180_'
                           f'{Path(str(file)).stem}.csv'))

    os.rename(os.path.join(ROOT_DIR, 'output_table_opposingpair_180mir.csv'),
              os.path.join(ROOT_DIR,
                           f'ignacio_experiments/ts_data/'
                           f'entanglement_opposingpair_180mir_'
                           f'{Path(str(file)).stem}.csv'))