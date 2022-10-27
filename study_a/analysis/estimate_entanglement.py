import os
from pathlib import Path

import torch

import main
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from pathlib import Path
from tqdm import tqdm

files = Path('brick_experiments/videos').glob('*')
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))
checkpoint_path = os.path.join(ROOT_DIR, 'models', 'checkpoint_iter_370000.pth')
net = PoseEstimationWithMobileNet()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
load_state(net, checkpoint)
net = net.eval()

for file in tqdm(files, desc='File processing'):
    main.run(
        video=str(file),
        show_livestream=False,
        save_livestream=False,
        save_output_table=True,
        save_camera_input=False,
        synch_metric="2pax_90",
        cpu=True,
        net=net
    )
    os.rename(os.path.join(ROOT_DIR, 'output_table.csv'),
              os.path.join(ROOT_DIR,
                           f'brick_experiments/output_2pax90/output'
                           f'{Path(str(file)).stem}'))