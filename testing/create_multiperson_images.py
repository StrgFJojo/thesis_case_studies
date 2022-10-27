from pathlib import Path
import cv2
import numpy as np
import re

"""
This skript creates multi-person images from COCO single-person images.
"""

files_1person = Path('testing/images_1person').glob('*')
path_2persons = 'testing/images_2persons'
path_4persons = 'testing/images_4persons'
path_6persons = 'testing/images_6persons'
path_8persons = 'testing/images_8persons'
path_10persons = 'testing/images_10persons'

for img_file in files_1person:
    img = cv2.imread(f'testing/images_1person/{img_file.name}')
    if img.shape[0] < img.shape[1]:
        img_2stack = np.vstack([img, img])
    else:
        img_2stack = np.hstack([img, img])
    cv2.imwrite(f'testing/images_2persons/{img_file.name}', img_2stack)

    if img.shape[0] < img.shape[1]:
        img_4stack = np.hstack([img_2stack, img_2stack])
    else:
        img_4stack = np.vstack([img_2stack, img_2stack])
    cv2.imwrite(f'testing/images_4persons/{img_file.name}', img_4stack)

    if img.shape[0] < img.shape[1]:
        img_6stack = np.hstack([img_4stack, img_2stack])
    else:
        img_6stack = np.vstack([img_4stack, img_2stack])
    cv2.imwrite(f'testing/images_6persons/{img_file.name}', img_6stack)

    if img.shape[0] < img.shape[1]:
        img_8stack = np.hstack([img_4stack, img_4stack])
    else:
        img_8stack = np.vstack([img_4stack, img_4stack])
    cv2.imwrite(f'testing/images_8persons/{img_file.name}', img_8stack)

    if img.shape[0] < img.shape[1]:
        img_10stack = np.hstack([img_8stack, img_2stack])
    else:
        img_10stack = np.vstack([img_8stack, img_2stack])
    cv2.imwrite(f'testing/images_10persons/{img_file.name}', img_10stack)



