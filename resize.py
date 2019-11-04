import os
import pickle

import cv2 as cv
from tqdm import tqdm

from config import data_file, image_folder
from utils import ensure_folder


def resize(filename):
    old_path = os.path.join(image_folder, filename)
    img = cv.imread(old_path)
    new_path = os.path.join(new_folder, filename)
    img = cv.resize(img, (im_size, im_size))
    cv.imwrite(new_path, img)


if __name__ == "__main__":
    im_size = 256
    new_folder = 'data/256x256'

    ensure_folder(new_folder)

    with open(data_file, 'rb') as f:
        samples = pickle.load(f)

    for sample in tqdm(samples):
        before = sample['before']
        resize(before)
        after = sample['after']
        resize(after)
