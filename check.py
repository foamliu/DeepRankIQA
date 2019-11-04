import os
import pickle

import cv2 as cv
from tqdm import tqdm

from config import data_file, image_folder

if __name__ == "__main__":
    with open(data_file, 'rb') as f:
        samples = pickle.load(f)

    filenames = set()

    for sample in tqdm(samples):
        before = sample['before']
        fullpath = os.path.join(image_folder, before)
        img = cv.imread(fullpath)
        assert (img is not None)
        filenames.add(before)

        after = sample['after']
        fullpath = os.path.join(image_folder, before)
        img = cv.imread(fullpath)
        assert (img is not None)
        filenames.add(after)

    num_samples = len(list(filenames))
    print('num_samples: ' + str(num_samples))
