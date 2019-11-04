import os
import pickle
from subprocess import Popen, PIPE

from tqdm import tqdm

from config import data_file

filename = 'data/photo.csv'
folder = 'data/photo'

if __name__ == '__main__':
    with open(filename, 'r') as f:
        lines = f.readlines()

    samples = []
    for line in tqdm(lines[1:]):
        tokens = line.split(',')
        before_file = tokens[0].strip()
        after_file = tokens[1].strip()

        before_filename = before_file[before_file.rfind("/") + 1:]
        before_fullname = os.path.join(folder, before_filename)
        if not os.path.isfile(before_fullname):
            process = Popen(["wget", '-N', before_file, "-P", folder], stdout=PIPE)
            (output, err) = process.communicate()
            exit_code = process.wait()

        after_filename = after_file[after_file.rfind("/") + 1:]
        after_fullname = os.path.join(folder, after_filename)
        if not os.path.isfile(after_fullname):
            process = Popen(["wget", '-N', after_file, "-P", folder], stdout=PIPE)
            (output, err) = process.communicate()
            exit_code = process.wait()

        samples.append({'before': before_filename, 'after': after_filename})

        with open(data_file, 'wb') as f:
            pickle.dump(samples, f)
