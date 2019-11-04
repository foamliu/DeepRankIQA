import json
import pickle

from config import num_train, data_file

if __name__ == "__main__":
    with open(data_file, 'rb') as f:
        samples = pickle.load(f)

    train = []
    valid = []

    for i, sample in enumerate(samples):
        before = sample['before']
        after = sample['after']
        if i < num_train:
            train.append({'before': before, 'after': after})
        else:
            valid.append({'before': before, 'after': after})

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))

    with open('train.json', 'w') as file:
        json.dump(train, file, indent=4)

    with open('valid.json', 'w') as file:
        json.dump(valid, file, indent=4)
