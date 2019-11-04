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
            train.append({'img_path': before, 'label': 0})
            train.append({'img_path': after, 'label': 1})
        else:
            valid.append({'img_path': before, 'label': 0})
            valid.append({'img_path': after, 'label': 1})

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))

    with open('train.json', 'w') as file:
        json.dump(train, file, indent=4)

    with open('valid.json', 'w') as file:
        json.dump(valid, file, indent=4)
