import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

from config import num_workers
from data_gen import DeepIQADataset
from utils import parse_args, AverageMeter, accuracy

classes = np.array(['Lou Yu', 'Kuang Jia'])


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def test():
    print('device: cpu')

    # checkpoint = 'BEST_checkpoint.tar'
    # checkpoint = torch.load(checkpoint, map_location='cpu')
    # model = checkpoint['model'].module
    scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
    model = torch.jit.load(scripted_quantized_model_file)
    model = model.to('cpu')
    model.eval()

    args = parse_args()
    valid_dataset = DeepIQADataset('valid')
    num_samples = len(valid_dataset)
    print('num_samples: {}'.format(num_samples))

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=num_workers)

    accs = AverageMeter('Acc@1', ':6.2f')

    y_true = []
    y_pred = []

    elapsed = 0

    # Batches
    for (img, label) in tqdm(valid_loader):
        # Move to CPU, if available
        img = img.to('cpu')  # [N, 3, 224, 224]
        label = label.to('cpu')  # [N, 2]

        # Forward prop.
        with torch.no_grad():
            start = time.time()
            out = model(img)
            end = time.time()
            elapsed = elapsed + (end - start)

        acc = accuracy(out, label)

        # Keep track of metrics
        accs.update(acc)

        out = out.cpu().numpy()
        out = np.argmax(out, axis=-1).astype(np.int).tolist()
        # print('out: ' + str(out))
        y_pred += out

        label = label.cpu().numpy().astype(np.int).tolist()
        # print('label: ' + str(label))
        y_true += label

    print('Elapsed: {0:.5f} ms'.format(elapsed / num_samples * 1000))
    print('Accuracy: {0:.2f}'.format(accs.avg))

    # print('y_true: ' + str(y_true))
    # print('y_pred: ' + str(y_pred))
    # plot_confusion_matrix(y_true, y_pred, classes)
    # plt.show()


if __name__ == '__main__':
    test()
