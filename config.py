import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 224
num_classes = 2

num_samples = 90679
num_train = 89679
num_valid = 1000
# image_folder = 'data/photo'
image_folder = 'data/256x256'
anno_file = 'data/koniq10k_scores_and_distributions.csv'
data_file = 'data/data.pkl'


# Training parameters
num_workers = 8  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
