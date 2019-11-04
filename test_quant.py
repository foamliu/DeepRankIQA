import os
import time

import torch
from torch import nn

from config import num_workers
from data_gen import DeepIQADataset
from utils import AverageMeter, load_model


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    cnt = 0
    elapsed = 0
    with torch.no_grad():
        for image, target in data_loader:
            start = time.time()
            output = model(image)
            end = time.time()
            elapsed = elapsed + (end - start)
            # loss = criterion(output, target)
            cnt += 1
            acc1 = accuracy(output, target)
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            if cnt >= neval_batches:
                break

    print('\nElapsed: {}{:.5f}{} sec'.format(bcolors.OKGREEN, elapsed, bcolors.ENDC))
    return top1


def prepare_data_loaders():
    train_dataset = DeepIQADataset('train')
    print('num_train: {}'.format(len(train_dataset)))
    valid_dataset = DeepIQADataset('valid')
    print('num_valid: {}'.format(len(valid_dataset)))

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True,
                                              num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=30, shuffle=False,
                                               num_workers=num_workers)
    return data_loader, valid_loader


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        print('.', end='')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1 = accuracy(output, target)
        top1.update(acc1[0].item(), image.size(0))
        avgloss.update(loss.item(), image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f}'.format(top1=top1))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f}'.format(top1=top1))
    return


def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end - start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %s %3.0f %s ms' % (bcolors.OKGREEN, elapsed / num_images * 1000, bcolors.ENDC))
    return elapsed


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size: {}{}{} (MB)'.format(bcolors.OKBLUE, os.path.getsize("temp.p") / 1e6, bcolors.ENDC))
    os.remove('temp.p')


def test():
    train_batch_size = 30
    eval_batch_size = 30

    scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
    scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

    data_loader, data_loader_test = prepare_data_loaders()
    criterion = nn.CrossEntropyLoss()

    filename = 'image_classification_quantized.pt'
    print('loading {}...'.format(filename))
    start = time.time()
    float_model = load_model(filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    float_model.eval()

    # Fuses modules
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inverted Residual Block: After fusion\n\n', float_model.features[1].conv)

    num_eval_batches = 10

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(float_model), scripted_float_model_file)

    # 4. Post-training static quantization
    print(bcolors.HEADER + '\nPost-training static quantization' + bcolors.ENDC)
    num_calibration_batches = 10

    myModel = load_model(filename).to('cpu')
    myModel.eval()

    # Fuse Conv, bn and relu
    myModel.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    myModel.qconfig = torch.quantization.default_qconfig
    print(myModel.qconfig)
    torch.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

    # Calibrate with the training set
    print('Calibrate with the training set')
    evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(myModel, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
          myModel.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(myModel)

    top1 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

    per_channel_quantized_model = load_model(filename)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(per_channel_quantized_model.qconfig)

    torch.quantization.prepare(per_channel_quantized_model, inplace=True)
    print('Calibrate with the training set')
    evaluate(per_channel_quantized_model, criterion, data_loader, num_calibration_batches)
    torch.quantization.convert(per_channel_quantized_model, inplace=True)
    top1 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(per_channel_quantized_model), scripted_quantized_model_file)

    # Speedup from quantization
    print(bcolors.HEADER + '\nSpeedup from quantization' + bcolors.ENDC)
    run_benchmark(scripted_float_model_file, data_loader_test)
    run_benchmark(scripted_quantized_model_file, data_loader_test)

    # 5. Quantization-aware training
    print(bcolors.HEADER + '\nQuantization-aware training' + bcolors.ENDC)
    qat_model = load_model(filename)
    qat_model.fuse_model()

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=0.0001)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    torch.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',
          qat_model.features[1].conv)

    num_train_batches = 20

    # Train and check accuracy after each epoch
    for nepoch in range(8):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            qat_model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Check the accuracy after each epoch
        quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f' % (
            nepoch, num_eval_batches * eval_batch_size, top1.avg))


if __name__ == '__main__':
    test()
