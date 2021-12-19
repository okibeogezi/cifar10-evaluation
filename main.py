'''Train CIFAR10 with PyTorch.'''

from time import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', '-e', default=5, type=int,
                    help='number of training epochs')
parser.add_argument('--model', '-m', choices=['shufflenet_v2', 'mobilenet_v2'], required=True, type=str,
                    help='model to train')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best val accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('Preparing data...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val_or_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

root = '~/Downloads/'


def split_train():
    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    return random_split(
        train_set, [45000, 5000], generator=torch.Generator().manual_seed(42))[0]


def split_val():
    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_val_or_test)
    return random_split(
        train_set, [45000, 5000], generator=torch.Generator().manual_seed(42))[1]


train_loader = torch.utils.data.DataLoader(
    split_train(), batch_size=128, shuffle=False, num_workers=2)
val_loader = torch.utils.data.DataLoader(
    split_val(), batch_size=128, shuffle=False, num_workers=2)

test_set = torchvision.datasets.CIFAR10(
    root=root, train=False, download=True, transform=transform_val_or_test)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
model_dict = {
    'shufflenet_v2': ShuffleNetV2(1),
    'mobilenet_v2': MobileNetV2(),
}

model = model_dict[args.model]
model_name = model.__class__.__name__
print('Building {} model...'.format(model_name))

model = model.to(device)
if device == 'cuda':
    model = nn.DataParallel(model)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('Resuming from checkpoint...')
    assert os.path.isdir(
        'checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoints/ckpt-{}.pth'.format(model_name)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    training_loss, val_loss = 0, 0
    training_correct, val_correct = 0, 0
    training_total, val_total = 0, 0

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        _, predicted = outputs.max(1)
        training_total += targets.size(0)
        training_correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)'
                     % (training_loss/(batch_idx+1), 100.*training_correct/training_total, training_correct, training_total))

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        val_total += targets.size(0)
        val_correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(val_loader), 'Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
                     % (val_loss/(batch_idx+1), 100.*val_correct/val_total, val_correct, val_total))

    training_acc = 100. * (training_correct / training_total)
    val_acc = 100. * (val_correct / val_total)

    training_info = {
        'training_loss': training_loss,
        'training_accuracy': training_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
    }

    file_path = './logs/{}-{}-{}.json'.format(model_name, int(time()), epoch)

    return training_info, file_path


def test():
    test_loss = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(start_epoch, args.epochs):
    training_info, file_path = train(epoch)

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    f = open(file_path, mode='w')
    f.write(json.dumps(training_info, indent=2))
    f.close()

    val_accuracy = training_info['val_accuracy']
    if val_accuracy > best_acc:
        print('Saving new best model...')
        state = {
            'model': model.state_dict(),
            'accuracy': val_accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        checkpoint_path = './checkpoints/ckpt-{}.pth'.format(model_name)
        torch.save(state, checkpoint_path)
        best_acc = val_accuracy
    elif best_acc > 0:
        persisted_state = torch.load(checkpoint_path)
        persisted_state['epoch'] = epoch
        torch.save(persisted_state, checkpoint_path)

    scheduler.step()

test()
