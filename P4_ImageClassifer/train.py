import argparse
import sys
import os
import torch
from torchvision import transforms, datasets
import common
import torchvision.models as models
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from collections import OrderedDict
import torch.nn as nn

parser = argparse.ArgumentParser(description='Train new network')
parser.add_argument('data_directory', nargs=1, 
    help='a directory of training images')
parser.add_argument('--save_dir', default=['.'], nargs=1, 
    help='a directory to save checkpoint')
parser.add_argument('--arch', default=['vgg19'], nargs=1, 
    help='network architecture(vgg19, resnet50, densenet121 supported)')
parser.add_argument('--learning_rate', default=[0.001], nargs=1, type=float, 
    help='hyper parameter: learning rate')
parser.add_argument('--hidden_units', default=[500], nargs=1, type=int, 
    help='hypter parameter: hidden units')
parser.add_argument('--epochs', default=[20], nargs=1, type=int, 
    help='hyper parameter: epochs')
parser.add_argument('--gpu', action='store_true', 
    help='train model in GPU mode')

args = parser.parse_args()

data_directory = args.data_directory[0]
save_dir = args.save_dir[0]
arch = args.arch[0]
learning_rate = args.learning_rate[0]
hidden_units = args.hidden_units[0]
epochs = args.epochs[0]
gpu = args.gpu

print('save to {}'.format(save_dir))
print('training data {}'.format(data_directory))
print('learning rate: {}'.format(learning_rate))
print('hidden units: {}'.format(hidden_units))
print('epochs: {}'.format(epochs))
print('training on GPU: {}'.format(gpu))

supported_archs = ['vgg19', 'resnet50', 'densenet121']

if arch not in supported_archs:
    print('arch {} not supported, must be one of {}'.format(arch, supported_archs))
    sys.exit(1)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory, x), 
    data_transforms[x]) for x in ['train', 'valid', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) 
               for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

model = None
if arch == 'vgg19':
    model = common.VGG19FineTune(hidden_units)
elif arch == 'resnet50':
    model = common.Resnet50FineTune(hidden_units)
elif arch == 'densenet121':
    model = common.Densenet121FineTune(hidden_units)

optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)
criterion = CrossEntropyLoss()

print('training model {}'.format(arch))
model = common.train_model(model, dataloaders, dataset_sizes, gpu, criterion, optimizer, epochs)

print('{} on test set'.format(arch))

common.test_model(model, criterion, dataloaders, gpu, dataset_sizes)

checkpoint = {
    arch: model.state_dict(),
    'hidden_units': hidden_units,
    'classes': len(image_datasets['train'].classes),
    'class_to_idx': image_datasets['train'].class_to_idx,
}

torch.save(checkpoint, '{}/{}_checkpoint.pth'.format(save_dir, arch))
print('{}_checkpoint.pth saved in {}'.format(arch, save_dir))
