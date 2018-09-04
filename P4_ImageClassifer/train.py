import argparse
import sys
import os
import torch
from torchvision import transforms, datasets
import models
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
parser.add_argument('--hidden_units', default=[4096], nargs=1, type=int, 
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

supported_archs = ['vgg19', 'resnet50']

if arch not in supported_archs:
    print('arch {} not supported, must be one of {}'.format(arch, supported_archs))
    sys.exit(1)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory, x), 
    data_transforms[x]) for x in ['train', 'valid', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) 
               for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

model = None
if arch == 'vgg19':
    # model = models.VGG19FineTune(hidden_units, len(image_datasets['train'].classes))
    model = models.resnet50(pretrained=True)
    num_features = model.classifier[0].in_features
elif arch == 'resnet50':
    # model = models.Resnet50FineTune(hidden_units, len(image_datasets['train'].classes))
    model = models.resnet50(pretrained=True)
    num_features = model.classifier.in_features

for param in model.parameters():
    param.require_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(num_features, 2000, bias=True)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(2000, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc3', nn.Linear(hidden_units, 102)),
    ('output', nn.Softmax(dim=1))
]))

model.classifier = classifier

criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)
print('training model {}'.format(arch))
model = models.train_model(model, dataloaders, dataset_sizes, gpu, criterion, optimizer, epochs)

print('{} on test set'.format(arch))

models.test_model(model, criterion, dataloaders, gpu, dataset_sizes)

checkpoint = {
    arch: model.state_dict(),
    'hidden_units': hidden_units,
    'classes': len(image_datasets['train'].classes),
    'class_to_idx': image_datasets['train'].class_to_idx,
}

torch.save(checkpoint, '{}/{}_checkpoint.pth'.format(save_dir, arch))
print('{}_checkpoint.pth saved in {}'.format(arch, save_dir))


