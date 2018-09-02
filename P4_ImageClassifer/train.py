import argparse
import sys
import os
import torch
from torchvision import transforms, datasets
import models
from torch.nn import CrossEntropyLoss
import torch.optim as optim

parser = argparse.ArgumentParser(description='Train new network')
parser.add_argument('data_directory', nargs=1, 
    help='a directory of training images')
parser.add_argument('--save_dir', default='.', nargs=1, 
    help='a directory to save checkpoint')
parser.add_argument('--arch', default='vgg19', nargs=1, 
    help='network architecture(vgg19, resnet50, densenet121 supported)')
parser.add_argument('--learning_rate', default=0.001, nargs=1, type=float, 
    help='hyper parameter: learning rate')
parser.add_argument('--hidden_units', default=[512], nargs=1, type=int, 
    help='hypter parameter: hidden units')
parser.add_argument('--epochs', default=20, nargs=1, type=int, 
    help='hyper parameter: epochs')
parser.add_argument('--gpu', action='store_true', 
    help='train model in GPU mode')

args = parser.parse_args()

supported_archs = ['vgg19', 'resnet50']

if args.arch not in supported_archs:
    print('arch {} not supported, must be one of {}'.format(args.arch[0], supported_archs))
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

image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_directory, x), 
    data_transforms[x]) for x in ['train', 'valid', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) 
               for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = None
if args.arch == 'vgg19':
    model = models.VGG19FineTune(args.hidden_units, 
    len(image_datasets['train'].classes))
elif args.arch == 'resnet50':
    model = models.Resnet50FineTune(args.hidden_units, 
    len(image_datasets['train'].classes))

if args.gpu:
    model.cuda()

criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.classifer.parameters(), lr=args.learning_rate, momentum=0.9)
print('training model {}'.format(args.arch))
model = models.train_model(model, dataloaders, dataset_sizes, 
    device, criterion, optimizer, args.epochs)

print('{} on test set'.format(args.arch))

models.test_model(model, criterion, dataloaders, device, dataset_sizes)

checkpoint = {
    args.arch: model.state_dict(),
    'hidden_units': args.hidden_units,
    'classes': len(image_datasets['train'].classes),
    'class_to_idx': image_datasets['train'].class_to_idx,
}

torch.save(checkpoint, '{}/{}_checkpoint.pth'.format(args.save_dir, args.arch))
print('{}_checkpoint.pth saved in {}'.format(args.arch, args.save_dir))


