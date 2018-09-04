import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from torchvision import models
from PIL import Image
import numpy as np

vgg19 = models.vgg19(pretrained=True)
class VGG19FineTune(nn.Module):
    def __init__(self, hidden_units):
        super(VGG19FineTune, self).__init__()
        self.features = vgg19.features
        for param in self.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=hidden_units, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_units, out_features=102, bias=True),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

resnet50 = models.resnet50(pretrained=True)
class Resnet50FineTune(nn.Module):
    def __init__(self, hidden_units):
        super(Resnet50FineTune, self).__init__()
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        for param in self.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=102, bias=True))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

densenet121 = models.densenet121(pretrained=True)
class Densenet121FineTune(nn.Module):
    def __init__(self, hidden_units):
        super(Densenet121FineTune, self).__init__()
        self.features = densenet121.features
        for param in self.parameters():
            param.require_grad = False
        # only change classifer layer
        self.classifier = nn.Linear(in_features=1024, out_features=102, bias=True)
    def forward(self, x):
        # copy from Densenet implementation in pytorch/vision
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def train_model(model, dataloaders, dataset_sizes, gpu, criterion, optimizer, num_epochs=25):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    if gpu and torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs)
                labels = Variable(labels)
                
                if gpu and torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            
            print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        print()
    
    print('best acc {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_weights)
    return model

def test_model(model, criterion, dataloaders, gpu, dataset_sizes):
    if gpu and torch.cuda.is_available():
        model.cuda()

    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
            
    for inputs, labels in dataloaders['test']:
        inputs = Variable(inputs)
        labels = Variable(labels)
                
        if gpu and torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
                
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print('test loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))

def rebuild_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model = None
    if 'vgg19' in checkpoint_file:
        model = VGG19FineTune(checkpoint['hidden_units'])
        model.load_state_dict(checkpoint['vgg19'])
    elif 'resnet50' in checkpoint_file:
        model = Resnet50FineTune(checkpoint['hidden_units'])
        model.load_state_dict(checkpoint['resnet50'])
    elif 'densenet121' in checkpoint_file:
        model = Densenet121FineTune(checkpoint['hidden_units'])
        model.load_state_dict(checkpoint['densenet121'])
    class_to_idx = checkpoint['class_to_idx']
    return model, class_to_idx

def process_image(image):
    w, h = image.size
    if w <= h:
        h = int(256 * h / w)
        w = 256
    else:
        w = int(256 * w / h)
        h = 256
    im = image.resize((w, h))
    w, h = im.size
    nw, nh = 224, 224
    left = (w - nw) / 2
    top = (h - nh) / 2
    right = (w + nw) / 2
    down = (h + nh) / 2
    im = im.crop((left, top, right, down))
    np_im = np.array(im)
    np_im = np_im / 255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - means) / stds
    return torch.from_numpy(np_im.transpose(2, 0, 1))

def predict(image_path, model, class_to_idx, is_gpu, topk=5):
    im = Image.open(image_path)
    im = process_image(im)
    im = im.view(1, *im.shape)
    im = Variable(im)
    model.eval()
    im = im.float()
    if is_gpu and torch.cuda.is_available():
        model.cuda()
        im = im.cuda()
    output = model(im)
    output = F.softmax(output, dim=1)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    probs, indices = output.topk(topk)
    indices = indices[0]
    classes = [idx_to_class[index.item()] for index in indices]
    probs = probs.cpu()
    return probs.data.numpy()[0], classes
