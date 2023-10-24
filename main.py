'''Train CIFAR with PyTorch.'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
from models.reducenet import * 

from utils import progress_bar
from torch.utils.data import DataLoader
from torchsummaryX import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

## Settings for model
parser.add_argument('-m', '--model', default='reduce20', help='Model Type.')
parser.add_argument('--expansion', default=1, type=int, help='expansion factor for bottleneck')
parser.add_argument('-ws','--width_scaler', default=1,type=int, help='network width scaler')

## Settings for data
parser.add_argument('-d', '--dataset', default='cifar10',choices=['cifar10', 'cifar100'], help='Dataset name.')
parser.add_argument('--data_dir', default='./data', help='data path')

## Settings for training
parser.add_argument('--multi_gpu', default=0, help='Model Type.')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--seed', default=666, type=int, help='number of random seed')
parser.add_argument('--epoch', default=200, type=int, help='total training epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')

## Settings for optimizer
parser.add_argument('--schedule', nargs='+', default=[100, 150, 180], type=int)
parser.add_argument('-opt', '--optmizer', default='cos',choices=['cos', 'step'], help='Dataset name.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate gamma')
parser.add_argument('-wd','--weight_decay', default=1e-4, type=float)

args = parser.parse_args()


SEED= args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

#torch.set_float32_matmul_precision('high')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
  

if args.dataset == 'cifar10':
   print('==> Preparing data cifar10')
   num_classes = 10
   #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD=(0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
   print('num_classes is {}'.format(num_classes))
   datagen = torchvision.datasets.CIFAR10

else:
   print('==> Preparing data cifar100')
   num_classes = 100
   #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD = (0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762)
   print('num_classes is {}'.format(num_classes))
   datagen = torchvision.datasets.CIFAR100


transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     #transforms.Normalize(CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD),
                    ])

transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD),
                    ])

trainset = datagen(root=args.data_dir, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

testset = datagen(root=args.data_dir, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.workers)



############# stage1:train teacher model#############
# create teacher model
if args.model == 'reduce20':
   net = reducenet20(num_classes,expansion=args.expansion,width_scaler=args.width_scaler)
   print('reducenet20 is loaded')

else:
    net = reducenet56(num_classes,expansion=args.expansion,width_scaler=args.width_scaler)
    print('reducenet56 is loaded')
print('num_classes is {}'.format(num_classes))

summary(net, torch.zeros((1, 3, 32, 32)))

#for pytorch2.0+
#net = torch.compile(net)
net = net.to(device)
if device == 'cuda' and args.multi_gpu==1:
    net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch,model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

# train teacher model
optimizer1 = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
if args.optmizer == 'cos':
   scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epoch)
else:
   scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer1, milestones=args.schedule, gamma=args.gamma)

net.scaler.data=torch.tensor(1.0)
for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch,optimizer1)
    test(epoch,net)
    scheduler1.step()


tpath = './checkpoint/model_{}_dataset_{}_expansion_{}_width_{}_teacher.pth'.format(args.model,args.dataset,args.expansion,args.width_scaler)
torch.save(net.state_dict(),tpath)

if args.model == 'reduce20':
   snet = reducenet20(num_classes,expansion=args.expansion,width_scaler=args.width_scaler)
   print('reducenet20 is loaded')

else:
    snet = reducenet56(num_classes,expansion=args.expansion,width_scaler=args.width_scaler)
    print('reducenet56 is loaded')



############# stage2: distill######################
# create student model, reusing weights from teacher.
snet = snet.to(device)
snet.load_state_dict(torch.load(tpath))
snet.scaler.data=torch.tensor(0.)
print('student model is loaded')

def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.7, T=5.0):
    KD_loss = (alpha * T * T)*nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) + (1. - alpha)*F.cross_entropy(outputs, labels) 

    return KD_loss

def distill(epoch,optimizer):
    print('\nEpoch: %d' % epoch)
    net.eval() #teacher model cannot be trained
    snet.train()
    train_loss = 0
    correct = 0
    total = 0
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        t_outputs = net(inputs)
        s_outputs = snet(inputs)
        loss = loss_fn_kd(s_outputs, targets, t_outputs) 
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = s_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# distill
optimizer2 = optim.SGD(snet.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epoch)

for epoch in range(start_epoch, start_epoch+(args.epoch)):
    distill(epoch,optimizer2)
    test(epoch,snet)
    scheduler2.step()

# spath: student model path 
spath = './checkpoint/model_{}_dataset_{}_expansion_{}_width_{}_student.pth'.format(args.model,args.dataset,args.expansion,args.width_scaler)
torch.save(snet.state_dict(),spath)
