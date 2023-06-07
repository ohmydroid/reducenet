'''Train CIFAR10 with PyTorch.'''
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

## Settings for model
parser.add_argument('-g', '--multi_gpu', default=0, help='Model Type.')
parser.add_argument('-m', '--model', default='reduce20', help='Model Type.')

## Settings for data
parser.add_argument('-d', '--dataset', default='cifar10',choices=['cifar10', 'cifar100'], help='Dataset name.')

## Settings for fast training

parser.add_argument('--workers', default=4, type=int, help='number of workers')

parser.add_argument('--expansion', default=1, type=int, help='expansion')
parser.add_argument('--seed', default=666, type=int, help='number of random seed')
## Settings for optimizer 
parser.add_argument('--schedule', nargs='+', default=[100, 150, 180], type=int)
parser.add_argument('-opt', '--optmizer', default='cos',choices=['cos', 'step'], help='Dataset name.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate gamma')
parser.add_argument('-wd','--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch', default=200, type=int, help='total training epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
args = parser.parse_args()


SEED= args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

#torch.set_float32_matmul_precision('high')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch


if args.dataset == 'cifar10':

   num_classes = 10
   print('num_classes is {}'.format(num_classes))
   #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD=(0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
    
else:
    num_classes = 100
    #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD = (0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762)



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
  




if args.dataset == 'cifar10':

   # Data
   print('==> Preparing data cifar10')
   trainset = torchvision.datasets.CIFAR10(root='/home/onedroid/apps/projects/pytorh-vision/data', train=True, download=False, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

   testset = torchvision.datasets.CIFAR10(root='/home/onedroid/apps/projects/pytorh-vision/data', train=False, download=False, transform=transform_test)
   testloader = torch.utils.data.DataLoader(testset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.workers)

else:
    # Data
   print('==> Preparing data..')
   trainset = torchvision.datasets.CIFAR100(root='/home/onedroid/apps/projects/pytorh-vision/data', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
   testset = torchvision.datasets.CIFAR100(root='/home/onedroid/apps/projects/pytorh-vision/data', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(testset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.workers)


if args.model == 'reduce20':
   net = reducenet20(num_classes,expansion=args.expansion)
   print('reducenet20 is loaded')
elif args.model == 'res20':
   net = resnet20(num_classes)
   print('resnet20 is loaded')
else:
    net = reducenet56(num_classes,expansion=args.expansion)
    print('reducenet56 is loaded')
print('num_classes is {}'.format(num_classes))

summary(net, torch.zeros((1, 3, 32, 32)))
#net = torch.compile(net)
net = net.to(device)
if device == 'cuda': 
    if args.multi_gpu==1:
       net = torch.nn.DataParallel(net)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}_{}_{}_{}_{}_ckpt.pth'.format(args.dataset,args.model,args.expansion, args.weight_decay,args.epoch))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch,optimizer,scaler=1.0):
    net.scaler.data=torch.tensor(scaler)

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


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if epoch == args.epoch:
       print(net.scaler.cpu().detach().numpy())


'''
optimizer0 = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
if args.optmizer == 'cos':
   scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=args.epoch)
else:
   scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer0, milestones=args.schedule, gamma=args.gamma)

for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch,optimizer0,scaler=0.)
    test(epoch)
    scheduler0.step()

'''

optimizer1 = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
if args.optmizer == 'cos':
   scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epoch)
else:
   scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer1, milestones=args.schedule, gamma=args.gamma)

for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch,optimizer1)
    test(epoch)
    scheduler1.step()


torch.save(net.state_dict(),'./checkpoint/expansion_{}_teacher.pth'.format(args.expansion))
#net.load_state_dict(torch.load('./checkpoint/expansion_{}_teacher.pth'.format(args.expansion)))

net._weights_freeze()
#net._weights_init()

#net = net.to('cpu')
#summary(net, torch.zeros((1, 3, 32, 32)))
#net = torch.compile(net)
#net = net.to(device)

optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()) , lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
if args.optmizer == 'cos':
   scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epoch)
else:
   scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer2, milestones=args.schedule, gamma=args.gamma)

for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch,optimizer2,scaler=0.)
    test(epoch)
    scheduler2.step()
