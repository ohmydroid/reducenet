import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1,scaler=1.,expansion=1):
        super(BasicBlock, self).__init__()

        self.shortcut = True if stride==1 else False
        self.scaler = scaler
         
        self.branch1 = nn.Sequential(
                                    nn.Conv2d(in_planes, expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                    nn.BatchNorm2d(expansion*planes),
                                    nn.ReLU(),
                                    nn.Conv2d(expansion*planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(planes),
                                    nn.ReLU())

        self.branch2 = nn.Sequential(
                                     nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                     nn.BatchNorm2d(planes),
                                     nn.ReLU()
                                     )


        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
                                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(planes)
                                        )

    def forward(self, x):
        out = self.scaler*self.branch1(x)+self.branch2(x)

        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) 
        return out

 
class ReduceNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_scaler=1, expansion=1):
        super(ReduceNet, self).__init__()

        self.requires_grad =nn.Parameter(torch.tensor(True), requires_grad=False)
        self.scaler = nn.Parameter(torch.tensor(1.), requires_grad=False)

        self.in_planes = 16*width_scaler

        self.conv1 = nn.Conv2d(3, 16*width_scaler, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*width_scaler)

        self.layer1 = self._make_layer(block, 16*width_scaler, num_blocks[0], stride=1, scaler=self.scaler, expansion=expansion)
        self.layer2 = self._make_layer(block, 32*width_scaler, num_blocks[1], stride=2, scaler=self.scaler, expansion=expansion)
        self.layer3 = self._make_layer(block, 64*width_scaler, num_blocks[2], stride=2, scaler=self.scaler, expansion=expansion)
        self.linear = nn.Linear(64*width_scaler, num_classes)

        self._weights_init()

    def _make_layer(self, block, planes, num_blocks, stride, scaler,expansion):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, scaler, expansion))
            self.in_planes = planes 

        return nn.Sequential(*layers)
    
    def _weights_init(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
               init.kaiming_normal_(m.weight)
            
    def _weights_freeze(self):
        for name, m in self.named_modules():     
            #if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear) :
            if isinstance(m, nn.Linear) :
               m.weight.requires_grad = False 

    def forward(self, x):
        
        out = self.bn1(self.conv1(x))
        out = F.relu(out,inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def reducenet20(num_classes,expansion):
    return ReduceNet(BasicBlock, [3, 3, 3],num_classes, expansion=expansion)


def reducenet56(num_classes,expansion):
    return ReduceNet(BasicBlock, [9, 9, 9],num_classes, expansion=expasion)
