import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1,scaler=0.,expansion=1):
        ## expansion factor allows bottleneck structure for basi block. There exists high dimensional intermediate representation during training, but only low dimensional representation will be left during inference.
        
        super(BasicBlock, self).__init__()

        self.shortcut = True if stride==1 else False
        self.scaler = scaler

        self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion*planes)

        self.conv2 = nn.Conv2d(expansion*planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)


    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.scaler*F.relu(out) + (1-self.scaler)*out
        # out = nn.Dropout2d(drop_rate)(out), to be tested
        out = self.bn2(self.conv2(out))
        
        out = self.bn3(self.conv3(out))
        if self.shortcut:
           out += x 
        return out

    '''    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        #out = self.scaler*F.relu(out) + (1-self.scaler)*out
        #out = self.bn2(self.conv2(out))

        if self.shortcut:
           out = out+x 
        
        out = F.relu(out)
        return out
    '''
class ReduceNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_scaler=1):
        super(ReduceNet, self).__init__()

        self.scaler = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.in_planes = 16*width_scaler

        self.conv1 = nn.Conv2d(3, 16*width_scaler, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*width_scaler)

        self.layer1 = self._make_layer(block, 16*width_scaler, num_blocks[0], stride=1, scaler=self.scaler)
        self.layer2 = self._make_layer(block, 32*width_scaler, num_blocks[1], stride=2, scaler=self.scaler)
        self.layer3 = self._make_layer(block, 64*width_scaler, num_blocks[2], stride=2, scaler=self.scaler)
        self.linear = nn.Linear(64*width_scaler, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, scaler):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, scaler))
            self.in_planes = planes 

        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.bn1(self.conv1(x))
        
        out = F.relu(out)
        #out = self.scaler*F.relu(out) + (1-self.scaler)*out

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def reducenet20(num_classes):
    return ReduceNet(BasicBlock, [3, 3, 3],num_classes)


def reducenet56(num_classes):
    return ReduceNet(BasicBlock, [9, 9, 9],num_classes)







