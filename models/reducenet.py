import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1,scaler=torch.tensor(1.0),expansion=1,use_lora=False):
        super(BasicBlock, self).__init__()

        self.scaler = scaler 

        self.use_lora=use_lora
         
        self.branch1 = nn.Sequential(
                                    nn.Conv2d(in_planes, expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                    nn.BatchNorm2d(expansion*planes),
                                    nn.ReLU(),
                                    nn.Conv2d(expansion*planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
                                    )

        self.branch2 = nn.Sequential(
                                     nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                     )
        if use_lora:
           self.lora_branch = nn.Sequential(
                                    nn.Conv2d(in_planes, expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                    nn.Conv2d(expansion*planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                                    )

       
        self.fuse = nn.Sequential(nn.BatchNorm2d(planes),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(planes))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
                                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(planes)
                                        )

    def forward(self, x):
        out1 = self.scaler*self.branch1(x)

        if  self.use_lora:
            out2 = self.branch2(x) + self.lora_branch(x)
        else:
            out2 = self.branch2(x)

        out = out1+out2
        out = self.fuse(out)
        out = out + self.shortcut(x)
        return out

 
class ReduceNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_scaler=1, expansion=1,use_lora=True):
        super(ReduceNet, self).__init__()
        
        self.scaler = nn.Parameter(torch.tensor(1.), requires_grad=False)

        self.in_planes = 16*width_scaler

        self.conv1 = nn.Sequential(nn.Conv2d(3,self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_planes),
                                   nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(block, 16*width_scaler, num_blocks[0], stride=1, scaler=self.scaler, expansion=expansion,use_lora=use_lora)
        self.layer2 = self._make_layer(block, 32*width_scaler, num_blocks[1], stride=2, scaler=self.scaler, expansion=expansion,use_lora=use_lora)
        self.layer3 = self._make_layer(block, 64*width_scaler, num_blocks[2], stride=2, scaler=self.scaler, expansion=expansion,use_lora=use_lora)
        self.linear = nn.Linear(64*width_scaler, num_classes)

        #self._weights_init()

    def _make_layer(self, block, planes, num_blocks, stride, scaler,expansion,use_lora):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, scaler, expansion,use_lora))
            self.in_planes = planes 

        return nn.Sequential(*layers)

    def _weights_init(self):
        for name, m in self.named_modules():

         for stage_name, stage in self.named_children():
            for named_block, block in stage.named_children():
                for named_layer, layer in block.named_children():
                    if named_layer=='branch2':
                       init.kaiming_normal_(layer.weight)

         for stage_name, stage in self.named_children():
            if stage_name not in ['linear','conv1']:
               for named_block, block in stage.named_children():
                    for named_layer, layer in block.named_children():
                        if named_layer in ['branch2','lora']:
                           for named_op, op in named_layer.named_children():
                               if isinstance(op, nn.Conv2d):
                                  init.kaiming_normal_(op.weight)
                               
                               elif isinstance(op, nn.BatchNorm2d):
                                    init.constant_(op.weight, 1)
                                    if op.bias is not None:
                                       init.constant_(op.bias, 0.0001)
                                    init.constant_(op.running_mean, 0)


    def _weights_freeze(self):
        for stage_name, stage in self.named_children():
            if stage_name in ['linear','conv1']:
               stage.parameters().requires_grad = False
            else:
                for named_block, block in stage.named_children():
                    for named_layer, layer in block.named_children():
                        if named_layer=='fuse':
                           layer.parameters().requires_grad = False
            

    def forward(self, x):
        
        out = self.conv1(x)

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
