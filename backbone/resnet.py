import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channel, out_channel, stride, padding):
# 3*3 convolution
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, dilation =1, groups=1, bias=False)


def conv1x1(in_channel, out_channel, stride, padding):
# 1*1 convolution
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=padding, dilation =1, groups=1, bias=False)

class basic_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, down=None):
        super().__init__()
        if down is not None:
            # stride 값을 늘려서 downsampling 시킴
            self.conv1=conv3x3(inplanes,planes,stride=2, padding=1)  
        else:
            self.conv1= conv3x3(inplanes, planes, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace = True)
        if planes==64:
            self.downsample = nn.Conv2d(inplanes,planes,kernel_size=3, stride=1, padding=1)
        else:
            self.downsample = nn.Conv2d(inplanes,planes,kernel_size=3, stride=2, padding=1)
        self.down =down
    
    def forward(self,x):
        ## 정의한 layer를 진행시킨다
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.down is not None:
            residual=self.downsample(residual)
        out=out+residual
        out=self.relu(out)

        return out

class bottleneck_block(nn.Module):

    expansion: int=4

    def __init__(self, inplanes, planes, stride=1, groups=1, down=None):
        super().__init__()
        if down is not None:
            # stride 값을 늘려서 downsampling 시킴
            self.conv1=conv1x1(inplanes,planes,stride=2, padding=1)  
        else:
            self.conv1= conv1x1(inplanes, planes, stride=1, padding=1) 
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes,planes*self.expansion,stride=1, padding=1)
        self.bn3= nn.BatchNorm2d(planes*self.expansion)
        self.relu= nn.ReLU(inplace=True)  #inplace=true -> the input is modified directly

        if inplanes==64:
            self.downsample = nn.Conv2d(inplanes,planes,kernel_size=3, stride=1, padding=1)
        else:
            self.downsample = nn.Conv2d(inplanes,planes,kernel_size=3, stride=2, padding=1)
        self.down =down


    def forward(self, x):
          ## 정의한 layer를 진행시킨다.
        residual =x 
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv3(out)
        out=self.bn3(out)

        if self.down is not None:
            residual=self.downsample(residual)
        out=out+residual
        out=self.relu(out)

       
        return out

    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, zero_init_residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, dilation =1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## The outputs of bottleneck block is actually num_block*expansion
        self.layer1 = self.make_layer(block, 64, 64, num_blocks[0])
        self.layer2 = self.make_layer(block, 64, 128, num_blocks[1])
        self.layer3 = self.make_layer(block, 128, 256, num_blocks[2])
        self.layer4 = self.make_layer(block, 256, 512, num_blocks[3])
        

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, 10)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, bottleneck_block):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, basic_block):
                    nn.init.constant_(m.bn2.weight, 0)
                    




    def make_layer(self, block, inplanes, planes, num_block): 
        layers = []

        if planes==64:
            tmp=None
        else:
            tmp=1
        layers.append(block(inplanes,planes, down=tmp))
        for i in range(1,num_block):
            layers.append(block(planes,planes,down=None))

        return nn.Sequential(*layers)        
        
         
        
       


    def forward(self, x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        c1=self.layer1(out)
        c2=self.layer2(out)
        c3=self.layer3(out)
        c4=self.layer4(out)

        
        #out = self.avgpool(out)
        #out = Tensor.flatten(out,1)
        #out = self.linear(out)
    
        return c1, c2, c3, c4

def ResNet18():
    return ResNet(basic_block, [2, 2, 2, 2])
def ResNet34():
    return ResNet(basic_block, [3, 4, 6, 3])
def ResNet50():
    return ResNet(bottleneck_block, [3, 4, 6, 3])
def ResNet101():
    return ResNet(bottleneck_block, [3, 4, 23, 3])
def ResNet152():
    return ResNet(bottleneck_block, [3, 8, 36, 3])