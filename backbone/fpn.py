import torch
import torch.nn.functional as F
from torch import nn
import resnet as r

class FPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.bottom_up = r.ResNet50()

        # Top layer
        self.toplayer = nn.Conv2d(2048,256,kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)             


    def upsample(self, x, y):

        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode="nearest")

    def forward(self,x):
        
        c1,c2,c3,c4=self.bottom_up(x)
        
        p5 = self.toplayer(c4)
        p4 = self.upsample(p5,self.latlayer1(c4))+self.latlayer1(c4)
        p3 = self.upsample(p4,self.latlayer2(c3))+self.latlayer2(c3)
        p2 = self.upsample(p3,self.latlayer3(c2))+self.latlayer3(c2)
    
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5


        